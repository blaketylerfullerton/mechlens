"""Confirm our SAE features are the same features Neuronpedia labelled.

Run this *before* importing the explanation export. A wrong mapping is the
failure mode that looks completely fine: you get fluent, confident labels for
the wrong dictionary, and nothing anywhere throws.

The check is empirical rather than nominal. Neuronpedia's feature endpoint
returns the top activating text chunks for a feature, with that feature's
activation on every token. So we can take their tokens, push them through our
model and our SAE, and ask whether we reproduce their numbers:

    argmax    the token we peak on must be the token they peak on.
              A mismatch means a different dictionary, full stop.
    corr      Pearson across all positions, ~1.0. Confirms it is the same
              feature rather than two features that happen to like one token.
    scale     our peak against their maxValue. This is the one that catches
              the *same layer at a different L0* — that case still correlates
              well but comes out at the wrong magnitude.

Usage, from `backend/`:

    python scripts/verify_neuronpedia_mapping.py
    python scripts/verify_neuronpedia_mapping.py --layers 0,12,25 -n 3
    python scripts/verify_neuronpedia_mapping.py --layers 20 --features 12082

Costs one gemma load plus one SAE per layer touched, so keep --layers short.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch

# `app` is a package under backend/; this script is run as a file, so put
# backend/ on the path the same way tests/conftest.py does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_cache import get_model  # noqa: E402
from app.sae_cache import DEFAULT_WIDTH, get_sae, neuronpedia_id  # noqa: E402

API = "https://www.neuronpedia.org/api/feature/{model}/{source_set}/{index}"
TIMEOUT_S = 30

# Verdict thresholds. Deliberately strict: the whole point is to catch a
# near-miss, and a genuine match comes out at corr > 0.99 with scale ~1.00.
CORR_PASS = 0.95
SCALE_LO, SCALE_HI = 0.9, 1.1


# --------------------------------------------------------------------------
# their side
# --------------------------------------------------------------------------


@dataclass
class Example:
    """One of Neuronpedia's top activating chunks for a feature."""

    tokens: list[str]  # raw sentencepiece pieces, e.g. '▁DOG'
    values: list[float]  # this feature's activation on each token
    max_value: float
    max_index: int
    explanation: str | None


def fetch(model_id: str, source_set: str, index: int) -> Example:
    url = API.format(model=model_id, source_set=source_set, index=index)
    response = requests.get(url, timeout=TIMEOUT_S)
    response.raise_for_status()
    data = response.json()

    activations = data.get("activations") or []
    if not activations:
        raise LookupError(f"{source_set}:{index} has no recorded activations")

    # Sort rather than trust the order — we want the strongest example, since
    # a weak one gives a flat activation profile that correlates with anything.
    top = max(activations, key=lambda a: a.get("maxValue") or 0.0)

    explanations = data.get("explanations") or []
    return Example(
        tokens=top["tokens"],
        values=[float(v) for v in top["values"]],
        max_value=float(top["maxValue"]),
        max_index=int(top["maxValueTokenIndex"]),
        explanation=explanations[0]["description"] if explanations else None,
    )


# --------------------------------------------------------------------------
# our side
# --------------------------------------------------------------------------


def to_ids(model, pieces: list[str]) -> list[int]:
    """Token pieces -> ids, without going back through the tokenizer's parser.

    Re-tokenizing ''.join(pieces) is the tempting shortcut and it is wrong:
    the tokenizer re-segments the text and you end up comparing two different
    sequences position by position, which shows up as a mediocre correlation
    that looks like a mapping problem but is not.
    """
    ids = model.tokenizer.convert_tokens_to_ids(pieces)
    unknown = [p for p, i in zip(pieces, ids) if i is None]
    if unknown:
        raise LookupError(f"tokenizer does not know {unknown[:5]}")
    return ids


def our_activations(model, sae, layer: int, ids: list[int], prepend_bos: bool) -> np.ndarray:
    """This feature's activation at every position of `ids`, our numbers.

    Returns a vector aligned to `ids` — the BOS column, when we prepend one, is
    dropped before returning so both alignments are directly comparable.
    """
    bos = model.tokenizer.bos_token_id
    row = ([bos] + ids) if prepend_bos else ids
    tokens = torch.tensor([row], device=model.cfg.device)

    hook = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        # stop_at_layer skips the rest of the stack — we only need this site.
        _, cache = model.run_with_cache(tokens, names_filter=hook, stop_at_layer=layer + 1)
        # The SAE is fp32 (see sae_cache) while the model runs bf16.
        resid = cache[hook][0].float().to(sae.W_enc.device)
        acts = sae.encode(resid)  # [seq, d_sae]

    return acts.cpu().numpy()[1:] if prepend_bos else acts.cpu().numpy()


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


@dataclass
class Result:
    layer: int
    index: int
    corr: float
    scale: float
    argmax_ok: bool
    prepended_bos: bool
    n_tokens: int
    explanation: str | None

    @property
    def verdict(self) -> str:
        if not self.argmax_ok or self.corr < CORR_PASS:
            return "FAIL"
        if not SCALE_LO <= self.scale <= SCALE_HI:
            return "SCALE"  # right feature, suspect variant
        return "ok"


def correlate(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Pearson, with the degenerate cases spelled out rather than warned about."""
    if ours.std() < 1e-9 or np.std(theirs) < 1e-9:
        return 0.0  # one side is flat; nothing to correlate
    return float(np.corrcoef(ours, theirs)[0, 1])


def check(model, sae, layer: int, index: int, example: Example) -> Result:
    ids = to_ids(model, example.tokens)
    theirs = np.array(example.values[: len(ids)], dtype=np.float64)

    # Neuronpedia's chunks are corpus slices and may carry no BOS, while our
    # capture always prepends one. Gemma's position 0 is its own weather system
    # (see phase 2), so try both alignments rather than guessing which
    # convention this chunk followed.
    candidates: list[Result] = []
    for prepend in (False, True):
        column = our_activations(model, sae, layer, ids, prepend)[:, index]
        ours = column.astype(np.float64)[: len(theirs)]
        result = Result(
            layer=layer,
            index=index,
            corr=correlate(ours, theirs),
            scale=float(ours.max() / example.max_value) if example.max_value else 0.0,
            argmax_ok=int(ours.argmax()) == example.max_index,
            prepended_bos=prepend,
            n_tokens=len(theirs),
            explanation=example.explanation,
        )
        candidates.append(result)

    # Both alignments routinely correlate at ~1.000, because dropping BOS
    # rescales this feature's activations without reshaping them. Picking on
    # correlation alone therefore picks on noise, and lands on the no-BOS run
    # about half the time — where Gemma, robbed of its attention sink, comes in
    # 30-45% low and the scale check cries mismatch at a perfectly good feature.
    # So: correlation decides only when it actually separates the two.
    return max(candidates, key=lambda r: (round(r.corr, 2), -abs(r.scale - 1.0)))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def pick_features(sae, n: int, seed: int) -> list[int]:
    """Uniform over the dictionary — an unbiased sample of the mapping.

    Sampling from a trace's top features instead would only tell us about the
    features this model likes, which is not the population we are checking.
    """
    return np.random.default_rng(seed).choice(sae.cfg.d_sae, size=n, replace=False).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", default="0,12,20", help="comma-separated (default: 0,12,20)")
    p.add_argument("-n", "--n-features", type=int, default=2, help="features per layer")
    p.add_argument("--features", help="explicit indices, e.g. '12082'; implies one layer")
    p.add_argument("--width", default=DEFAULT_WIDTH)
    p.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    model = get_model()

    results: list[Result] = []
    for layer in layers:
        model_id, source_set = neuronpedia_id(layer, args.width)
        sae = get_sae(layer, args.width, args.device)
        indices = (
            [int(x) for x in args.features.split(",")]
            if args.features
            else pick_features(sae, args.n_features, args.seed + layer)
        )
        print(f"\nlayer {layer}  ->  {model_id}/{source_set}")

        for index in indices:
            try:
                example = fetch(model_id, source_set, index)
            except (requests.RequestException, LookupError) as exc:
                print(f"  #{index:<6} skipped — {exc}")
                continue

            result = check(model, sae, layer, index, example)
            results.append(result)
            bos = "with BOS" if result.prepended_bos else "no BOS"
            print(
                f"  #{index:<6} corr {result.corr:+.3f}  scale {result.scale:.2f}  "
                f"argmax {'✓' if result.argmax_ok else '✗'}  "
                f"[{result.n_tokens} tok, {bos}]  {result.verdict}"
            )
            if result.explanation:
                print(f"           {result.explanation[:90]}")

    summarise(results)


def summarise(results: list[Result]) -> None:
    if not results:
        raise SystemExit("\nnothing was checked — every feature was skipped")

    verdicts = [r.verdict for r in results]
    n_ok = verdicts.count("ok")
    print(
        f"\n{n_ok}/{len(results)} features matched  "
        f"(median corr {np.median([r.corr for r in results]):+.3f}, "
        f"median scale {np.median([r.scale for r in results]):.2f})"
    )

    if n_ok == len(results):
        print("mapping confirmed — safe to import the explanation export")
        return

    if "FAIL" in verdicts:
        print(
            "⚠ MISMATCH. Our features are not their features. Check the width and the\n"
            "  canonical variant before importing anything — labels would be for the\n"
            "  wrong dictionary and nothing downstream would notice."
        )
    else:
        print(
            "⚠ correlations are fine but the scale is off. Likely the right layer at a\n"
            "  different L0 — check which average_l0_* the canonical id resolves to."
        )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
