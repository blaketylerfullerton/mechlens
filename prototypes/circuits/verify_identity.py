"""Does the replacement model explain the same model mechlens traced?

Three questions, in order of how badly a wrong answer would invalidate CIR-01:

1. Do the token ids match?  mechlens saved token ids; circuit-tracer must consume
   those exact ids, BOS included, with the same position indexing.
2. Is the target excluded from its own prefix?
3. Do the logits agree?  The replacement model is a HookedTransformer with
   transcoder machinery attached. If a plain forward pass through it disagrees
   with the plain forward pass mechlens does, then every graph built on it
   explains a different model, and that is a no-go.

Exits non-zero on any failure. Tolerances are stated up front and the *measured*
error is always reported, because "passed" without a number is not evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import common

# Tolerances must be expressed in units the dtype can actually represent. An
# absolute 1e-2 on a logit near 7.5 is finer than bfloat16's resolution there
# (one ULP is 0.03125), so it would fail on two bit-identical-to-within-rounding
# models. Logits are therefore compared in ULPs of the dtype they were computed
# in; probabilities, which are derived in fp32 after a softmax, keep an absolute
# bound.
LOGIT_TOL_ULPS = 2
PROB_TOL = 1e-3


def bf16_ulps_apart(a: float, b: float) -> float:
    """How many representable bfloat16 steps separate a and b.

    0 means the two values are the same bfloat16 number; 1 means they are
    adjacent and no closer agreement is expressible.
    """
    ta = torch.tensor([a], dtype=torch.bfloat16).view(torch.int16).item()
    tb = torch.tensor([b], dtype=torch.bfloat16).view(torch.int16).item()
    return abs(ta - tb)


def scores(logits: torch.Tensor, target_token_id: int) -> dict:
    """Target token's scores at the last prefix position."""
    last = logits[0, -1].float()
    logprobs = torch.log_softmax(last, dim=-1)
    return {
        "logit": last[target_token_id].item(),
        "log_probability": logprobs[target_token_id].item(),
        "probability": logprobs[target_token_id].exp().item(),
        "argmax_token_id": int(last.argmax()),
    }


def check_tokens(model, rec: dict, results: list) -> None:
    prefix = rec["prefix_token_ids"]
    bos = model.tokenizer.bos_token_id

    results.append({
        "check": "bos_present_at_position_0",
        "expected": bos,
        "actual": prefix[0],
        "ok": prefix[0] == bos,
        "note": "mechlens indexes positions including BOS; so must the prefix",
    })

    results.append({
        "check": "target_excluded_from_own_prefix",
        "target_position": rec["target_position"],
        "prefix_len": len(prefix),
        "ok": len(prefix) == rec["target_position"],
        "note": "prefix is token_ids[:j], target is token_ids[j]",
    })

    # Round-tripping the decoded prefix is what we must NOT rely on. Checking it
    # anyway tells the report whether decode/re-encode would have been lossy here.
    decoded = model.tokenizer.decode(prefix)
    reencoded = model.tokenizer(decoded).input_ids
    # Informational, not a gate: the pipeline never re-tokenizes decoded text, so
    # a lossy round-trip here is a hazard avoided rather than a failure. It is
    # recorded because it is the evidence for why ids are passed directly.
    results.append({
        "check": "decode_reencode_roundtrip_is_lossless",
        "ok": True,
        "informational": True,
        "lossless": reencoded == prefix,
        "prefix_len": len(prefix),
        "reencoded_len": len(reencoded),
        "decoded": decoded,
        "note": ("if lossless is False, re-tokenizing decoded trace text would have "
                 "produced different ids than the model actually ran"),
    })

    # The library zeroes transcoder activations at BOS (ReplacementModel sets
    # zero_positions = slice(0, 1) for non-gemma-3-it models). Position 0 will
    # therefore carry no feature nodes; record it so the UI does not read that
    # absence as "nothing happened here".
    results.append({
        "check": "bos_transcoder_zeroing_documented",
        "zero_positions": str(getattr(model, "zero_positions", None)),
        "ok": True,
        "note": "features are suppressed at these positions by design, not measurement",
    })


def check_baseline(replacement, rec: dict, results: list) -> dict:
    """Replacement model vs. a plain HookedTransformer built the mechlens way."""
    from transformer_lens import HookedTransformer

    prefix = rec["prefix_token_ids"]
    target = rec["target_token_id"]
    tokens = torch.tensor([prefix], device=common.DEVICE)

    with torch.no_grad():
        repl = scores(replacement(tokens), target)

    # backend/app/model_cache.py uses from_pretrained_no_processing: no LayerNorm
    # folding, no weight centering. Same call here or the comparison is rigged.
    plain = HookedTransformer.from_pretrained_no_processing(
        "gemma-2-2b", device=common.DEVICE, dtype=common.DTYPE
    )
    plain.eval()
    with torch.no_grad():
        base = scores(plain(tokens), target)
    del plain
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    d_logit = abs(repl["logit"] - base["logit"])
    d_prob = abs(repl["probability"] - base["probability"])
    ulps = bf16_ulps_apart(repl["logit"], base["logit"])
    results.append({
        "check": "baseline_agreement_with_mechlens_model",
        "replacement": repl,
        "mechlens_plain": base,
        "abs_logit_error": d_logit,
        "logit_error_bf16_ulps": ulps,
        "abs_probability_error": d_prob,
        "logit_tolerance_ulps": LOGIT_TOL_ULPS,
        "probability_tolerance": PROB_TOL,
        "argmax_matches": repl["argmax_token_id"] == base["argmax_token_id"],
        "ok": ulps <= LOGIT_TOL_ULPS and d_prob <= PROB_TOL,
        "note": ("both from_pretrained_no_processing, bfloat16, same device, "
                 "identical token ids; logit error reported in bfloat16 ULPs "
                 "because absolute error below one ULP is not representable"),
    })
    return {"replacement": repl, "mechlens_plain": base}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True, help="JSON from export_trace.py")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    rec = json.loads(args.export.read_text())
    model, load_s = common.load_replacement_model()
    print(f"model + transcoders loaded in {load_s:.1f}s\n")

    results: list[dict] = []
    check_tokens(model, rec, results)
    check_baseline(model, rec, results)

    for r in results:
        print(f"[{'ok ' if r['ok'] else 'FAIL'}] {r['check']}")
        for k, v in r.items():
            if k not in ("check", "ok"):
                print(f"         {k}: {v}")
        print()

    failed = [r["check"] for r in results if not r["ok"]]
    record = common.provenance(
        export=str(args.export),
        source_trace=rec.get("trace_id"),
        tolerances={"logit_bf16_ulps": LOGIT_TOL_ULPS, "probability": PROB_TOL},
        results=results,
        failed=failed,
        peak_memory_mb=common.peak_memory_mb(),
    )
    out = args.out or common.OUT / f"identity-{args.export.stem}.json"
    common.save_json(out, record)
    print(f"wrote {out}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nall identity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
