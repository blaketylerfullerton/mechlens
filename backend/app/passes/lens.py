"""Phase 4: decode every layer's residual stream through the unembed.

The logit lens asks the model's own output head what it would say if it had to
answer *now*, at layer L, instead of at layer 25. Run it at every depth and you
can watch an answer crystallise:

    L14  ' a'      2.1%
    L18  ' Paris' 11.4%      <- the answer first appears
    L21  ' Paris' 38.0%
    L25  ' Paris' 71.9%      <- and this is the model's real output

Cheap, because the residuals are already on disk. Three things make it wrong in
ways that still look plausible, so all three are guarded:

  softcap      Gemma 2 caps its logits at 30 with `cap * tanh(x / cap)`, and
               TransformerLens applies that in `forward`, *after* `unembed` --
               not inside it. Skip it and top-1 still looks about right while
               every probability and entropy is off. See phase4.md.
  ln_final     Applied via `model.ln_final`, never reimplemented: Gemma's
               `(1 + w)` RMSNorm scaling is already folded into the weight by
               TransformerLens's conversion, and the module upcasts to float32
               before taking the scale.
  the identity Layer `n_layers - 1` is `resid_post` of the last block, which is
               exactly what `forward` hands to `ln_final`. So its lens must
               reproduce the captured next-token distribution, position for
               position. `final_layer_agreement` on the PassRecord says whether
               it did, and anything below 1.0 means one of the above is wrong.

Unlike the SAE and label passes, this one needs the model -- `W_U` is
2304 x 256_000, too big to sit in a sidecar next to every trace. It is the one
pass that pays gemma's ~10s load.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utilities.activation_functions import apply_softcap

from ..capture import RESID_HOOK, logit_summary
from ..schema import LogitLens, PassRecord, Trace

DEFAULT_TOP_K = 5


@dataclass
class LogitLensPass:
    """Fills LayerState.logit_lens for every (token, layer)."""

    name: str = field(default="lens", init=False)
    top_k: int = DEFAULT_TOP_K
    layers: list[int] | None = None  # None = every layer
    verbose: bool = True

    # Injectable the way SAEPass.saes and LabelsPass.store are: tests stand in
    # tiny-stories-1M on CPU, and a long-lived server passes the handle it
    # already holds instead of going through the cache.
    model: HookedTransformer | None = None

    def run(self, trace: Trace, residuals: np.ndarray) -> PassRecord:
        model = self.model if self.model is not None else _default_model()
        _check_compatible(trace, model)

        layers = self.layers if self.layers is not None else list(range(trace.n_layers))
        n_positions = len(trace.steps)
        t0 = time.time()

        # Per-layer curves, indexed by position in `layers`.
        entropy_by_layer: list[float] = []
        agreement_by_layer: list[float] = []
        echo_by_layer: list[float] = []

        # What the model actually ended up predicting at each position. The
        # crystallisation curve is "how often does depth L already agree with
        # this", so it is read from the trace, not recomputed.
        final_top1 = [step.logits.top_k[0].token_id for step in trace.steps]
        # And what token is sitting *at* each position, for the echo curve.
        current = [step.token.token_id for step in trace.steps]

        for layer in layers:
            summaries = self._decode_layer(model, residuals, layer, n_positions)

            for pos, summary in enumerate(summaries):
                trace.steps[pos].layers[layer].logit_lens = LogitLens(
                    top_k=summary.top_k, entropy=summary.entropy
                )

            entropy_by_layer.append(float(np.mean([s.entropy for s in summaries])))
            agreement_by_layer.append(
                sum(s.top_k[0].token_id == final_top1[pos] for pos, s in enumerate(summaries))
                / n_positions
            )
            echo_by_layer.append(
                sum(s.top_k[0].token_id == current[pos] for pos, s in enumerate(summaries))
                / n_positions
            )

        elapsed = time.time() - t0
        stats = _crystallisation_stats(
            layers, entropy_by_layer, agreement_by_layer, echo_by_layer
        )
        stats.update(_final_layer_check(trace, layers))

        if self.verbose:
            _report(trace, layers, stats, elapsed)

        return PassRecord(
            name=self.name,
            params={
                "hook": RESID_HOOK,
                "top_k": self.top_k,
                "model": model.cfg.model_name,
                # 0.0 means the model has no cap; recorded either way, because
                # "was the softcap applied?" is the first question to ask of a
                # lens whose probabilities look wrong.
                "output_logits_soft_cap": float(model.cfg.output_logits_soft_cap or 0.0),
                "n_layers_decoded": len(layers),
            },
            stats=stats,
            elapsed_s=elapsed,
        )

    def _decode_layer(
        self,
        model: HookedTransformer,
        residuals: np.ndarray,
        layer: int,
        n_positions: int,
    ) -> list:
        """Every position of one layer, decoded through ln_final and W_U.

        One layer at a time on purpose: [n_positions, d_vocab] is 32MB of
        float32 for a 31-token gemma trace, where all 26 layers at once would
        be 825MB.

        float32 goes in, not a bf16 downcast. The sidecar holds an exact
        widening of the bf16 activations capture saw, and RMSNorm upcasts to
        float32 before taking the scale -- so feeding float32 reproduces the
        forward pass exactly, while casting down first would discard precision
        the norm is about to ask for.
        """
        x = torch.from_numpy(np.ascontiguousarray(residuals[:n_positions, layer])).to(
            model.cfg.device
        )

        with torch.no_grad():
            logits = model.unembed(model.ln_final(x))
            # The step everyone forgets. Identity when the model has no cap.
            logits = apply_softcap(logits, model.cfg.output_logits_soft_cap)

        return [
            logit_summary(model, logits[pos], self.top_k, chosen_id=None)
            for pos in range(n_positions)
        ]


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


# Two logits that land on the same bf16 value are a real occurrence, not a
# rounding curiosity: gemma's late-layer logits sit around 28 where bf16 steps
# by 0.125, so near-ties collapse to exact ties often enough to see on a
# 16-token trace. `torch.topk` then orders them however it likes, and the two
# call sites -- capture indexing [1, seq, vocab], the lens indexing
# [seq, vocab] -- need not agree. Same distribution, different argmax.
TIE_ATOL = 1e-5


def _final_layer_check(trace: Trace, layers: list[int]) -> dict[str, float]:
    """The correctness test, recorded so a saved trace can be judged on sight.

    resid_post of the last block is exactly what `forward` feeds to ln_final,
    so that layer's lens is not an approximation of the model's output -- it is
    the model's output, recomputed. Anything short of a match means the
    softcap, the norm or the hook site is wrong.

    "Match" is deliberately not "the same top-1 token id". Under bf16 the top
    two logits are sometimes bit-identical, and which one topk returns first is
    then arbitrary -- a strict id comparison reports a failure that is really a
    coin flip, and sends the reader hunting for a softcap bug that is not
    there. A position counts as agreeing when the ids match *or* the two top-1
    probabilities are equal, and the ties are counted separately rather than
    quietly absorbed.

    `max_entropy_delta` is the check that does not care about any of this:
    entropy is taken over the full vocabulary, so it is blind to tie-breaking
    and still moves sharply if the softcap is skipped.

    Returns nothing when the last layer was not decoded (`--layers 0-5`): a
    missing check is better than a check that silently passes on the wrong
    layer.
    """
    last = trace.n_layers - 1
    if last not in layers:
        return {}

    matched = exact = ties = 0
    max_prob_delta = 0.0
    max_entropy_delta = 0.0

    for step in trace.steps:
        lens = step.layers[last].logit_lens
        if lens is None:
            continue
        got, want = lens.top_k[0], step.logits.top_k[0]

        delta = abs(got.prob - want.prob)
        max_prob_delta = max(max_prob_delta, delta)
        max_entropy_delta = max(max_entropy_delta, abs(lens.entropy - step.logits.entropy))

        if got.token_id == want.token_id:
            exact += 1
            matched += 1
        elif delta <= TIE_ATOL:
            ties += 1
            matched += 1

    n = len(trace.steps)
    return {
        "final_layer_agreement": matched / n,
        "final_layer_exact_top1": exact / n,
        "final_layer_argmax_ties": float(ties),
        # Not asserted to be zero: capture ran the matmul on CUDA in bf16 and
        # the lens may re-run it elsewhere. It should be ~1e-3, not ~1e-1.
        "final_layer_max_prob_delta": max_prob_delta,
        "final_layer_max_entropy_delta": max_entropy_delta,
    }


def _crystallisation_stats(
    layers: list[int],
    entropy_by_layer: list[float],
    agreement_by_layer: list[float],
    echo_by_layer: list[float],
) -> dict[str, float | list[float]]:
    """The "done when" of phase 4, as numbers rather than an eyeball.

    `top1_agreement_by_layer` rising toward 1.0 *is* the crystallisation: the
    fraction of positions where depth L already holds the answer the model
    finally gives. `crossover_layer` is the first depth past half, i.e. roughly
    where the model makes up its mind.

    One caveat on reading the tail of that curve: it is a plain token-id
    comparison, so an argmax tie (see TIE_ATOL) knocks a position off it even
    though the distribution matched. `top1_agreement_by_layer[-1]` can
    therefore sit at 0.94 on a trace where `final_layer_agreement` is a clean
    1.0 — the latter is the tie-aware number and the one to trust. The curve is
    descriptive; the check is the check.

    `echo_by_layer` is the correction to the naive reading of that. Early
    residuals are dominated by the token embedding, so the lens decodes them
    back to the token *already at* this position rather than to a prediction —
    on gemma-2-2b, layers 0-9 confidently "predict" the current token at 60-85%.
    That is why entropy is not the headline: it is *low* early for a reason
    that has nothing to do with the model being sure of an answer, then rises
    through the middle layers before falling again. A reader watching entropy
    alone would conclude the model gets less certain with depth, which is
    backwards. Agreement is the honest curve; echo says how much of the early
    part of it to discount.
    """
    crossover = next(
        (layer for layer, a in zip(layers, agreement_by_layer) if a >= 0.5),
        -1.0,  # never crosses; -1 rather than None so the stats stay numeric
    )
    return {
        "entropy_by_layer": entropy_by_layer,
        "top1_agreement_by_layer": agreement_by_layer,
        "echo_by_layer": echo_by_layer,
        "crossover_layer": float(crossover),
        "agreement_first_layer": agreement_by_layer[0],
        "agreement_last_layer": agreement_by_layer[-1],
    }


def _report(trace: Trace, layers: list[int], stats: dict, elapsed: float) -> None:
    agreement = stats.get("final_layer_agreement")
    print(
        f"logit lens over {len(layers)} layers x {len(trace.steps)} tokens "
        f"in {elapsed:.1f}s"
    )
    # The tail of this curve is a plain id comparison, so it can read a point
    # or two below the tie-aware check on the next line. Not a contradiction.
    print(
        f"  top-1 agreement with the final answer "
        f"{stats['agreement_first_layer']:.0%} -> {stats['agreement_last_layer']:.0%} "
        f"| crossover at layer {stats['crossover_layer']:.0f}"
    )
    if agreement is None:
        print("  final layer not decoded — correctness check skipped")
    elif agreement == 1.0:
        ties = int(stats["final_layer_argmax_ties"])
        note = f", {ties} by an argmax tie" if ties else ""
        print(
            f"  layer {trace.n_layers - 1} reproduces the model's output on "
            f"{len(trace.steps)}/{len(trace.steps)} positions{note} "
            f"(max prob delta {stats['final_layer_max_prob_delta']:.1e})"
        )
    else:
        print(
            f"  ⚠ layer {trace.n_layers - 1} agrees with the model on only "
            f"{agreement:.1%} of positions — softcap or norm is wrong, see phase4.md"
        )


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def _check_compatible(trace: Trace, model: HookedTransformer) -> None:
    """Refuse the combinations that produce plausible nonsense.

    Note what is *not* checked, unlike passes/sae.py: `trace.normalization`.
    LayerNorm folding moves resid_post off the distribution Gemma Scope's SAEs
    were fitted on, so the SAE pass has to refuse it -- but folding is exactly
    `RMSNormPre(x) @ W_U_folded == RMSNorm(x) @ W_U`, the same arithmetic. The
    lens only needs ln_final and W_U to come from one model, which they do.
    """
    if trace.residuals is None:
        raise ValueError(f"trace {trace.trace_id} has no residuals attached")

    if trace.residuals.hook != RESID_HOOK:
        raise ValueError(
            f"trace {trace.trace_id} captured {trace.residuals.hook}; the lens needs "
            f"{RESID_HOOK} — the final layer's identity against the model's own "
            f"output is what makes it verifiable, and only resid_post has it"
        )

    if (trace.n_layers, trace.d_model) != (model.cfg.n_layers, model.cfg.d_model):
        raise ValueError(
            f"trace {trace.trace_id} is {trace.n_layers}x{trace.d_model} but "
            f"{model.cfg.model_name} is {model.cfg.n_layers}x{model.cfg.d_model}"
        )

    if trace.model != model.cfg.model_name:
        raise ValueError(
            f"trace {trace.trace_id} was captured with {trace.model}, but the lens "
            f"was handed {model.cfg.model_name} — a different unembed decodes these "
            f"residuals into a different vocabulary"
        )


def _default_model() -> HookedTransformer:
    from ..model_cache import get_model

    return get_model()
