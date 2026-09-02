"""Phase 1: token-by-token generation with the residual stream captured.

The loop here is deliberately *not* `model.generate`. Every step is an explicit
forward pass we own, which is what makes the later phases possible: an
intervention (steering, ablation, patching) is a hook added inside this loop,
and its effect shows up in every subsequent step's residuals.

Cost note: each step re-runs the whole prefix (no KV cache), so generation is
O(n^2) in sequence length. For a 2B model and the ~20-100 token traces this
tool is built for, that is a few seconds — and it keeps the loop simple enough
that adding interventions later does not mean fighting a cache. Revisit with
HookedTransformerKeyValueCache if traces get long.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .schema import (
    LayerState,
    LogitSummary,
    TokenInfo,
    TokenStep,
    TopToken,
    Trace,
)

# Which residual-stream site we snapshot. resid_post is the layer's output —
# what the next layer reads and what Gemma Scope's SAEs are trained on.
RESID_HOOK = "hook_resid_post"

# (layer, hook_fn) — hook_fn has TransformerLens's usual (tensor, HookPoint) ->
# tensor signature and is registered on that layer's RESID_HOOK for the
# duration of one generate_trace call. This is the intervention point the
# module docstring above refers to.
Intervention = tuple[int, Callable]

DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_TOP_K = 10


@dataclass
class CaptureResult:
    """A trace plus the raw tensor it references.

    Kept apart because the tensor never goes into the JSON: `store.save_trace`
    writes the two halves side by side and attaches the `ResidualRef` that ties
    them together — until then `trace.residuals` is None, because a tensor that
    is not on disk yet has no path to point at.
    """

    trace: Trace
    residuals: np.ndarray  # [n_tokens, n_layers, d_model], float32
    hook: str = RESID_HOOK


def _hook_name(layer: int) -> str:
    return f"blocks.{layer}.{RESID_HOOK}"


def _token_text(model: HookedTransformer, token_id: int) -> str:
    """Decode one token, preserving its leading whitespace."""
    return model.tokenizer.decode([token_id])


def logit_summary(
    model: HookedTransformer,
    logits_row: torch.Tensor,  # [d_vocab] for a single position
    top_k: int,
    chosen_id: int | None,
) -> LogitSummary:
    """Summarise one position's next-token distribution.

    Public because the logit lens (phase 4) calls it too: a lens summary and a
    capture summary have to be built the same way for the last layer's lens to
    be comparable against the model's real output, which is the lens's whole
    correctness test. Two parallel softmax-and-topk implementations would make
    that test prove nothing.
    """
    # float32 for the softmax: entropy over 256k bf16 entries loses real
    # precision in the tail, and the tail is most of the entropy.
    row = logits_row.float()
    probs = torch.softmax(row, dim=-1)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())

    values, indices = probs.topk(top_k)
    top = [
        TopToken(
            token_id=int(i),
            text=_token_text(model, int(i)),
            logit=float(row[i]),
            prob=float(p),
        )
        for p, i in zip(values.tolist(), indices.tolist())
    ]

    chosen = None
    if chosen_id is not None:
        chosen = TopToken(
            token_id=chosen_id,
            text=_token_text(model, chosen_id),
            logit=float(row[chosen_id]),
            prob=float(probs[chosen_id]),
        )

    return LogitSummary(top_k=top, entropy=entropy, chosen=chosen)


def generate_trace(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    stop_at_eos: bool = True,
    trace_id: str | None = None,
    intervention: Intervention | None = None,
) -> CaptureResult:
    """Greedily generate from `prompt`, capturing every layer at every token.

    Returns the populated `Trace` (minus the fields later phases fill in) and
    the `[n_tokens, n_layers, d_model]` residual tensor it points at. Positions
    cover the prompt as well as the generation: the prompt's representations
    are where most attribution work ends up pointing.

    `intervention`, when given, is registered on the model for the duration
    of this call (see `Intervention`), so the residuals captured below —
    and everything downstream of them, generation included — are the
    *steered* activations. `None` (the default) is exactly today's
    unsteered behavior.
    """
    cfg = model.cfg
    n_layers, d_model = cfg.n_layers, cfg.d_model

    tokens = model.to_tokens(prompt)  # [1, seq]; BOS prepended
    n_prompt = tokens.shape[1]

    # One row per position we will ever hold; trimmed at the end if we stop early.
    residuals = np.empty((n_prompt + max_new_tokens, n_layers, d_model), dtype=np.float32)
    summaries: dict[int, LogitSummary] = {}

    eos_id = model.tokenizer.eos_token_id
    names_filter = lambda name: name.endswith(RESID_HOOK)  # noqa: E731

    filled = 0  # positions whose residuals are already stored
    stop_reason = "max_tokens"
    t0 = time.time()

    if intervention is not None:
        layer, hook_fn = intervention
        model.add_hook(_hook_name(layer), hook_fn)

    try:
        for step in range(max_new_tokens + 1):
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

            seq = tokens.shape[1]

            # Causal attention means a position's residual never changes once
            # computed, so each pass only has to bank what is new: the whole prompt
            # on the first pass, one token on every pass after it.
            for layer in range(n_layers):
                acts = cache[_hook_name(layer)][0]  # [seq, d_model]
                residuals[filled:seq, layer] = acts[filled:seq].float().cpu().numpy()

            # Every position predicts a next token. For positions inside the
            # sequence the "chosen" token is simply whatever sits at position+1.
            for pos in range(filled, seq - 1):
                summaries[pos] = logit_summary(
                    model, logits[0, pos], top_k, chosen_id=int(tokens[0, pos + 1])
                )
            filled = seq

            # The last position is the one actually driving generation, so its
            # summary waits until we know whether we are appending its argmax.
            last_logits = logits[0, -1]
            next_id = int(last_logits.argmax())
            del cache

            at_budget = step == max_new_tokens
            hit_eos = stop_at_eos and next_id == eos_id
            if at_budget or hit_eos:
                if hit_eos:
                    stop_reason = "eos"
                summaries[seq - 1] = logit_summary(model, last_logits, top_k, chosen_id=None)
                break

            summaries[seq - 1] = logit_summary(model, last_logits, top_k, chosen_id=next_id)
            tokens = torch.cat(
                [tokens, last_logits.new_tensor([[next_id]], dtype=tokens.dtype)], dim=1
            )
    finally:
        if intervention is not None:
            model.reset_hooks()

    elapsed = time.time() - t0

    n_tokens = filled
    residuals = residuals[:n_tokens]
    token_ids = tokens[0].tolist()
    token_texts = model.to_str_tokens(tokens[0])

    steps = [
        TokenStep(
            step=pos,
            token=TokenInfo(
                position=pos,
                token_id=token_ids[pos],
                text=token_texts[pos],
                source="prompt" if pos < n_prompt else "generated",
            ),
            logits=summaries[pos],
            layers=[
                LayerState(
                    layer=layer,
                    resid_norm=float(np.linalg.norm(residuals[pos, layer])),
                )
                for layer in range(n_layers)
            ],
        )
        for pos in range(n_tokens)
    ]

    trace = Trace(
        trace_id=trace_id or uuid.uuid4().hex[:12],
        model=cfg.model_name,
        device=str(model.cfg.device),
        dtype=str(cfg.dtype),
        normalization=cfg.normalization_type,
        n_layers=n_layers,
        d_model=d_model,
        prompt=prompt,
        completion=model.to_string(tokens[0, n_prompt:]),
        n_prompt_tokens=n_prompt,
        n_generated_tokens=n_tokens - n_prompt,
        stop_reason=stop_reason,
        elapsed_s=elapsed,
        steps=steps,
    )

    return CaptureResult(trace=trace, residuals=residuals)
