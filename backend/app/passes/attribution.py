"""Phase 5: decompose each layer's residual stream into attributable edges.

For every `(layer, position)`, `resid_post = resid_pre + attn_out + mlp_out` —
not an approximation but the literal structure of the residual stream, so the
decomposition is exact (modulo bf16 rounding) rather than something to
estimate with gradients or patching:

  resid   one edge, carried over from the same position at `layer - 1`
          (already on disk — `resid_pre(L) == resid_post(L-1)`)
  attn    one edge per source position, from a per-head, per-source
          decomposition of this layer's attention output
  mlp     one edge, the MLP output at this position — already position-wise,
          so there is nothing further to decompose it across

Unlike the SAE and label passes, this needs a fresh forward pass — the
residual sidecar only ever held `hook_resid_post`. Unlike the lens, it is not
enough to read `W_U`: attention needs its own pattern and per-head values, so
this is the one pass that takes the model *and* re-runs it.

Two non-obvious things make the per-source attention decomposition correct
rather than merely plausible, both checked against the installed
TransformerLens source rather than assumed:

  the sandwich norm   Gemma 2 applies a post-attention RMSNorm (`ln1_post`)
                      to the *summed* attention output before it reaches the
                      residual stream. RMSNorm's scale and gain are computed
                      from that whole vector, not per source, so — computed
                      once from the total and applied identically to every
                      per-source term — it distributes over the sum exactly.
                      Push it through the wrong way (normalizing each source
                      individually) and the pieces do not add back up.
  grouped-query attn  `hook_v` sits at `n_key_value_heads` granularity for a
                      GQA model, not `n_heads` — TransformerLens expands V to
                      match the attention pattern's head count later, inside
                      `calculate_z_scores`, not at the hook. Skipping the
                      expansion here silently pairs the wrong value vector
                      with each pattern head.

`b_O` (the attention output projection's bias) is a per-layer constant, not
tied to any source position — `Edge.source` has no "no source" slot for it,
so it is left out of the edges and only its norm is reported, the same way
`resid_norm`'s only proof against the edges is a checked gap, not silence
about the difference. See design.md, Decisions 3 and 6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformer_lens import HookedTransformer

from ..capture import RESID_HOOK
from ..schema import Edge, NodeRef, PassRecord, Trace

DEFAULT_TOP_K = 8

# Below this an "edge" is just softmax noise on a causally-masked future
# position — keeping it would mean every position has edges pointing at
# tokens that come after it.
MIN_EDGE_WEIGHT = 1e-8


@dataclass
class AttributionPass:
    """Fills LayerState.edges with resid / attn / mlp contributions."""

    name: str = field(default="attribution", init=False)
    top_k: int = DEFAULT_TOP_K
    layers: list[int] | None = None  # None = every layer
    verbose: bool = True

    # Injectable the way SAEPass.saes and LogitLensPass.model are: tests stand
    # in a tiny model, a long-lived server passes the handle it already holds.
    model: HookedTransformer | None = None

    def run(self, trace: Trace, residuals: np.ndarray) -> PassRecord:
        model = self.model if self.model is not None else _default_model()
        _check_compatible(trace, model)

        layers = self.layers if self.layers is not None else list(range(trace.n_layers))
        n_positions = len(trace.steps)
        t0 = time.time()

        cache = _run_with_hooks(model, trace, layers)

        recon_gaps: list[float] = []  # block-level: resid_pre+attn+mlp vs captured resid_post
        decomp_gaps: list[float] = []  # attn-only: my per-source sum vs the model's hook_attn_out
        coverages: list[float] = []
        bias_fracs: list[float] = []

        for layer in layers:
            attn_out, mlp_out, contrib, bias, decomp_gap = _decompose_layer(
                model, cache, layer, n_positions
            )
            decomp_gaps.append(decomp_gap)

            for pos in range(n_positions):
                edges: list[Edge] = []

                if layer > 0:
                    edges.append(
                        Edge(
                            source=NodeRef(layer=layer - 1, position=pos),
                            weight=trace.steps[pos].layers[layer - 1].resid_norm,
                            kind="resid",
                        )
                    )

                weights = torch.linalg.norm(contrib[pos], dim=-1)  # [n_positions] (source axis)
                total = float(weights.sum())
                kept_idx = [
                    s for s in torch.argsort(weights, descending=True).tolist()
                    if weights[s] > MIN_EDGE_WEIGHT
                ][: self.top_k]
                kept = sorted(kept_idx)  # stored source-position order, not weight order
                for s in kept:
                    edges.append(
                        Edge(
                            source=NodeRef(layer=layer, position=s),
                            weight=float(weights[s]),
                            kind="attn",
                        )
                    )
                kept_norm = sum(float(weights[s]) for s in kept)
                coverages.append(kept_norm / total if total > 0 else 1.0)
                bias_norm = float(torch.linalg.norm(bias[pos]))
                attn_norm = float(torch.linalg.norm(attn_out[pos]))
                if attn_norm > 0:
                    bias_fracs.append(bias_norm / attn_norm)

                edges.append(
                    Edge(
                        source=NodeRef(layer=layer, position=pos),
                        weight=float(torch.linalg.norm(mlp_out[pos])),
                        kind="mlp",
                    )
                )

                trace.steps[pos].layers[layer].edges = edges

            if layer > 0:
                # residuals is always CPU numpy (mmap'd from the sidecar);
                # attn_out/mlp_out live wherever the model does.
                prev = torch.from_numpy(np.array(residuals[:n_positions, layer - 1])).to(
                    attn_out.device
                )
                actual = torch.from_numpy(np.array(residuals[:n_positions, layer])).to(
                    attn_out.device
                )
                recon = prev + attn_out + mlp_out
                gap = torch.linalg.norm(actual - recon, dim=-1) / torch.linalg.norm(
                    actual, dim=-1
                ).clamp_min(1e-12)
                recon_gaps.append(float(gap.max()))

            if self.verbose:
                print(
                    f"\rlayer {layer:>2}  attn top-{self.top_k} coverage "
                    f"{coverages[-1]:.3f}  reconstruction gap "
                    f"{recon_gaps[-1] if recon_gaps else float('nan'):.1e}",
                    end="" if layer != layers[-1] else "\n",
                    flush=True,
                )

        return PassRecord(
            name=self.name,
            params={
                "top_k": self.top_k,
                "hook_attn_out": "hook_attn_out",
                "hook_mlp_out": "hook_mlp_out",
                "hook_pattern": "attn.hook_pattern",
                "hook_v": "attn.hook_v",
                "model": model.cfg.model_name,
                "n_layers_processed": len(layers),
            },
            stats={
                # Should read ~0 by construction (see module docstring); a
                # real gap means a hook or a sign is wrong, not that top_k
                # truncated too much — this is computed from the untruncated
                # sums.
                "reconstruction_max_rel_gap": max(recon_gaps) if recon_gaps else float("nan"),
                # A second, narrower check: does *my* per-source recomposition
                # of attn_out agree with the model's own hook_attn_out? Keeps
                # "the decomposition's own math is right" separate from "the
                # block-level identity holds", the same way the lens pass
                # keeps final-layer agreement separate from crossover.
                "attn_decomposition_max_rel_gap": max(decomp_gaps) if decomp_gaps else float("nan"),
                # Honest like the SAE pass's explained_variance: expected < 1.0
                # on prompts where attention spreads across many positions,
                # not a target to force to 1.0.
                "attn_topk_coverage": float(np.mean(coverages)) if coverages else float("nan"),
                "attn_topk_coverage_min": float(np.min(coverages)) if coverages else float("nan"),
                # b_O is not attributable to a source (see module docstring);
                # this says how much of attn_out it typically accounts for,
                # so the gap it leaves in the edges is legible rather than a
                # silent shortfall.
                "attn_bias_mean_norm_frac": float(np.mean(bias_fracs)) if bias_fracs else 0.0,
            },
            elapsed_s=time.time() - t0,
        )


# --------------------------------------------------------------------------
# the forward pass
# --------------------------------------------------------------------------


def _hook_names(layer: int) -> tuple[str, str, str, str]:
    return (
        f"blocks.{layer}.hook_attn_out",
        f"blocks.{layer}.hook_mlp_out",
        f"blocks.{layer}.attn.hook_pattern",
        f"blocks.{layer}.attn.hook_v",
    )


def _run_with_hooks(model: HookedTransformer, trace: Trace, layers: list[int]) -> dict:
    """One forward pass over the trace's own tokens, caching just enough.

    The token sequence is rebuilt from `trace.steps`, not re-tokenized from
    `trace.text` — it already carries the exact ids capture saw, BOS
    included, and re-tokenizing risks a different id stream (whitespace,
    special tokens) for what is supposed to be the identical sequence.

    A single pass computes every layer regardless of `layers` — the block
    stack is sequential, so there is no way to obtain layer 20's attention
    without running layers 0-19 first — but `names_filter` still bounds
    memory to the requested layers: pattern/v/attn_out/mlp_out together cost
    a few MB per layer, not the ~800MB a full `use_attn_result` cache would.
    """
    token_ids = [step.token.token_id for step in trace.steps]
    tokens = torch.tensor([token_ids], device=model.cfg.device, dtype=torch.long)

    wanted = {name for layer in layers for name in _hook_names(layer)}
    names_filter = lambda name: name in wanted  # noqa: E731

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)
    return cache


def _decompose_layer(
    model: HookedTransformer, cache: dict, layer: int, n_positions: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """One layer's attn_out/mlp_out plus a per-(dest, source) attn decomposition.

    Returns `(attn_out, mlp_out, contrib, bias, decomp_gap)`:
      attn_out  [pos, d_model]        the model's own hook_attn_out
      mlp_out   [pos, d_model]        the model's own hook_mlp_out
      contrib   [pos, pos, d_model]   contrib[d, s] = source s's share of
                                      attn_out at destination d
      bias      [pos, d_model]        b_O's share, per destination — not a
                                      source, not stored as an edge
      decomp_gap  max relative gap between attn_out and
                  sum_s contrib[d, s] + bias[d], over all d
    """
    names = _hook_names(layer)
    attn_out = cache[names[0]][0, :n_positions].float()
    mlp_out = cache[names[1]][0, :n_positions].float()
    pattern = cache[names[2]][0, :, :n_positions, :n_positions].float()  # [heads, dest, src]
    v = cache[names[3]][0, :n_positions].float()  # [pos, kv_heads, d_head]

    # W_O/b_O/ln1_post.w are model parameters and carry requires_grad=True
    # even though this pass never backpropagates; detach so building this
    # tensor does not build an autograd graph nobody uses.
    attn = model.blocks[layer].attn
    W_O = attn.W_O.float().detach()  # [heads, d_head, d_model]
    b_O = attn.b_O.float().detach()  # [d_model]

    repeat = getattr(attn, "repeat_kv_heads", 1)
    v_expanded = v.repeat_interleave(repeat, dim=1) if repeat > 1 else v  # [pos, heads, d_head]

    # A silent mismatch here would pair the wrong value vector with each
    # pattern head — fail loudly rather than let an off-by-a-group-size error
    # through as a plausible-looking einsum broadcast.
    if v_expanded.shape[1] != pattern.shape[0]:
        raise ValueError(
            f"layer {layer}: expanded hook_v has {v_expanded.shape[1]} heads but "
            f"hook_pattern has {pattern.shape[0]} — grouped-query-attention expansion "
            f"(repeat_kv_heads={repeat}) did not line up"
        )

    # head_out[s, h] = this source position's value vector for head h,
    # projected through that head's slice of W_O — the per-head, per-source
    # piece attention mixes over destinations.
    head_out = torch.einsum("shd,hde->she", v_expanded, W_O)  # [src, heads, d_model]
    # contrib[d, s] = sum_h pattern[h, d, s] * head_out[s, h] — every source's
    # share of the *pre-post-norm* attention output at destination d.
    contrib = torch.einsum("hds,she->dse", pattern, head_out)  # [dest, src, d_model]
    raw_attn_out = contrib.sum(dim=1) + b_O  # [dest, d_model]

    ln1_post = getattr(model.blocks[layer], "ln1_post", None)
    if ln1_post is not None and not isinstance(ln1_post, torch.nn.Identity):
        gain = ln1_post.w.float().detach()
        rms = torch.sqrt(raw_attn_out.pow(2).mean(dim=-1, keepdim=True) + ln1_post.eps)
        scale = gain / rms  # [dest, d_model] — same multiplier for every source's term
        contrib = contrib * scale.unsqueeze(1)
        bias = b_O.unsqueeze(0) * scale
        recomposed = raw_attn_out * scale
    else:
        bias = b_O.unsqueeze(0).expand(n_positions, -1)
        recomposed = raw_attn_out

    gap = torch.linalg.norm(attn_out - recomposed, dim=-1) / torch.linalg.norm(
        attn_out, dim=-1
    ).clamp_min(1e-12)

    return attn_out, mlp_out, contrib, bias, float(gap.max())


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def _check_compatible(trace: Trace, model: HookedTransformer) -> None:
    """Refuse combinations where the re-run forward pass would not match
    the residuals already on disk.

    LayerNorm folding is not refused, unlike the SAE pass: folding does not
    change the block's own arithmetic (`resid_post = resid_pre + attn_out +
    mlp_out` holds unconditionally), so it does not threaten this pass's
    correctness the way it threatens SAE features fitted on unfolded
    activations. Same reasoning the lens pass already uses.
    """
    if trace.residuals is None:
        raise ValueError(f"trace {trace.trace_id} has no residuals attached")

    if trace.residuals.hook != RESID_HOOK:
        raise ValueError(
            f"trace {trace.trace_id} captured {trace.residuals.hook}; attribution "
            f"reconstructs resid_post from resid_pre + attn_out + mlp_out, so the "
            f"sidecar has to hold {RESID_HOOK}"
        )

    if (trace.n_layers, trace.d_model) != (model.cfg.n_layers, model.cfg.d_model):
        raise ValueError(
            f"trace {trace.trace_id} is {trace.n_layers}x{trace.d_model} but "
            f"{model.cfg.model_name} is {model.cfg.n_layers}x{model.cfg.d_model}"
        )

    if trace.model != model.cfg.model_name:
        raise ValueError(
            f"trace {trace.trace_id} was captured with {trace.model}, but attribution "
            f"was handed {model.cfg.model_name} — different attention weights decompose "
            f"these residuals differently"
        )


def _default_model() -> HookedTransformer:
    from ..model_cache import get_model

    return get_model()
