"""Phase 2: encode each residual through its layer's Gemma Scope SAE.

For every (token, layer) this writes the top-k active features into the trace,
plus `l0` — the true number of features that fired. The full sparse vector is
16384 mostly-zero entries; the top 16 of them are what anyone ever looks at.

Two diagnostics come along for free, because a wrong answer here looks
plausible and only the numbers give it away:

  L0                  ~100 for the canonical 16k SAEs. 16384 means the JumpReLU
                      threshold was skipped; 0 means the wrong activation site.
  explained variance  how much of the residual decode(encode(x)) recovers.
                      ~0.7-0.95 is healthy; near 0 means the SAE and the
                      activations do not belong together.

Both are recorded on the PassRecord so a saved trace can be judged without
re-running anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..capture import RESID_HOOK
from ..sae_cache import DEFAULT_WIDTH, RELEASE, SAE_HOOK, load_layers, pick_device
from ..schema import Feature, PassRecord, Trace

DEFAULT_TOP_K = 16

# Position 0 is BOS. Gemma Scope SAEs see an activation there unlike anything
# in their training data and respond with huge, meaningless features — a known
# artifact, but one that would drag every average off if left in.
BOS_POSITION = 0


@dataclass
class SAEPass:
    """Fills LayerState.features and LayerState.l0."""

    name: str = field(default="sae", init=False)
    width: str = DEFAULT_WIDTH
    top_k: int = DEFAULT_TOP_K
    layers: list[int] | None = None  # None = every layer
    device: str | None = None
    verbose: bool = True

    # Pre-loaded SAEs keyed by layer. Skips the loader — for a long-lived
    # server that holds them, and for tests that stand in a fake.
    saes: dict[int, object] | None = None

    def run(self, trace: Trace, residuals: np.ndarray) -> PassRecord:
        _check_compatible(trace)

        device = self.device or pick_device()
        layers = self.layers if self.layers is not None else list(range(trace.n_layers))
        saes = self.saes if self.saes is not None else load_layers(layers, self.width, device)

        n_tokens = len(trace.steps)
        l0_by_layer: list[float] = []
        ev_by_layer: list[float] = []
        t0 = time.time()

        for layer in layers:
            sae = saes[layer]
            # [n_tokens, d_model] — one layer's slice for the whole sequence.
            # np.array (a copy) rather than asarray: a memory-mapped trace is
            # read-only, and torch.from_numpy warns on non-writable buffers.
            x = torch.from_numpy(np.array(residuals[:, layer])).to(device)

            with torch.no_grad():
                acts = sae.encode(x)  # [n_tokens, d_sae], JumpReLU-gated
                recon = sae.decode(acts)

            active = acts > 0
            l0 = active.sum(dim=-1)  # [n_tokens]
            ev = _explained_variance(x, recon)  # [n_tokens]

            k = min(self.top_k, acts.shape[-1])
            values, indices = acts.topk(k, dim=-1)

            for pos in range(n_tokens):
                state = trace.steps[pos].layers[layer]
                state.l0 = int(l0[pos])
                # topk pads with zeros once a token has fewer than k active
                # features; those are not features, they are absence.
                state.features = [
                    Feature(index=int(i), activation=float(v))
                    for v, i in zip(values[pos].tolist(), indices[pos].tolist())
                    if v > 0
                ]

            keep = slice(1, None) if n_tokens > 1 else slice(None)  # drop BOS
            l0_by_layer.append(float(l0[keep].float().mean()))
            ev_by_layer.append(float(ev[keep].mean()))

            if self.verbose:
                print(
                    f"\rlayer {layer:>2}  L0 {l0_by_layer[-1]:6.1f}  "
                    f"explained variance {ev_by_layer[-1]:.3f}",
                    end="" if layer != layers[-1] else "\n",
                    flush=True,
                )

        return PassRecord(
            name=self.name,
            params={
                "release": RELEASE,
                "width": self.width,
                "top_k": self.top_k,
                "hook": SAE_HOOK,
                "device": device,
                "n_layers": len(layers),
            },
            stats={
                "l0_mean": float(np.mean(l0_by_layer)),
                "l0_min": float(np.min(l0_by_layer)),
                "l0_max": float(np.max(l0_by_layer)),
                "explained_variance_mean": float(np.mean(ev_by_layer)),
                "explained_variance_min": float(np.min(ev_by_layer)),
                "l0_by_layer": l0_by_layer,
                "explained_variance_by_layer": ev_by_layer,
            },
            elapsed_s=time.time() - t0,
        )


def _explained_variance(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-token 1 - Var(residual error) / Var(x), the standard SAE metric."""
    err = (x - recon).var(dim=-1)
    total = x.var(dim=-1)
    return 1.0 - err / total.clamp_min(1e-12)


def _check_compatible(trace: Trace) -> None:
    """Refuse activations these SAEs were not trained on.

    Both of these produce features that look perfectly reasonable and mean
    nothing, so they are worth failing loudly over:

      - the wrong hook site (resid_pre / mlp_out instead of resid_post)
      - LayerNorm-folded weights (MECHLENS_PROCESS_WEIGHTS), which shift
        resid_post away from the distribution Gemma Scope was fitted on
    """
    if trace.residuals is None:
        raise ValueError(
            f"trace {trace.trace_id} has no residuals — the SAE pass reads the "
            f".npy sidecar, so run the capture first"
        )
    if trace.residuals.hook != SAE_HOOK:
        raise ValueError(
            f"{RELEASE} is trained on {SAE_HOOK}, but this trace captured "
            f"{trace.residuals.hook!r}. Re-capture with capture.RESID_HOOK = {SAE_HOOK!r}."
        )
    assert RESID_HOOK == SAE_HOOK, "capture and SAE hook sites have diverged"

    # TransformerLens marks folded weights by suffixing the norm type: "RMS"
    # is raw, "RMSPre" has been folded.
    if (trace.normalization or "").endswith("Pre"):
        raise ValueError(
            f"trace {trace.trace_id} was captured with LayerNorm-folded weights "
            f"(normalization={trace.normalization!r}). Gemma Scope is trained on raw "
            f"activations — re-capture with MECHLENS_PROCESS_WEIGHTS unset."
        )
