"""Phase 2: encode each residual through its layer's SAE.

For every (token, layer) this writes the top-k active features into the trace,
plus `l0` — the true number of features that fired. The full sparse vector is
d_sae mostly-zero entries (16384 for Gemma Scope 16k); the top 16 of them are
what anyone ever looks at. Which SAEs is the trace model's profile's release.

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

from ..capture import RESID_HOOK, ProgressCallback
from ..profiles import SAERelease, get_release, reads_layer, release_for, same_model
from ..sae_cache import (
    SAE_HOOK,
    center_input,
    expects_centered,
    load_layers,
    pick_device,
)
from ..schema import Feature, PassRecord, Trace

DEFAULT_TOP_K = 16

# Position 0 is BOS. SAEs (Gemma Scope especially) see an activation there unlike anything
# in their training data and respond with huge, meaningless features — a known
# artifact, but one that would drag every average off if left in.
BOS_POSITION = 0


@dataclass
class SAEPass:
    """Fills LayerState.features and LayerState.l0."""

    name: str = field(default="sae", init=False)
    width: str | None = None  # None = the release's default width
    # SAELens release. None = the trace model's; with injected `saes` and a
    # model that has none, the default model's.
    release: str | None = None
    top_k: int = DEFAULT_TOP_K
    layers: list[int] | None = None  # None = every layer
    device: str | None = None
    verbose: bool = True
    # Local dictionaries carry an immutable content ID, never a published release's identity.
    artifact_id: str | None = None
    feature_count: int | None = None

    # Pre-loaded SAEs keyed by layer. Skips the loader — for a long-lived
    # server that holds them, and for tests that stand in a fake.
    saes: dict[int, object] | None = None

    # Vouches for where the residual array came from, for a caller that holds
    # the tensor but has no sidecar on disk to point a ResidualRef at (the HTTP
    # service). Only consulted when `trace.residuals` is absent: a ref
    # describes bytes that actually landed on disk, so it wins whenever it is
    # there. Same field, same rule, as LogitLensPass.hook.
    hook: str | None = None

    # Called with (layers_encoded, layers_to_encode) as each layer finishes.
    # This pass really does walk layers one at a time, so — like the lens and
    # unlike the capture loop — it can report a depth sweep honestly. Advisory
    # only: nothing in the PassRecord depends on it.
    on_progress: ProgressCallback | None = None

    def run(self, trace: Trace, residuals: np.ndarray) -> PassRecord:
        _check_compatible(trace, self.hook)

        # A local dictionary is its own identity; only a published one has a release.
        spec = None if self.artifact_id else self._release(trace)
        width = self.width if spec is None else spec.check_width(self.width)
        device = self.device or pick_device()
        layers = self.layers if self.layers is not None else list(range(trace.n_layers))
        saes = self.saes if self.saes is not None else load_layers(layers, width, device, spec.release if spec else None)

        n_tokens = len(trace.steps)
        l0_by_layer: list[float] = []
        ev_by_layer: list[float] = []
        t0 = time.time()

        for encoded, layer in enumerate(layers, start=1):
            sae = saes[layer].to(device)
            metadata = getattr(getattr(sae, "cfg", None), "metadata", None)
            expected_hook = getattr(metadata, "hook_name", None)
            if expected_hook and not reads_layer(expected_hook, layer):
                raise ValueError(f"SAE expects hook {expected_hook}, not layer {layer} resid_post")
            expected_size = self.feature_count if spec is None else spec.widths[width]
            actual_size = getattr(getattr(sae, "cfg", None), "d_sae", None)
            if actual_size is not None and actual_size != expected_size:
                raise ValueError(f"SAE size {actual_size} does not match width {width}")
            expected_model = getattr(metadata, "model_name", None)
            if expected_model and not same_model(expected_model, trace.model):
                raise ValueError(f"SAE expects {expected_model}, trace uses {trace.model}")
            # [n_tokens, d_model] — one layer's slice for the whole sequence.
            # np.array (a copy) rather than asarray: a memory-mapped trace is
            # read-only, and torch.from_numpy warns on non-writable buffers.
            x = center_input(sae, torch.from_numpy(np.array(residuals[:, layer])).to(device))

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

            if self.on_progress is not None:
                self.on_progress(encoded, len(layers))

        return PassRecord(
            name=self.name,
            params={
                "release": f"local/{self.artifact_id}" if spec is None else spec.release,
                "model": trace.model,
                "width": width,
                "top_k": self.top_k,
                "hook": SAE_HOOK,
                # Mean-subtracted before encoding, for SAEs trained on a
                # centered model (sae_cache.expects_centered).
                "centered": any(expects_centered(saes[layer]) for layer in layers),
                "device": device,
                "n_layers": len(layers),
                # Which layers actually ran, not just how many. A caller may
                # encode a subset (gemma's 26 resident 16k SAEs come to ~7.9GB), and a
                # consumer has to be able to tell a layer that was skipped from
                # a layer in which nothing fired. Comma-joined for the same
                # reason LabelsPass joins its explainer mix: `params` values are
                # scalars. See the `layers` field above.
                "layers": ",".join(str(layer) for layer in layers),
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


    def _release(self, trace: Trace) -> SAERelease:
        if self.release is not None:
            return get_release(self.release)
        found = release_for(trace.model)
        if found is not None:
            return found
        if self.saes is None:
            raise ValueError(f"{trace.model} has no published SAE release; train a dictionary for it instead")
        return get_release()  # injected SAEs on a model without one: the service's tests


def _explained_variance(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-token 1 - Var(residual error) / Var(x), the standard SAE metric."""
    err = (x - recon).var(dim=-1)
    total = x.var(dim=-1)
    return 1.0 - err / total.clamp_min(1e-12)


def _check_compatible(trace: Trace, vouched_hook: str | None = None) -> None:
    """Refuse activations these SAEs were not trained on.

    Both of these produce features that look perfectly reasonable and mean
    nothing, so they are worth failing loudly over:

      - the wrong hook site (resid_pre / mlp_out instead of resid_post)
      - LayerNorm-folded weights (MECHLENS_PROCESS_WEIGHTS), which shift
        resid_post away from the distribution the SAEs were fitted on

    Where the hook site is recorded depends on the caller, exactly as it does
    for the lens: a caller enriching a saved trace loads the array *through*
    `trace.residuals`, so the ref is the provenance. A caller holding the array
    in memory (the HTTP service, which writes no sidecar) passes it as an
    argument, so only its origin is in question and `vouched_hook` carries
    that. The ref wins when both are available.
    """
    captured_hook = trace.residuals.hook if trace.residuals is not None else vouched_hook

    if captured_hook is None:
        raise ValueError(
            f"trace {trace.trace_id} has no residuals attached — to encode an "
            f"in-memory capture, pass the hook it was captured at (SAEPass(hook=...))"
        )
    if captured_hook != SAE_HOOK:
        raise ValueError(
            f"SAEs here are trained on {SAE_HOOK} (or the equivalent next-block "
            f"resid_pre), but this trace captured "
            f"{captured_hook!r}. Re-capture with capture.RESID_HOOK = {SAE_HOOK!r}."
        )
    assert RESID_HOOK == SAE_HOOK, "capture and SAE hook sites have diverged"

    # TransformerLens marks folded weights by suffixing the norm type: "RMS"
    # is raw, "RMSPre" has been folded.
    if (trace.normalization or "").endswith("Pre"):
        raise ValueError(
            f"trace {trace.trace_id} was captured with LayerNorm-folded weights "
            f"(normalization={trace.normalization!r}). SAEs are trained on raw "
            f"activations — re-capture with MECHLENS_PROCESS_WEIGHTS unset."
        )
