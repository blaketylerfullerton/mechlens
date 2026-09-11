"""Controlled feature interventions on an identical token sequence.

These measure effects on a chosen next-token score, not a whole causal graph.
Ablation removes the feature's decoded contribution while preserving the rest
of the residual, including SAE reconstruction error.
"""
from __future__ import annotations

import hashlib
import math
from importlib.metadata import version

import torch

from .capture import RESID_HOOK
from .sae_cache import RELEASE, DEFAULT_WIDTH, sae_id


def measure_feature(
    model, sae, prompt: str, layer: int, feature_idx: int, target_token_id: int,
    coefficients: list[float], *, width: str = DEFAULT_WIDTH, position: int | None = None,
) -> dict:
    """Baseline, zero control, ablation and additions at one selected position.

    All variants use precisely the same prefix. The measured output is the
    target token's probability after the final prefix token, even when the
    intervention position is earlier. No generated continuation is fed back.
    """
    if not 0 <= layer < model.cfg.n_layers:
        raise ValueError("layer out of range")
    if not 0 <= feature_idx < sae.W_dec.shape[0]:
        raise ValueError("feature index out of range")
    if not 0 <= target_token_id < model.cfg.d_vocab_out:
        raise ValueError("target token ID out of range")
    if not coefficients or not all(math.isfinite(c) for c in coefficients):
        raise ValueError("provide finite steering coefficients")
    if (model.cfg.normalization_type or "").endswith("Pre"):
        raise ValueError("feature experiments require unprocessed model weights")
    metadata = getattr(getattr(sae, "cfg", None), "metadata", None)
    expected_model = getattr(metadata, "model_name", None)
    if expected_model and expected_model != model.cfg.model_name:
        raise ValueError("SAE and model identities do not match")
    if sae.W_dec.shape[1] != model.cfg.d_model:
        raise ValueError("SAE residual dimensions do not match model")
    tokens = model.to_tokens(prompt)
    if not 0 < tokens.shape[1] <= model.cfg.n_ctx:
        raise ValueError("prompt exceeds the model context window")
    position = tokens.shape[1] - 1 if position is None else position
    if not 0 <= position < tokens.shape[1]:
        raise ValueError("intervention position out of range")
    hook_name = f"blocks.{layer}.{RESID_HOOK}"
    expected_hook = getattr(metadata, "hook_name", None)
    if expected_hook and expected_hook != hook_name:
        raise ValueError("SAE was trained at a different hook")
    direction = sae.W_dec[feature_idx].detach()
    rows = []
    model.eval()
    modes = [("baseline", None), ("zero_control", 0.0), ("ablation", None)]
    modes += [("steering", float(c)) for c in coefficients]
    for mode, coefficient in modes:
        observed = {}
        def intervene(resid, hook):
            x = resid[:, position, :]
            encoder_device = getattr(sae, "W_enc", sae.W_dec).device
            a = sae.encode(x.to(device=encoder_device, dtype=sae.W_dec.dtype))[:, feature_idx]
            observed["activation_before"] = float(a[0])
            if mode == "baseline":
                return resid
            delta = -a if mode == "ablation" else x.new_full((x.shape[0],), coefficient)
            out = resid.clone()
            out[:, position, :] += delta.to(x).unsqueeze(-1) * direction.to(x)
            after = sae.encode(out[:, position, :].to(device=encoder_device, dtype=sae.W_dec.dtype))
            observed["activation_after"] = float(after[0, feature_idx])
            return out
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, intervene)]):
            logits = model(tokens)[0, -1].float()
            log_probs = logits.log_softmax(dim=-1)
        if not torch.isfinite(log_probs).all():
            raise ValueError(f"non-finite output during {mode}; reduce the intervention strength")
        rows.append({
            "mode": mode, "coefficient": coefficient,
            "target_probability": float(log_probs[target_token_id].exp()),
            "target_log_probability": float(log_probs[target_token_id]),
            "target_logit": float(logits[target_token_id]),
            **observed,
        })
    baseline = rows[0]
    for row in rows:
        row["delta_probability"] = row["target_probability"] - baseline["target_probability"]
        row["delta_log_probability"] = row["target_log_probability"] - baseline["target_log_probability"]
    zero_error = max(abs(rows[1][k] - baseline[k]) for k in
                     ("target_probability", "target_log_probability", "target_logit"))
    if zero_error > 1e-5:
        raise ValueError(f"zero-control failed (maximum target-score difference {zero_error})")
    return {
        "experiment_version": "1.0", "model": model.cfg.model_name,
        "model_revision": getattr(model.cfg, "revision", None),
        "device": str(model.cfg.device), "dtype": str(model.cfg.dtype),
        "runtime": {name: version(name) for name in ("torch", "transformer-lens", "sae-lens")},
        "prompt": prompt, "token_ids": tokens[0].tolist(),
        "layer": layer, "position": position, "feature_idx": feature_idx,
        "target_token_id": target_token_id,
        "release": RELEASE, "width": width, "sae_id": sae_id(layer, width), "hook": hook_name,
        "direction_sha256": hashlib.sha256(direction.float().cpu().numpy().tobytes()).hexdigest(),
        "zero_control_max_error": zero_error, "measurements": rows,
        "interpretation": "Effect of a residual-direction intervention at one position on a fixed-prefix next-token score; not proof of a complete circuit.",
    }
