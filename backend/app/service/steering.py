"""Builds the `capture.Intervention` for SAE feature steering.

A steering request names one labeled feature — a layer and an index into
that layer's SAE — plus a coefficient. This turns that into the
`(layer, hook_fn)` pair `generate_trace` expects: `hook_fn` adds
`coefficient * W_dec[feature_idx]`, the feature's decoder direction, to
`hook_resid_post` at that layer, at every position, on every forward pass
during generation.

`feature_direction` and `make_hook` are split out from `build_intervention`
so a test can exercise the hook's numerical effect against a synthetic
direction, without downloading a real SAE.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..capture import Intervention
from ..sae_cache import DEFAULT_WIDTH, get_sae


def feature_direction(
    layer: int, feature_idx: int, width: str = DEFAULT_WIDTH, sae: object | None = None
) -> torch.Tensor:
    """The decoder direction for one SAE feature, in residual space.

    `sae`, like `SAEPass.saes`, lets a caller (or a test) hand in an
    already-loaded or fake SAE and skip the loader.
    """
    sae = sae if sae is not None else get_sae(layer, width)
    return sae.W_dec[feature_idx].detach()


def make_hook(direction: torch.Tensor, coefficient: float) -> Callable:
    """A TransformerLens hook adding `coefficient * direction` to resid_post.

    Cast to the residual's own dtype/device at hook time rather than the
    direction's: the direction is loaded once in the SAE's fp32, but the
    model it steers may run in bf16 on a different device.
    """

    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        return resid + coefficient * direction.to(dtype=resid.dtype, device=resid.device)

    return hook_fn


def build_intervention(
    layer: int,
    feature_idx: int,
    coefficient: float,
    width: str = DEFAULT_WIDTH,
    sae: object | None = None,
) -> Intervention:
    """`(layer, hook_fn)` for `generate_trace(..., intervention=...)`."""
    direction = feature_direction(layer, feature_idx, width, sae=sae)
    return layer, make_hook(direction, coefficient)
