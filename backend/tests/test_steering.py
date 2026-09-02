"""Feature steering, against a stand-in SAE and the tiny CPU test model.

A real Gemma Scope SAE is 302MB per layer, so `build_intervention`'s numeric
effect is tested against a fake `W_dec` — the same pattern test_sae_pass.py
uses for the SAE pass itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.capture import generate_trace
from app.service.steering import build_intervention, feature_direction, make_hook

D_MODEL = 8


class FakeSAE:
    def __init__(self, w_dec: torch.Tensor):
        self.W_dec = w_dec


def test_feature_direction_reads_the_named_row():
    w_dec = torch.arange(3 * D_MODEL, dtype=torch.float32).reshape(3, D_MODEL)
    direction = feature_direction(layer=0, feature_idx=1, sae=FakeSAE(w_dec))
    torch.testing.assert_close(direction, w_dec[1])


def test_hook_adds_scaled_direction():
    direction = torch.ones(D_MODEL)
    hook_fn = make_hook(direction, coefficient=2.0)
    resid = torch.zeros(1, 5, D_MODEL)

    out = hook_fn(resid, hook=None)

    torch.testing.assert_close(out, torch.full((1, 5, D_MODEL), 2.0))


def test_hook_casts_to_the_residual_dtype():
    direction = torch.ones(D_MODEL, dtype=torch.float32)
    hook_fn = make_hook(direction, coefficient=1.0)
    resid = torch.zeros(1, 1, D_MODEL, dtype=torch.bfloat16)

    out = hook_fn(resid, hook=None)

    assert out.dtype == torch.bfloat16


@pytest.fixture(scope="module")
def model():
    from transformer_lens import HookedTransformer

    try:
        m = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"tiny-stories-1M unavailable: {exc}")
    m.eval()
    return m


PROMPT = "Once upon a time there was a"


def test_build_intervention_steers_the_named_layer(model):
    """End to end through generate_trace: a nonzero coefficient must change
    the captured residual at the target layer by coefficient * W_dec[idx]."""
    layer = 1
    d_model = model.cfg.d_model
    w_dec = torch.zeros(4, d_model)
    w_dec[2] = torch.ones(d_model)  # feature 2's direction
    coefficient = 3.0

    baseline = generate_trace(model, PROMPT, max_new_tokens=0)
    intervention = build_intervention(layer, feature_idx=2, coefficient=coefficient, sae=FakeSAE(w_dec))
    steered = generate_trace(model, PROMPT, max_new_tokens=0, intervention=intervention)

    np.testing.assert_allclose(
        steered.residuals[:, layer],
        baseline.residuals[:, layer] + coefficient,
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        steered.residuals[:, :layer], baseline.residuals[:, :layer], rtol=1e-5, atol=1e-5
    )


def test_zero_coefficient_matches_the_unsteered_trace(model):
    """The steering spec's no-op scenario: coefficient=0 must reproduce an
    unsteered trace of the same prompt exactly."""
    layer = 1
    d_model = model.cfg.d_model
    w_dec = torch.ones(4, d_model)

    baseline = generate_trace(model, PROMPT, max_new_tokens=6, top_k=5)
    intervention = build_intervention(layer, feature_idx=0, coefficient=0.0, sae=FakeSAE(w_dec))
    steered = generate_trace(model, PROMPT, max_new_tokens=6, top_k=5, intervention=intervention)

    assert steered.trace.completion == baseline.trace.completion
    np.testing.assert_array_equal(steered.residuals, baseline.residuals)
