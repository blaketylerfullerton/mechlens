"""SAE pass logic, against a stand-in SAE.

A real Gemma Scope SAE is 302MB per layer, so the bookkeeping — top-k
selection, l0, which stats exclude BOS, the guards — is tested against a fake
whose activations are dictated by the test. The one test that uses a real SAE
is at the bottom and skips unless MECHLENS_SLOW=1.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from app.passes import apply
from app.passes.sae import SAEPass
from app.schema import ResidualRef
from factories import D_MODEL, N_LAYERS, N_TOKENS, make_result

D_SAE = 32


class FakeSAE:
    """Returns activations the test dictates, so assertions can be exact."""

    def __init__(self, acts: torch.Tensor, recon: torch.Tensor | None = None):
        self.acts = acts
        self.recon = recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.acts

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return self.recon if self.recon is not None else torch.zeros(N_TOKENS, D_MODEL)


def sparse_acts(active_per_token: dict[int, dict[int, float]]) -> torch.Tensor:
    """{token: {feature: activation}} -> a [n_tokens, d_sae] activation matrix."""
    acts = torch.zeros(N_TOKENS, D_SAE)
    for token, feats in active_per_token.items():
        for index, value in feats.items():
            acts[token, index] = value
    return acts


def traced(**ref_kwargs):
    """A synthetic trace carrying a plausible ResidualRef."""
    result = make_result()
    defaults = dict(
        path="x.npy", hook="hook_resid_post", shape=(N_TOKENS, N_LAYERS, D_MODEL), dtype="float32"
    )
    result.trace.residuals = ResidualRef(**{**defaults, **ref_kwargs})
    return result


def run_pass(result, acts, recon=None, **kwargs):
    saes = {layer: FakeSAE(acts, recon) for layer in range(N_LAYERS)}
    pass_ = SAEPass(saes=saes, verbose=False, device="cpu", **kwargs)
    return apply(pass_, result.trace, result.residuals)


# --------------------------------------------------------------------------
# what lands in the trace
# --------------------------------------------------------------------------


def test_top_k_features_are_kept_in_descending_order():
    result = traced()
    acts = sparse_acts({1: {5: 1.0, 9: 8.0, 2: 4.0, 30: 0.5, 17: 2.0}})
    run_pass(result, acts, top_k=3)

    state = result.trace.steps[1].layers[0]
    assert [f.index for f in state.features] == [9, 2, 17]
    assert [f.activation for f in state.features] == [8.0, 4.0, 2.0]


def test_l0_records_the_full_count_not_the_truncated_one():
    """The point of l0: top-3 looks the same whether 3 or 3000 features fired."""
    result = traced()
    acts = sparse_acts({1: {i: float(i + 1) for i in range(12)}})
    run_pass(result, acts, top_k=3)

    state = result.trace.steps[1].layers[0]
    assert len(state.features) == 3
    assert state.l0 == 12


def test_inactive_features_are_not_recorded():
    """topk pads with zeros; a zero is an absent feature, not a weak one."""
    result = traced()
    acts = sparse_acts({2: {7: 3.0, 8: 1.0}})
    run_pass(result, acts, top_k=6)

    state = result.trace.steps[2].layers[0]
    assert [f.index for f in state.features] == [7, 8]
    assert state.l0 == 2

    # a token with nothing active gets an empty list, not six zeros
    quiet = result.trace.steps[3].layers[0]
    assert quiet.features == []
    assert quiet.l0 == 0


def test_only_the_requested_layers_are_touched():
    result = traced()
    run_pass(result, sparse_acts({1: {3: 1.0}}), layers=[1])

    assert result.trace.steps[1].layers[1].l0 == 1
    assert result.trace.steps[1].layers[0].l0 is None
    assert result.trace.steps[1].layers[0].features == []


# --------------------------------------------------------------------------
# the diagnostics
# --------------------------------------------------------------------------


def test_bos_is_excluded_from_the_summary_stats():
    """Position 0 activates wildly under Gemma Scope; it must not skew L0."""
    result = traced()
    acts = sparse_acts({0: {i: 1.0 for i in range(D_SAE)}, 1: {0: 1.0, 1: 1.0}, 2: {0: 1.0, 1: 1.0}, 3: {0: 1.0, 1: 1.0}})
    record = run_pass(result, acts)

    assert result.trace.steps[0].layers[0].l0 == D_SAE  # still recorded per token
    assert record.stats["l0_mean"] == 2.0  # but averaged without it


def test_explained_variance_reflects_the_reconstruction():
    result = traced()
    acts = sparse_acts({1: {0: 1.0}})
    exact = torch.from_numpy(result.residuals[:, 0].copy())  # layer 0's own residual

    perfect = run_pass(result, acts, recon=exact, layers=[0])
    assert perfect.stats["explained_variance_mean"] == pytest.approx(1.0, abs=1e-5)

    useless = run_pass(result, acts, recon=torch.zeros(N_TOKENS, D_MODEL), layers=[0])
    assert useless.stats["explained_variance_mean"] < 0.5


def test_pass_record_carries_its_provenance():
    result = traced()
    record = run_pass(result, sparse_acts({1: {0: 1.0}}), top_k=5, width="16k")

    assert record.name == "sae"
    assert record.params["width"] == "16k"
    assert record.params["top_k"] == 5
    assert record.params["hook"] == "hook_resid_post"
    assert "release" in record.params
    assert len(record.stats["l0_by_layer"]) == N_LAYERS


def test_rerunning_the_pass_replaces_its_record():
    """Otherwise a trace claims two provenances for one set of features."""
    result = traced()
    run_pass(result, sparse_acts({1: {0: 1.0}}), top_k=4)
    run_pass(result, sparse_acts({1: {0: 1.0}}), top_k=8)

    assert [p.name for p in result.trace.passes] == ["sae"]
    assert result.trace.pass_record("sae").params["top_k"] == 8


# --------------------------------------------------------------------------
# the guards — each of these otherwise yields plausible nonsense
# --------------------------------------------------------------------------


def test_wrong_hook_site_is_refused():
    result = traced(hook="hook_resid_pre")
    with pytest.raises(ValueError, match="trained on hook_resid_post"):
        run_pass(result, sparse_acts({1: {0: 1.0}}))


def test_layernorm_folded_capture_is_refused():
    result = traced()
    result.trace.normalization = "RMSPre"
    with pytest.raises(ValueError, match="folded"):
        run_pass(result, sparse_acts({1: {0: 1.0}}))


def test_trace_without_residuals_is_refused():
    result = make_result()  # no ResidualRef attached
    with pytest.raises(ValueError, match="no residuals"):
        run_pass(result, sparse_acts({1: {0: 1.0}}))


# --------------------------------------------------------------------------
# the real thing
# --------------------------------------------------------------------------


@pytest.mark.skipif(not os.getenv("MECHLENS_SLOW"), reason="downloads a 302MB SAE")
def test_real_gemma_scope_sae_is_jumprelu_gated():
    """Pin the encoder's actual behaviour: gate on a threshold, not a ReLU.

    Note what this test deliberately does *not* do: feed it random noise and
    expect a sane L0. Gaussian noise is nowhere near the manifold gemma's
    residuals live on, and these thresholds are calibrated for that manifold —
    off-manifold input lights up thousands of features. L0 only means something
    on real activations, which is why it is measured by the pass, on a trace.
    """
    from app.sae_cache import get_sae

    sae = get_sae(20, device="cpu")
    assert sae.cfg.d_in == 2304
    assert sae.cfg.d_sae == 16384

    x = torch.randn(4, 2304) * 30
    with torch.no_grad():
        acts = sae.encode(x)
        pre = x @ sae.W_enc + sae.b_enc
        expected = pre * (pre > sae.threshold)

    torch.testing.assert_close(acts, expected, rtol=1e-4, atol=1e-4)
    # everything that survives is above its own threshold, and nothing is
    # merely "positive but small" the way a plain ReLU would leave it
    assert bool((acts[acts > 0] > sae.threshold.expand_as(acts)[acts > 0]).all())
