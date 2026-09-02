"""Attribution pass tests.

Three tiers, the same shape as test_lens_pass.py:

  - the real thing: tiny-stories-1M through a real `AttributionPass` run,
    checking the identity the whole decomposition rests on
    (resid_post = resid_pre + attn_out + mlp_out) and the bookkeeping
    around it.
  - what tiny-stories cannot exercise: it has neither a sandwich norm nor
    grouped-query attention, Gemma 2's two real gotchas, so a small
    synthetic HookedTransformerConfig stands in for both — the same move
    test_lens_pass.py makes to test the softcap, which tiny-stories's own
    config sets to zero.
  - bookkeeping against a stub, for the parts that need no model at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.capture import RESID_HOOK, generate_trace
from app.passes import apply
from app.passes.attribution import AttributionPass, _decompose_layer, _run_with_hooks
from app.schema import ResidualRef
from factories import D_MODEL, N_LAYERS, N_TOKENS, make_result

TEST_MODEL = "tiny-stories-1M"
PROMPT = "Once upon a time there was a"


# --------------------------------------------------------------------------
# the real thing
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    from transformer_lens import HookedTransformer

    try:
        m = HookedTransformer.from_pretrained(TEST_MODEL, device="cpu")
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"{TEST_MODEL} unavailable: {exc}")
    m.eval()
    return m


def _capture(model, prompt=PROMPT, max_new_tokens=4, top_k=5):
    result = generate_trace(model, prompt, max_new_tokens=max_new_tokens, top_k=top_k)
    result.trace.residuals = ResidualRef(
        path="x.npy",
        hook=RESID_HOOK,
        shape=tuple(result.residuals.shape),
        dtype="float32",
    )
    return result


@pytest.fixture(scope="module")
def attributed(model):
    result = _capture(model)
    record = apply(
        AttributionPass(model=model, top_k=3, verbose=False), result.trace, result.residuals
    )
    return result.trace, record


def test_layer_zero_has_no_resid_edge(attributed):
    """Layer 0 has no prior LayerState for a resid edge to point at."""
    trace, _ = attributed
    for step in trace.steps:
        kinds = [e.kind for e in step.layers[0].edges]
        assert "resid" not in kinds
        assert kinds.count("mlp") == 1


def test_every_other_layer_has_exactly_one_resid_and_mlp_edge(attributed):
    trace, _ = attributed
    for step in trace.steps:
        for layer in range(1, trace.n_layers):
            kinds = [e.kind for e in step.layers[layer].edges]
            assert kinds.count("resid") == 1
            assert kinds.count("mlp") == 1
            assert kinds.count("attn") >= 1


def test_resid_edge_points_at_the_previous_layer_same_position(attributed):
    trace, _ = attributed
    step = trace.steps[-1]
    for layer in range(1, trace.n_layers):
        resid_edge = next(e for e in step.layers[layer].edges if e.kind == "resid")
        assert resid_edge.source.layer == layer - 1
        assert resid_edge.source.position == step.step
        # Not recomputed — read straight from the LayerState it already sits on.
        assert resid_edge.weight == pytest.approx(step.layers[layer - 1].resid_norm)


def test_mlp_edge_points_at_the_same_layer_and_position(attributed):
    trace, _ = attributed
    step = trace.steps[2]
    for layer in range(trace.n_layers):
        mlp_edge = next(e for e in step.layers[layer].edges if e.kind == "mlp")
        assert mlp_edge.source.layer == layer
        assert mlp_edge.source.position == step.step


def test_no_sae_edges_are_ever_produced(attributed):
    """Feature-level attribution is explicitly out of scope for this pass."""
    trace, _ = attributed
    for step in trace.steps:
        for state in step.layers:
            assert all(e.kind != "sae" for e in state.edges)


def test_reconstruction_gap_reads_near_zero(attributed):
    """The correctness test of phase 5: resid_pre + attn_out + mlp_out holds
    by construction, the same class of guarantee as the lens's final-layer
    identity — a real gap means a hook or a sign is wrong."""
    _, record = attributed
    assert record.stats["reconstruction_max_rel_gap"] < 1e-3
    assert record.stats["attn_decomposition_max_rel_gap"] < 1e-3


def test_pass_record_carries_its_provenance(attributed):
    trace, record = attributed
    assert record.name == "attribution"
    assert record.params["top_k"] == 3
    assert record.params["n_layers_processed"] == trace.n_layers
    assert "attn_topk_coverage" in record.stats
    assert "attn_bias_mean_norm_frac" in record.stats


def test_attn_edges_are_truncated_to_top_k(attributed):
    trace, _ = attributed
    for step in trace.steps:
        for layer in range(trace.n_layers):
            attn_edges = [e for e in step.layers[layer].edges if e.kind == "attn"]
            assert len(attn_edges) <= 3


def test_kept_attn_edges_are_the_highest_weight_ones(model):
    """Proves the *set* kept is the top-k by weight, not just any k sources —
    computed independently via _decompose_layer rather than trusting the
    pass's own bookkeeping."""
    k = 2
    result = _capture(model)
    trace = result.trace
    apply(AttributionPass(model=model, top_k=k, verbose=False), trace, result.residuals)

    cache = _run_with_hooks(model, trace, list(range(trace.n_layers)))
    n = len(trace.steps)
    layer = trace.n_layers - 1
    _, _, contrib, _, _ = _decompose_layer(model, cache, layer, n)
    dest = n - 1
    weights = torch.linalg.norm(contrib[dest], dim=-1)
    expected = set(torch.argsort(weights, descending=True)[:k].tolist())

    got = {e.source.position for e in trace.steps[dest].layers[layer].edges if e.kind == "attn"}
    assert got == expected


def test_topk_coverage_is_exact_when_k_covers_every_source(model):
    result = _capture(model, prompt="Hi", max_new_tokens=1, top_k=3)
    record = apply(
        AttributionPass(model=model, top_k=64, verbose=False), result.trace, result.residuals
    )
    assert record.stats["attn_topk_coverage"] == pytest.approx(1.0, abs=1e-6)


def test_only_the_requested_layers_are_touched(model):
    result = _capture(model)
    trace = result.trace
    apply(
        AttributionPass(model=model, layers=[1], verbose=False), trace, result.residuals
    )
    assert trace.steps[0].layers[1].edges
    assert trace.steps[0].layers[0].edges == []


def test_rerunning_the_pass_replaces_its_record(model):
    result = _capture(model)
    apply(AttributionPass(model=model, top_k=2, verbose=False), result.trace, result.residuals)
    apply(AttributionPass(model=model, top_k=5, verbose=False), result.trace, result.residuals)

    assert [p.name for p in result.trace.passes] == ["attribution"]
    assert result.trace.pass_record("attribution").params["top_k"] == 5


# --------------------------------------------------------------------------
# what tiny-stories cannot exercise: sandwich norm + grouped-query attention
# --------------------------------------------------------------------------


def _gqa_sandwich_model():
    """A tiny Gemma-2-shaped config: grouped-query attention (4 heads sharing
    2 KV heads) and the post-attention/post-MLP RMSNorm ("sandwich norm").
    Random weights are fine — the identity being tested holds for any."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=3,
        d_model=32,
        n_ctx=16,
        n_heads=4,
        d_head=8,
        n_key_value_heads=2,
        d_vocab=50,
        act_fn="relu",
        normalization_type="RMS",
        use_normalization_before_and_after=True,
    )
    m = HookedTransformer(cfg)
    m.eval()
    return m


def test_sandwich_norm_and_gqa_reconstruct_attn_out_exactly():
    """Gemma-2-2b is too heavy for a test run, but its two gotchas are
    model-independent: RMSNorm's scale/gain are computed from the whole
    vector, so they distribute over a per-source decomposition; GQA's
    hook_v sits at kv-head granularity and must be expanded before use.
    Get either wrong and this gap stops being near-zero."""
    m = _gqa_sandwich_model()
    assert type(m.blocks[0].attn).__name__ == "GroupedQueryAttention"
    assert not isinstance(m.blocks[0].ln1_post, torch.nn.Identity)

    torch.manual_seed(0)
    tokens = torch.randint(0, m.cfg.d_vocab, (1, 7))
    with torch.no_grad():
        _, cache = m.run_with_cache(tokens)

    n = tokens.shape[1]
    for layer in range(m.cfg.n_layers):
        attn_out, mlp_out, contrib, bias, decomp_gap = _decompose_layer(m, cache, layer, n)
        assert decomp_gap < 1e-4

        resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0].float()
        resid_post = cache[f"blocks.{layer}.hook_resid_post"][0].float()
        recon = resid_pre + attn_out + mlp_out
        gap = (
            torch.linalg.norm(resid_post - recon, dim=-1)
            / torch.linalg.norm(resid_post, dim=-1).clamp_min(1e-12)
        ).max()
        assert gap < 1e-4


def test_a_gqa_head_mismatch_fails_loudly():
    """A silent shape mismatch here would pair the wrong value vector with
    each pattern head — this must raise, not produce a plausible-looking
    wrong answer."""
    m = _gqa_sandwich_model()
    torch.manual_seed(0)
    tokens = torch.randint(0, m.cfg.d_vocab, (1, 5))
    with torch.no_grad():
        _, cache = m.run_with_cache(tokens)

    m.blocks[0].attn.repeat_kv_heads = 1  # really 2 for this config
    with pytest.raises(ValueError, match="did not line up"):
        _decompose_layer(m, cache, 0, tokens.shape[1])


def test_hand_computed_single_source_single_head_contribution():
    """The decomposition's smallest unit, checked against arithmetic written
    out by hand rather than against the pass's own reconstruction — a bug
    that mis-attributes between two sources while still summing correctly
    would pass the reconstruction-gap check but fail this one."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1, d_model=4, n_ctx=8, n_heads=1, d_head=3, d_vocab=10,
        act_fn="relu", normalization_type="RMS",
    )
    m = HookedTransformer(cfg)
    m.eval()
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.d_vocab, (1, 2))
    with torch.no_grad():
        _, cache = m.run_with_cache(tokens)

    _, _, contrib, _, _ = _decompose_layer(m, cache, 0, 2)

    pattern = cache["blocks.0.attn.hook_pattern"][0]  # [1 head, 2 dest, 2 src]
    v = cache["blocks.0.attn.hook_v"][0]  # [2 pos, 1 head, 3 d_head]
    W_O = m.blocks[0].attn.W_O.detach()  # [1 head, 3 d_head, 4 d_model]

    hand = pattern[0, 1, 0] * (v[0, 0, :] @ W_O[0])
    assert torch.allclose(hand, contrib[1, 0], atol=1e-5)


# --------------------------------------------------------------------------
# bookkeeping, against a stub
# --------------------------------------------------------------------------


def _traced(**ref_kwargs):
    result = make_result()
    defaults = dict(
        path="x.npy", hook=RESID_HOOK, shape=(N_TOKENS, N_LAYERS, D_MODEL), dtype="float32"
    )
    result.trace.residuals = ResidualRef(**{**defaults, **ref_kwargs})
    return result


def test_wrong_hook_site_is_refused():
    result = _traced(hook="hook_resid_pre")
    with pytest.raises(ValueError, match="attribution reconstructs"):
        apply(
            AttributionPass(model=object(), verbose=False), result.trace, result.residuals
        )


def test_trace_without_residuals_is_refused():
    result = make_result()
    with pytest.raises(ValueError, match="no residuals"):
        apply(AttributionPass(model=object(), verbose=False), result.trace, result.residuals)
