"""Schema + storage tests. No model, no GPU — these must stay fast."""

import numpy as np
import pytest
from pydantic import ValidationError

from capture import CaptureResult
from schema import (
    SCHEMA_VERSION,
    Edge,
    Feature,
    LayerState,
    LogitLens,
    LogitSummary,
    NodeRef,
    TokenInfo,
    TokenStep,
    TopToken,
    Trace,
)
from store import load, load_residuals, load_trace, save_trace

N_TOKENS, N_LAYERS, D_MODEL = 4, 3, 8


def make_result(seed: int = 0) -> CaptureResult:
    rng = np.random.default_rng(seed)
    residuals = rng.standard_normal((N_TOKENS, N_LAYERS, D_MODEL), dtype=np.float32)

    steps = [
        TokenStep(
            step=pos,
            token=TokenInfo(
                position=pos,
                token_id=100 + pos,
                text=f" t{pos}",
                source="prompt" if pos < 2 else "generated",
            ),
            logits=LogitSummary(
                top_k=[TopToken(token_id=100 + pos + 1, text=" next", logit=1.0, prob=0.5)],
                entropy=1.23,
                chosen=(
                    None
                    if pos == N_TOKENS - 1
                    else TopToken(token_id=100 + pos + 1, text=" next", logit=1.0, prob=0.5)
                ),
            ),
            layers=[
                LayerState(
                    layer=layer,
                    resid_norm=float(np.linalg.norm(residuals[pos, layer])),
                )
                for layer in range(N_LAYERS)
            ],
        )
        for pos in range(N_TOKENS)
    ]

    trace = Trace(
        trace_id="test0001",
        model="test-model",
        device="cpu",
        dtype="torch.float32",
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        prompt=" t0 t1",
        completion=" t2 t3",
        n_prompt_tokens=2,
        n_generated_tokens=2,
        steps=steps,
    )
    return CaptureResult(trace=trace, residuals=residuals)


def test_roundtrip_preserves_trace_and_residuals(tmp_path):
    result = make_result()
    json_path = save_trace(result, tmp_path)

    trace, residuals = load(json_path)

    assert trace.schema_version == SCHEMA_VERSION
    assert trace.trace_id == "test0001"
    assert trace.text == " t0 t1 t2 t3"
    assert len(trace.steps) == N_TOKENS
    assert all(len(step.layers) == N_LAYERS for step in trace.steps)
    np.testing.assert_array_equal(residuals, result.residuals)


def test_save_records_sidecar_relative_to_the_json(tmp_path):
    json_path = save_trace(make_result(), tmp_path)
    trace = load(json_path)[0]

    assert trace.residuals.path == "test0001.residuals.npy"
    assert tuple(trace.residuals.shape) == (N_TOKENS, N_LAYERS, D_MODEL)
    assert trace.residuals.dtype == "float32"

    # Moving the pair together must not break the reference.
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for f in (json_path, json_path.with_name(trace.residuals.path)):
        f.rename(moved / f.name)

    _, residuals = load(moved / json_path.name)
    assert residuals.shape == (N_TOKENS, N_LAYERS, D_MODEL)


def test_shape_mismatch_between_halves_is_caught(tmp_path):
    result = make_result()
    json_path = save_trace(result, tmp_path)
    np.save(json_path.with_name(result.trace.residuals.path), np.zeros((1, 1, 1), np.float32))

    trace = load_trace(json_path)
    with pytest.raises(ValueError, match="expected"):
        load_residuals(trace, json_path)


def test_mmap_load_does_not_copy(tmp_path):
    json_path = save_trace(make_result(), tmp_path)
    trace = load(json_path)[0]
    assert isinstance(load_residuals(trace, json_path, mmap=True), np.memmap)


def test_later_phases_can_enrich_a_saved_trace(tmp_path):
    """features / logit_lens / edges are the contract for phases 2-4."""
    json_path = save_trace(make_result(), tmp_path)
    trace = load(json_path)[0]

    layer = trace.steps[2].layers[1]
    layer.features = [Feature(index=4023, activation=7.5, label="golden gate bridge")]
    layer.logit_lens = LogitLens(
        top_k=[TopToken(token_id=7, text=" San", logit=2.0, prob=0.8)], entropy=0.4
    )
    layer.edges = [Edge(source=NodeRef(layer=0, position=1, feature=12), weight=-0.3, kind="sae")]

    reread = Trace.model_validate_json(trace.model_dump_json())
    enriched = reread.steps[2].layers[1]
    assert enriched.features[0].label == "golden gate bridge"
    assert enriched.logit_lens.top_k[0].text == " San"
    assert enriched.edges[0].source.feature == 12
    # untouched layers keep their empty defaults
    assert reread.steps[0].layers[0].features == []


def test_token_source_is_constrained():
    with pytest.raises(ValidationError):
        TokenInfo(position=0, token_id=1, text="x", source="hallucinated")
