"""Schema + storage tests. No model, no GPU — these must stay fast."""

import numpy as np
import pytest
from pydantic import ValidationError

from app.schema import (
    SCHEMA_VERSION,
    Edge,
    Feature,
    FeatureLabel,
    LogitLens,
    NodeRef,
    TokenInfo,
    TopToken,
    Trace,
    label_key,
)
from app.store import load, load_residuals, load_trace, save_trace
from factories import D_MODEL, N_LAYERS, N_TOKENS, make_result


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
    layer.features = [Feature(index=4023, activation=7.5)]
    trace.labels[label_key(1, 4023)] = FeatureLabel(
        text="golden gate bridge", explainer="gpt-4o-mini"
    )
    layer.logit_lens = LogitLens(
        top_k=[TopToken(token_id=7, text=" San", logit=2.0, prob=0.8)], entropy=0.4
    )
    layer.edges = [Edge(source=NodeRef(layer=0, position=1, feature=12), weight=-0.3, kind="sae")]

    reread = Trace.model_validate_json(trace.model_dump_json())
    enriched = reread.steps[2].layers[1]
    assert enriched.features[0].index == 4023
    assert reread.label(1, 4023).text == "golden gate bridge"
    assert enriched.logit_lens.top_k[0].text == " San"
    assert enriched.edges[0].source.feature == 12
    # untouched layers keep their empty defaults
    assert reread.steps[0].layers[0].features == []


def test_token_source_is_constrained():
    with pytest.raises(ValidationError):
        TokenInfo(position=0, token_id=1, text="x", source="hallucinated")
