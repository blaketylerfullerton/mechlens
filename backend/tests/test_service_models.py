"""Request/response models round-trip through validate/dump."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.service.jobs import JobProgress
from app.service.models import (
    FeatureResponse,
    JobResponse,
    JobStatusResponse,
    SteerRequest,
    TraceRequest,
)
from factories import make_result


def test_trace_request_round_trips():
    payload = {"prompt": "hello", "max_tokens": 10, "passes": ["lens"], "sae_layers": None, "live": False}
    model = TraceRequest.model_validate(payload)
    assert model.model_dump() == payload


def test_trace_request_defaults_to_no_passes():
    """Capture-only stays the default: a client that says nothing about passes
    must not start paying for the lens."""
    model = TraceRequest.model_validate({"prompt": "hello", "max_tokens": 10})
    assert model.passes == []
    assert model.model_dump() == {
        "prompt": "hello",
        "max_tokens": 10,
        "passes": [],
        "sae_layers": None,
        "live": False,
    }


def test_trace_request_rejects_an_unknown_pass():
    with pytest.raises(ValidationError):
        TraceRequest.model_validate({"prompt": "hello", "max_tokens": 10, "passes": ["nope"]})


@pytest.mark.parametrize("passes", [["sae"], ["labels", "sae"], ["sae", "labels", "lens"]])
def test_trace_request_accepts_the_sae_and_label_passes(passes):
    assert TraceRequest.model_validate(
        {"prompt": "hello", "max_tokens": 10, "passes": passes}
    ).passes == passes


def test_trace_request_rejects_labels_without_sae():
    """The labels pass reads the features the SAE pass records, so the
    dependency is a boundary check, not a worker-thread surprise."""
    with pytest.raises(ValidationError, match="sae"):
        TraceRequest.model_validate(
            {"prompt": "hello", "max_tokens": 10, "passes": ["labels"]}
        )


@pytest.mark.parametrize("sae_layers", [[], [-1], [0, 0]])
def test_trace_request_rejects_a_malformed_layer_subset(sae_layers):
    with pytest.raises(ValidationError):
        TraceRequest.model_validate(
            {"prompt": "hello", "max_tokens": 10, "passes": ["sae"], "sae_layers": sae_layers}
        )


def test_trace_request_accepts_a_layer_subset():
    model = TraceRequest.model_validate(
        {"prompt": "hello", "max_tokens": 10, "passes": ["sae"], "sae_layers": [0, 5, 20]}
    )
    assert model.sae_layers == [0, 5, 20]


def test_steer_request_round_trips():
    payload = {"prompt": "hello", "max_tokens": 10, "layer": 3, "feature_idx": 42, "coefficient": 1.5}
    model = SteerRequest.model_validate(payload)
    assert model.model_dump() == payload


def test_job_response_round_trips():
    payload = {"job_id": "abc123"}
    assert JobResponse.model_validate(payload).model_dump() == payload


def test_job_status_response_round_trips_with_a_trace():
    trace = make_result().trace
    payload = {"status": "done", "trace": trace.model_dump(mode="json"), "error": None}
    model = JobStatusResponse.model_validate(payload)
    assert model.status == "done"
    assert model.trace.trace_id == trace.trace_id


def test_job_status_response_round_trips_pending():
    payload = {"status": "pending", "trace": None, "partial_trace": None, "error": None, "progress": None}
    assert JobStatusResponse.model_validate(payload).model_dump() == payload


def test_job_status_response_carries_a_progress_reading():
    payload = {
        "status": "running",
        "trace": None,
        "partial_trace": None,
        "error": None,
        "progress": {"phase": "lens", "done": 4, "total": 26},
    }
    assert JobStatusResponse.model_validate(payload).model_dump() == payload


def test_job_status_progress_accepts_the_worker_side_dataclass():
    """The route hands over `JobRecord.progress` directly, so the response model
    has to read a frozen dataclass and not just a dict."""
    model = JobStatusResponse(
        status="running", trace=None, error=None, progress=JobProgress("generating", 3, 20)
    )
    assert model.model_dump()["progress"] == {"phase": "generating", "done": 3, "total": 20}


def test_feature_response_round_trips():
    payload = {
        "layer": 20,
        "feature_idx": 12082,
        "label": "references to dogs",
        "explainer": "gpt-4o-mini",
        "explanation_type": "oai_token-act-pair",
        "score": None,
        "url": "https://neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/12082",
    }
    assert FeatureResponse.model_validate(payload).model_dump() == payload
