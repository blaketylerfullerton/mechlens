"""Request/response models round-trip through validate/dump."""

from __future__ import annotations

from app.service.models import (
    FeatureResponse,
    JobResponse,
    JobStatusResponse,
    SteerRequest,
    TraceRequest,
)
from factories import make_result


def test_trace_request_round_trips():
    payload = {"prompt": "hello", "max_tokens": 10}
    model = TraceRequest.model_validate(payload)
    assert model.model_dump() == payload


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
    payload = {"status": "pending", "trace": None, "error": None}
    assert JobStatusResponse.model_validate(payload).model_dump() == payload


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
