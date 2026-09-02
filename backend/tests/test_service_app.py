"""The FastAPI routes, against the tiny CPU test model and a throwaway label DB.

Mirrors test_capture.py's approach: gemma is too heavy to load in every test
run, so the tiny model stands in for it, and a fake SAE stands in for a real
Gemma Scope one (whose dimensions would not even match the tiny model's).
"""

from __future__ import annotations

import time

import pytest
import torch
from fastapi.testclient import TestClient

from app.labels import LabelRow, LabelStore
from app.service import jobs
from app.service.app import create_app

PROMPT = "Once upon a time there was a"


class FakeSAE:
    def __init__(self, n_features: int, d_model: int):
        self.W_dec = torch.zeros(n_features, d_model)


@pytest.fixture(scope="module")
def tiny_model():
    from transformer_lens import HookedTransformer

    try:
        m = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"tiny-stories-1M unavailable: {exc}")
    m.eval()
    return m


@pytest.fixture
def client(tiny_model, tmp_path):
    jobs.JOBS.clear()
    sae_provider = lambda layer: FakeSAE(n_features=16, d_model=tiny_model.cfg.d_model)  # noqa: E731
    app = create_app(
        model=tiny_model, label_db_path=tmp_path / "labels.db", sae_provider=sae_provider
    )
    with TestClient(app) as c:
        yield c


def _poll(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/trace/{job_id}")
        if resp.json()["status"] in ("done", "error"):
            return resp
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


# -- POST /trace -------------------------------------------------------


def test_post_trace_returns_a_job_id(client):
    resp = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 3})
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_post_trace_rejects_empty_prompt(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post("/trace", json={"prompt": "", "max_tokens": 3})
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


def test_post_trace_rejects_non_positive_max_tokens(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 0})
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


# -- GET /trace/{id} -----------------------------------------------------


def test_get_trace_reaches_done_with_a_full_trace(client):
    job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 3}).json()["job_id"]
    resp = _poll(client, job_id)
    body = resp.json()
    assert body["status"] == "done"
    assert body["trace"]["prompt"] == PROMPT
    assert body["error"] is None


def test_get_trace_unknown_id_is_404(client):
    resp = client.get("/trace/does-not-exist")
    assert resp.status_code == 404


# -- POST /steer -----------------------------------------------------------


def test_post_steer_returns_a_pollable_job(client):
    resp = client.post(
        "/steer",
        json={"prompt": PROMPT, "max_tokens": 3, "layer": 2, "feature_idx": 5, "coefficient": 1.0},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    result = _poll(client, job_id)
    body = result.json()
    assert body["status"] == "done"
    assert body["trace"]["steering"] == {"layer": 2, "feature_idx": 5, "coefficient": 1.0}


def test_post_steer_rejects_out_of_range_layer(client, tiny_model):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/steer",
        json={
            "prompt": PROMPT,
            "max_tokens": 3,
            "layer": tiny_model.cfg.n_layers,  # one past the last valid layer
            "feature_idx": 0,
            "coefficient": 1.0,
        },
    )
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


def test_post_steer_rejects_out_of_range_feature_idx(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/steer",
        json={
            "prompt": PROMPT,
            "max_tokens": 3,
            "layer": 2,
            "feature_idx": 999,  # the fake SAE only has 16 features
            "coefficient": 1.0,
        },
    )
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


# -- GET /feature/{layer}/{idx} ---------------------------------------------


LAYER = 2
SOURCE_SET = "2-gemmascope-res-16k"


def test_get_feature_returns_a_known_label(client, tmp_path):
    with LabelStore(tmp_path / "labels.db") as store:
        store.upsert(
            [
                LabelRow(
                    source_set=SOURCE_SET,
                    feature=12082,
                    text="references to dogs as pets",
                    explainer="gpt-4o-mini",
                )
            ]
        )

    resp = client.get(f"/feature/{LAYER}/12082")
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "references to dogs as pets"
    assert body["explainer"] == "gpt-4o-mini"
    assert body["url"]


def test_get_feature_distinguishes_no_label_from_not_found(client, tmp_path):
    with LabelStore(tmp_path / "labels.db") as store:
        store.upsert([LabelRow(source_set=SOURCE_SET, feature=7, text=None)])

    looked_up_but_unexplained = client.get(f"/feature/{LAYER}/7")
    assert looked_up_but_unexplained.status_code == 200
    assert looked_up_but_unexplained.json()["label"] is None

    never_looked_up = client.get(f"/feature/{LAYER}/8")  # in range, but no row at all
    assert never_looked_up.status_code == 200
    assert never_looked_up.json()["label"] is None


def test_get_feature_out_of_range_layer_is_404(client, tiny_model):
    resp = client.get(f"/feature/{tiny_model.cfg.n_layers}/0")
    assert resp.status_code == 404


def test_get_feature_out_of_range_index_is_404(client):
    resp = client.get(f"/feature/{LAYER}/99999999")
    assert resp.status_code == 404


# -- model loading -----------------------------------------------------


def test_model_is_loaded_at_most_once_across_requests(monkeypatch, tiny_model, tmp_path):
    calls = {"n": 0}

    def fake_get_model():
        calls["n"] += 1
        return tiny_model

    monkeypatch.setattr("app.service.app.model_cache.get_model", fake_get_model)
    jobs.JOBS.clear()
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    with TestClient(app) as c:
        assert calls["n"] == 1  # loaded once at startup, by the lifespan handler
        job_id = c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1}).json()["job_id"]
        _poll(c, job_id)
        c.get(f"/trace/{job_id}")

    assert calls["n"] == 1
