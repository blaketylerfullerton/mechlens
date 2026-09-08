"""The FastAPI routes, against the tiny CPU test model and a throwaway label DB.

Mirrors test_capture.py's approach: gemma is too heavy to load in every test
run, so the tiny model stands in for it, and a fake SAE stands in for a real
Gemma Scope one (whose dimensions would not even match the tiny model's).
"""

from __future__ import annotations

import threading
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


def _wait_ready(client, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/health").json()["status"]
        if status != "loading":
            return status
        time.sleep(0.02)
    raise AssertionError("model did not become ready in time")


def test_model_is_loaded_at_most_once_across_requests(monkeypatch, tiny_model, tmp_path):
    calls = {"n": 0}

    def fake_get_model():
        calls["n"] += 1
        return tiny_model

    monkeypatch.setattr("app.service.app.model_cache.get_model", fake_get_model)
    jobs.JOBS.clear()
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    with TestClient(app) as c:
        assert _wait_ready(c) == "ready"  # loaded by the lifespan's warm-up thread
        assert calls["n"] == 1
        job_id = c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1}).json()["job_id"]
        _poll(c, job_id)
        c.get(f"/trace/{job_id}")

    assert calls["n"] == 1


def test_health_is_ready_when_a_model_is_injected(client):
    assert client.get("/health").json() == {"status": "ready", "detail": None}


def test_routes_answer_503_while_the_model_is_still_loading(monkeypatch, tmp_path):
    """The point of loading off-thread: the port is open and the API answers,
    with a status a client can act on, before the model is in memory."""
    release = threading.Event()

    def slow_get_model():
        release.wait(timeout=5)
        raise AssertionError("test never lets the load finish")

    monkeypatch.setattr("app.service.app.model_cache.get_model", slow_get_model)
    jobs.JOBS.clear()
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    try:
        with TestClient(app) as c:
            assert c.get("/health").json()["status"] == "loading"
            resp = c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1})
            assert resp.status_code == 503
            assert "still loading" in resp.json()["detail"]
    finally:
        release.set()


def test_health_reports_a_failed_load(monkeypatch, tmp_path):
    def broken_get_model():
        raise RuntimeError("no weights on disk")

    monkeypatch.setattr("app.service.app.model_cache.get_model", broken_get_model)
    jobs.JOBS.clear()
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    with TestClient(app) as c:
        assert _wait_ready(c) == "error"
        assert "no weights on disk" in c.get("/health").json()["detail"]
        resp = c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1})
        assert resp.status_code == 503
        assert "failed to load" in resp.json()["detail"]


def test_cors_allows_the_vite_dev_server_on_a_fallback_port(client):
    """vite moves to 5174+ when 5173 is taken; the browser there must not be
    told "access control checks" by our own middleware."""
    for origin in ("http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5175"):
        resp = client.post(
            "/trace", json={"prompt": PROMPT, "max_tokens": 1}, headers={"Origin": origin}
        )
        assert resp.headers["access-control-allow-origin"] == origin


# -- POST /trace with enrichment passes ------------------------------------


def test_requesting_the_lens_pass_fills_per_layer_readouts(client, tiny_model):
    """The brain view reads `logit_lens` per (token, layer); it only exists if
    the trace job ran the pass, since the service writes no sidecar to enrich
    later."""
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["lens"]}
    ).json()["job_id"]

    body = _poll(client, job_id).json()
    assert body["status"] == "done", body.get("error")
    trace = body["trace"]

    assert [p["name"] for p in trace["passes"]] == ["lens"]
    for step in trace["steps"]:
        assert len(step["layers"]) == tiny_model.cfg.n_layers
        for layer in step["layers"]:
            assert layer["logit_lens"] is not None
            assert layer["logit_lens"]["top_k"]


def test_the_lens_record_carries_the_crystallisation_stats(client):
    """`crossover_layer` is what the brain marks, so it has to survive the trip."""
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["lens"]}
    ).json()["job_id"]

    trace = _poll(client, job_id).json()["trace"]
    stats = trace["passes"][0]["stats"]

    assert "crossover_layer" in stats
    assert "top1_agreement_by_layer" in stats
    assert "final_layer_agreement" in stats


def test_a_trace_with_no_passes_is_capture_only(client):
    """The default must stay exactly what it was before `passes` existed."""
    job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 3}).json()["job_id"]

    trace = _poll(client, job_id).json()["trace"]

    assert trace["passes"] == []
    assert all(
        layer["logit_lens"] is None for step in trace["steps"] for layer in step["layers"]
    )
    # capture data is still all there
    assert trace["steps"][0]["layers"][0]["resid_norm"] > 0


def test_an_unknown_pass_is_rejected_without_enqueueing(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["not-a-pass"]}
    )
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


def test_a_failing_pass_fails_the_job_rather_than_returning_a_bare_trace(
    client, tiny_model, monkeypatch
):
    """A capture-only trace returned after a requested pass raised would be
    indistinguishable from one that never asked for the pass."""
    from app.service import app as app_module

    class Exploding:
        name = "lens"

        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _trace, _residuals):
            raise RuntimeError("lens exploded")

    monkeypatch.setattr(app_module, "LogitLensPass", Exploding)

    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["lens"]}
    ).json()["job_id"]

    body = _poll(client, job_id).json()
    assert body["status"] == "error"
    assert "lens exploded" in body["error"]
    assert body["trace"] is None


# -- GET /trace/{id} progress ----------------------------------------------


def test_progress_is_absent_for_a_queued_job(client):
    """"Pending" and "running, at token 0" have to stay distinguishable."""
    first = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 4}).json()["job_id"]
    second = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 4}).json()["job_id"]

    body = client.get(f"/trace/{second}").json()
    if body["status"] == "pending":  # the worker may already have picked it up
        assert body["progress"] is None

    _poll(client, first)
    _poll(client, second)


def test_progress_reports_both_phases_over_the_life_of_a_job(client, tiny_model):
    """Polled readings must identify the phase and count that phase's own unit
    of work: tokens while generating, layers while running the lens."""
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 4, "passes": ["lens"]}
    ).json()["job_id"]

    readings: list[tuple[str, int, int]] = []
    deadline = time.time() + 10.0
    while time.time() < deadline:
        body = client.get(f"/trace/{job_id}").json()
        p = body["progress"]
        if p is not None:
            reading = (p["phase"], p["done"], p["total"])
            if not readings or readings[-1] != reading:
                readings.append(reading)
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.005)

    assert readings, "no progress was ever reported"
    phases = [phase for phase, _d, _t in readings]
    assert "generating" in phases
    assert "lens" in phases
    # phases never interleave or go back
    assert phases == sorted(phases, key=["generating", "lens"].index)

    generating = [(d, t) for phase, d, t in readings if phase == "generating"]
    assert all(total == 4 for _d, total in generating)
    assert [d for d, _t in generating] == sorted(d for d, _t in generating)

    lens = [(d, t) for phase, d, t in readings if phase == "lens"]
    assert all(total == tiny_model.cfg.n_layers for _d, total in lens)
    assert lens[-1][0] == tiny_model.cfg.n_layers  # the sweep reaches the last layer


def test_progress_never_goes_backwards_across_polls(client):
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 4, "passes": ["lens"]}
    ).json()["job_id"]

    order = {"generating": 0, "lens": 1}
    previous: tuple[int, int] | None = None
    deadline = time.time() + 10.0
    while time.time() < deadline:
        body = client.get(f"/trace/{job_id}").json()
        p = body["progress"]
        if p is not None:
            current = (order[p["phase"]], p["done"])
            if previous is not None:
                assert current >= previous, f"progress went backwards: {previous} -> {current}"
            previous = current
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.005)

    assert previous is not None


def test_a_finished_job_reports_its_status_without_depending_on_progress(client):
    job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 2}).json()["job_id"]
    body = _poll(client, job_id).json()

    assert body["status"] == "done"
    assert body["trace"] is not None
    assert "progress" in body  # the field is always present, even if null


def test_steer_jobs_still_poll_through_the_same_endpoint(client):
    """Adding `progress` to the response must not have changed /steer's shape."""
    job_id = client.post(
        "/steer",
        json={"prompt": PROMPT, "max_tokens": 2, "layer": 2, "feature_idx": 5, "coefficient": 1.0},
    ).json()["job_id"]

    body = _poll(client, job_id).json()
    assert body["status"] == "done"
    assert body["trace"]["steering"] == {"layer": 2, "feature_idx": 5, "coefficient": 1.0}
    assert body["error"] is None
