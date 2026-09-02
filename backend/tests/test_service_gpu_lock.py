"""The defensive forward-pass lock (design.md's GPU serialization decision).

Not exercising real concurrency here — the worker thread's single-consumer
design is what actually serializes jobs (see test_jobs.py). This just checks
the lock is held for the duration of the forward-pass call inside a job, so
a future second caller of `generate_trace` would be blocked by it.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.service import app as app_module
from app.service import jobs
from app.service.app import create_app

PROMPT = "Once upon a time there was a"


@pytest.fixture(scope="module")
def tiny_model():
    from transformer_lens import HookedTransformer

    try:
        m = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    except Exception as exc:
        pytest.skip(f"tiny-stories-1M unavailable: {exc}")
    m.eval()
    return m


def test_forward_lock_is_held_during_generation(monkeypatch, tiny_model, tmp_path):
    observed = {"locked": None}
    real_generate_trace = app_module.generate_trace

    def spy(*args, **kwargs):
        observed["locked"] = app_module._forward_lock.locked()
        return real_generate_trace(*args, **kwargs)

    monkeypatch.setattr(app_module, "generate_trace", spy)
    jobs.JOBS.clear()
    app = create_app(model=tiny_model, label_db_path=tmp_path / "labels.db")

    with TestClient(app) as client:
        job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 1}).json()["job_id"]
        deadline = time.time() + 5
        while time.time() < deadline and jobs.get(job_id).status not in ("done", "error"):
            time.sleep(0.01)

    assert observed["locked"] is True
    assert not app_module._forward_lock.locked()  # released once the job finished
