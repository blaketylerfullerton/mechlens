"""GET /stats: the header readout's source.

The numbers themselves are the host's and cannot be asserted, so these pin the
two properties the interface actually depends on — that the route answers
without a model, and that a missing reading stays distinguishable from a zero
one rather than being rendered as free capacity.
"""

from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from app.service import stats
from app.service.app import create_app


@pytest.fixture
def stats_client(tmp_path):
    # model=None deliberately: /stats has to answer during the warm-up window,
    # which is the longest wait in the product and the one where "is this
    # machine doing anything" is least answerable from the rest of the screen.
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")
    with TestClient(app) as c:
        yield c


def test_stats_answers_while_the_model_is_still_loading(stats_client):
    body = stats_client.get("/stats").json()
    assert stats_client.get("/health").json()["status"] == "loading"
    assert body["ram_total_bytes"] > 0
    assert 0 < body["ram_used_bytes"] <= body["ram_total_bytes"]


def test_absent_gpu_utilisation_is_null_not_zero(monkeypatch, stats_client):
    """An unreadable reading and an idle GPU must not print the same thing:
    "0%" claims the machine is doing nothing, which is a fabricated number."""
    monkeypatch.setattr(stats, "_smi_cache", (0.0, None))
    monkeypatch.setattr(
        stats.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError)
    )
    assert stats_client.get("/stats").json()["gpu_util_pct"] is None


@pytest.mark.parametrize(
    "stdout",
    ["[N/A]\n", "", "not a number\n"],
    ids=["not-supported", "empty", "garbage"],
)
def test_unparseable_utilisation_is_null(monkeypatch, stdout):
    """Some parts report [N/A] for fields they do not expose (GB10 does this
    for memory). An unparseable number is a missing number."""
    monkeypatch.setattr(stats, "_smi_cache", (0.0, None))
    monkeypatch.setattr(
        stats.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout, ""),
    )
    assert stats._gpu_utilisation() is None


def test_utilisation_is_cached_rather_than_forked_per_poll(monkeypatch):
    """The browser drives this on a 2s timer; each reading is a process spawn
    unless the TTL holds."""
    calls = {"n": 0}

    def counting_run(*_a, **_k):
        calls["n"] += 1
        return subprocess.CompletedProcess([], 0, "42\n", "")

    monkeypatch.setattr(stats, "_smi_cache", (0.0, None))
    monkeypatch.setattr(stats.subprocess, "run", counting_run)

    assert stats._gpu_utilisation() == 42
    assert stats._gpu_utilisation() == 42
    assert calls["n"] == 1
