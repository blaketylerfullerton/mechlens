"""The in-process job queue: sequencing, non-blocking submit, error isolation."""

from __future__ import annotations

import threading
import time

from app.service import jobs


def setup_function(_fn):
    """Each test gets its own worker: JOBS/_QUEUE are module-global, so a
    stale record from a previous test must not leak into a lookup here."""
    jobs.JOBS.clear()
    jobs.start_worker()


def test_submit_returns_before_the_job_runs():
    release = threading.Event()

    def slow():
        release.wait(timeout=2)
        return "done"

    job_id = jobs.submit(slow)

    # `slow` is blocked on `release`, which nothing has set yet, so submit()
    # returning at all proves it did not wait for the job to finish.
    assert jobs.get(job_id).result is None
    assert jobs.get(job_id).status in ("pending", "running")

    release.set()
    _wait_for(job_id)
    assert jobs.get(job_id).result == "done"


def test_second_job_does_not_start_until_the_first_is_done():
    order: list[str] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def first():
        first_started.set()
        order.append("first-start")
        release_first.wait(timeout=2)
        order.append("first-end")
        return 1

    def second():
        order.append("second-start")
        return 2

    id1 = jobs.submit(first)
    id2 = jobs.submit(second)

    first_started.wait(timeout=2)
    time.sleep(0.05)  # give a (buggy) concurrent worker a chance to also start `second`
    assert jobs.get(id2).status == "pending"
    assert order == ["first-start"]

    release_first.set()
    _wait_for(id1)
    _wait_for(id2)

    assert order == ["first-start", "first-end", "second-start"]


def test_job_raising_is_recorded_as_error_and_worker_keeps_going():
    def boom():
        raise ValueError("kaboom")

    def fine():
        return 42

    id1 = jobs.submit(boom)
    _wait_for(id1)
    assert jobs.get(id1).status == "error"
    assert "kaboom" in jobs.get(id1).error

    id2 = jobs.submit(fine)
    _wait_for(id2)
    assert jobs.get(id2).status == "done"
    assert jobs.get(id2).result == 42


def _wait_for(job_id: str, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if jobs.get(job_id).status in ("done", "error"):
            return
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish in time")
