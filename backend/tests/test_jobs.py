"""The in-process job queue: sequencing, non-blocking submit, error isolation,
and the progress readings a running job publishes."""

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

    def slow(_report):
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

    def first(_report):
        first_started.set()
        order.append("first-start")
        release_first.wait(timeout=2)
        order.append("first-end")
        return 1

    def second(_report):
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
    def boom(_report):
        raise ValueError("kaboom")

    def fine(_report):
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


# --------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------


def test_progress_is_absent_until_the_job_publishes_one():
    """"Queued" and "running, at token 0" are different answers, so a job that
    has published nothing must report no reading rather than a zeroed one."""
    release = threading.Event()
    started = threading.Event()

    def waits(_report):
        started.set()
        release.wait(timeout=2)
        return "done"

    blocker = jobs.submit(waits)
    queued = jobs.submit(lambda _report: "done")

    started.wait(timeout=2)
    assert jobs.get(queued).status == "pending"
    assert jobs.get(queued).progress is None
    assert jobs.get(blocker).progress is None  # running, but has reported nothing

    release.set()
    _wait_for(blocker)
    _wait_for(queued)


def test_progress_is_readable_while_the_job_is_still_running():
    published = threading.Event()
    release = threading.Event()

    def reports(report):
        report("generating", 3, 20)
        published.set()
        release.wait(timeout=2)
        return "done"

    job_id = jobs.submit(reports)
    published.wait(timeout=2)

    # Read from this thread while the worker is parked inside `reports`.
    progress = jobs.get(job_id).progress
    assert jobs.get(job_id).status == "running"
    assert (progress.phase, progress.done, progress.total) == ("generating", 3, 20)

    release.set()
    _wait_for(job_id)


def test_terminal_status_does_not_depend_on_a_progress_reading():
    """A job that never reports, and one that reports then fails, both have to
    reach a terminal status a client can act on."""

    def silent(_report):
        return 7

    def reports_then_fails(report):
        report("generating", 1, 20)
        raise ValueError("kaboom")

    quiet = jobs.submit(silent)
    _wait_for(quiet)
    assert jobs.get(quiet).status == "done"
    assert jobs.get(quiet).result == 7
    assert jobs.get(quiet).progress is None

    failed = jobs.submit(reports_then_fails)
    _wait_for(failed)
    assert jobs.get(failed).status == "error"
    assert "kaboom" in jobs.get(failed).error


def test_progress_never_goes_backwards_within_a_phase():
    """Readings are published forward-only, so a repeated or out-of-order
    callback cannot make a polling client watch progress rewind."""
    seen: list[tuple[str, int, int]] = []
    release = threading.Event()

    def reports(report):
        for done in (1, 5, 3, 5, 9):  # 3 and the repeated 5 must not take effect
            report("generating", done, 20)
            p = jobs.get(job_id).progress
            seen.append((p.phase, p.done, p.total))
        report("lens", 4, 26)
        seen.append(_reading(job_id))
        report("generating", 20, 20)  # an earlier phase never displaces a later one
        seen.append(_reading(job_id))
        release.wait(timeout=2)
        return "done"

    job_id = jobs.submit(reports)
    release.set()
    _wait_for(job_id)

    assert seen == [
        ("generating", 1, 20),
        ("generating", 5, 20),
        ("generating", 5, 20),
        ("generating", 5, 20),
        ("generating", 9, 20),
        ("lens", 4, 26),
        ("lens", 4, 26),
    ]


def test_the_sae_phase_sits_between_generating_and_lens():
    """A trace job runs the SAE pass before the lens, so a reading from either
    neighbour must not displace the phase the job has actually reached."""
    seen: list[tuple[str, int, int]] = []
    release = threading.Event()

    def reports(report):
        report("generating", 4, 20)
        seen.append(_reading(job_id))
        report("sae", 3, 26)
        seen.append(_reading(job_id))
        report("generating", 20, 20)  # behind sae: ignored
        seen.append(_reading(job_id))
        report("lens", 1, 26)
        seen.append(_reading(job_id))
        report("sae", 26, 26)  # behind lens: ignored
        seen.append(_reading(job_id))
        release.wait(timeout=2)
        return "done"

    job_id = jobs.submit(reports)
    release.set()
    _wait_for(job_id)

    assert seen == [
        ("generating", 4, 20),
        ("sae", 3, 26),
        ("sae", 3, 26),
        ("lens", 1, 26),
        ("lens", 1, 26),
    ]


def test_phase_order_lists_every_phase():
    """A phase missing from _PHASE_ORDER raises ValueError inside the reporter,
    on the worker thread, where it would surface as a failed job."""
    from typing import get_args

    assert set(get_args(jobs.JobPhase)) == set(jobs._PHASE_ORDER)


def test_a_queued_job_reports_no_running_phase_until_the_first_finishes():
    """Progress reporting must not have loosened the single-worker guarantee:
    only one job can be mid-phase at a time."""
    first_reported = threading.Event()
    release_first = threading.Event()

    def first(report):
        report("generating", 2, 20)
        first_reported.set()
        release_first.wait(timeout=2)
        return 1

    def second(report):
        report("generating", 1, 20)
        return 2

    id1 = jobs.submit(first)
    id2 = jobs.submit(second)

    first_reported.wait(timeout=2)
    time.sleep(0.05)  # a concurrent worker would have started `second` by now
    assert jobs.get(id1).status == "running"
    assert jobs.get(id2).status == "pending"
    assert jobs.get(id2).progress is None

    release_first.set()
    _wait_for(id1)
    _wait_for(id2)
    assert jobs.get(id2).progress.done == 1


def _reading(job_id: str) -> tuple[str, int, int]:
    p = jobs.get(job_id).progress
    return (p.phase, p.done, p.total)


def test_a_dropped_record_does_not_kill_the_worker():
    """The queue can outlive a record — a caller clearing JOBS while a job is
    still queued. There is only one worker, so it has to survive that: before
    this was handled, the KeyError escaped `_worker_loop` and every job
    submitted afterwards hung forever."""
    release = threading.Event()
    started = threading.Event()

    def blocks(_report):
        started.set()
        release.wait(timeout=2)
        return "first"

    blocker = jobs.submit(blocks)
    started.wait(timeout=2)
    orphan = jobs.submit(lambda _report: "orphan")

    del jobs.JOBS[orphan]  # the record goes away while the job sits in the queue
    release.set()
    _wait_for(blocker)

    # The worker survived the orphan and is still serving new work.
    after = jobs.submit(lambda _report: "after")
    _wait_for(after)
    assert jobs.get(after).result == "after"
    assert jobs.get(orphan) is None


def test_partial_snapshot_is_detached_and_survives_failure():
    from factories import make_result

    ready, release = threading.Event(), threading.Event()
    trace = make_result().trace

    def run(report):
        report.publish(trace)
        trace.completion = "mutated after publication"
        ready.set()
        release.wait(2)
        raise ValueError("analysis failed")

    job_id = jobs.submit(run)
    try:
        assert ready.wait(2)
        job = jobs.get(job_id)
        assert job.status == "running"
        assert job.result is None
        assert job.partial_trace.completion != trace.completion
    finally:
        release.set()
    _wait_for(job_id)
    assert job.status == "error"
    assert job.partial_trace is not None
