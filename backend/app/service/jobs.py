"""The in-process job queue: one worker thread, one job at a time.

`generate_trace` is synchronous, GPU-bound torch code — running it inside an
`async def` route would block the event loop for the whole trace. A single
background thread pulling from a `queue.Queue` keeps the FastAPI handlers
non-blocking (they just enqueue and return) while guaranteeing jobs never
overlap: there is exactly one consumer.

No persistence, no retries: `JOBS` is a plain in-memory dict, so a process
restart drops whatever was in flight. Acceptable for a v1 whose traces take
seconds-to-tens-of-seconds and whose only client is local dev/viewer use —
see design.md's Non-Goals. Progress readings are the same kind of thing: job
metadata that lives and dies with the process, never written into a trace.

Progress crosses a thread boundary — the worker writes it, the event loop
serving `GET /trace/{id}` reads it — so a reading is published by assigning a
whole frozen `JobProgress` to `JobRecord.progress`, never by mutating one in
place. A reader therefore sees either the previous reading or the next one,
never a half-written one, and no lock sits on the reporting path.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

from ..schema import Trace

JobStatus = Literal["pending", "running", "done", "error"]

# The phases a trace job passes through, in order. `generating` counts tokens
# because that is the granularity a client can actually observe — one forward
# pass per token. `lens` and `sae` count layers because both passes genuinely
# walk them one at a time. See design.md on progress granularity for why the
# generating phase does not report a layer.
JobPhase = Literal["generating", "sae", "lens"]


@dataclass(frozen=True)
class JobProgress:
    """How far through `phase` a running job is.

    Frozen because it is published by whole-object assignment across a thread
    boundary; see the module docstring.
    """

    phase: JobPhase
    done: int
    total: int


# What a job callable is handed so it can publish progress: (phase, done, total).
class Reporter(Protocol):
    def __call__(self, phase: JobPhase, done: int, total: int) -> None: ...
    def publish(self, trace: Trace) -> None: ...

# A unit of work for the worker. It takes a `Reporter`; a job with nothing to
# report simply ignores the argument.
JobFn = Callable[[Reporter], object]

# Phase order, used to reject a stale reading from an earlier phase. Adding a
# phase means adding it here, in the order jobs reach it. `sae` precedes `lens`
# because the trace job runs the passes in that order — the SAE encoder reads
# only the residual array, so it finishes before the lens starts its per-layer
# unembed and a client watching progress never sees the phase move backwards.
_PHASE_ORDER: tuple[JobPhase, ...] = ("generating", "sae", "lens")


@dataclass
class JobRecord:
    id: str
    status: JobStatus = "pending"
    result: object | None = None
    partial_trace: Trace | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    # None until the job publishes its first reading, so "queued" stays
    # distinguishable from "running and at token 0".
    progress: JobProgress | None = None


JOBS: dict[str, JobRecord] = {}
_QUEUE: queue.Queue[tuple[str, JobFn]] = queue.Queue()
_worker: threading.Thread | None = None


def submit(fn: JobFn) -> str:
    """Enqueue `fn`, to be run on the worker thread with a progress reporter.

    Returns the job id immediately; `fn` has not necessarily started, let
    alone finished, by the time this returns.
    """
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = JobRecord(id=job_id)
    _QUEUE.put((job_id, fn))
    return job_id


def get(job_id: str) -> JobRecord | None:
    return JOBS.get(job_id)


def _reporter_for(job: JobRecord) -> Reporter:
    """A `Reporter` that publishes onto `job`, forward only.

    The forward-only check lives here rather than in each caller so the
    "counters never go backwards within a phase" guarantee holds for every
    producer, and so a callback that fires out of order cannot make a client
    watch progress rewind.
    """

    def report(phase: JobPhase, done: int, total: int) -> None:
        previous = job.progress
        if previous is not None:
            if _PHASE_ORDER.index(phase) < _PHASE_ORDER.index(previous.phase):
                return
            if phase == previous.phase and done <= previous.done:
                return
        job.progress = JobProgress(phase=phase, done=done, total=total)

    class Publisher:
        def __call__(self, phase: JobPhase, done: int, total: int) -> None:
            report(phase, done, total)

        def publish(self, trace: Trace) -> None:
            # Never expose a document the worker will keep mutating.
            job.partial_trace = trace.model_copy(deep=True)

    return Publisher()


def _run_one(job_id: str, fn: JobFn) -> None:
    job = JOBS.get(job_id)
    if job is None:
        # The record was dropped while the job waited in the queue, so nothing
        # can observe a result any more and there is nothing worth running.
        # Skipped rather than raised: this is the only worker, and an
        # exception escaping here used to kill it for the life of the process.
        return
    job.status = "running"
    try:
        job.result = fn(_reporter_for(job))
        job.status = "done"
        job.partial_trace = None
    except Exception as exc:  # noqa: BLE001 - reported on the job, not raised
        job.error = str(exc)
        job.status = "error"


def _worker_loop() -> None:
    while True:
        job_id, fn = _QUEUE.get()
        try:
            _run_one(job_id, fn)
        except Exception:  # noqa: BLE001
            # A failing *job* is recorded on its record by _run_one; reaching
            # here means the bookkeeping around it failed instead. Either way
            # the sole worker has to survive, or every job queued after this
            # one waits forever.
            pass
        finally:
            _QUEUE.task_done()


def start_worker() -> None:
    """Start the background worker thread, once per process.

    Idempotent so `app.service.app`'s startup hook can call it without
    worrying whether a test or an earlier import already has.
    """
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_worker_loop, daemon=True)
        _worker.start()
