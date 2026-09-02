"""The in-process job queue: one worker thread, one job at a time.

`generate_trace` is synchronous, GPU-bound torch code — running it inside an
`async def` route would block the event loop for the whole trace. A single
background thread pulling from a `queue.Queue` keeps the FastAPI handlers
non-blocking (they just enqueue and return) while guaranteeing jobs never
overlap: there is exactly one consumer.

No persistence, no retries: `JOBS` is a plain in-memory dict, so a process
restart drops whatever was in flight. Acceptable for a v1 whose traces take
seconds-to-tens-of-seconds and whose only client is local dev/viewer use —
see design.md's Non-Goals.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

JobStatus = Literal["pending", "running", "done", "error"]


@dataclass
class JobRecord:
    id: str
    status: JobStatus = "pending"
    result: object | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


JOBS: dict[str, JobRecord] = {}
_QUEUE: queue.Queue[tuple[str, Callable[[], object]]] = queue.Queue()
_worker: threading.Thread | None = None


def submit(fn: Callable[[], object]) -> str:
    """Enqueue `fn`, to be run with no arguments on the worker thread.

    Returns the job id immediately; `fn` has not necessarily started, let
    alone finished, by the time this returns.
    """
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = JobRecord(id=job_id)
    _QUEUE.put((job_id, fn))
    return job_id


def get(job_id: str) -> JobRecord | None:
    return JOBS.get(job_id)


def _run_one(job_id: str, fn: Callable[[], object]) -> None:
    job = JOBS[job_id]
    job.status = "running"
    try:
        job.result = fn()
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - reported on the job, not raised
        job.error = str(exc)
        job.status = "error"


def _worker_loop() -> None:
    while True:
        job_id, fn = _QUEUE.get()
        try:
            _run_one(job_id, fn)
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
