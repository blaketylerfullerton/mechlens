"""SQLite run journal. Each operation owns its connection; readers never share cursors."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS metrics (run_id TEXT, seq INTEGER, body TEXT, PRIMARY KEY(run_id, seq))")
            db.execute("CREATE TABLE IF NOT EXISTS interp_jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.root / "runs.sqlite3", timeout=30)

    def create(self, config: dict) -> dict:
        run = dict(id=uuid.uuid4().hex, config=config, status="queued", phase="queued",
                   created_at=time.time(), updated_at=time.time(), tokens=0,
                   error=None, checkpoint=None, evaluation=None, provenance=None,
                   resume_supported=False)
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?)", (run["id"], json.dumps(run)))
        return run


    def create_batch(self, configs: list[dict], model: str) -> list[dict]:
        batch_id = uuid.uuid4().hex
        runs = []
        with self.connect() as db:
            for index, config in enumerate(configs):
                run = dict(id=uuid.uuid4().hex, config=config, model=model,
                           batch_id=batch_id, batch_index=index, batch_size=len(configs),
                           status="queued", phase="queued", created_at=time.time(), updated_at=time.time(),
                           tokens=0, error=None, checkpoint=None, evaluation=None,
                           provenance=None, resume_supported=False)
                db.execute("INSERT INTO runs VALUES (?, ?)", (run["id"], json.dumps(run)))
                runs.append(run)
        return runs

    def run_directory(self, run_id: str) -> Path:
        root = self.root.resolve()
        path = root / run_id
        if not run_id or path.parent != root or path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Invalid training run directory")
        return path

    def storage_bytes(self, run_id: str) -> int:
        path = self.run_directory(run_id)
        total = 0
        for directory, _, filenames in os.walk(path, followlinks=False):
            for name in filenames:
                file = Path(directory) / name
                try:
                    if not file.is_symlink():
                        total += file.stat().st_size
                except FileNotFoundError:
                    pass  # A checkpoint may have been replaced during this read.
        return total

    def prune_checkpoints(self, run_id: str, keep: str):
        """Called only after the new checkpoint is registered, under the compute lock."""
        directory = self.run_directory(run_id)
        current = (self.root / keep).resolve()
        if current.parent != directory or not current.is_dir():
            raise ValueError("The retained checkpoint must belong to this run")
        for path in directory.glob("checkpoint-*"):
            if path != current and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)

    def delete(self, run_id: str):
        """Remove a terminal run and its owned files; caller excludes compute jobs."""
        source = self.run_directory(run_id)
        trash = self.root / f".deleted-{uuid.uuid4().hex}"
        moved = False
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if json.loads(row[0])["status"] not in TERMINAL:
                    raise ValueError("Stop this run before deleting it")
                if source.exists():
                    source.rename(trash)
                    moved = True
                db.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
                # Parse JSON here for compatibility with SQLite builds without JSON1.
                for job_id, body in db.execute("SELECT id, body FROM interp_jobs").fetchall():
                    if json.loads(body)["run_id"] == run_id:
                        db.execute("DELETE FROM interp_jobs WHERE id=?", (job_id,))
                db.execute("DELETE FROM runs WHERE id=?", (run_id,))
        except Exception:
            if moved:
                trash.rename(source)
            raise
        if moved:
            shutil.rmtree(trash)

    def get(self, run_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0])

    def update(self, run_id: str, **fields) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = json.loads(row[0])
            if run["status"] == "cancelled" and fields.get("status") in {"running", "completed"}:
                fields["status"] = "cancelled"
            # Cancellation wins races with worker progress and completion.
            if run["status"] == "cancelling" and fields.get("status") in {"running", "completed"}:
                fields["status"] = "cancelled" if fields["status"] == "completed" else "cancelling"
            run.update(fields, updated_at=time.time())
            db.execute("UPDATE runs SET body=? WHERE id=?", (json.dumps(run, allow_nan=False), run_id))
        return run

    def list(self, limit: int | None = 100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT body FROM runs ORDER BY rowid DESC LIMIT ?", (limit if limit is not None else -1,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def recover(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for run_id, body in db.execute("SELECT id, body FROM runs").fetchall():
                run = json.loads(body)
                if run["status"] not in TERMINAL:
                    run.update(status="interrupted", updated_at=time.time(),
                               error="Backend stopped before the run finished. Saved checkpoints remain available.")
                    db.execute("UPDATE runs SET body=? WHERE id=?", (json.dumps(run), run_id))

    def cancel(self, run_id: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = json.loads(row[0])
            if run["status"] not in TERMINAL:
                run.update(status="cancelled" if run["status"] == "queued" else "cancelling",
                           updated_at=time.time())
                db.execute("UPDATE runs SET body=? WHERE id=?", (json.dumps(run), run_id))
        return run

    def resume(self, run_id: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = json.loads(row[0])
            if run["status"] not in {"failed", "cancelled", "interrupted"} or not run.get("resume_supported"):
                raise ValueError("Only stopped runs with resumable checkpoints can be resumed")
            run.update(status="queued", phase="queued for resume", error=None, updated_at=time.time())
            db.execute("UPDATE runs SET body=? WHERE id=?", (json.dumps(run), run_id))
        return run

    def metric(self, run_id: str, values: dict):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM metrics WHERE run_id=?", (run_id,)).fetchone()[0]
            body = dict(values, seq=seq, timestamp=time.time())
            db.execute("INSERT INTO metrics VALUES (?, ?, ?)", (run_id, seq, json.dumps(body, allow_nan=False)))
            # Preserve a bounded recent window; the final evaluation lives on the run.
            db.execute("DELETE FROM metrics WHERE run_id=? AND seq <= ?", (run_id, seq - 2000))

    def metrics(self, run_id: str, after: int = 0) -> list[dict]:
        self.get(run_id)
        with self.connect() as db:
            rows = db.execute("SELECT body FROM metrics WHERE run_id=? AND seq>? ORDER BY seq LIMIT 2000",
                              (run_id, after)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def create_interp_job(self, run_id: str, artifact_id: str, feature_ids: list[int], endpoint: str, model: str) -> dict:
        """Create a durable auto-interp work record before dispatching it."""
        job = dict(id=uuid.uuid4().hex, run_id=run_id, artifact_id=artifact_id,
                   feature_ids=feature_ids, endpoint=endpoint, model=model,
                   status="queued", phase="queued", created_at=time.time(), updated_at=time.time(),
                   completed=0, total=len(feature_ids), error=None, result_path=None)
        with self.connect() as db:
            db.execute("INSERT INTO interp_jobs VALUES (?, ?)", (job["id"], json.dumps(job)))
        return job

    def list_interp_jobs(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT body FROM interp_jobs ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_interp_job(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT body FROM interp_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def update_interp_job(self, job_id: str, **fields) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM interp_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = json.loads(row[0])
            if job["status"] == "cancelled" and fields.get("status") in {"running", "completed"}:
                fields["status"] = "cancelled"
            job.update(fields, updated_at=time.time())
            db.execute("UPDATE interp_jobs SET body=? WHERE id=?", (json.dumps(job, allow_nan=False), job_id))
        return job
