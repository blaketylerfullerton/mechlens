"""SQLite run journal. Each operation owns its connection; readers never share cursors."""
from __future__ import annotations

import json
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

    def list(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT body FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
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
