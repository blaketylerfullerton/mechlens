"""SQLite journal of atlas builds, beside the training runs they read.

Same shape as `training.store.RunStore`: one JSON body per row, each call owns
its connection, cancellation wins races with progress, and a restart marks
anything unfinished as interrupted rather than leaving it claiming to run.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class AtlasBuildStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.outputs = self.root / "atlas-builds"
        self.outputs.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS builds (id TEXT PRIMARY KEY, body TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.root / "atlas_builds.sqlite3", timeout=30)

    def directory(self, build_id: str) -> Path:
        return self.outputs / build_id

    def create(self, build_id: str, spec: dict) -> tuple[dict, bool]:
        """(record, created). Re-posting the same id returns the existing
        record, so a caller retrying after a lost response starts nothing new."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM builds WHERE id=?", (build_id,)).fetchone()
            if row is not None:
                existing = json.loads(row[0])
                if existing["spec"] != spec:
                    raise ValueError("A different atlas build already uses this id")
                return existing, False
            now = time.time()
            build = dict(id=build_id, spec=spec, status="queued", phase="queued", done=None,
                         total=None, error=None, result=None, created_at=now, updated_at=now)
            db.execute("INSERT INTO builds VALUES (?, ?)", (build_id, json.dumps(build)))
        return build, True

    def get(self, build_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT body FROM builds WHERE id=?", (build_id,)).fetchone()
        if row is None:
            raise KeyError(build_id)
        return json.loads(row[0])

    def list(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT body FROM builds ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def update(self, build_id: str, **fields) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM builds WHERE id=?", (build_id,)).fetchone()
            if row is None:
                raise KeyError(build_id)
            build = json.loads(row[0])
            # A stopped build stays stopped whatever the child reports late.
            if build["status"] in TERMINAL:
                return build
            if build["status"] == "cancelling" and fields.get("status") in {"running", "completed"}:
                fields["status"] = "cancelling" if fields["status"] == "running" else "cancelled"
            build.update(fields, updated_at=time.time())
            db.execute("UPDATE builds SET body=? WHERE id=?", (json.dumps(build, allow_nan=False), build_id))
        return build

    def cancel(self, build_id: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM builds WHERE id=?", (build_id,)).fetchone()
            if row is None:
                raise KeyError(build_id)
            build = json.loads(row[0])
            if build["status"] not in TERMINAL:
                build.update(status="cancelled" if build["status"] == "queued" else "cancelling",
                             updated_at=time.time())
                db.execute("UPDATE builds SET body=? WHERE id=?", (json.dumps(build), build_id))
        return build

    def recover(self):
        """Mark unfinished builds interrupted. Nothing is resumed on purpose:
        a restarted service should not silently start a long job."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for build_id, body in db.execute("SELECT id, body FROM builds").fetchall():
                build = json.loads(body)
                if build["status"] not in TERMINAL:
                    build.update(status="interrupted", updated_at=time.time(),
                                 error="The GPU service stopped before this atlas finished. Retry to build it again.")
                    db.execute("UPDATE builds SET body=? WHERE id=?", (json.dumps(build), build_id))
