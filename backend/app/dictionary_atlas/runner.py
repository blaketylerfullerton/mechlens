"""Runs queued atlas builds one at a time, each in its own child process.

A separate process rather than the service's compute worker: UMAP is minutes
of CPU, and running it on the worker would block every trace behind it. The
build reads weights from disk and never touches the loaded model, so it takes
no compute lock. Cancelling terminates the process.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .store import AtlasBuildStore

# The directory holding the `app` package, so `python -m app...` resolves the
# same code the service is running, installed or not.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class AtlasBuildRunner:
    def __init__(self, store: AtlasBuildStore, python: str = sys.executable, poll_seconds: float = 0.5):
        self.store = store
        self.python = python
        self.poll_seconds = poll_seconds
        self.pending: queue.Queue[str] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def submit(self, build_id: str) -> None:
        self.pending.put(build_id)
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, name="atlas-builds", daemon=True)
                self.thread.start()

    def _loop(self) -> None:
        while True:
            build_id = self.pending.get()
            try:
                self.execute(build_id)
            except Exception as exc:  # never let one build kill the queue
                self.store.update(build_id, status="failed", phase="failed",
                                  error=f"Could not run the build: {exc}")
            finally:
                self.pending.task_done()

    def execute(self, build_id: str) -> None:
        build = self.store.get(build_id)
        if build["status"] != "queued":
            return
        out = self.store.directory(build_id)
        out.mkdir(parents=True, exist_ok=True)
        spec_path = out / "spec.json"
        spec_path.write_text(json.dumps(build["spec"]))
        self.store.update(build_id, status="running", phase="starting", started_at=time.time())

        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(PACKAGE_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]))
        with (out / "build.log").open("wb") as log:
            process = subprocess.Popen(
                [self.python, "-m", "app.dictionary_atlas.build", str(spec_path), str(out)],
                stdout=subprocess.PIPE, stderr=log, cwd=PACKAGE_ROOT, env=env,
            )
            errors: list[str] = []
            reader = threading.Thread(target=self._read, args=(build_id, process, errors), daemon=True)
            reader.start()
            while process.poll() is None:
                if self.store.get(build_id)["status"] == "cancelling":
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(self.poll_seconds)
            reader.join(timeout=10)

        if self.store.get(build_id)["status"] == "cancelling":
            self.store.update(build_id, status="cancelled", phase="stopped")
        elif process.returncode == 0:
            self.store.update(build_id, status="completed", phase="done", result=self._result(out),
                              finished_at=time.time())
        else:
            message = errors[-1] if errors else self._log_tail(out) or f"Build exited with code {process.returncode}"
            self.store.update(build_id, status="failed", phase="failed", error=message,
                              finished_at=time.time())

    def _read(self, build_id: str, process: subprocess.Popen, errors: list[str]) -> None:
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if "error" in event:
                errors.append(str(event["error"]))
            elif event.get("phase") and event["phase"] != "done":
                self.store.update(build_id, status="running", phase=event["phase"],
                                  done=event.get("done"), total=event.get("total"))

    @staticmethod
    def _result(out: Path) -> dict:
        record = json.loads((out / "record.json").read_text())
        body = (out / "atlas.json").read_bytes()
        return dict(record=record, atlas_bytes=len(body), atlas_sha256=hashlib.sha256(body).hexdigest())

    @staticmethod
    def _log_tail(out: Path) -> str:
        try:
            lines = (out / "build.log").read_text(errors="replace").strip().splitlines()
        except OSError:
            return ""
        return lines[-1][:500] if lines else ""
