"""Stable export identities and retryable disk snapshots; no model work."""
import hashlib
import json
import re
from pathlib import Path

from ..export_run import export_run

KEY = re.compile(r"[a-f0-9]{64}")


def snapshot_info(store, run_id):
    run = store.get(run_id)
    if run["status"] not in {"completed", "failed", "cancelled", "interrupted"} or not run.get("checkpoint"):
        raise ValueError("Finish or stop training with a checkpoint before saving")
    with store.connect() as db:
        interpretations = [json.loads(row[0]) for row in db.execute("SELECT body FROM interp_jobs ORDER BY id")]
    interpretations = [job for job in interpretations if job.get("run_id") == run_id]
    # Labels/examples alter the snapshot even when the SAE weights are unchanged.
    body = json.dumps(dict(run=run, interp_jobs=interpretations), sort_keys=True, allow_nan=False).encode()
    return dict(snapshot_key=hashlib.sha256(body).hexdigest(), artifact_id=run["checkpoint"]["artifact_id"])


def snapshot_path(store, run_id, key):
    if not re.fullmatch(r"[a-f0-9]{32}", run_id) or not KEY.fullmatch(key):
        raise ValueError("Invalid snapshot identity")
    return Path(store.root) / ".cloud-exports" / run_id / (key + ".zip")


def prepare_snapshot(store, run_id, key):
    path = snapshot_path(store, run_id, key)
    if path.is_file():
        return path
    if snapshot_info(store, run_id)["snapshot_key"] != key:
        raise ValueError("The dictionary changed before saving. Save the updated snapshot instead.")
    export_run(store.root, run_id, path)
    return path
