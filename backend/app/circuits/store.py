"""Saved analyses on disk, one JSON document each.

Deliberately the same shape as `app/store.py`: a directory of documents written
atomically, no database. An analysis is a few hundred KB once pruned -- the
dense adjacency it came from is not kept, because re-running attribution costs
about six seconds and keeping it would cost ~114MB per graph.

Saved analyses must stay readable with no model loaded and no network, so
everything the UI needs to redraw a graph is in the document.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..store import atomic_write_text
from .schema import Analysis

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "circuits"


class AnalysisStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or DEFAULT_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, analysis_id: str) -> Path:
        # Ids are generated here, but a caller-supplied one must not be able to
        # walk out of the directory.
        if not analysis_id.isalnum():
            raise KeyError(analysis_id)
        return self.root / f"{analysis_id}.json"

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def save(self, analysis: Analysis) -> Path:
        path = self._path(analysis.analysis_id)
        atomic_write_text(path, analysis.model_dump_json(indent=2))
        return path

    def get(self, analysis_id: str) -> Analysis:
        path = self._path(analysis_id)
        if not path.exists():
            raise KeyError(analysis_id)
        return Analysis.model_validate_json(path.read_text())

    def delete(self, analysis_id: str) -> None:
        self._path(analysis_id).unlink(missing_ok=True)

    def list(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Newest first. Reads summaries only — never the graph bodies."""
        paths = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for path in paths[offset : offset + limit]:
            try:
                out.append(Analysis.model_validate_json(path.read_text()).summary())
            except Exception:  # noqa: BLE001
                # A document this version cannot read is skipped rather than
                # breaking the whole listing.
                continue
        return out

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*.json"))

    def recover(self) -> int:
        """Mark analyses that a restart interrupted.

        Jobs live in memory, so anything still queued or running when the
        process died can never finish. Recording that is better than leaving a
        document that claims to be running forever.
        """
        changed = 0
        for path in self.root.glob("*.json"):
            try:
                analysis = Analysis.model_validate_json(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            if analysis.status in ("queued", "running"):
                analysis.status = "interrupted"
                analysis.error = "interrupted by a service restart"
                self.save(analysis)
                changed += 1
        return changed
