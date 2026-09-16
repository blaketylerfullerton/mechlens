"""The stored shape of a circuits analysis.

Measured data and interpretation are separated on purpose. Everything in
`Analysis` up to and including `coverage` is measurement plus provenance and is
never rewritten; annotations live in their own field and can be edited without
touching a number.
"""
from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "circuits/1"

AnalysisStatus = Literal["queued", "running", "done", "error", "interrupted"]
NodeKind = Literal["feature", "error", "embed", "logit"]


class CircuitNode(BaseModel):
    id: str
    kind: NodeKind
    layer: int | None = None
    position: int | None = None
    # Index into the transcoder's feature space -- NOT a Gemma Scope residual
    # SAE index. The two share no identity even at the same layer.
    feature_idx: int | None = None
    activation: float | None = None
    influence: float


class CircuitEdge(BaseModel):
    source: str
    target: str
    weight: float


class SourceSnapshot(BaseModel):
    """The immutable record of what was explained.

    Holds token ids, never text to be re-tokenized: decoding a prefix and
    encoding it again is lossy, and for gemma-2-2b it silently doubles the BOS.
    """

    trace_id: str
    model: str
    prompt: str | None = None
    prefix_token_ids: list[int]
    prefix_texts: list[str] | None = None
    target_token_id: int
    target_text: str | None = None
    target_position: int


class AnalysisSettings(BaseModel):
    max_feature_nodes: int = Field(default=5000, ge=100, le=10000)
    batch_size: int = Field(default=256, ge=1, le=1024)
    node_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    edge_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    max_edges_per_node: int = Field(default=10, ge=1, le=100)


class Annotations(BaseModel):
    """Interpretation, kept apart from measurement.

    A group label or a note is a claim by whoever wrote it, not a property of
    the model, so it carries its own field and never edits a graph value.
    """

    notes: dict[str, str] = Field(default_factory=dict)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    pinned: list[str] = Field(default_factory=list)


class Analysis(BaseModel):
    schema_version: str = SCHEMA_VERSION
    analysis_id: str
    created_at: float = Field(default_factory=time.time)
    status: AnalysisStatus = "queued"
    error: str | None = None

    source: SourceSnapshot
    settings: AnalysisSettings = Field(default_factory=AnalysisSettings)

    # Populated when the run completes.
    provenance: dict = Field(default_factory=dict)
    coverage: dict = Field(default_factory=dict)
    target_probability: float | None = None
    elapsed_s: float | None = None
    nodes: list[CircuitNode] = Field(default_factory=list)
    edges: list[CircuitEdge] = Field(default_factory=list)

    annotations: Annotations = Field(default_factory=Annotations)

    def summary(self) -> dict:
        """Everything except the graph body — what a list view needs."""
        return self.model_dump(exclude={"nodes", "edges"})
