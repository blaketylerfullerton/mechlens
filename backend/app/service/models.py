"""Request/response shapes for the HTTP layer.

Deliberately separate from `schema.py`: `Trace` is the on-disk/wire trace
contract, read by the CLI and any future consumer; these are just what one
HTTP call in and out of this service looks like.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schema import Trace
from .jobs import JobPhase

# The enrichment passes `POST /trace` knows how to run as part of a trace job.
# A Literal rather than a plain `list[str]`: an unrecognised name has to be a
# 422 at the boundary, before a job is enqueued, and this is the one place the
# known set is written down.
TracePass = Literal["lens"]


class TraceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    # Opt-in, and empty by default: the lens pass roughly doubles a short
    # trace's wall time, so a client that does not need per-layer readouts
    # should not pay for them.
    passes: list[TracePass] = Field(default_factory=list)


class SteerRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    layer: int = Field(ge=0)
    feature_idx: int = Field(ge=0)
    coefficient: float


class HealthResponse(BaseModel):
    """`status` is about the model, not the process: the server binds its port
    before gemma is in memory, so "loading" is a normal answer for the first
    few seconds (minutes, if the weights are not in the HF cache yet)."""

    status: Literal["loading", "ready", "error"]
    detail: str | None = None


class JobResponse(BaseModel):
    job_id: str


class JobProgressResponse(BaseModel):
    """How far through its current phase a running job is.

    `from_attributes` so the route can hand over `JobRecord.progress` (a frozen
    dataclass on the worker side) without restating its fields here.
    """

    model_config = ConfigDict(from_attributes=True)

    phase: JobPhase
    done: int
    total: int


class JobStatusResponse(BaseModel):
    status: Literal["pending", "running", "done", "error"]
    trace: Trace | None = None
    error: str | None = None
    # Absent for a queued job and for a running one that has not reported yet,
    # so "pending" stays distinguishable from "running, at token 0".
    progress: JobProgressResponse | None = None


class FeatureResponse(BaseModel):
    layer: int
    feature_idx: int
    label: str | None = None  # None: feature exists but Neuronpedia has no explanation
    explainer: str | None = None
    explanation_type: str | None = None
    score: float | None = None
    url: str
