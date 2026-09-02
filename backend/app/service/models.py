"""Request/response shapes for the HTTP layer.

Deliberately separate from `schema.py`: `Trace` is the on-disk/wire trace
contract, read by the CLI and any future consumer; these are just what one
HTTP call in and out of this service looks like.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schema import Trace


class TraceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)


class SteerRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    layer: int = Field(ge=0)
    feature_idx: int = Field(ge=0)
    coefficient: float


class JobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    status: Literal["pending", "running", "done", "error"]
    trace: Trace | None = None
    error: str | None = None


class FeatureResponse(BaseModel):
    layer: int
    feature_idx: int
    label: str | None = None  # None: feature exists but Neuronpedia has no explanation
    explainer: str | None = None
    explanation_type: str | None = None
    score: float | None = None
    url: str
