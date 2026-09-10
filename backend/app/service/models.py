"""Request/response shapes for the HTTP layer.

Deliberately separate from `schema.py`: `Trace` is the on-disk/wire trace
contract, read by the CLI and any future consumer; these are just what one
HTTP call in and out of this service looks like.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schema import Trace
from .jobs import JobPhase

# The enrichment passes `POST /trace` knows how to run as part of a trace job.
# A Literal rather than a plain `list[str]`: an unrecognised name has to be a
# 422 at the boundary, before a job is enqueued, and this is the one place the
# known set is written down.
TracePass = Literal["lens", "sae", "labels"]


class TraceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    # Opt-in, and empty by default: the lens pass roughly doubles a short
    # trace's wall time, so a client that does not need per-layer readouts
    # should not pay for them.
    passes: list[TracePass] = Field(default_factory=list)
    # Which layers the SAE pass encodes. None = every layer, which is what a
    # client should ask for; the subset exists because 26 resident 16k SAEs come
    # to ~7.9GB, fine on a unified-memory box and not fine on a 16GB discrete
    # GPU. Ignored unless `passes` includes "sae".
    sae_layers: list[int] | None = None

    @model_validator(mode="after")
    def _check_pass_dependencies(self) -> TraceRequest:
        """Reject a pass whose input another pass has to produce.

        The labels pass reads `LayerState.features`, so without the SAE pass it
        raises on an empty trace *on the worker thread* — a 500-shaped failure
        for a request that was wrong when it arrived. Caught here it is a 422
        and no job is enqueued.
        """
        if "labels" in self.passes and "sae" not in self.passes:
            raise ValueError(
                "the 'labels' pass labels the features the 'sae' pass records, "
                "so it cannot run without it — request passes=['sae', 'labels']"
            )
        if self.sae_layers is not None:
            if not self.sae_layers:
                raise ValueError("sae_layers must name at least one layer, or be omitted")
            if any(layer < 0 for layer in self.sae_layers):
                raise ValueError("sae_layers must not contain a negative layer")
            if len(set(self.sae_layers)) != len(self.sae_layers):
                raise ValueError("sae_layers must not repeat a layer")
        return self


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


class AtlasNodes(BaseModel):
    """The node table, transposed and quantised.

    Column-wise rather than a list of objects because it is the same four
    values repeated ~98,000 times: `[{"layer":0,"feature":0,...}, ...]` spends
    most of its bytes on repeated key names. `xyz` is flat int16 triples over
    `extent` — the quantisation error is far below a pixel at any camera
    distance and it halves the payload. Identical to the idle asset's shape,
    deliberately: one wire format, one parser on the client.
    """

    layer: list[int]
    feature: list[int]
    cluster: list[int]
    xyz: list[int]


class AtlasArea(BaseModel):
    """One cluster of the atlas.

    `name` is null whenever the naming was not earned, and that is a
    measurement outcome to be shown rather than a gap to be filled — the build
    names a cluster only when its members' label coherence beats its own random
    baseline by the recorded margin. `coherence` and `baseline_coherence` are
    carried so a consumer can say *how* it was earned or missed, and a null
    coherence ("never measured") stays distinct from a low one ("measured, and
    it lost").
    """

    cluster: int
    name: str | None = None
    n_members: int
    centroid: tuple[float, float, float]
    spread: float
    coherence: float | None = None
    baseline_coherence: float | None = None
    explainers: str = ""


class AtlasResponse(BaseModel):
    """The atlas as the brain consumes it.

    Field names match `build_feature_atlas.py`'s idle asset rather than this
    module's usual style, because the client parses both with one function: the
    static asset is what the brain draws before a trace exists, and this is the
    same atlas served whole. A second shape here would mean a second parser and
    two ways for them to disagree.
    """

    atlas_version: str
    source: str
    note: str
    # The atlas's own fidelity, carried so the interface can publish how much
    # of the original neighbourhood structure survived instead of asking the
    # reader to trust the picture. None means not measured — never 0.
    knn_preservation: float | None = None
    knn_k: float | None = None
    explainer_ami: float | None = None
    # Identity, so a trace drawn against a different atlas is detectable.
    positions_sha256: str
    release: str
    width: str
    seed: int
    layers: str = ""
    extent: float
    quantisation: str = "int16, position = value / 32767 * extent"
    n_sampled: int
    n_total: int
    nodes: AtlasNodes
    areas: list[AtlasArea] = Field(default_factory=list)


class FeatureResponse(BaseModel):
    layer: int
    feature_idx: int
    label: str | None = None  # None: feature exists but Neuronpedia has no explanation
    explainer: str | None = None
    explanation_type: str | None = None
    score: float | None = None
    url: str
