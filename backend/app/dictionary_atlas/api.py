"""Build a feature atlas from trained dictionaries on this machine.

    POST /atlas/builds/check          validate inputs, start nothing
    POST /atlas/builds                start (idempotent on build_id)
    GET  /atlas/builds                history
    GET  /atlas/builds/{id}           status, progress, error, provenance
    POST /atlas/builds/{id}/cancel
    GET  /atlas/builds/{id}/atlas     the finished atlas.json

Every call returns quickly; the build runs in `runner`, so a relayed request
never waits on UMAP.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ..training.store import RunStore
from .runner import AtlasBuildRunner
from .store import AtlasBuildStore

# What must match across dictionaries for their decoder rows to share one
# space: the same residual stream of the same model.
IDENTITY_KEYS = ("model", "model_revision", "tokenizer", "tokenizer_revision", "d_in", "normalization")
MAX_INPUTS = 64
MAX_FEATURES = 1_000_000


class AtlasSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    method: Literal["umap", "pca"] = "umap"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    n_neighbors: int = Field(default=15, ge=2, le=200)
    min_dist: float = Field(default=0.1, ge=0, le=1)
    # Smaller than the Gemma Scope build's 200: a local dictionary is
    # thousands of features, not hundreds of thousands.
    min_cluster_size: int = Field(default=50, ge=2, le=10_000)
    pca_dim: int = Field(default=0, ge=0, le=1024)
    max_nodes: int = Field(default=20_000, ge=1000, le=100_000)


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: list[str] = Field(min_length=1, max_length=MAX_INPUTS)
    # The checkpoints the caller saved; a GPU copy that differs is refused so
    # the atlas describes exactly what was saved.
    expected_artifact_ids: dict[str, str] = Field(default_factory=dict)


class BuildRequest(CheckRequest):
    build_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    settings: AtlasSettings = Field(default_factory=AtlasSettings)


def resolve_inputs(store: RunStore, req: CheckRequest) -> tuple[list[dict], dict]:
    """The checkpoints to build from, in layer order, plus their shared model
    identity — or a 422 saying which dictionary is the problem."""
    if len(set(req.run_ids)) != len(req.run_ids):
        raise HTTPException(422, "The same dictionary was selected twice")
    inputs = []
    for run_id in req.run_ids:
        try:
            run = store.get(run_id)
        except KeyError:
            raise HTTPException(422, f"Dictionary {run_id[:8]} is not on this GPU. "
                                     "Connect the GPU that trained it.") from None
        layer = run["config"]["layer"]
        saved = run.get("checkpoint")
        if run["status"] != "completed" or not saved:
            raise HTTPException(422, f"Layer {layer} has not finished training")
        expected = req.expected_artifact_ids.get(run_id)
        if expected and expected != saved["artifact_id"]:
            raise HTTPException(422, f"Layer {layer} on this GPU differs from the saved checkpoint")
        if saved.get("hook") != f"blocks.{layer}.hook_resid_post" or saved.get("normalize_activations") != "none":
            raise HTTPException(422, f"Layer {layer} uses an unsupported hook or preprocessing")
        path = (store.root / saved["path"]).resolve()
        if not path.is_relative_to(store.root.resolve()) or not path.is_dir():
            raise HTTPException(422, f"Layer {layer}: checkpoint files are missing on this GPU")
        inputs.append(dict(run_id=run_id, layer=layer, artifact_id=saved["artifact_id"],
                           features=saved["features"], hook=saved["hook"], tokens=saved.get("tokens"),
                           checkpoint_dir=str(path), manifest=saved))

    first = inputs[0]
    for item in inputs[1:]:
        for key in IDENTITY_KEYS:
            if item["manifest"].get(key) != first["manifest"].get(key):
                raise HTTPException(
                    422,
                    f"Layer {item['layer']} has {key} {item['manifest'].get(key)!r} but layer "
                    f"{first['layer']} has {first['manifest'].get(key)!r}. "
                    "An atlas can only combine dictionaries from the same model.")
    layers = [item["layer"] for item in inputs]
    if len(set(layers)) != len(layers):
        raise HTTPException(422, "Choose one dictionary per layer")
    if sum(item["features"] for item in inputs) > MAX_FEATURES:
        raise HTTPException(422, f"Too many features for one atlas (limit {MAX_FEATURES:,})")

    identity = {key: first["manifest"].get(key) for key in IDENTITY_KEYS}
    identity["n_layers"] = first["manifest"].get("n_layers")
    inputs.sort(key=lambda item: item["layer"])
    for item in inputs:
        del item["manifest"]
    return inputs, identity


def public(build: dict) -> dict:
    """A build record without local filesystem paths."""
    spec = build["spec"]
    inputs = [{k: v for k, v in item.items() if k != "checkpoint_dir"} for item in spec["inputs"]]
    return dict(build, spec=dict(spec, inputs=inputs))


def router(training_store: RunStore, store: AtlasBuildStore, runner: AtlasBuildRunner) -> APIRouter:
    api = APIRouter(prefix="/atlas/builds", tags=["atlas"])

    def get_build(build_id: str) -> dict:
        try:
            return store.get(build_id)
        except KeyError:
            raise HTTPException(404, "Unknown atlas build") from None

    @api.post("/check")
    def check(req: CheckRequest):
        inputs, identity = resolve_inputs(training_store, req)
        return dict(ok=True, identity=identity, layers=[item["layer"] for item in inputs],
                    features=sum(item["features"] for item in inputs))

    @api.post("", status_code=202)
    def create(req: BuildRequest):
        inputs, identity = resolve_inputs(training_store, req)
        spec = dict(inputs=inputs, identity=identity, settings=req.settings.model_dump())
        try:
            build, created = store.create(req.build_id, spec)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if created:
            runner.submit(build["id"])
        return public(build)

    @api.get("")
    def list_builds():
        return [public(build) for build in store.list()]

    @api.get("/{build_id}")
    def read(build_id: str):
        return public(get_build(build_id))

    @api.post("/{build_id}/cancel")
    def cancel(build_id: str):
        get_build(build_id)
        return public(store.cancel(build_id))

    @api.get("/{build_id}/atlas")
    def atlas_file(build_id: str):
        if get_build(build_id)["status"] != "completed":
            raise HTTPException(409, "This atlas has not finished building")
        path = store.directory(build_id) / "atlas.json"
        if not path.is_file():
            raise HTTPException(410, "The atlas file is no longer on this GPU")
        return FileResponse(path, media_type="application/json")

    return api
