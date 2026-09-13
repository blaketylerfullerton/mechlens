"""Training endpoints share the service's existing compute queue and model."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..capture import generate_trace
from ..passes import apply
from ..passes.sae import SAEPass
from ..passes.lens import LogitLensPass
from ..service import jobs
from ..service.scheduling import serialized
from .runner import Cancelled, TrainingConfig, load_artifact, train
from .store import RunStore


class InspectRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    max_tokens: int = Field(default=16, ge=1, le=256)


class ModelRequest(BaseModel):
    repository: str


def router(get_model, compute_lock, store: RunStore, prepare_model=None):
    api = APIRouter(prefix="/training", tags=["training"])

    def get_run(run_id):
        try:
            return store.get(run_id)
        except KeyError:
            raise HTTPException(404, "Unknown training run") from None

    def summary(run):
        # Long private corpus text is retained on disk, not repeated in history polling.
        return dict(run, config={k: v for k, v in run["config"].items()
                                 if k not in {"training_text", "evaluation_text"}})

    @api.get("/options")
    def options():
        from ..model_cache import MODEL_NAME, SUPPORTED_TRAINING_MODELS, pick_device
        try:
            model = get_model()
            model_name, layers, dimension = model.cfg.model_name, model.cfg.n_layers, model.cfg.d_model
            readiness = "ready"
            device = str(model.cfg.device)
        except HTTPException as exc:
            model_name, layers, dimension = MODEL_NAME, None, None
            readiness = str(exc.detail)
            device = pick_device()[0]
        return dict(model=model_name, layers=layers, d_in=dimension, device=device,
                    supported_models=list(SUPPORTED_TRAINING_MODELS),
                    readiness=readiness, architecture="standard", datasets=["tiny-stories", "text"],
                    resume_supported=True, compute_policy="One training or inference job at a time",
                    estimate_note="Runtime and memory presets are unbenchmarked on this host.")

    @api.post("/model", status_code=202)
    @serialized
    def prepare(req: ModelRequest):
        if prepare_model is None:
            raise HTTPException(409, "Model switching is unavailable in this backend")
        return {"job_id": prepare_model(req.repository)}

    @api.get("/runs")
    def list_runs():
        return [summary(run) for run in store.list()]

    @api.post("/runs", status_code=202)
    @serialized
    def create_run(config: TrainingConfig):
        model = get_model()
        if config.layer >= model.cfg.n_layers:
            raise HTTPException(422, f"Layer must be below {model.cfg.n_layers}")
        if config.context_size > model.cfg.n_ctx:
            raise HTTPException(422, f"Context size must be at most {model.cfg.n_ctx}")
        if str(model.cfg.normalization_type).endswith("Pre"):
            raise HTTPException(422, "Training needs unprocessed model weights")
        run = store.create(config.model_dump())
        enqueue(run, model)
        return summary(run)

    def enqueue(run, model, resume=False):
        def execute(_report):
            if store.get(run["id"])["status"] == "cancelled":
                return
            try:
                with compute_lock:
                    train(store, run["id"], model, **({"resume": True} if resume else {}))
            except Cancelled:
                store.update(run["id"], status="cancelled", phase="stopped")
            except Exception as exc:
                store.update(run["id"], status="failed", error=str(exc))
            finally:
                # The training result belongs in SQLite, not the ephemeral trace queue.
                if str(model.cfg.device).startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()

        jobs.submit(execute)

    @api.post("/runs/{run_id}/resume", status_code=202)
    @serialized
    def resume_run(run_id: str):
        get_run(run_id)
        model = get_model()
        try:
            run = store.resume(run_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        enqueue(run, model, resume=True)
        return summary(run)

    @api.get("/runs/{run_id}")
    def read_run(run_id: str):
        return summary(get_run(run_id))

    @api.get("/runs/{run_id}/metrics")
    def read_metrics(run_id: str, after: int = Query(default=0, ge=0)):
        get_run(run_id)
        return store.metrics(run_id, after)

    @api.post("/runs/{run_id}/cancel")
    def cancel(run_id: str):
        get_run(run_id)
        return summary(store.cancel(run_id))

    @api.get("/runs/{run_id}/files/{filename}")
    def artifact_file(run_id: str, filename: str):
        saved = get_run(run_id)["checkpoint"]
        if not saved or filename not in {"cfg.json", "sae_weights.safetensors", "manifest.json"}:
            raise HTTPException(404, "Artifact file is unavailable")
        path = (store.root / saved["path"] / filename).resolve()
        if not path.is_relative_to(store.root.resolve()) or not path.is_file():
            raise HTTPException(404, "Artifact file is unavailable")
        return FileResponse(path, filename=filename)

    @api.post("/runs/{run_id}/trace", status_code=202)
    @serialized
    def inspect(run_id: str, req: InspectRequest):
        saved = get_run(run_id)["checkpoint"]
        if not saved:
            raise HTTPException(422, "No checkpoint has been saved yet")
        model = get_model()
        def execute(report):
            with compute_lock:
                sae, manifest = load_artifact(store, run_id, model)
                captured = generate_trace(model, req.prompt, max_new_tokens=req.max_tokens,
                    on_progress=lambda done, total: report("generating", done, total))
                apply(SAEPass(layers=[manifest["layer"]], saes={manifest["layer"]: sae},
                    artifact_id=manifest["artifact_id"], feature_count=manifest["features"],
                    width=str(manifest["features"]), hook=captured.hook,
                    device=str(model.cfg.device), verbose=False), captured.trace, captured.residuals)
                report("sae", 1, 1)
                apply(LogitLensPass(model=model, hook=captured.hook, verbose=False,
                    on_progress=lambda done, total: report("lens", done, total)), captured.trace, captured.residuals)
                return captured.trace
        return {"job_id": jobs.submit(execute)}

    return api


def default_root():
    return Path(os.environ.get("MECHLENS_TRAINING_DIR", Path(__file__).resolve().parents[2] / "data" / "training"))
