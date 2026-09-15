"""Training endpoints share the service's existing compute queue and model."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..capture import generate_trace
from ..passes import apply
from ..passes.sae import SAEPass
from ..passes.lens import LogitLensPass
from ..service import jobs
from ..service.scheduling import serialized
from .auto_interp import generate, load_profiles
from .examples import collect_feature_examples, save_feature_examples
from .runner import Cancelled, TrainingConfig, load_artifact, train
from .store import RunStore, TERMINAL


class BatchRequest(BaseModel):
    config: TrainingConfig
    layers: list[int] = Field(min_length=1, max_length=256)
    model_repository: str


class InspectRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    max_tokens: int = Field(default=16, ge=1, le=256)


class ModelRequest(BaseModel):
    repository: str


class FeatureExamplesRequest(BaseModel):
    feature_ids: list[int] = Field(min_length=1, max_length=32)
    max_examples: int = Field(default=20, ge=1, le=50)
    max_sequences: int = Field(default=128, ge=1, le=2048)


class AutoInterpRequest(BaseModel):
    feature_ids: list[int] = Field(min_length=1, max_length=32)
    profile: str = Field(default="local", min_length=1, max_length=80)

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
        return dict(model=model_name, model_repository=next((repo for repo, name in SUPPORTED_TRAINING_MODELS.items()
                    if name == model_name), None) if readiness == "ready" else None,
                    layers=layers, d_in=dimension, device=device,
                    supported_models=list(SUPPORTED_TRAINING_MODELS), multi_layer=True, storage_management=True,
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
        return [summary(run) for run in store.list(limit=None)]

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

    @api.post("/batches", status_code=202)
    @serialized
    def create_batch(req: BatchRequest):
        from ..model_cache import SUPPORTED_TRAINING_MODELS
        model = get_model()
        if SUPPORTED_TRAINING_MODELS.get(req.model_repository) != model.cfg.model_name:
            raise HTTPException(409, "Prepare the selected model before starting training")
        layers = sorted(set(req.layers))
        if len(layers) != len(req.layers) or any(layer < 0 or layer >= model.cfg.n_layers for layer in layers):
            raise HTTPException(422, f"Choose unique layers between 0 and {model.cfg.n_layers - 1}")
        if req.config.context_size > model.cfg.n_ctx:
            raise HTTPException(422, f"Context size must be at most {model.cfg.n_ctx}")
        if str(model.cfg.normalization_type).endswith("Pre"):
            raise HTTPException(422, "Training needs unprocessed model weights")
        configs = [dict(req.config.model_dump(), layer=layer) for layer in layers]
        batch = store.create_batch(configs, model.cfg.model_name)
        for run in batch:
            enqueue(run, model)
        return dict(batch_id=batch[0]["batch_id"], runs=[summary(run) for run in batch])

    @api.post("/batches/{batch_id}/cancel")
    @serialized
    def cancel_batch(batch_id: str):
        batch = [run for run in store.list(limit=None) if run.get("batch_id") == batch_id]
        if not batch:
            raise HTTPException(404, "Unknown training batch")
        return [summary(store.cancel(run["id"])) for run in batch]

    def compute_busy():
        return any(job.status in {"pending", "running"} for job in list(jobs.JOBS.values()))

    @api.get("/storage")
    def storage():
        occupied = compute_busy()
        records = []
        for run in store.list(limit=None):
            reason = "Stop this run before deleting it" if run["status"] not in TERMINAL else (
                "Wait for queued and running operations to finish before deleting" if occupied else None)
            records.append(dict(id=run["id"], model=run.get("model") or (run.get("provenance") or {}).get("model"),
                layer=run["config"]["layer"], features=run["config"]["features"], status=run["status"],
                created_at=run["created_at"], has_checkpoint=bool(run.get("checkpoint")),
                bytes=store.storage_bytes(run["id"]), delete_blocked_reason=reason))
        disk = shutil.disk_usage(store.root)
        return dict(runs=records, total_bytes=sum(run["bytes"] for run in records),
                    free_bytes=disk.free, checkpoint_policy="latest")

    @api.delete("/runs/{run_id}")
    @serialized
    def delete_run(run_id: str):
        get_run(run_id)
        # Submission and model switching use this same scheduling lock. Waiting
        # jobs may still need the checkpoint, even when the training run is terminal.
        if compute_busy():
            raise HTTPException(409, "Wait for queued and running operations to finish before deleting")
        if not compute_lock.acquire(blocking=False):
            raise HTTPException(409, "Compute is using training files. Try again when it finishes")
        try:
            store.delete(run_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            compute_lock.release()
        return dict(deleted=run_id)

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

    @api.get("/runs/{run_id}/report")
    def validation_report(run_id: str):
        run = summary(get_run(run_id))
        return JSONResponse(dict(schema_version=1, run_id=run_id, status=run["status"],
            config=run["config"], provenance=run.get("provenance"),
            checkpoint=run.get("checkpoint"), correctness=run.get("validation"),
            quality=run.get("evaluation")), headers={
                "Content-Disposition": f'attachment; filename="training-{run_id}-report.json"'})

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

    @api.post("/runs/{run_id}/examples", status_code=202)
    @serialized
    def collect_examples(run_id: str, req: FeatureExamplesRequest):
        run = get_run(run_id)
        saved = run["checkpoint"]
        if not saved:
            raise HTTPException(422, "No checkpoint has been saved yet")
        feature_ids = sorted(set(req.feature_ids))
        if any(feature < 0 or feature >= saved["features"] for feature in feature_ids):
            raise HTTPException(422, "Feature IDs must belong to this dictionary")

        def execute(report):
            with compute_lock:
                model = get_model()
                sae, manifest = load_artifact(store, run_id, model)
                result = collect_feature_examples(
                    model, sae, manifest, TrainingConfig(**run["config"]), feature_ids,
                    req.max_examples, req.max_sequences,
                    progress=lambda done, total: report("sae", done, total))
                path = save_feature_examples(store.root, run_id, result)
                store.update(run_id, examples=dict(path=str(path.relative_to(store.root)),
                    artifact_id=manifest["artifact_id"], feature_ids=feature_ids,
                    sequences_scanned=result["sequences_scanned"], tokens_scanned=result["tokens_scanned"]))
                return result

        return {"job_id": jobs.submit(execute)}

    @api.get("/runs/{run_id}/examples/jobs/{job_id}")
    def example_job(run_id: str, job_id: str):
        get_run(run_id)
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown feature-example job")
        return dict(status=job.status, error=job.error,
                    progress=None if job.progress is None else dict(phase=job.progress.phase,
                                                                     done=job.progress.done,
                                                                     total=job.progress.total))

    @api.get("/runs/{run_id}/examples")
    def read_examples(run_id: str):
        record = get_run(run_id).get("examples")
        if not record:
            raise HTTPException(404, "No feature examples have been collected for this run")
        path = (store.root / record["path"]).resolve()
        if not path.is_relative_to(store.root.resolve()) or not path.is_file():
            raise HTTPException(404, "Feature examples are unavailable")
        return json.loads(path.read_text())

    @api.get("/interp-jobs")
    def list_interp_jobs():
        """Durable queue records for the forthcoming self-hosted auto-interp worker."""
        return store.list_interp_jobs()

    @api.get("/runs/{run_id}/interp-jobs")
    def list_run_interp_jobs(run_id: str):
        get_run(run_id)
        return [job for job in store.list_interp_jobs() if job["run_id"] == run_id]

    @api.get("/interp-profiles")
    def interp_profiles():
        return [profile.public() for profile in load_profiles(store.root).values()]

    @api.post("/runs/{run_id}/interp-jobs", status_code=202)
    @serialized
    def create_interp_job(run_id: str, req: AutoInterpRequest):
        run = get_run(run_id)
        saved, examples = run["checkpoint"], run.get("examples")
        if not saved or not examples:
            raise HTTPException(422, "Collect activation examples before creating an auto-interp job")
        profile = load_profiles(store.root).get(req.profile)
        if profile is None:
            raise HTTPException(422, "No matching local auto-interp profile is configured")
        path = (store.root / examples["path"]).resolve()
        if not path.is_relative_to(store.root.resolve()) or not path.is_file():
            raise HTTPException(422, "Saved activation examples are unavailable")
        report = json.loads(path.read_text())
        feature_ids = sorted(set(req.feature_ids))
        if report.get("artifact_id") != saved["artifact_id"] or any(not report["examples"].get(str(fid)) for fid in feature_ids):
            raise HTTPException(422, "Each requested feature needs examples from this exact checkpoint")
        job = store.create_interp_job(run_id, saved["artifact_id"], feature_ids, profile.base_url, profile.model)
        store.update_interp_job(job["id"], profile=profile.public(), source_examples=examples["path"])
        def execute(_reporter):
            store.update_interp_job(job["id"], status="running", phase="generating")
            records = []
            try:
                for index, feature_id in enumerate(feature_ids, 1):
                    if store.get_interp_job(job["id"])["status"] == "cancelled":
                        return store.get_interp_job(job["id"])
                    response = generate(profile, saved["artifact_id"], feature_id, report["examples"][str(feature_id)])
                    records.append(dict(feature_id=feature_id, status="unverified", **response))
                    store.update_interp_job(job["id"], completed=index, records=records)
                result_path = store.root / run_id / f"auto-interp-{job['id']}.json"
                temporary = result_path.with_name(f".{result_path.name}")
                temporary.write_text(json.dumps(dict(schema_version=1, job_id=job["id"], artifact_id=saved["artifact_id"], records=records)))
                os.replace(temporary, result_path)
                return store.update_interp_job(job["id"], status="completed", phase="completed", result_path=str(result_path.relative_to(store.root)))
            except Exception as exc:
                store.update_interp_job(job["id"], status="failed", phase="failed", error=str(exc), records=records)
                raise
        jobs.submit(execute)
        return store.get_interp_job(job["id"])

    @api.post("/interp-jobs/{job_id}/cancel")
    def cancel_interp_job(job_id: str):
        try:
            return store.update_interp_job(job_id, status="cancelled", phase="cancelled")
        except KeyError:
            raise HTTPException(404, "Unknown auto-interp job") from None

    return api


def default_root():
    return Path(os.environ.get("MECHLENS_TRAINING_DIR", Path(__file__).resolve().parents[2] / "data" / "training"))
