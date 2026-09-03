"""The FastAPI app: one process, one model, HTTP in front of trace/steer/feature.

`create_app(model=..., label_db_path=...)` takes the same dependency-injection
shape as `LogitLensPass.model` / `SAEPass.saes` — a test hands in the tiny CPU
model and a throwaway label DB instead of paying for gemma's load and the
69MB Neuronpedia export. Importing this module does not itself load a model;
the module-level `app` below only does so when the app actually starts
(`lifespan`) or a route first needs it, whichever comes first.

Concurrency: exactly one job runs at a time because there is exactly one
worker thread (`jobs.start_worker`). `_forward_lock` is a defensive guard
around the forward-pass section, not what enforces that — see design.md.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformer_lens import HookedTransformer

from .. import model_cache
from ..capture import generate_trace
from ..labels import LabelStore, feature_url
from ..sae_cache import DEFAULT_WIDTH, get_sae
from ..schema import SteeringInfo, Trace
from . import jobs
from .models import FeatureResponse, JobResponse, JobStatusResponse, SteerRequest, TraceRequest
from .steering import build_intervention

# Structural upper bound on a feature index for a width, without loading the
# SAE itself — GET /feature stays a cheap DB lookup, per design.md.
WIDTH_D_SAE = {"16k": 16384, "65k": 65536, "262k": 262144}

_forward_lock = threading.Lock()


def create_app(
    model: HookedTransformer | None = None,
    label_db_path: Path | str | None = None,
    sae_provider: Callable[[int], object] | None = None,
) -> FastAPI:
    """`sae_provider(layer) -> SAE-like` defaults to `sae_cache.get_sae`; a
    test overrides it with a fake so /steer does not need a real Gemma Scope
    SAE (which would not even match the tiny CPU test model's dimensions)."""
    state: dict[str, HookedTransformer | None] = {"model": model}
    sae_provider = sae_provider or (lambda layer: get_sae(layer))

    def get_model() -> HookedTransformer:
        if state["model"] is None:
            state["model"] = model_cache.get_model()
        return state["model"]

    def open_label_store() -> LabelStore:
        return LabelStore(label_db_path) if label_db_path is not None else LabelStore()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        get_model()  # load once, at startup, per model_cache's own doc comment
        jobs.start_worker()
        yield

    app = FastAPI(lifespan=lifespan)

    # Dev-only: lets the Vite frontend (localhost:5173) call this API directly
    # from the browser instead of going through a same-origin proxy.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/trace", response_model=JobResponse)
    def post_trace(req: TraceRequest) -> JobResponse:
        m = get_model()

        def run() -> Trace:
            with _forward_lock:
                return generate_trace(m, req.prompt, max_new_tokens=req.max_tokens).trace

        return JobResponse(job_id=jobs.submit(run))

    @app.get("/trace/{job_id}", response_model=JobStatusResponse)
    def get_trace(job_id: str) -> JobStatusResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        return JobStatusResponse(status=job.status, trace=job.result, error=job.error)

    @app.post("/steer", response_model=JobResponse)
    def post_steer(req: SteerRequest) -> JobResponse:
        m = get_model()
        if req.layer >= m.cfg.n_layers:
            raise HTTPException(status_code=422, detail=f"layer must be < {m.cfg.n_layers}")

        sae = sae_provider(req.layer)
        if req.feature_idx >= sae.W_dec.shape[0]:
            raise HTTPException(
                status_code=422,
                detail=f"feature_idx must be < {sae.W_dec.shape[0]} for layer {req.layer}",
            )

        intervention = build_intervention(req.layer, req.feature_idx, req.coefficient, sae=sae)

        def run() -> Trace:
            with _forward_lock:
                result = generate_trace(
                    m, req.prompt, max_new_tokens=req.max_tokens, intervention=intervention
                )
            result.trace.steering = SteeringInfo(
                layer=req.layer, feature_idx=req.feature_idx, coefficient=req.coefficient
            )
            return result.trace

        return JobResponse(job_id=jobs.submit(run))

    @app.get("/feature/{layer}/{idx}", response_model=FeatureResponse)
    def get_feature(layer: int, idx: int) -> FeatureResponse:
        m = get_model()
        if layer < 0 or layer >= m.cfg.n_layers:
            raise HTTPException(status_code=404, detail="layer out of range")

        max_idx = WIDTH_D_SAE.get(DEFAULT_WIDTH)
        if idx < 0 or (max_idx is not None and idx >= max_idx):
            raise HTTPException(status_code=404, detail="feature index out of range")

        with open_label_store() as store:
            label = store.get(layer, idx)

        return FeatureResponse(
            layer=layer,
            feature_idx=idx,
            label=label.text if label else None,
            explainer=label.explainer if label else None,
            explanation_type=label.explanation_type if label else None,
            score=label.score if label else None,
            url=feature_url(layer, idx),
        )

    return app


app = create_app()
