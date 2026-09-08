"""The FastAPI app: one process, one model, HTTP in front of trace/steer/feature.

`create_app(model=..., label_db_path=...)` takes the same dependency-injection
shape as `LogitLensPass.model` / `SAEPass.saes` — a test hands in the tiny CPU
model and a throwaway label DB instead of paying for gemma's load and the
69MB Neuronpedia export. Importing this module does not itself load a model;
the module-level `app` below loads one when it starts (`lifespan`), on a
warm-up thread so the port binds first — until that finishes, the routes that
need the model answer 503 and GET /health reports "loading".

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
from ..passes import apply
from ..passes.lens import LogitLensPass
from ..sae_cache import DEFAULT_WIDTH, get_sae
from ..schema import SteeringInfo, Trace
from . import jobs
from .models import (
    FeatureResponse,
    HealthResponse,
    JobResponse,
    JobStatusResponse,
    SteerRequest,
    TraceRequest,
)
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
    state: dict[str, object] = {"model": model, "load_error": None}
    sae_provider = sae_provider or (lambda layer: get_sae(layer))

    def load_model() -> None:
        """The blocking load, run once on the lifespan's warm-up thread.

        Doing this inline in `lifespan` would keep uvicorn from binding its
        port until gemma is in memory — which is minutes, not seconds, the
        first time the weights are fetched from the hub — so a browser calling
        the API during startup got a connection error rather than an answer.
        Off-thread, the port is open immediately and `get_model` below can say
        "still loading" instead.
        """
        try:
            state["model"] = model_cache.get_model()
        except Exception as exc:  # reported by /health and as a 503 per route
            state["load_error"] = exc

    def get_model() -> HookedTransformer:
        model_ = state["model"]
        if model_ is not None:
            return model_
        if state["load_error"] is not None:
            raise HTTPException(
                status_code=503, detail=f"model failed to load: {state['load_error']}"
            )
        raise HTTPException(status_code=503, detail="model is still loading, retry in a moment")

    def open_label_store() -> LabelStore:
        return LabelStore(label_db_path) if label_db_path is not None else LabelStore()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        jobs.start_worker()
        if state["model"] is None:  # a test that injected one needs no warm-up
            threading.Thread(target=load_model, name="model-warmup", daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    # Dev-only: lets the Vite frontend call this API directly from the browser
    # instead of going through a same-origin proxy. A regex rather than a fixed
    # 5173 pair because vite walks up to the next free port (5174, 5175, ...)
    # when 5173 is taken, and the browser on that port would otherwise be
    # refused by CORS with no hint as to why.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        """Readiness of the model, so a client can tell "not loaded yet" from
        "not running at all" — the two look identical from a failed fetch."""
        if state["load_error"] is not None:
            return HealthResponse(status="error", detail=str(state["load_error"]))
        return HealthResponse(status="ready" if state["model"] is not None else "loading")

    @app.post("/trace", response_model=JobResponse)
    def post_trace(req: TraceRequest) -> JobResponse:
        m = get_model()
        # Read off the request now: the job callable runs on the worker thread
        # long after this handler has returned.
        wants_lens = "lens" in req.passes

        def run(report: jobs.Reporter) -> Trace:
            with _forward_lock:
                result = generate_trace(
                    m,
                    req.prompt,
                    max_new_tokens=req.max_tokens,
                    on_progress=lambda done, total: report("generating", done, total),
                )

                if wants_lens:
                    # `residuals` is the array capture just built, still in
                    # memory -- the service writes no sidecar, so the pass is
                    # told which hook it came from instead of reading a
                    # ResidualRef. `apply` is what puts the record on the
                    # trace, which is how a client can tell the pass ran.
                    apply(
                        LogitLensPass(
                            model=m,
                            hook=result.hook,
                            verbose=False,
                            on_progress=lambda done, total: report("lens", done, total),
                        ),
                        result.trace,
                        result.residuals,
                    )

            # The residual array falls out of scope with the closure here; it is
            # ~7MB for a 31-token gemma trace and nothing downstream needs it.
            return result.trace

        return JobResponse(job_id=jobs.submit(run))

    @app.get("/trace/{job_id}", response_model=JobStatusResponse)
    def get_trace(job_id: str) -> JobStatusResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        return JobStatusResponse(
            status=job.status, trace=job.result, error=job.error, progress=job.progress
        )

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

        # `_report` is unused: steering does not run enrichment passes, and no
        # client drives a progress indicator off /steer today.
        def run(_report: jobs.Reporter) -> Trace:
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
