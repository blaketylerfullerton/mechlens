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

import numpy as np

from .. import atlas, model_cache
from ..capture import generate_trace
from ..labels import LabelStore, feature_url
from ..passes import apply
from ..passes.labels import LabelsPass
from ..passes.layout import DEFAULT_ATLAS_SOURCE, LayoutPass
from ..passes.lens import LogitLensPass
from ..passes.sae import SAEPass
from ..sae_cache import DEFAULT_WIDTH, get_sae
from ..schema import SteeringInfo, Trace
from . import jobs
from .models import (
    AtlasArea,
    AtlasNodes,
    AtlasResponse,
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
    atlas_version: str | None = None,
    atlas_source: str | None = DEFAULT_ATLAS_SOURCE,
) -> FastAPI:
    """`sae_provider(layer) -> SAE-like` defaults to `sae_cache.get_sae`; a
    test overrides it with a fake so /steer does not need a real Gemma Scope
    SAE (which would not even match the tiny CPU test model's dimensions).

    `atlas_version` pins one atlas; `atlas_source` (the default) picks the most
    recent build from one source representation. Several atlases coexist by
    design, so serving "whichever was built last" would hand clients a
    different map depending on the order the builds happened to run in.
    """
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
        wants_sae = "sae" in req.passes
        wants_labels = "labels" in req.passes
        sae_layers = req.sae_layers

        if wants_sae and sae_layers is not None:
            over = [layer for layer in sae_layers if layer >= m.cfg.n_layers]
            if over:
                raise HTTPException(
                    status_code=422,
                    detail=f"sae_layers must all be < {m.cfg.n_layers}; got {over}",
                )

        # Checked here rather than on the worker: a missing label DB is a
        # deployment fact that is already true at request time, so failing the
        # job for it would report a runtime error for something knowable now.
        if wants_labels:
            try:
                with open_label_store() as store:
                    if store.stats().get("labelled", 0) == 0:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "the label store holds no labels — build it with "
                                "scripts/import_neuronpedia.py before requesting "
                                "the 'labels' pass"
                            ),
                        )
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as a 422, not a 500
                raise HTTPException(
                    status_code=422,
                    detail=f"the 'labels' pass needs a label store, which could not be opened: {exc}",
                ) from exc

        def run(report: jobs.Reporter) -> Trace:
            with _forward_lock:
                result = generate_trace(
                    m,
                    req.prompt,
                    max_new_tokens=req.max_tokens,
                    on_progress=lambda done, total: report("generating", done, total),
                )

                # Before the lens, so the phases a client observes stay in
                # `_PHASE_ORDER` — and because the SAE pass reads only the
                # residual array, which is already in hand.
                if wants_sae:
                    # Resolved through `sae_provider` — the same injection point
                    # /steer uses — rather than `load_layers`. It is
                    # lru_cached, so the SAEs stay resident across requests
                    # (a 31-token trace encodes in ~0.3s once they are), and a
                    # test can stand in a fake for all of them at once.
                    layers = (
                        sae_layers if sae_layers is not None else list(range(m.cfg.n_layers))
                    )
                    apply(
                        SAEPass(
                            layers=layers,
                            saes={layer: sae_provider(layer) for layer in layers},
                            # The model's device, not `pick_device()`: the
                            # residuals are moved onto it to be encoded, so it
                            # has to be where the SAEs already are. Identical
                            # on the real box (both cuda); the difference shows
                            # up when the two disagree.
                            device=str(m.cfg.device),
                            hook=result.hook,
                            verbose=False,
                            on_progress=lambda done, total: report("sae", done, total),
                        ),
                        result.trace,
                        result.residuals,
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

            # Outside the forward lock: a lookup against SQLite, no GPU work,
            # so it must not hold the guard the forward passes share.
            if wants_labels:
                with open_label_store() as store:
                    apply(
                        LabelsPass(store=store, verbose=False),
                        result.trace,
                        result.residuals,
                    )

            # Not a requestable pass: positions are what makes the features
            # drawable, so a client asking for features wants them placed. It
            # runs whenever the SAE pass did, and reports the atlas's absence
            # rather than failing when none has been built — an unbuilt atlas
            # is a deployment state, not an error, and the record says which.
            if wants_sae:
                with open_label_store() as store:
                    apply(
                        LayoutPass(
                            store=store,
                            atlas_version=atlas_version,
                            source=atlas_source,
                            verbose=False,
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

    @app.get("/atlas", response_model=AtlasResponse)
    def get_atlas(sample: int | None = None, version: str | None = None) -> AtlasResponse:
        """The atlas, whole or sampled — positions, clusters, names, record.

        Deliberately not behind `get_model()`: the atlas is a precomputed table
        in SQLite, so this answers while gemma is still loading and on a
        deployment with no GPU at all. That is a requirement, not an
        optimisation — the brain draws its node cloud before any trace exists.

        `sample` caps the node count using the same deterministic subsample the
        idle asset is cut with, so a client asking twice gets the same nodes.
        """
        with open_label_store() as store:
            record = store.atlas_record(version or atlas_version, source=atlas_source)
            # 404 rather than an empty atlas: "no atlas has been built" and "an
            # atlas that placed nothing" are different facts, and a client that
            # cannot tell them apart will render one as the other.
            if record is None:
                asked = version or atlas_version
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"no atlas has been built"
                        + (f" for version {asked}" if asked else "")
                        + (f" from source '{atlas_source}'" if atlas_source and not asked else "")
                        + " — build one with scripts/build_feature_atlas.py"
                    ),
                )
            rows = store.layout_all(record.atlas_version)
            clusters = store.clusters(record.atlas_version)

        n_total = len(rows)
        if sample is not None and 0 < sample < n_total:
            picks = atlas.subsample_indices(n_total, sample, seed=record.seed)
            rows = [rows[i] for i in picks]

        positions = np.array([(r.x, r.y, r.z) for r in rows], dtype=np.float64).reshape(-1, 3)
        quantised, extent = atlas.quantise_positions(positions)

        metrics = record.metrics
        return AtlasResponse(
            atlas_version=record.atlas_version,
            source=str(record.params.get("source", "")),
            note=(
                "The feature atlas: one fixed position per SAE feature. Near means "
                "similar; far means nothing, because the projection preserves local "
                "neighbourhoods and distorts larger distances."
            ),
            knn_preservation=metrics.get("knn_preservation"),
            knn_k=metrics.get("knn_k"),
            explainer_ami=metrics.get("explainer_ami"),
            positions_sha256=record.positions_sha256,
            release=record.release,
            width=record.width,
            seed=record.seed,
            layers=str(record.params.get("layers", "")),
            extent=extent,
            n_sampled=len(rows),
            n_total=n_total,
            nodes=AtlasNodes(
                layer=[r.layer for r in rows],
                feature=[r.feature for r in rows],
                cluster=[r.cluster for r in rows],
                xyz=quantised.reshape(-1).tolist(),
            ),
            areas=[
                AtlasArea(
                    cluster=c.cluster,
                    name=c.name,
                    n_members=c.n_members,
                    centroid=c.centroid,
                    spread=c.spread,
                    coherence=c.coherence,
                    baseline_coherence=c.baseline_coherence,
                    explainers=c.explainers,
                )
                for c in clusters
            ],
        )

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
