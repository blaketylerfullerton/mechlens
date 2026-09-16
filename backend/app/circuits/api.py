"""HTTP surface for the Circuits workspace.

Mounted by `service/app.py` the way the training router is. The shape follows
the spec: a client posts a validated source snapshot, gets a job, polls it, then
reads a bounded graph.

The snapshot is posted rather than looked up by trace id on purpose. Service
trace jobs live in memory and die with the process, while the browser keeps its
last trace in IndexedDB -- so the client is the one that still has the trace
after a restart, and an analysis must be startable from what it holds.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..model_cache import MODEL_NAME
from ..service import jobs
from ..service.scheduling import serialized
from .adapter import MAX_PREFIX_TOKENS
from .schema import (
    Analysis,
    AnalysisSettings,
    Annotations,
    CircuitEdge,
    CircuitNode,
    SourceSnapshot,
)


class AnalysisRequest(BaseModel):
    source: SourceSnapshot
    settings: AnalysisSettings = Field(default_factory=AnalysisSettings)


class AnalysisResponse(BaseModel):
    analysis_id: str
    job_id: str


def _validate(source: SourceSnapshot) -> None:
    """Reject what cannot be explained, before any compute is scheduled.

    Deliberately needs no loaded model: an analysis can be validated, queued and
    rejected while the inference model is still warming up, and the bound that
    matters is the measured prefix ceiling rather than the context window.
    """
    if source.model != MODEL_NAME:
        raise HTTPException(
            422, f"attribution supports {MODEL_NAME!r}; this trace is {source.model!r}"
        )
    if not source.prefix_token_ids:
        raise HTTPException(422, "the explanatory prefix is empty")
    # The invariant the whole feature rests on: for a token generated at
    # position j, the prefix is token_ids[:j]. A target inside its own prefix
    # would be explaining a token by itself.
    if len(source.prefix_token_ids) != source.target_position:
        raise HTTPException(
            422,
            f"prefix length {len(source.prefix_token_ids)} does not match target "
            f"position {source.target_position}; the target must not be in its own prefix",
        )
    if len(source.prefix_token_ids) > MAX_PREFIX_TOKENS:
        raise HTTPException(
            422,
            f"prefix of {len(source.prefix_token_ids)} tokens exceeds the supported "
            f"ceiling of {MAX_PREFIX_TOKENS}",
        )
    if source.target_token_id < 0:
        raise HTTPException(422, "target token id out of range")


def router(store, compute_lock):
    api = APIRouter(prefix="/circuits", tags=["circuits"])

    def load(analysis_id: str) -> Analysis:
        try:
            return store.get(analysis_id)
        except KeyError:
            raise HTTPException(404, "Unknown analysis") from None

    @api.post("/analyses", response_model=AnalysisResponse)
    @serialized
    def create_analysis(request: AnalysisRequest) -> AnalysisResponse:
        _validate(request.source)

        analysis = Analysis(
            analysis_id=store.new_id(), source=request.source, settings=request.settings
        )
        store.save(analysis)

        def execute(report):
            from . import adapter

            record = store.get(analysis.analysis_id)
            record.status = "running"
            store.save(record)
            try:
                # The replacement model is a second model instance, so it takes
                # the same lock inference and training use rather than assuming
                # it can run beside them.
                with compute_lock:
                    result = adapter.attribute(
                        record.source.prefix_token_ids,
                        record.source.target_token_id,
                        max_feature_nodes=record.settings.max_feature_nodes,
                        batch_size=record.settings.batch_size,
                        node_threshold=record.settings.node_threshold,
                        edge_threshold=record.settings.edge_threshold,
                        max_edges_per_node=record.settings.max_edges_per_node,
                    )
            except Exception as exc:  # noqa: BLE001 - recorded on the document
                record = store.get(analysis.analysis_id)
                record.status = "error"
                record.error = str(exc)
                store.save(record)
                raise

            record = store.get(analysis.analysis_id)
            record.nodes = [CircuitNode(**vars(n)) for n in result.nodes]
            record.edges = [CircuitEdge(**vars(e)) for e in result.edges]
            record.coverage = result.coverage
            record.provenance = result.provenance
            record.target_probability = result.target_probability
            record.elapsed_s = result.elapsed_s
            record.status = "done"
            store.save(record)
            return analysis.analysis_id

        return AnalysisResponse(analysis_id=analysis.analysis_id, job_id=jobs.submit(execute))

    @api.get("/analyses")
    def list_analyses(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        return {"analyses": store.list(limit=limit, offset=offset), "total": store.count()}

    @api.get("/analyses/{analysis_id}")
    def get_analysis(analysis_id: str):
        """Lifecycle, provenance and coverage — never the graph body."""
        return load(analysis_id).summary()

    @api.get("/analyses/{analysis_id}/graph")
    def get_graph(
        analysis_id: str,
        max_edges_per_node: int | None = Query(None, ge=1, le=100),
        min_influence: float = Query(0.0, ge=0.0),
    ):
        """The drawable payload: retained nodes and their incoming edges.

        Narrowing here re-filters what was retained; it never computes more.
        Widening beyond what was retained needs a new analysis, which is why
        `coverage` travels with the graph.
        """
        analysis = load(analysis_id)
        if analysis.status != "done":
            raise HTTPException(409, f"analysis is {analysis.status}")

        nodes = [n for n in analysis.nodes if n.influence >= min_influence]
        keep = {n.id for n in nodes}
        edges = [e for e in analysis.edges if e.source in keep and e.target in keep]

        if max_edges_per_node is not None:
            by_target: dict[str, list] = {}
            for edge in sorted(edges, key=lambda e: abs(e.weight), reverse=True):
                bucket = by_target.setdefault(edge.target, [])
                if len(bucket) < max_edges_per_node:
                    bucket.append(edge)
            edges = [e for bucket in by_target.values() for e in bucket]

        return {
            "analysis_id": analysis_id,
            "source": analysis.source,
            "target_probability": analysis.target_probability,
            "nodes": nodes,
            "edges": edges,
            "coverage": {**analysis.coverage, "sent_nodes": len(nodes), "sent_edges": len(edges)},
            "annotations": analysis.annotations,
        }

    @api.put("/analyses/{analysis_id}/annotations")
    def put_annotations(analysis_id: str, annotations: Annotations):
        """Interpretation only. Cannot reach a measured value."""
        analysis = load(analysis_id)
        analysis.annotations = annotations
        store.save(analysis)
        return analysis.annotations

    @api.delete("/analyses/{analysis_id}")
    def delete_analysis(analysis_id: str):
        load(analysis_id)
        store.delete(analysis_id)
        return {"deleted": analysis_id}

    @api.get("/features/{layer}/{feature_idx}")
    def get_feature(layer: int, feature_idx: int,
                    max_examples: int = Query(8, ge=1, le=50)):
        """Transcoder feature evidence. Always unlabeled — see evidence.py."""
        from .evidence import describe

        try:
            return describe(layer, feature_idx, max_examples=max_examples)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                503, f"feature evidence is unavailable: {exc}"
            ) from None

    return api
