"""The narrow seam between mechlens and `circuit-tracer`.

Everything this package knows about the upstream library lives here. The rest
of the Circuits code deals in plain dataclasses, so replacing the attribution
engine later means rewriting this file and nothing else.

Two things it deliberately does not do:

* It never accepts a prompt string. Attribution runs on the exact token ids a
  trace recorded. Decoding a saved prefix and re-tokenizing it is lossy -- for
  gemma-2-2b it doubles the BOS -- so the ids travel end to end.
* It never claims an intervention tested more than its own operation.

Loading is cached per process the way `model_cache` caches the plain model, but
kept separate: a `ReplacementModel` carries transcoders and their hooks, and is
not interchangeable with the `HookedTransformer` the trace passes use.
"""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from importlib.metadata import version

import torch

from ..model_cache import MODEL_NAME, pick_device

# The per-layer GemmaScope transcoders. Not the residual-stream SAEs in
# sae_cache: a transcoder approximates an MLP block and defines its own feature
# space, so its indices, hooks and evidence share nothing with a Gemma Scope
# feature at the same layer/index. See the spec's identity rules.
TRANSCODER_SET = "mntss/gemma-scope-transcoders"

# HuggingFace name, as circuit-tracer expects it. model_cache uses the
# TransformerLens alias ("gemma-2-2b") for the same weights.
HF_MODEL_NAME = "google/gemma-2-2b"

SUPPORTED_MODELS = {MODEL_NAME: HF_MODEL_NAME}

# The supported prefix ceiling, set from measurement rather than from n_ctx.
# At 64 tokens attribution costs ~16-47s and peaks around 24GB depending on the
# node budget; the model's 8192-token window is irrelevant long before that
# becomes usable. See docs/circuits-cir01-report.md section 8.
MAX_PREFIX_TOKENS = 64


@dataclass(frozen=True)
class GraphNode:
    """One node of the attribution graph, in the caller's vocabulary."""

    id: str
    kind: str  # "feature" | "error" | "embed" | "logit"
    layer: int | None
    position: int | None
    feature_idx: int | None
    activation: float | None
    influence: float


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: float


@dataclass
class Attribution:
    """A completed attribution, already reduced to what mechlens stores.

    `nodes` and `edges` are the retained graph after pruning. `coverage` keeps
    the numbers that say how much was left out, because an omitted edge is not
    a zero effect and the UI has to be able to say so.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    input_token_ids: list[int]
    target_token_id: int
    target_probability: float
    coverage: dict
    provenance: dict
    elapsed_s: float


def transcoder_revision(repo: str = TRANSCODER_SET) -> str | None:
    """Artifact fingerprint, or None when the hub cannot be reached.

    None means unknown and is recorded as such. It is never back-filled from
    whatever happens to be in the local cache today.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().repo_info(repo).sha
    except Exception:
        return None


# One replacement model per process, held explicitly rather than in an
# lru_cache: unloading has to release hooks and drop the reference on purpose,
# and reaching into a cache's internals to do that is worse than owning a slot.
_CACHE: dict[tuple[str, str], object] = {}
_CACHE_LOCK = threading.Lock()


def get_replacement_model(model_name: str = MODEL_NAME, transcoder_set: str = TRANSCODER_SET):
    """Load once per process and reuse, like `model_cache.get_model`.

    Only one combination is held at a time: a second replacement model would
    double an already ~12GB residency for no benefit, since exactly one
    model/transcoder pair is supported.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model for attribution: {model_name!r}")

    key = (model_name, transcoder_set)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

        from circuit_tracer import ReplacementModel

        device, dtype = pick_device()
        t0 = time.time()
        model = ReplacementModel.from_pretrained(
            SUPPORTED_MODELS[model_name], transcoder_set, dtype=dtype, device=device
        )
        print(
            f"loaded replacement model {model_name} + {transcoder_set} "
            f"in {time.time() - t0:.1f}s | device={device} dtype={dtype}"
        )
        _CACHE.clear()
        _CACHE[key] = model
        return model


def unload() -> None:
    """Drop the replacement model and its transcoders.

    Called when the service switches models. Hooks are released explicitly
    because the library's `__del__` raises a TypeError during interpreter
    teardown, so leaving cleanup to garbage collection is not sound.
    """
    with _CACHE_LOCK:
        for model in _CACHE.values():
            reset = getattr(model, "reset_hooks", None)
            if reset is not None:
                try:
                    reset()
                except Exception:  # noqa: BLE001 - unloading must not raise
                    pass
        _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_loaded() -> bool:
    return bool(_CACHE)


def runtime_provenance(model, transcoder_set: str) -> dict:
    """Everything needed to say which weights and code produced a number."""
    return {
        "circuit_tracer": version("circuit-tracer"),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "transformer_lens": version("transformer-lens"),
        "model": MODEL_NAME,
        "model_hf_name": SUPPORTED_MODELS[MODEL_NAME],
        # Unknown rather than guessed when the hub is unreachable.
        "model_revision": getattr(model.cfg, "model_revision", None),
        "transcoder_set": transcoder_set,
        "transcoder_revision": transcoder_revision(transcoder_set),
        "device": str(model.cfg.device),
        "dtype": str(model.cfg.dtype),
    }


def _node_id(kind: str, layer: int | None, position: int | None, feature_idx: int | None) -> str:
    """Stable, human-readable node identity.

    Feature nodes are keyed by (layer, position, feature_idx) rather than by
    their row in the adjacency matrix, so an id survives re-running attribution
    with a different node budget.
    """
    if kind == "feature":
        return f"F{layer}_{position}_{feature_idx}"
    if kind == "error":
        return f"E{layer}_{position}"
    if kind == "embed":
        return f"I{position}"
    return f"L{feature_idx}"


def attribute(
    prefix_token_ids: list[int],
    target_token_id: int,
    *,
    model=None,
    transcoder_set: str = TRANSCODER_SET,
    max_feature_nodes: int = 5000,
    batch_size: int = 256,
    node_threshold: float = 0.7,
    edge_threshold: float = 0.9,
    max_edges_per_node: int = 10,
    progress=None,
) -> Attribution:
    """Attribute `target_token_id` back through the computation on `prefix_token_ids`.

    The prefix is token ids and the target is one explicit vocabulary index, so
    this explains the token a trace actually produced rather than whatever the
    library would have picked as salient.

    Pruning is separate from computation, and both are separate from what the
    API later chooses to send. `node_threshold`/`edge_threshold` decide what is
    *retained*; `max_edges_per_node` caps the strongest incoming edges kept per
    node, which is what makes the result small enough to draw. The counts that
    say how much was dropped are returned in `coverage`.

    There is no cancellation: the upstream call runs to completion once entered.
    `progress` is therefore only called at phase boundaries, and a caller that
    wants to stop this has to stop the process.
    """
    from circuit_tracer import attribute as ct_attribute
    from circuit_tracer.graph import compute_node_influence, prune_graph

    if not prefix_token_ids:
        raise ValueError("empty prefix")
    if model is None:
        model = get_replacement_model(transcoder_set=transcoder_set)
    if not 0 <= target_token_id < model.cfg.d_vocab_out:
        raise ValueError("target token id out of range")
    if len(prefix_token_ids) > model.cfg.n_ctx:
        raise ValueError("prefix exceeds the model context window")

    if progress:
        progress("attributing", 0, 1)

    t0 = time.time()
    graph = ct_attribute(
        list(prefix_token_ids),
        model,
        attribution_targets=torch.tensor([target_token_id]),
        max_feature_nodes=max_feature_nodes,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0

    if progress:
        progress("pruning", 1, 2)

    return _reduce(
        graph,
        model=model,
        transcoder_set=transcoder_set,
        prefix_token_ids=list(prefix_token_ids),
        target_token_id=target_token_id,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_edges_per_node=max_edges_per_node,
        max_feature_nodes=max_feature_nodes,
        elapsed_s=elapsed,
    )


def _reduce(graph, *, model, transcoder_set, prefix_token_ids, target_token_id,
            node_threshold, edge_threshold, max_edges_per_node,
            max_feature_nodes, elapsed_s) -> Attribution:
    """Turn a dense circuit-tracer Graph into retained nodes and edges.

    Node order in the adjacency matrix is
    [active_features, error_nodes, embed_nodes, logit_nodes], with
    n_layers * n_pos error nodes and one embed node per position. Rows are
    targets, columns are sources.
    """
    from circuit_tracer.graph import compute_node_influence, prune_graph

    n_pos = graph.n_pos
    n_layers = graph.cfg.n_layers
    n_features = int(graph.selected_features.shape[0])
    n_errors = n_layers * n_pos
    n_logits = len(graph.logit_targets)

    node_mask, edge_mask, _ = (t.cpu() for t in prune_graph(graph, node_threshold, edge_threshold))
    adjacency = graph.adjacency_matrix.cpu().float()

    logit_weights = torch.zeros(adjacency.shape[0])
    logit_weights[-n_logits:] = graph.logit_probabilities.cpu().float()
    influence = compute_node_influence(adjacency, logit_weights)

    # activation_values is aligned to active_features (one per active feature),
    # NOT to selected_features. Indexing it by position-within-selected reads a
    # different feature's activation entirely -- a wrong number that looks
    # perfectly plausible in a UI. Gather through selected_features so both
    # arrays below share one index space.
    selected = graph.selected_features.cpu()
    activations = graph.activation_values.cpu().float()[selected]
    feature_rows = graph.active_features.cpu()[selected]

    def describe(index: int) -> tuple[str, int | None, int | None, int | None, float | None]:
        if index < n_features:
            layer, position, feature_idx = (int(v) for v in feature_rows[index])
            return "feature", layer, position, feature_idx, float(activations[index])
        index -= n_features
        if index < n_errors:
            return "error", index // n_pos, index % n_pos, None, None
        index -= n_errors
        if index < n_pos:
            return "embed", None, index, None, None
        return "logit", None, None, target_token_id, None

    kept = node_mask.nonzero().flatten().tolist()
    nodes, index_to_id = [], {}
    for i in kept:
        kind, layer, position, feature_idx, activation = describe(i)
        node_id = _node_id(kind, layer, position, feature_idx)
        index_to_id[i] = node_id
        nodes.append(GraphNode(
            id=node_id, kind=kind, layer=layer, position=position,
            feature_idx=feature_idx, activation=activation,
            influence=float(influence[i]),
        ))

    # Keep only the strongest incoming edges per retained node. This is the step
    # that makes the payload drawable: the full pruned matrix is hundreds of
    # thousands of edges, and a backward-exploration UI reads one node's inputs
    # at a time. Dropped edges are counted, never silently discarded.
    pruned = adjacency.clone()
    pruned[~edge_mask] = 0
    edges, retained_total = [], 0
    for i in kept:
        row = pruned[i]
        sources = row.nonzero().flatten()
        sources = sources[torch.isin(sources, torch.tensor(kept))]
        retained_total += int(sources.numel())
        if sources.numel() > max_edges_per_node:
            order = row[sources].abs().argsort(descending=True)[:max_edges_per_node]
            sources = sources[order]
        for j in sources.tolist():
            edges.append(GraphEdge(
                source=index_to_id[j], target=index_to_id[i], weight=float(row[j])
            ))

    coverage = {
        "total_nodes": int(adjacency.shape[0]),
        "retained_nodes": len(nodes),
        "total_edges_nonzero": int((adjacency != 0).sum()),
        "edges_after_pruning": retained_total,
        "sent_edges": len(edges),
        "n_active_features": int(graph.active_features.shape[0]),
        "n_selected_features": n_features,
        "n_error_nodes": n_errors,
        "n_embed_nodes": n_pos,
        "n_layers": n_layers,
        "n_positions": n_pos,
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
        "max_edges_per_node": max_edges_per_node,
        "max_feature_nodes": max_feature_nodes,
        "note": (
            "Omitted nodes and edges are below the retention thresholds, not zero "
            "effects. Error nodes carry the computation the transcoders do not "
            "reconstruct and are never dropped for being unexplained."
        ),
    }

    return Attribution(
        nodes=nodes,
        edges=edges,
        input_token_ids=list(prefix_token_ids),
        target_token_id=target_token_id,
        target_probability=float(graph.logit_probabilities[0]),
        coverage=coverage,
        provenance=runtime_provenance(model, transcoder_set),
        elapsed_s=elapsed_s,
    )
