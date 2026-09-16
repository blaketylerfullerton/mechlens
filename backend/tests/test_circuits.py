"""Circuits: validation, storage, and the shape of what the API sends.

No model and no network. Attribution itself is exercised against the real model
in the CIR-01 prototype (see docs/circuits-cir01-report.md); what matters here is
everything around it — that a target can never be explained by itself, that
annotations cannot reach a measured value, that narrowing a graph never invents
data, and that a restart cannot leave a document claiming to be running.
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.circuits import adapter
from app.circuits.api import router
from app.circuits.schema import Analysis, CircuitEdge, CircuitNode, SourceSnapshot
from app.circuits.store import AnalysisStore


def make_source(**overrides) -> dict:
    source = {
        "trace_id": "abc123",
        "model": "gemma-2-2b",
        "prompt": "The Golden Gate Bridge is located in the city of",
        "prefix_token_ids": [2, 651, 17489, 3250],
        "prefix_texts": ["<bos>", "The", " Golden", " San"],
        "target_token_id": 12288,
        "target_text": " Francisco",
        "target_position": 4,
    }
    source.update(overrides)
    return source


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def client(tmp_path):
    store = AnalysisStore(tmp_path / "circuits")
    app = FastAPI()
    app.include_router(router(store, FakeLock()))
    from app.service import jobs

    jobs.start_worker()
    with TestClient(app) as c:
        yield c, store


def wait_for(client, analysis_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/circuits/analyses/{analysis_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"analysis {analysis_id} never settled")


# --------------------------------------------------------------------------
# the invariant the feature rests on
# --------------------------------------------------------------------------


def test_a_target_cannot_appear_in_its_own_prefix(client):
    c, _ = client
    # position 3 with a 4-token prefix would mean token_ids[:4] explaining
    # token_ids[3] — the answer smuggled into its own question.
    r = c.post("/circuits/analyses", json={"source": make_source(target_position=3)})
    assert r.status_code == 422
    assert "own prefix" in r.json()["detail"]


def test_prefix_and_target_position_must_agree_exactly(client):
    c, _ = client
    r = c.post("/circuits/analyses", json={"source": make_source(target_position=9)})
    assert r.status_code == 422


def test_validation_needs_no_loaded_model(client):
    """Every rejection above happens before any compute is scheduled, so an
    analysis can be refused while the inference model is still warming up."""
    c, store = client
    for bad in (make_source(model="gpt2"),
                make_source(prefix_token_ids=[], target_position=0),
                make_source(prefix_token_ids=[2] * 99, target_position=99)):
        assert c.post("/circuits/analyses", json={"source": bad}).status_code == 422
    assert store.count() == 0


def test_an_unsupported_model_is_named_in_the_error(client):
    c, _ = client
    detail = c.post("/circuits/analyses", json={"source": make_source(model="gpt2")}).json()["detail"]
    assert "gpt2" in detail and "gemma-2-2b" in detail


def test_the_prefix_ceiling_is_the_measured_one_not_the_context_window(client, monkeypatch):
    c, _ = client
    # Stubbed because accepting the request enqueues a job on the process-wide
    # worker, and a real attribution run there would block every later test.
    fake_attribution(monkeypatch)
    n = adapter.MAX_PREFIX_TOKENS
    ok = make_source(prefix_token_ids=[2] * n, target_position=n)
    too_long = make_source(prefix_token_ids=[2] * (n + 1), target_position=n + 1)
    assert c.post("/circuits/analyses", json={"source": too_long}).status_code == 422
    # The accepted one is far below gemma's 8192-token window on purpose.
    assert n < 8192
    accepted = c.post("/circuits/analyses", json={"source": ok})
    assert accepted.status_code == 200
    # Waited on, not fired and forgotten: the worker is process-wide, and a job
    # still queued when this test's stub is torn down would run the real
    # attribution and stall everything behind it.
    wait_for(c, accepted.json()["analysis_id"])


# --------------------------------------------------------------------------
# running, with attribution stubbed out
# --------------------------------------------------------------------------


def fake_attribution(monkeypatch, nodes=None, edges=None, coverage=None):
    nodes = nodes or [
        adapter.GraphNode("F1_3_7", "feature", 1, 3, 7, 4.0, 0.9),
        adapter.GraphNode("F0_1_2", "feature", 0, 1, 2, 1.0, 0.4),
        adapter.GraphNode("E1_3", "error", 1, 3, None, None, 0.2),
        adapter.GraphNode("I0", "embed", None, 0, None, None, 0.1),
        adapter.GraphNode("L12288", "logit", None, None, 12288, None, 1.0),
    ]
    edges = edges or [
        adapter.GraphEdge("F0_1_2", "F1_3_7", 2.0),
        adapter.GraphEdge("E1_3", "F1_3_7", -1.0),
        adapter.GraphEdge("I0", "F0_1_2", 0.5),
        adapter.GraphEdge("F1_3_7", "L12288", 3.0),
    ]

    def stub(prefix, target, **kwargs):
        return adapter.Attribution(
            nodes=nodes, edges=edges, input_token_ids=list(prefix),
            target_token_id=target, target_probability=0.97,
            coverage=coverage or {"total_nodes": 500, "retained_nodes": len(nodes),
                                  "total_edges_nonzero": 99999, "sent_edges": len(edges)},
            provenance={"model": "gemma-2-2b", "transcoder_set": adapter.TRANSCODER_SET,
                        "transcoder_revision": None},
            elapsed_s=6.3,
        )

    monkeypatch.setattr(adapter, "attribute", stub)


def test_a_completed_analysis_carries_its_graph_and_provenance(client, monkeypatch):
    c, _ = client
    fake_attribution(monkeypatch)
    aid = c.post("/circuits/analyses", json={"source": make_source()}).json()["analysis_id"]
    body = wait_for(c, aid)

    assert body["status"] == "done"
    assert body["target_probability"] == 0.97
    assert body["provenance"]["transcoder_set"] == adapter.TRANSCODER_SET
    # Unknown provenance stays null rather than being back-filled.
    assert body["provenance"]["transcoder_revision"] is None
    # The summary is a summary: no graph body in it.
    assert "nodes" not in body and "edges" not in body

    graph = c.get(f"/circuits/analyses/{aid}/graph").json()
    assert len(graph["nodes"]) == 5 and len(graph["edges"]) == 4


def test_error_nodes_survive_into_the_served_graph(client, monkeypatch):
    """Unexplained computation is not dropped for being unexplained."""
    c, _ = client
    fake_attribution(monkeypatch)
    aid = c.post("/circuits/analyses", json={"source": make_source()}).json()["analysis_id"]
    wait_for(c, aid)
    graph = c.get(f"/circuits/analyses/{aid}/graph").json()
    assert any(n["kind"] == "error" for n in graph["nodes"])


def test_coverage_reports_what_was_left_out(client, monkeypatch):
    c, _ = client
    fake_attribution(monkeypatch)
    aid = c.post("/circuits/analyses", json={"source": make_source()}).json()["analysis_id"]
    wait_for(c, aid)
    coverage = c.get(f"/circuits/analyses/{aid}/graph").json()["coverage"]
    # A client can always tell the served graph is a subset.
    assert coverage["retained_nodes"] < coverage["total_nodes"]
    assert coverage["sent_edges"] < coverage["total_edges_nonzero"]


def test_a_failed_run_is_recorded_on_the_document(client, monkeypatch):
    c, store = client

    def boom(*args, **kwargs):
        raise RuntimeError("transcoders unavailable")

    monkeypatch.setattr(adapter, "attribute", boom)
    aid = c.post("/circuits/analyses", json={"source": make_source()}).json()["analysis_id"]
    body = wait_for(c, aid)
    assert body["status"] == "error"
    assert "transcoders unavailable" in body["error"]


def test_the_graph_is_unavailable_until_the_run_finishes(client, monkeypatch):
    c, store = client
    analysis = Analysis(analysis_id=store.new_id(),
                        source=SourceSnapshot(**make_source()))
    store.save(analysis)
    r = c.get(f"/circuits/analyses/{analysis.analysis_id}/graph")
    assert r.status_code == 409 and "queued" in r.json()["detail"]


# --------------------------------------------------------------------------
# narrowing the graph
# --------------------------------------------------------------------------


def test_narrowing_filters_what_was_retained_and_never_adds_to_it(client, monkeypatch):
    c, _ = client
    fake_attribution(monkeypatch)
    aid = c.post("/circuits/analyses", json={"source": make_source()}).json()["analysis_id"]
    wait_for(c, aid)

    full = c.get(f"/circuits/analyses/{aid}/graph").json()
    narrow = c.get(f"/circuits/analyses/{aid}/graph?min_influence=0.5").json()

    assert len(narrow["nodes"]) < len(full["nodes"])
    assert all(n["influence"] >= 0.5 for n in narrow["nodes"])
    # No edge may dangle: both endpoints must still be present.
    ids = {n["id"] for n in narrow["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in narrow["edges"])
    # Narrowing cannot invent a node the analysis never retained.
    assert ids <= {n["id"] for n in full["nodes"]}


def test_capping_edges_per_node_keeps_the_strongest(client, monkeypatch):
    c, _ = client
    edges = [adapter.GraphEdge("F0_1_2", "F1_3_7", 2.0),
             adapter.GraphEdge("E1_3", "F1_3_7", -9.0),
             adapter.GraphEdge("I0", "F1_3_7", 0.1)]
    fake_attribution(monkeypatch, edges=edges)
    aid = c.post("/circuits/analyses", json={"source": make_source()}).json()["analysis_id"]
    wait_for(c, aid)

    capped = c.get(f"/circuits/analyses/{aid}/graph?max_edges_per_node=1").json()
    incoming = [e for e in capped["edges"] if e["target"] == "F1_3_7"]
    assert len(incoming) == 1
    # Strongest by magnitude, so a large negative contribution is not discarded
    # in favour of a small positive one.
    assert incoming[0]["weight"] == -9.0


# --------------------------------------------------------------------------
# annotations are interpretation, not measurement
# --------------------------------------------------------------------------


def test_annotations_cannot_alter_measured_graph_data(client, monkeypatch):
    c, _ = client
    fake_attribution(monkeypatch)
    aid = c.post("/circuits/analyses", json={"source": make_source()}).json()["analysis_id"]
    wait_for(c, aid)
    before = c.get(f"/circuits/analyses/{aid}/graph").json()

    c.put(f"/circuits/analyses/{aid}/annotations",
          json={"notes": {"F1_3_7": "San-city detector"},
                "groups": {"place names": ["F1_3_7", "F0_1_2"]},
                "pinned": ["F1_3_7"]})

    after = c.get(f"/circuits/analyses/{aid}/graph").json()
    assert after["nodes"] == before["nodes"]
    assert after["edges"] == before["edges"]
    assert after["coverage"] == before["coverage"]
    assert after["annotations"]["notes"]["F1_3_7"] == "San-city detector"


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


def test_a_saved_analysis_reopens_without_compute(tmp_path, monkeypatch):
    """Reading a saved graph must not need weights, transcoders or a network."""
    store = AnalysisStore(tmp_path)
    analysis = Analysis(
        analysis_id=store.new_id(), source=SourceSnapshot(**make_source()), status="done",
        nodes=[CircuitNode(id="F1_3_7", kind="feature", layer=1, position=3,
                           feature_idx=7, activation=4.0, influence=0.9)],
        edges=[CircuitEdge(source="F1_3_7", target="L12288", weight=3.0)],
    )
    store.save(analysis)

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("reading a saved analysis loaded a model")

    monkeypatch.setattr(adapter, "get_replacement_model", explode)
    reloaded = store.get(analysis.analysis_id)
    assert reloaded.nodes[0].feature_idx == 7
    assert reloaded.edges[0].weight == 3.0


def test_a_restart_marks_unfinished_analyses_interrupted(tmp_path):
    store = AnalysisStore(tmp_path)
    running = Analysis(analysis_id=store.new_id(),
                       source=SourceSnapshot(**make_source()), status="running")
    finished = Analysis(analysis_id=store.new_id(),
                        source=SourceSnapshot(**make_source()), status="done")
    store.save(running)
    store.save(finished)

    assert store.recover() == 1
    # Jobs live in memory, so a "running" document after a restart can never
    # finish; saying so beats leaving it running forever.
    assert store.get(running.analysis_id).status == "interrupted"
    assert store.get(finished.analysis_id).status == "done"


def test_an_analysis_id_cannot_walk_out_of_the_store(tmp_path):
    store = AnalysisStore(tmp_path)
    for hostile in ("../evil", "a/b", "..", "with-dash"):
        with pytest.raises(KeyError):
            store.get(hostile)


def test_listing_is_newest_first_and_omits_graph_bodies(tmp_path):
    store = AnalysisStore(tmp_path)
    first = Analysis(analysis_id=store.new_id(), source=SourceSnapshot(**make_source()))
    store.save(first)
    time.sleep(0.01)
    second = Analysis(analysis_id=store.new_id(), source=SourceSnapshot(**make_source()))
    store.save(second)

    listing = store.list()
    assert [a["analysis_id"] for a in listing] == [second.analysis_id, first.analysis_id]
    assert all("nodes" not in a and "edges" not in a for a in listing)


def test_an_unreadable_document_is_skipped_rather_than_breaking_the_listing(tmp_path):
    store = AnalysisStore(tmp_path)
    good = Analysis(analysis_id=store.new_id(), source=SourceSnapshot(**make_source()))
    store.save(good)
    (store.root / "deadbeefcafe.json").write_text("{not json")

    assert [a["analysis_id"] for a in store.list()] == [good.analysis_id]


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_feature_node_ids_are_stable_across_node_budgets():
    """Ids key on (layer, position, feature_idx), not on a row in the adjacency
    matrix, so the same feature keeps its id when attribution is re-run with a
    different budget and an annotation written against it still lands."""
    assert adapter._node_id("feature", 21, 11, 6271) == "F21_11_6271"
    assert adapter._node_id("error", 21, 11, None) == "E21_11"
    assert adapter._node_id("embed", None, 3, None) == "I3"


def test_transcoder_features_are_not_gemma_scope_features():
    """A transcoder approximates an MLP block and has its own feature space. The
    residual SAEs mechlens already serves are a different dictionary, so nothing
    may join them by layer/index."""
    from app.sae_cache import RELEASE

    assert adapter.TRANSCODER_SET != RELEASE
    assert "gemma-scope-2b-pt-res" not in adapter.TRANSCODER_SET


# --------------------------------------------------------------------------
# reducing a dense graph: the arithmetic that a UI cannot sanity-check by eye
# --------------------------------------------------------------------------


class FakeCfg:
    n_layers = 2
    d_vocab_out = 100
    # Read by runtime_provenance when it records what produced a number.
    device = "cpu"
    dtype = "torch.float32"
    model_revision = None


class FakeGraph:
    """A hand-built stand-in with the index-space trap baked in.

    `activation_values` is aligned to `active_features` upstream, while
    `selected_features` picks a subset of them. Anything that indexes the
    activations by position-within-selected reads the wrong feature.
    """

    def __init__(self):
        import torch

        self.cfg = FakeCfg()
        self.n_pos = 2
        # Four active features; activations deliberately all distinct.
        self.active_features = torch.tensor([[0, 0, 11], [0, 1, 22], [1, 0, 33], [1, 1, 44]])
        self.activation_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # Only the 2nd and 4th are selected, and out of order, so a wrong
        # indexing cannot accidentally agree.
        self.selected_features = torch.tensor([3, 1])

        n_features, n_errors, n_embed, n_logits = 2, 4, 2, 1
        size = n_features + n_errors + n_embed + n_logits
        self.adjacency_matrix = torch.zeros(size, size)
        self.adjacency_matrix[0, 1] = 5.0     # feature <- feature
        self.adjacency_matrix[-1, 0] = 7.0    # logit   <- feature
        self.logit_targets = [object()]
        self.logit_probabilities = torch.tensor([0.9])


def test_activations_follow_the_feature_they_belong_to(monkeypatch):
    """Regression: activation_values is indexed by active_features, so reading
    it by position-within-selected silently reports another feature's number."""
    import torch

    graph = FakeGraph()
    monkeypatch.setattr(
        "circuit_tracer.graph.prune_graph",
        lambda g, nt, et: (torch.ones(g.adjacency_matrix.shape[0], dtype=torch.bool),
                           g.adjacency_matrix != 0,
                           torch.zeros(g.adjacency_matrix.shape[0])),
    )

    monkeypatch.setattr(adapter, "transcoder_revision", lambda repo=None: None)
    result = adapter._reduce(
        graph, model=type("M", (), {"cfg": FakeCfg()})(),
        transcoder_set="fake", prefix_token_ids=[2, 3], target_token_id=12288,
        node_threshold=0.0, edge_threshold=0.0, max_edges_per_node=10,
        max_feature_nodes=10, elapsed_s=1.0,
    )

    by_id = {n.id: n for n in result.nodes if n.kind == "feature"}
    # selected_features = [3, 1] -> active rows [1,1,44] and [0,1,22],
    # whose activations are 4.0 and 2.0 respectively.
    assert by_id["F1_1_44"].activation == 4.0
    assert by_id["F0_1_22"].activation == 2.0


def test_reduce_labels_every_node_kind_from_its_position_in_the_matrix(monkeypatch):
    """Node order is [features, errors, embeds, logits]; mislabelling a band
    would turn error nodes into features and hide unexplained computation."""
    import torch

    graph = FakeGraph()
    monkeypatch.setattr(
        "circuit_tracer.graph.prune_graph",
        lambda g, nt, et: (torch.ones(g.adjacency_matrix.shape[0], dtype=torch.bool),
                           g.adjacency_matrix != 0,
                           torch.zeros(g.adjacency_matrix.shape[0])),
    )
    monkeypatch.setattr(adapter, "transcoder_revision", lambda repo=None: None)
    result = adapter._reduce(
        graph, model=type("M", (), {"cfg": FakeCfg()})(),
        transcoder_set="fake", prefix_token_ids=[2, 3], target_token_id=12288,
        node_threshold=0.0, edge_threshold=0.0, max_edges_per_node=10,
        max_feature_nodes=10, elapsed_s=1.0,
    )
    kinds = {}
    for node in result.nodes:
        kinds.setdefault(node.kind, 0)
        kinds[node.kind] += 1

    assert kinds["feature"] == 2
    assert kinds["error"] == FakeCfg.n_layers * graph.n_pos  # 2 layers x 2 positions
    assert kinds["embed"] == graph.n_pos
    assert kinds["logit"] == 1
    assert result.coverage["n_error_nodes"] == 4
