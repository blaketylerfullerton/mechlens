"""The FastAPI routes, against the tiny CPU test model and a throwaway label DB.

Mirrors test_capture.py's approach: gemma is too heavy to load in every test
run, so the tiny model stands in for it, and a fake SAE stands in for a real
Gemma Scope one (whose dimensions would not even match the tiny model's).
"""

from __future__ import annotations

import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient

from app.labels import LabelRow, LabelStore
from app.service import jobs
from app.service.app import create_app

PROMPT = "Once upon a time there was a"


class FakeSAE:
    """Stands in for a Gemma Scope SAE for both /steer and the SAE pass.

    `W_dec` is all /steer needs. `encode` fires a deterministic three features
    per token position — enough for the pass's bookkeeping (top-k, l0) to be
    asserted exactly, and sparse the way a real JumpReLU SAE is, without the
    302MB.
    """

    def __init__(self, n_features: int, d_model: int):
        self.n_features = n_features
        self.W_dec = torch.zeros(n_features, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        acts = torch.zeros(x.shape[0], self.n_features)
        for token in range(x.shape[0]):
            for offset, value in enumerate((3.0, 2.0, 1.0)):
                acts[token, (token * 3 + offset) % self.n_features] = value
        return acts

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return torch.zeros(a.shape[0], self.W_dec.shape[1])


# The features FakeSAE.encode fires at a given position, strongest first.
def expected_features(token: int, n_features: int = 16) -> list[int]:
    return [(token * 3 + offset) % n_features for offset in range(3)]


@pytest.fixture(scope="module")
def tiny_model():
    from transformer_lens import HookedTransformer

    try:
        # `_no_processing` for the same reason model_cache uses it: plain
        # `from_pretrained` folds LayerNorm into the weights, which moves
        # resid_post off the distribution the SAEs were fitted on — and the SAE
        # pass rightly refuses such a trace. The service's real captures are
        # unfolded, so the stand-in has to be too.
        m = HookedTransformer.from_pretrained_no_processing("tiny-stories-1M", device="cpu")
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"tiny-stories-1M unavailable: {exc}")
    m.eval()
    return m


@pytest.fixture
def client(tiny_model, tmp_path):
    jobs.JOBS.clear()
    sae_provider = lambda layer: FakeSAE(n_features=16, d_model=tiny_model.cfg.d_model)  # noqa: E731
    app = create_app(
        model=tiny_model, label_db_path=tmp_path / "labels.db", sae_provider=sae_provider
    )
    with TestClient(app) as c:
        yield c


def _poll(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/trace/{job_id}")
        if resp.json()["status"] in ("done", "error"):
            return resp
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


# -- POST /trace -------------------------------------------------------


def test_post_trace_returns_a_job_id(client):
    resp = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 3})
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_post_trace_rejects_empty_prompt(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post("/trace", json={"prompt": "", "max_tokens": 3})
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


def test_post_trace_rejects_non_positive_max_tokens(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 0})
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


# -- GET /trace/{id} -----------------------------------------------------


def test_get_trace_reaches_done_with_a_full_trace(client):
    job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 3}).json()["job_id"]
    resp = _poll(client, job_id)
    body = resp.json()
    assert body["status"] == "done"
    assert body["trace"]["prompt"] == PROMPT
    assert body["error"] is None


def test_get_trace_unknown_id_is_404(client):
    resp = client.get("/trace/does-not-exist")
    assert resp.status_code == 404


# -- the sae and labels passes -------------------------------------------


def _seed_labels(db_path, n_layers, features=range(16)):
    """A label for every (layer, feature) the fake SAE can fire — all 16 of
    them, since which ones fire depends on how many token positions there are."""
    from app.sae_cache import neuronpedia_id

    with LabelStore(db_path) as store:
        for layer in range(n_layers):
            _, source_set = neuronpedia_id(layer, store.width)
            store.upsert(
                [
                    LabelRow(
                        source_set=source_set,
                        feature=feature,
                        text=f"stand-in label for L{layer} #{feature}",
                        explainer="gpt-4o-mini",
                    )
                    for feature in features
                ]
            )


def test_trace_with_the_sae_pass_carries_features_on_every_layer(client, tiny_model):
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 2, "passes": ["sae"]}
    ).json()["job_id"]
    trace = _poll(client, job_id).json()["trace"]

    # `layout` rides along with `sae` unasked: positions are what make the
    # features drawable, and its record states the atlas's absence here rather
    # than being omitted. See test_a_trace_with_no_atlas_states_the_absence.
    assert [p["name"] for p in trace["passes"]] == ["sae", "layout"]
    for step in trace["steps"]:
        assert len(step["layers"]) == tiny_model.cfg.n_layers
        for state in step["layers"]:
            assert state["l0"] == 3
            assert [f["index"] for f in state["features"]] == expected_features(step["step"])
            assert [f["activation"] for f in state["features"]] == [3.0, 2.0, 1.0]


def test_trace_records_which_layers_the_sae_pass_ran(client, tiny_model):
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 1, "passes": ["sae"]}
    ).json()["job_id"]
    trace = _poll(client, job_id).json()["trace"]

    record = next(p for p in trace["passes"] if p["name"] == "sae")
    assert record["params"]["layers"] == ",".join(str(l) for l in range(tiny_model.cfg.n_layers))
    assert record["params"]["n_layers"] == tiny_model.cfg.n_layers


def test_trace_can_ask_for_a_subset_of_layers(client, tiny_model):
    """26 resident 16k SAEs are ~7.9GB, so a smaller deployment must be able to
    ask for fewer — and a skipped layer has to stay distinguishable from a
    layer in which nothing fired."""
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 1, "passes": ["sae"], "sae_layers": [1]}
    ).json()["job_id"]
    trace = _poll(client, job_id).json()["trace"]

    record = next(p for p in trace["passes"] if p["name"] == "sae")
    assert record["params"]["layers"] == "1"

    for step in trace["steps"]:
        assert step["layers"][1]["features"]
        assert step["layers"][1]["l0"] == 3
        for layer in range(tiny_model.cfg.n_layers):
            if layer != 1:
                assert step["layers"][layer]["features"] == []
                assert step["layers"][layer]["l0"] is None


def test_trace_rejects_sae_layers_past_the_model(client, tiny_model):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/trace",
        json={
            "prompt": PROMPT,
            "max_tokens": 1,
            "passes": ["sae"],
            "sae_layers": [tiny_model.cfg.n_layers],
        },
    )
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


def test_trace_with_the_labels_pass_carries_labels(client, tiny_model, tmp_path):
    _seed_labels(tmp_path / "labels.db", tiny_model.cfg.n_layers)

    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 2, "passes": ["sae", "labels"]}
    ).json()["job_id"]
    trace = _poll(client, job_id).json()["trace"]

    assert {p["name"] for p in trace["passes"]} == {"sae", "labels", "layout"}
    assert trace["labels"], "the labels pass recorded nothing"
    for step in trace["steps"]:
        for state in step["layers"]:
            for feature in state["features"]:
                key = f"{state['layer']}/{feature['index']}"
                assert trace["labels"][key]["text"].startswith("stand-in label")


def test_trace_rejects_labels_without_the_sae_pass(client):
    """The labels pass labels the features the SAE pass records, so without it
    it would raise on the worker — a 500-shaped failure for a request that was
    already wrong when it arrived."""
    n_jobs_before = len(jobs.JOBS)
    resp = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 1, "passes": ["labels"]})
    assert resp.status_code == 422
    assert "sae" in resp.text
    assert len(jobs.JOBS) == n_jobs_before


def test_trace_rejects_labels_when_the_store_holds_none(client):
    """The client fixture's label DB is empty, so this is the unseeded case."""
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 1, "passes": ["sae", "labels"]}
    )
    assert resp.status_code == 422
    assert "label store" in resp.text
    assert len(jobs.JOBS) == n_jobs_before


# -- the layout ------------------------------------------------------------


ATLAS = "test-atlas-v1"


def _seed_atlas(db_path, pairs, atlas_version=ATLAS, source="labels", **record):
    """An atlas placing exactly `pairs`, so what it cannot place is controlled.

    Positions are derived from the pair so a test can assert a feature got
    *its own* position rather than merely some position.
    """
    from app.labels import AtlasRecord, LayoutRow

    with LabelStore(db_path) as store:
        store.put_layout(
            atlas_version,
            [
                LayoutRow(
                    layer=layer, feature=feature, x=float(layer), y=float(feature), z=0.5,
                    cluster=layer,
                )
                for layer, feature in pairs
            ],
        )
        defaults = dict(
            atlas_version=atlas_version,
            release="gemma-scope-2b-pt-res-canonical",
            width="16k",
            seed=0,
            params={"source": source, "layers": "0,1"},
            metrics={"knn_preservation": 0.292, "knn_k": 20},
            positions_sha256="c" * 64,
            n_features=len(pairs),
            n_clusters=2,
        )
        store.put_atlas_record(AtlasRecord(**{**defaults, **record}))


def _traced_with_features(client, max_tokens=2):
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": max_tokens, "passes": ["sae"]}
    ).json()["job_id"]
    return _poll(client, job_id).json()["trace"]


def _reported_pairs(trace):
    return {
        (state["layer"], feature["index"])
        for step in trace["steps"]
        for state in step["layers"]
        for feature in state["features"]
    }


def test_trace_carries_a_position_for_every_feature_the_atlas_places(client, tmp_path):
    """3.2: features, an available atlas, and the atlas's identity alongside."""
    trace = _traced_with_features(client)
    pairs = _reported_pairs(trace)
    _seed_atlas(tmp_path / "labels.db", pairs)

    trace = _traced_with_features(client)
    assert _reported_pairs(trace) == pairs

    assert set(trace["layout"]) == {f"{layer}/{feature}" for layer, feature in pairs}
    for layer, feature in pairs:
        position = trace["layout"][f"{layer}/{feature}"]
        assert (position["x"], position["y"], position["cluster"]) == (
            float(layer),
            float(feature),
            layer,
        )

    record = next(p for p in trace["passes"] if p["name"] == "layout")
    assert record["params"]["atlas_available"] is True
    assert record["params"]["atlas_version"] == ATLAS
    assert record["params"]["positions_sha256"] == "c" * 64
    assert record["params"]["source"] == "labels"
    assert record["stats"]["features_placed"] == len(pairs)
    assert record["stats"]["features_unplaced"] == 0
    assert record["stats"]["knn_preservation"] == pytest.approx(0.292)


def test_a_feature_the_atlas_cannot_place_is_absent_rather_than_zeroed(client, tmp_path):
    """3.3: unplaced has to stay distinguishable from placed at the origin."""
    trace = _traced_with_features(client)
    pairs = sorted(_reported_pairs(trace))
    placed, unplaced = pairs[:-1], pairs[-1:]
    _seed_atlas(tmp_path / "labels.db", placed)

    trace = _traced_with_features(client)

    # Still reported as a feature — it fired, the atlas just has no row for it.
    assert unplaced[0] in _reported_pairs(trace)
    layer, feature = unplaced[0]
    assert f"{layer}/{feature}" not in trace["layout"]
    assert set(trace["layout"]) == {f"{l}/{f}" for l, f in placed}

    record = next(p for p in trace["passes"] if p["name"] == "layout")
    assert record["stats"]["features_placed"] == len(placed)
    assert record["stats"]["features_unplaced"] == len(unplaced)
    assert record["stats"]["coverage"] == pytest.approx(len(placed) / len(pairs))


def test_a_trace_with_no_atlas_states_the_absence(client):
    """3.4: the client fixture's DB has no atlas — features, no positions, and
    a record saying why, rather than a layout full of zeroes."""
    trace = _traced_with_features(client)

    assert _reported_pairs(trace), "the SAE pass recorded nothing to place"
    assert trace["layout"] == {}

    record = next(p for p in trace["passes"] if p["name"] == "layout")
    assert record["params"]["atlas_available"] is False
    assert "atlas_version" not in record["params"]
    assert record["stats"]["features_placed"] == 0
    assert record["stats"]["features_unplaced"] == record["stats"]["features_wanted"]


def test_a_trace_without_the_sae_pass_has_no_layout_and_no_layout_record(client, tmp_path):
    """There are no features to position, so the pass does not run at all —
    which is a different fact from running and placing nothing."""
    _seed_atlas(tmp_path / "labels.db", [(0, 0), (1, 1)])

    job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 1}).json()["job_id"]
    trace = _poll(client, job_id).json()["trace"]

    assert trace["layout"] == {}
    assert [p["name"] for p in trace["passes"]] == []


def test_the_layout_pass_picks_its_atlas_by_source_not_by_recency(client, tmp_path):
    """Both sources are retained by design, so "most recently built" is not a
    statement about which map anyone wants — the decoder atlas built last must
    not displace the label one the interface shows."""
    trace = _traced_with_features(client)
    pairs = sorted(_reported_pairs(trace))

    _seed_atlas(
        tmp_path / "labels.db", pairs, atlas_version="labels-atlas", source="labels",
        built_at="2026-01-01T00:00:00+00:00",
    )
    _seed_atlas(
        tmp_path / "labels.db", pairs, atlas_version="decoder-atlas", source="decoder",
        built_at="2026-06-01T00:00:00+00:00",
    )

    trace = _traced_with_features(client)
    record = next(p for p in trace["passes"] if p["name"] == "layout")
    assert record["params"]["atlas_version"] == "labels-atlas"
    assert record["params"]["source"] == "labels"


# -- GET /atlas ------------------------------------------------------------


def _seed_areas(db_path, rows, atlas_version=ATLAS):
    from app.labels import ClusterRow

    with LabelStore(db_path) as store:
        store.put_clusters(atlas_version, rows)


def test_get_atlas_serves_positions_clusters_and_the_record(client, tmp_path):
    """3.5: everything a client needs to draw the atlas, in one call."""
    from app.labels import ClusterRow

    pairs = [(0, 1), (0, 2), (1, 3)]
    _seed_atlas(tmp_path / "labels.db", pairs)
    _seed_areas(
        tmp_path / "labels.db",
        [
            ClusterRow(
                cluster=0, n_members=2, centroid=(0.1, 0.2, 0.3), spread=0.4,
                name="URLs and web links", name_source=(0, 1),
                coherence=0.47, baseline_coherence=0.23, explainers="gpt-4o-mini:2",
            ),
            ClusterRow(
                cluster=1, n_members=1, centroid=(0.0, 0.0, 0.0), spread=0.0,
                coherence=0.24, baseline_coherence=0.23,
            ),
        ],
    )

    body = client.get("/atlas").json()

    assert body["atlas_version"] == ATLAS
    assert body["source"] == "labels"
    assert body["n_total"] == len(pairs) and body["n_sampled"] == len(pairs)

    # The record's identity and its published fidelity travel with the nodes.
    assert body["positions_sha256"] == "c" * 64
    assert body["knn_preservation"] == pytest.approx(0.292)
    assert body["knn_k"] == 20

    nodes = body["nodes"]
    assert list(zip(nodes["layer"], nodes["feature"])) == pairs
    assert nodes["cluster"] == [0, 0, 1]
    assert len(nodes["xyz"]) == 3 * len(pairs)

    # int16 over `extent` round-trips to the seeded positions.
    extent = body["extent"]
    restored = [v / 32767 * extent for v in nodes["xyz"]]
    for i, (layer, feature) in enumerate(pairs):
        assert restored[i * 3] == pytest.approx(float(layer), abs=1e-3)
        assert restored[i * 3 + 1] == pytest.approx(float(feature), abs=1e-3)

    named, unnamed = body["areas"]
    assert named["name"] == "URLs and web links"
    assert named["coherence"] == pytest.approx(0.47)
    assert unnamed["name"] is None, "an unearned name must not be synthesised"
    assert unnamed["coherence"] == pytest.approx(0.24)


def test_get_atlas_answers_without_the_model(monkeypatch, tmp_path):
    """3.5: the atlas is a precomputed table, so it must answer while gemma is
    still loading — the brain draws its node cloud before any trace exists."""
    release = threading.Event()

    def slow_get_model():
        release.wait(timeout=5)
        raise AssertionError("test never lets the load finish")

    monkeypatch.setattr("app.service.app.model_cache.get_model", slow_get_model)
    jobs.JOBS.clear()
    _seed_atlas(tmp_path / "labels.db", [(0, 1)])
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    try:
        with TestClient(app) as c:
            assert c.get("/health").json()["status"] == "loading"
            # The routes that need gemma are 503 right now; this one is not.
            assert c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1}).status_code == 503

            resp = c.get("/atlas")
            assert resp.status_code == 200
            assert resp.json()["atlas_version"] == ATLAS
    finally:
        release.set()


def test_get_atlas_reports_absence_rather_than_an_empty_layout(client):
    """3.6: the client fixture's DB has no atlas at all."""
    resp = client.get("/atlas")
    assert resp.status_code == 404
    assert "no atlas has been built" in resp.text


def test_an_atlas_that_placed_nothing_is_not_a_missing_atlas(client, tmp_path):
    """3.6: the two cases a client has to tell apart. An atlas exists and holds
    no positions — that is a 200 with an empty node list, not a 404."""
    _seed_atlas(tmp_path / "labels.db", [], n_features=0, n_clusters=0)

    resp = client.get("/atlas")
    assert resp.status_code == 200
    body = resp.json()
    assert body["atlas_version"] == ATLAS
    assert body["n_total"] == 0
    assert body["nodes"]["layer"] == [] and body["nodes"]["xyz"] == []


def test_get_atlas_sample_is_a_deterministic_subset(client, tmp_path):
    pairs = [(layer, feature) for layer in range(2) for feature in range(40)]
    _seed_atlas(tmp_path / "labels.db", pairs)

    first = client.get("/atlas", params={"sample": 10}).json()
    second = client.get("/atlas", params={"sample": 10}).json()

    assert first["n_sampled"] == 10 and first["n_total"] == len(pairs)
    assert first["nodes"] == second["nodes"], "the same request returned different nodes"
    served = set(zip(first["nodes"]["layer"], first["nodes"]["feature"]))
    assert served < set(pairs), "the sample must be a strict subset of the atlas"


def test_get_atlas_can_ask_for_a_named_version(client, tmp_path):
    _seed_atlas(tmp_path / "labels.db", [(0, 1)], atlas_version="labels-atlas", source="labels")
    _seed_atlas(tmp_path / "labels.db", [(0, 2)], atlas_version="decoder-atlas", source="decoder")

    assert client.get("/atlas").json()["atlas_version"] == "labels-atlas"
    picked = client.get("/atlas", params={"version": "decoder-atlas"}).json()
    assert picked["atlas_version"] == "decoder-atlas"
    assert picked["source"] == "decoder"


def test_trace_reports_the_sae_phase_while_running(client, tiny_model):
    """The SAE pass genuinely walks layers, so — unlike generating — it can
    report a depth sweep."""
    seen: list[dict] = []
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 4, "passes": ["sae"]}
    ).json()["job_id"]

    deadline = time.time() + 5.0
    while time.time() < deadline:
        body = client.get(f"/trace/{job_id}").json()
        if body["progress"] is not None:
            seen.append(body["progress"])
        if body["status"] in ("done", "error"):
            break

    phases = [p["phase"] for p in seen]
    assert "sae" in phases, f"no sae reading among {phases}"
    for reading in (p for p in seen if p["phase"] == "sae"):
        assert reading["total"] == tiny_model.cfg.n_layers
        assert 1 <= reading["done"] <= reading["total"]
    # generating precedes sae; a reading never moves back to an earlier phase.
    assert phases == sorted(phases, key=lambda p: ("generating", "sae", "lens").index(p))


# -- POST /steer -----------------------------------------------------------


def test_post_steer_returns_a_pollable_job(client):
    resp = client.post(
        "/steer",
        json={"prompt": PROMPT, "max_tokens": 3, "layer": 2, "feature_idx": 5, "coefficient": 1.0},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    result = _poll(client, job_id)
    body = result.json()
    assert body["status"] == "done"
    assert body["trace"]["steering"] == {"layer": 2, "feature_idx": 5, "coefficient": 1.0}


def test_post_steer_rejects_out_of_range_layer(client, tiny_model):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/steer",
        json={
            "prompt": PROMPT,
            "max_tokens": 3,
            "layer": tiny_model.cfg.n_layers,  # one past the last valid layer
            "feature_idx": 0,
            "coefficient": 1.0,
        },
    )
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


def test_post_steer_rejects_out_of_range_feature_idx(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/steer",
        json={
            "prompt": PROMPT,
            "max_tokens": 3,
            "layer": 2,
            "feature_idx": 999,  # the fake SAE only has 16 features
            "coefficient": 1.0,
        },
    )
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


# -- GET /feature/{layer}/{idx} ---------------------------------------------


LAYER = 2
SOURCE_SET = "2-gemmascope-res-16k"


def test_get_feature_returns_a_known_label(client, tmp_path):
    with LabelStore(tmp_path / "labels.db") as store:
        store.upsert(
            [
                LabelRow(
                    source_set=SOURCE_SET,
                    feature=12082,
                    text="references to dogs as pets",
                    explainer="gpt-4o-mini",
                )
            ]
        )

    resp = client.get(f"/feature/{LAYER}/12082")
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "references to dogs as pets"
    assert body["explainer"] == "gpt-4o-mini"
    assert body["url"]


def test_get_feature_distinguishes_no_label_from_not_found(client, tmp_path):
    with LabelStore(tmp_path / "labels.db") as store:
        store.upsert([LabelRow(source_set=SOURCE_SET, feature=7, text=None)])

    looked_up_but_unexplained = client.get(f"/feature/{LAYER}/7")
    assert looked_up_but_unexplained.status_code == 200
    assert looked_up_but_unexplained.json()["label"] is None

    never_looked_up = client.get(f"/feature/{LAYER}/8")  # in range, but no row at all
    assert never_looked_up.status_code == 200
    assert never_looked_up.json()["label"] is None


def test_get_feature_out_of_range_layer_is_404(client, tiny_model):
    resp = client.get(f"/feature/{tiny_model.cfg.n_layers}/0")
    assert resp.status_code == 404


def test_get_feature_out_of_range_index_is_404(client):
    resp = client.get(f"/feature/{LAYER}/99999999")
    assert resp.status_code == 404


# -- model loading -----------------------------------------------------


def _wait_ready(client, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/health").json()["status"]
        if status != "loading":
            return status
        time.sleep(0.02)
    raise AssertionError("model did not become ready in time")


def test_model_is_loaded_at_most_once_across_requests(monkeypatch, tiny_model, tmp_path):
    calls = {"n": 0}

    def fake_get_model():
        calls["n"] += 1
        return tiny_model

    monkeypatch.setattr("app.service.app.model_cache.get_model", fake_get_model)
    jobs.JOBS.clear()
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    with TestClient(app) as c:
        assert _wait_ready(c) == "ready"  # loaded by the lifespan's warm-up thread
        assert calls["n"] == 1
        job_id = c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1}).json()["job_id"]
        _poll(c, job_id)
        c.get(f"/trace/{job_id}")

    assert calls["n"] == 1


def test_health_is_ready_when_a_model_is_injected(client):
    assert client.get("/health").json() == {"status": "ready", "detail": None}


def test_routes_answer_503_while_the_model_is_still_loading(monkeypatch, tmp_path):
    """The point of loading off-thread: the port is open and the API answers,
    with a status a client can act on, before the model is in memory."""
    release = threading.Event()

    def slow_get_model():
        release.wait(timeout=5)
        raise AssertionError("test never lets the load finish")

    monkeypatch.setattr("app.service.app.model_cache.get_model", slow_get_model)
    jobs.JOBS.clear()
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    try:
        with TestClient(app) as c:
            assert c.get("/health").json()["status"] == "loading"
            resp = c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1})
            assert resp.status_code == 503
            assert "still loading" in resp.json()["detail"]
    finally:
        release.set()


def test_health_reports_a_failed_load(monkeypatch, tmp_path):
    def broken_get_model():
        raise RuntimeError("no weights on disk")

    monkeypatch.setattr("app.service.app.model_cache.get_model", broken_get_model)
    jobs.JOBS.clear()
    app = create_app(model=None, label_db_path=tmp_path / "labels.db")

    with TestClient(app) as c:
        assert _wait_ready(c) == "error"
        assert "no weights on disk" in c.get("/health").json()["detail"]
        resp = c.post("/trace", json={"prompt": PROMPT, "max_tokens": 1})
        assert resp.status_code == 503
        assert "failed to load" in resp.json()["detail"]


def test_cors_allows_the_vite_dev_server_on_a_fallback_port(client):
    """vite moves to 5174+ when 5173 is taken; the browser there must not be
    told "access control checks" by our own middleware."""
    for origin in ("http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5175"):
        resp = client.post(
            "/trace", json={"prompt": PROMPT, "max_tokens": 1}, headers={"Origin": origin}
        )
        assert resp.headers["access-control-allow-origin"] == origin


# -- POST /trace with enrichment passes ------------------------------------


def test_requesting_the_lens_pass_fills_per_layer_readouts(client, tiny_model):
    """The brain view reads `logit_lens` per (token, layer); it only exists if
    the trace job ran the pass, since the service writes no sidecar to enrich
    later."""
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["lens"]}
    ).json()["job_id"]

    body = _poll(client, job_id).json()
    assert body["status"] == "done", body.get("error")
    trace = body["trace"]

    assert [p["name"] for p in trace["passes"]] == ["lens"]
    for step in trace["steps"]:
        assert len(step["layers"]) == tiny_model.cfg.n_layers
        for layer in step["layers"]:
            assert layer["logit_lens"] is not None
            assert layer["logit_lens"]["top_k"]


def test_the_lens_record_carries_the_crystallisation_stats(client):
    """`crossover_layer` is what the brain marks, so it has to survive the trip."""
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["lens"]}
    ).json()["job_id"]

    trace = _poll(client, job_id).json()["trace"]
    stats = trace["passes"][0]["stats"]

    assert "crossover_layer" in stats
    assert "top1_agreement_by_layer" in stats
    assert "final_layer_agreement" in stats


def test_a_trace_with_no_passes_is_capture_only(client):
    """The default must stay exactly what it was before `passes` existed."""
    job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 3}).json()["job_id"]

    trace = _poll(client, job_id).json()["trace"]

    assert trace["passes"] == []
    assert all(
        layer["logit_lens"] is None for step in trace["steps"] for layer in step["layers"]
    )
    # capture data is still all there
    assert trace["steps"][0]["layers"][0]["resid_norm"] > 0


def test_an_unknown_pass_is_rejected_without_enqueueing(client):
    n_jobs_before = len(jobs.JOBS)
    resp = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["not-a-pass"]}
    )
    assert resp.status_code == 422
    assert len(jobs.JOBS) == n_jobs_before


def test_a_failing_pass_fails_the_job_rather_than_returning_a_bare_trace(
    client, tiny_model, monkeypatch
):
    """A capture-only trace returned after a requested pass raised would be
    indistinguishable from one that never asked for the pass."""
    from app.service import app as app_module

    class Exploding:
        name = "lens"

        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _trace, _residuals):
            raise RuntimeError("lens exploded")

    monkeypatch.setattr(app_module, "LogitLensPass", Exploding)

    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 3, "passes": ["lens"]}
    ).json()["job_id"]

    body = _poll(client, job_id).json()
    assert body["status"] == "error"
    assert "lens exploded" in body["error"]
    assert body["trace"] is None


# -- GET /trace/{id} progress ----------------------------------------------


def test_progress_is_absent_for_a_queued_job(client):
    """"Pending" and "running, at token 0" have to stay distinguishable."""
    first = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 4}).json()["job_id"]
    second = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 4}).json()["job_id"]

    body = client.get(f"/trace/{second}").json()
    if body["status"] == "pending":  # the worker may already have picked it up
        assert body["progress"] is None

    _poll(client, first)
    _poll(client, second)


def test_progress_reports_both_phases_over_the_life_of_a_job(client, tiny_model):
    """Polled readings must identify the phase and count that phase's own unit
    of work: tokens while generating, layers while running the lens."""
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 4, "passes": ["lens"]}
    ).json()["job_id"]

    readings: list[tuple[str, int, int]] = []
    deadline = time.time() + 10.0
    while time.time() < deadline:
        body = client.get(f"/trace/{job_id}").json()
        p = body["progress"]
        if p is not None:
            reading = (p["phase"], p["done"], p["total"])
            if not readings or readings[-1] != reading:
                readings.append(reading)
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.005)

    assert readings, "no progress was ever reported"
    phases = [phase for phase, _d, _t in readings]
    assert "generating" in phases
    assert "lens" in phases
    # phases never interleave or go back
    assert phases == sorted(phases, key=["generating", "lens"].index)

    generating = [(d, t) for phase, d, t in readings if phase == "generating"]
    assert all(total == 4 for _d, total in generating)
    assert [d for d, _t in generating] == sorted(d for d, _t in generating)

    lens = [(d, t) for phase, d, t in readings if phase == "lens"]
    assert all(total == tiny_model.cfg.n_layers for _d, total in lens)
    assert lens[-1][0] == tiny_model.cfg.n_layers  # the sweep reaches the last layer


def test_progress_never_goes_backwards_across_polls(client):
    job_id = client.post(
        "/trace", json={"prompt": PROMPT, "max_tokens": 4, "passes": ["lens"]}
    ).json()["job_id"]

    order = {"generating": 0, "lens": 1}
    previous: tuple[int, int] | None = None
    deadline = time.time() + 10.0
    while time.time() < deadline:
        body = client.get(f"/trace/{job_id}").json()
        p = body["progress"]
        if p is not None:
            current = (order[p["phase"]], p["done"])
            if previous is not None:
                assert current >= previous, f"progress went backwards: {previous} -> {current}"
            previous = current
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.005)

    assert previous is not None


def test_a_finished_job_reports_its_status_without_depending_on_progress(client):
    job_id = client.post("/trace", json={"prompt": PROMPT, "max_tokens": 2}).json()["job_id"]
    body = _poll(client, job_id).json()

    assert body["status"] == "done"
    assert body["trace"] is not None
    assert "progress" in body  # the field is always present, even if null


def test_steer_jobs_still_poll_through_the_same_endpoint(client):
    """Adding `progress` to the response must not have changed /steer's shape."""
    job_id = client.post(
        "/steer",
        json={"prompt": PROMPT, "max_tokens": 2, "layer": 2, "feature_idx": 5, "coefficient": 1.0},
    ).json()["job_id"]

    body = _poll(client, job_id).json()
    assert body["status"] == "done"
    assert body["trace"]["steering"] == {"layer": 2, "feature_idx": 5, "coefficient": 1.0}
    assert body["error"] is None


def test_live_trace_publishes_before_analysis_finishes(tiny_model, tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    seen = []
    snapshots = []
    original_reporter = jobs._reporter_for

    def recording_reporter(job):
        reporter = original_reporter(job)
        publish = reporter.publish

        def record(trace):
            publish(trace)
            snapshots.append(job.partial_trace)

        reporter.publish = record
        return reporter

    monkeypatch.setattr(jobs, "_reporter_for", recording_reporter)

    class RecordingSAE(FakeSAE):
        def encode(self, x):
            seen.append(x.shape[0])
            return super().encode(x)

    sae = RecordingSAE(16, tiny_model.cfg.d_model)

    def provider(layer):
        entered.set()
        assert release.wait(10)
        return sae

    _seed_labels(tmp_path / "live.db", tiny_model.cfg.n_layers)
    app = create_app(model=tiny_model, label_db_path=tmp_path / "live.db", sae_provider=provider)
    with TestClient(app) as client:
        job_id = client.post('/trace', json={
            'prompt': PROMPT, 'max_tokens': 3, 'live': True,
            'passes': ['sae', 'labels', 'lens'], 'sae_layers': [0],
        }).json()['job_id']
        try:
            assert entered.wait(10)
            response = client.get(f'/trace/{job_id}').json()
            assert response['status'] == 'running'
            assert response['trace'] is None
            partial = response['partial_trace']
            assert partial['completion']
            assert partial['n_generated_tokens'] == 1
            assert len(partial['steps']) == partial['n_prompt_tokens']
            assert partial['steps'][-1]['logits']['chosen'] is not None
        finally:
            release.set()
        response = _poll(client, job_id, timeout=30).json()
        assert response['status'] == 'done', response['error']
        assert response['partial_trace'] is None
        trace = response['trace']
        assert trace['trace_id'] == partial['trace_id']
        assert all(s['layers'][0]['features'] for s in trace['steps'])
        assert all(s['layers'][0]['logit_lens'] for s in trace['steps'])
        # First the prompt, then one new position at a time, then the final
        # whole-trace validation pass. No growing prefixes during generation.
        assert seen[0] == partial['n_prompt_tokens']
        assert all(count == 1 for count in seen[1:-1])
        assert seen[-1] == len(trace['steps'])
        enriched = [s for s in snapshots if s.steps[-1].layers[0].logit_lens]
        assert enriched
        assert len(enriched[0].steps) == partial['n_prompt_tokens']
        assert enriched[0].steps[-1].layers[0].features
        assert enriched[0].labels
        assert all(not p.stats for p in enriched[0].passes)
        assert any(p['stats'] for p in trace['passes'])
