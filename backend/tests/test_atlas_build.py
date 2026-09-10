"""The atlas build, end to end, against a fake SAE.

The real thing loads ~7.9GB of Gemma Scope weights and takes tens of minutes,
so the pipeline is exercised here on synthetic decoder directions with planted
structure: three groups of features that genuinely point in different
directions, so a working build has something real to find and a broken one has
nowhere to hide.

`umap.UMAP(random_state=...)` is single-threaded and slow, so the fixtures are
deliberately small and module-scoped.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from app import atlas
from app.labels import LabelRow, LabelStore
from app.sae_cache import neuronpedia_id

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_feature_atlas.py"

N_LAYERS = 3
D_SAE = 64
D_MODEL = 32
N_GROUPS = 3


def _load_script():
    spec = importlib.util.spec_from_file_location("build_feature_atlas", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_module():
    return _load_script()


class PlantedSAE:
    """`W_dec` rows on N_GROUPS well-separated arcs.

    Two levels of structure, both deliberate:

    - **between groups**, three near-orthogonal centre directions, so the
      groups are separable and clustering has something real to find;
    - **within a group**, a smooth one-dimensional sweep from the centre toward
      a second direction, so a feature's nearest neighbours are its neighbours
      *along that sweep* and are therefore well defined.

    The within-group sweep is what makes a kNN-preservation assertion
    meaningful at all. An earlier version of this fixture planted only the
    group centres plus isotropic noise; each group was then a tight ball in
    which "the 5 nearest" was decided purely by noise, so no projection could
    preserve it and the measurement correctly reported ~0.26. That was the
    fixture being unmeasurable, not the build being broken.
    """

    def __init__(self, layer: int, d_sae: int = D_SAE, d_model: int = D_MODEL):
        shared = np.random.default_rng(99)
        centres = shared.normal(size=(N_GROUPS, d_model))
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        offsets = shared.normal(size=(N_GROUPS, d_model))
        offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)

        rng = np.random.default_rng(1000 + layer)
        assignment = np.arange(d_sae) % N_GROUPS
        # Position along the group's sweep, distinct per feature.
        t = (np.arange(d_sae) // N_GROUPS) / max(1, (d_sae // N_GROUPS) - 1)

        self.group = assignment
        self.W_dec = (
            centres[assignment] * 4.0
            + offsets[assignment] * t[:, None] * 3.0
            + rng.normal(scale=0.05, size=(d_sae, d_model))
        ).astype(np.float32)


def _provider(layer: int):
    return PlantedSAE(layer)


def _seed_labels(
    db_path: Path,
    *,
    with_embeddings: bool,
    coherent: bool = True,
    one_explainer: bool = False,
) -> None:
    """Labels for every feature, optionally with embeddings.

    When `coherent`, a feature's embedding follows its planted group, so a
    cluster's members agree and naming should be earned. When not, embeddings
    are noise and every cluster should come back unnamed *with* a measured
    coherence — the "we measured and it was bad" case.
    """
    rng = np.random.default_rng(5)
    group_vectors = np.random.default_rng(6).normal(size=(N_GROUPS, 16))
    group_vectors /= np.linalg.norm(group_vectors, axis=1, keepdims=True)

    with LabelStore(db_path) as store:
        for layer in range(N_LAYERS):
            _, source_set = neuronpedia_id(layer, store.width)
            sae = PlantedSAE(layer)
            rows = []
            for feature in range(D_SAE):
                group = int(sae.group[feature])
                embedding = None
                if with_embeddings:
                    embedding = (
                        group_vectors[group] * 6.0 + rng.normal(scale=0.2, size=16)
                        if coherent
                        else rng.normal(size=16)
                    ).astype(np.float32)
                rows.append(
                    LabelRow(
                        source_set=source_set,
                        feature=feature,
                        text=f"group {group} concept, L{layer} #{feature}",
                        # The export's real split, so the recorded mix is not
                        # uniform by construction.
                        explainer=(
                            "gpt-4o-mini"
                            if one_explainer
                            else ("gemini-2.5-flash-lite" if layer == 1 else "gpt-4o-mini")
                        ),
                        embedding=embedding,
                    )
                )
            store.upsert(rows)


def _build(build_module, db_path, **kwargs):
    defaults = dict(
        layers=list(range(N_LAYERS)),
        seed=0,
        pca_dim=8,
        n_neighbors=10,
        min_cluster_size=12,
        knn_k=5,
        knn_sample=60,
        cross_sample=200,
        subsample=40,
        db_path=db_path,
        sae_provider=_provider,
        verbose=False,
    )
    return build_module.build(**{**defaults, **kwargs})


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def test_decoder_directions_are_stacked_with_one_key_each(build_module):
    directions, keys = build_module.load_decoder_directions(
        [0, 1, 2], sae_provider=_provider, verbose=False
    )
    assert directions.shape == (N_LAYERS * D_SAE, D_MODEL)
    assert len(keys) == N_LAYERS * D_SAE
    assert len(set(keys)) == len(keys), "a (layer, feature) pair appeared twice"
    assert keys[0] == (0, 0)
    assert keys[-1] == (2, D_SAE - 1)


def test_every_layer_and_feature_gets_a_key(build_module):
    _, keys = build_module.load_decoder_directions([0, 2], sae_provider=_provider, verbose=False)
    assert {layer for layer, _ in keys} == {0, 2}
    for layer in (0, 2):
        assert sorted(f for l, f in keys if l == layer) == list(range(D_SAE))


def test_no_layers_is_an_error_not_an_empty_atlas(build_module):
    with pytest.raises(ValueError, match="no layers"):
        build_module.load_decoder_directions([], sae_provider=_provider, verbose=False)


def test_parse_layers_accepts_ranges_lists_and_singles(build_module):
    assert build_module.parse_layers("0-3") == [0, 1, 2, 3]
    assert build_module.parse_layers("0,4,8") == [0, 4, 8]
    assert build_module.parse_layers("20") == [20]
    assert build_module.parse_layers("2-4,0") == [0, 2, 3, 4]
    assert build_module.parse_layers("") == list(range(26))
    assert build_module.parse_layers("1,1,1") == [1]


def test_parse_layers_rejects_nonsense(build_module):
    with pytest.raises(ValueError):
        build_module.parse_layers(",,")


# --------------------------------------------------------------------------
# the whole build
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(build_module, tmp_path_factory):
    db = tmp_path_factory.mktemp("atlas") / "labels.db"
    _seed_labels(db, with_embeddings=True)
    out = tmp_path_factory.mktemp("asset") / "atlas-idle.json"
    summary = _build(build_module, db, subsample_out=out)
    return summary, db, out


def test_every_feature_gets_exactly_one_position(built):
    summary, db, _ = built
    assert summary["n_features"] == N_LAYERS * D_SAE

    with LabelStore(db) as store:
        pairs = [(layer, feature) for layer in range(N_LAYERS) for feature in range(D_SAE)]
        got = store.layout(summary["atlas_version"], pairs)
    assert len(got) == N_LAYERS * D_SAE


def test_positions_land_inside_the_shell(built):
    summary, db, _ = built
    with LabelStore(db) as store:
        rows = store.layout(
            summary["atlas_version"],
            [(layer, feature) for layer in range(N_LAYERS) for feature in range(D_SAE)],
        )
    positions = np.array([[r.x, r.y, r.z] for r in rows.values()])
    atlas.assert_finite(positions)
    assert atlas.within_shell(positions)


def test_the_projection_preserves_the_planted_structure(built):
    """A sanity floor on the pipeline, stated against the chance floor.

    A bare threshold on this number would be arbitrary — how much of a 2304-d
    neighbourhood survives three dimensions is exactly the thing the atlas
    measures rather than assumes. What *is* assertable is that it beats what a
    random layout would score on the same data, by a wide margin.
    """
    summary, _, _ = built
    metrics = summary["metrics"]
    k, n = metrics["knn_k"], summary["n_features"]

    # A random layout puts k of n-1 others nearest by luck.
    chance = k / (n - 1)
    assert metrics["knn_preservation"] > 8 * chance, (
        f"{metrics['knn_preservation']:.3f} is not meaningfully above the "
        f"{chance:.3f} a random layout would score"
    )


def test_nearest_neighbours_in_the_layout_come_from_the_same_planted_group(built):
    """The claim the atlas actually makes: near means similar. Asserted at the
    level a viewer reads — which group a node's neighbours belong to — rather
    than on the identity of the exact k nearest."""
    summary, db, _ = built
    pairs = [(layer, feature) for layer in range(N_LAYERS) for feature in range(D_SAE)]

    with LabelStore(db) as store:
        rows = store.layout(summary["atlas_version"], pairs)

    keys = list(rows.keys())
    positions = np.array([[rows[k].x, rows[k].y, rows[k].z] for k in keys])
    groups = np.array([int(PlantedSAE(layer).group[feature]) for layer, feature in keys])

    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :5]

    same_group = (groups[nearest] == groups[:, None]).mean()
    assert same_group > 0.9, f"only {same_group:.1%} of near neighbours share a planted group"


def test_the_record_carries_every_measurement(built):
    summary, db, _ = built
    with LabelStore(db) as store:
        record = store.atlas_record(summary["atlas_version"])

    assert record is not None
    for key in (
        "knn_preservation",
        "knn_k",
        "knn_sample",
        "knn_high_metric",
        "knn_low_metric",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "cross_source_ari",
        "n_named_clusters",
        "n_unnamed_clusters",
        "naming_attempted",
    ):
        assert key in record.metrics, f"{key} missing from the atlas record"

    assert record.positions_sha256 == summary["positions_sha256"]
    assert record.params["metric"] == "cosine"
    assert record.params["coherence_margin"] == atlas.DEFAULT_COHERENCE_MARGIN
    assert record.release and record.width and record.built_at


def test_every_feature_is_clustered_or_explicitly_unclustered(built):
    summary, db, _ = built
    with LabelStore(db) as store:
        rows = store.layout(
            summary["atlas_version"],
            [(layer, feature) for layer in range(N_LAYERS) for feature in range(D_SAE)],
        )
        clusters = {c.cluster for c in store.clusters(summary["atlas_version"])}

    for row in rows.values():
        assert row.cluster == -1 or row.cluster in clusters, (
            f"L{row.layer}#{row.feature} points at cluster {row.cluster}, "
            f"which has no row"
        )


def test_cluster_member_counts_match_the_positions(built):
    summary, db, _ = built
    with LabelStore(db) as store:
        rows = store.layout(
            summary["atlas_version"],
            [(layer, feature) for layer in range(N_LAYERS) for feature in range(D_SAE)],
        )
        clusters = store.clusters(summary["atlas_version"])

    counted: dict[int, int] = {}
    for row in rows.values():
        counted[row.cluster] = counted.get(row.cluster, 0) + 1
    for cluster in clusters:
        assert cluster.n_members == counted.get(cluster.cluster, 0)


def test_clusters_record_their_explainer_mix(built):
    """One of the three planted layers uses a different explainer, so a mix
    that came back uniform would mean the field was never populated."""
    summary, db, _ = built
    with LabelStore(db) as store:
        clusters = store.clusters(summary["atlas_version"])

    assert clusters
    assert any("gemini-2.5-flash-lite" in c.explainers for c in clusters)
    assert any("gpt-4o-mini" in c.explainers for c in clusters)


def test_a_named_cluster_points_at_a_real_member(built):
    summary, db, _ = built
    with LabelStore(db) as store:
        clusters = store.clusters(summary["atlas_version"])
        named = [c for c in clusters if c.name is not None]
        assert named, "coherent embeddings should have earned at least one name"

        for cluster in named:
            layer, feature = cluster.name_source
            label = store.get(layer, feature)
            assert label is not None and label.text == cluster.name
            row = store.layout(summary["atlas_version"], [(layer, feature)])[(layer, feature)]
            assert row.cluster == cluster.cluster, "the name came from a non-member"


def test_a_named_cluster_beat_its_own_baseline(built):
    summary, db, _ = built
    with LabelStore(db) as store:
        named = [c for c in store.clusters(summary["atlas_version"]) if c.name]
    for cluster in named:
        assert cluster.baseline_coherence is not None
        margin = cluster.coherence - cluster.baseline_coherence
        assert margin >= atlas.DEFAULT_COHERENCE_MARGIN, (
            f"cluster {cluster.cluster} was named on a margin of only {margin:.3f}"
        )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_a_rebuild_from_the_same_seed_reproduces_the_atlas(build_module, tmp_path):
    """The one property the whole artifact depends on: an atlas that reshuffles
    between builds is worthless, because familiarity with it never accrues."""
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    first = _build(build_module, db, dry_run=True)
    second = _build(build_module, db, dry_run=True)

    assert first["atlas_version"] == second["atlas_version"]
    assert first["positions_sha256"] == second["positions_sha256"]
    assert first["metrics"]["knn_preservation"] == second["metrics"]["knn_preservation"]
    assert first["metrics"]["n_clusters"] == second["metrics"]["n_clusters"]


def test_a_different_seed_is_a_different_atlas(build_module, tmp_path):
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    a = _build(build_module, db, seed=0, dry_run=True)
    b = _build(build_module, db, seed=1, dry_run=True)

    assert a["atlas_version"] != b["atlas_version"]
    assert a["positions_sha256"] != b["positions_sha256"]


def test_two_atlases_coexist_rather_than_overwriting(build_module, tmp_path):
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    a = _build(build_module, db, seed=0)
    b = _build(build_module, db, seed=1)

    with LabelStore(db) as store:
        assert store.atlas_record(a["atlas_version"]) is not None
        assert store.atlas_record(b["atlas_version"]) is not None
        pos_a = store.layout(a["atlas_version"], [(0, 0)])[(0, 0)]
        pos_b = store.layout(b["atlas_version"], [(0, 0)])[(0, 0)]
    assert (pos_a.x, pos_a.y, pos_a.z) != (pos_b.x, pos_b.y, pos_b.z)


def test_a_dry_run_writes_nothing(build_module, tmp_path):
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    summary = _build(build_module, db, dry_run=True)

    with LabelStore(db) as store:
        assert store.atlas_record(summary["atlas_version"]) is None
        assert store.layout(summary["atlas_version"], [(0, 0)]) == {}
        assert store.clusters(summary["atlas_version"]) == []


# --------------------------------------------------------------------------
# naming when it cannot be earned
# --------------------------------------------------------------------------


def test_incoherent_labels_leave_clusters_unnamed_but_measured(build_module, tmp_path):
    """The distinction the spec turns on: measured and rejected, not skipped."""
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True, coherent=False)

    summary = _build(build_module, db, coherence_margin=0.9)

    with LabelStore(db) as store:
        clusters = store.clusters(summary["atlas_version"])

    assert clusters
    assert all(c.name is None for c in clusters)
    assert all(c.coherence is not None for c in clusters), "coherence was not measured"
    assert summary["metrics"]["naming_attempted"] is True
    assert summary["metrics"]["n_named_clusters"] == 0


def test_no_embeddings_builds_an_unnamed_decoder_atlas_and_says_so(build_module, tmp_path):
    """`--embeddings` was never imported: naming is not attempted at all, which
    is a different fact from attempting it and failing.

    Scoped to the decoder source, which is the only one that can be built
    without embeddings at all — the label source *is* the embeddings.
    """
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=False)

    summary = _build(build_module, db, source="decoder")

    with LabelStore(db) as store:
        clusters = store.clusters(summary["atlas_version"])
        record = store.atlas_record(summary["atlas_version"])

    assert clusters, "clusters are still produced without embeddings"
    assert all(c.name is None for c in clusters)
    assert all(c.coherence is None for c in clusters), "nothing to measure with"
    assert record.metrics["naming_attempted"] is False
    assert record.metrics["cross_source_ari"] is None
    assert "too few explanation embeddings" in record.metrics["cross_source_note"]
    # A decoder atlas's positions do not depend on labels at all.
    with_labels_db = tmp_path / "with.db"
    _seed_labels(with_labels_db, with_embeddings=True)
    assert _build(build_module, with_labels_db, source="decoder", dry_run=True)[
        "positions_sha256"
    ] == summary["positions_sha256"]


def test_label_text_never_moves_a_feature_in_a_decoder_atlas(build_module, tmp_path):
    """The decoder source's defining property: two label sets that disagree
    completely must produce byte-identical positions.

    This is what makes the decoder atlas worth keeping alongside the label one.
    The label atlas makes the opposite trade deliberately — its positions *are*
    the label text — which is why the source is recorded rather than inferred.
    """
    coherent = tmp_path / "coherent.db"
    noise = tmp_path / "noise.db"
    _seed_labels(coherent, with_embeddings=True, coherent=True)
    _seed_labels(noise, with_embeddings=True, coherent=False)

    a = _build(build_module, coherent, source="decoder", dry_run=True)
    b = _build(build_module, noise, source="decoder", dry_run=True)

    assert a["positions_sha256"] == b["positions_sha256"]
    assert a["atlas_version"] == b["atlas_version"]


def test_label_text_does_move_a_feature_in_a_label_atlas(build_module, tmp_path):
    """The complement, stated so the difference between the two sources is a
    tested fact and not a comment."""
    coherent = tmp_path / "coherent.db"
    noise = tmp_path / "noise.db"
    _seed_labels(coherent, with_embeddings=True, coherent=True)
    _seed_labels(noise, with_embeddings=True, coherent=False)

    a = _build(build_module, coherent, source="labels", dry_run=True)
    b = _build(build_module, noise, source="labels", dry_run=True)

    assert a["positions_sha256"] != b["positions_sha256"]


def test_a_label_atlas_needs_embeddings_and_says_which_flag_supplies_them(
    build_module, tmp_path
):
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=False)

    with pytest.raises(ValueError, match="--embeddings"):
        _build(build_module, db, source="labels")


def test_an_unknown_source_is_refused(build_module, tmp_path):
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    with pytest.raises(ValueError, match="source must be one of"):
        _build(build_module, db, source="vibes")


def test_the_source_is_recorded_on_the_atlas(build_module, tmp_path):
    """Recorded rather than inferred from the positions: the two sources answer
    different questions and a consumer has to know which it is reading."""
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    for source in ("labels", "decoder"):
        summary = _build(build_module, db, source=source)
        with LabelStore(db) as store:
            record = store.atlas_record(summary["atlas_version"])
        assert record.params["source"] == source
        assert record.metrics["source"] == source
        assert record.metrics["source_dim"] > 0


def test_the_two_sources_coexist_rather_than_overwriting(build_module, tmp_path):
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    a = _build(build_module, db, source="labels")
    b = _build(build_module, db, source="decoder")

    assert a["atlas_version"] != b["atlas_version"]
    with LabelStore(db) as store:
        assert store.atlas_record(a["atlas_version"]).params["source"] == "labels"
        assert store.atlas_record(b["atlas_version"]).params["source"] == "decoder"
        pos_a = store.layout(a["atlas_version"], [(0, 0)])[(0, 0)]
        pos_b = store.layout(b["atlas_version"], [(0, 0)])[(0, 0)]
    assert (pos_a.x, pos_a.y, pos_a.z) != (pos_b.x, pos_b.y, pos_b.z)


def test_explainer_influence_is_measured_and_recorded(build_module, tmp_path):
    """The hazard a text-derived layout carries. The planted fixture gives
    layer 1 a different explainer, so there are two to tell apart."""
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True)

    summary = _build(build_module, db, source="labels")
    ami = summary["metrics"]["explainer_ami"]
    assert ami is not None, "explainer influence was not measured"
    # Adjusted mutual information is corrected for chance, so a small negative
    # value is a normal reading and means "no association" — not an error, and
    # not something to clamp to zero. The planted fixture assigns explainers by
    # layer and embeddings by group, which are independent, so ~0 is the right
    # answer here.
    assert -0.1 <= ami <= 1.0
    assert abs(ami) < 0.2, f"explainer explains the clustering more than expected: {ami}"


def test_explainer_influence_says_so_when_there_is_only_one_explainer(
    build_module, tmp_path
):
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True, one_explainer=True)

    summary = _build(build_module, db, source="labels")
    assert summary["metrics"]["explainer_ami"] is None
    assert "only one explainer" in summary["metrics"]["explainer_ami_note"]


# --------------------------------------------------------------------------
# cross-source agreement
# --------------------------------------------------------------------------


def test_cross_source_agreement_is_recorded_when_embeddings_agree(built):
    summary, _, _ = built
    ari = summary["metrics"]["cross_source_ari"]
    assert ari is not None
    # The planted groups drive both the decoder directions and the embeddings,
    # so the two clusterings should broadly agree.
    assert ari > 0.2


def test_low_agreement_is_recorded_not_acted_on(build_module, tmp_path):
    """A low ARI must not make the build substitute the embedding clustering."""
    db = tmp_path / "labels.db"
    _seed_labels(db, with_embeddings=True, coherent=False)

    summary = _build(build_module, db, source="decoder")
    assert summary["metrics"]["cross_source_ari"] is not None
    assert summary["metrics"]["cross_source_ari"] < 0.2

    # Positions are still the decoder ones: same hash as the coherent build.
    coherent = tmp_path / "coherent.db"
    _seed_labels(coherent, with_embeddings=True, coherent=True)
    assert _build(build_module, coherent, source="decoder", dry_run=True)[
        "positions_sha256"
    ] == summary["positions_sha256"]


# --------------------------------------------------------------------------
# the idle subsample
# --------------------------------------------------------------------------


def test_the_subsample_is_a_strict_subset_of_the_atlas(built):
    summary, db, out = built
    payload = json.loads(out.read_text())

    assert payload["atlas_version"] == summary["atlas_version"]
    assert payload["n_sampled"] == 40
    assert payload["n_total"] == summary["n_features"]
    assert payload["n_sampled"] < payload["n_total"]
    assert "Not the set a trace's features are placed against" in payload["note"]

    nodes = payload["nodes"]
    assert len(nodes["layer"]) == len(nodes["feature"]) == len(nodes["cluster"]) == 40
    assert len(nodes["xyz"]) == 40 * 3

    with LabelStore(db) as store:
        pairs = list(zip(nodes["layer"], nodes["feature"]))
        placed = store.layout(summary["atlas_version"], pairs)
    assert len(placed) == 40, "the subsample named a feature the atlas has no position for"


def test_the_subsample_positions_round_trip_through_int16(built):
    summary, db, out = built
    payload = json.loads(out.read_text())
    extent = payload["extent"]
    xyz = np.array(payload["nodes"]["xyz"], dtype=np.float64).reshape(-1, 3)
    restored = xyz / 32767.0 * extent

    with LabelStore(db) as store:
        pairs = list(zip(payload["nodes"]["layer"], payload["nodes"]["feature"]))
        placed = store.layout(summary["atlas_version"], pairs)

    exact = np.array([[placed[p].x, placed[p].y, placed[p].z] for p in pairs])
    assert np.abs(restored - exact).max() < 1e-4
    assert payload["max_quantisation_error"] < 1e-4


def test_the_subsample_carries_the_areas_including_unnamed_ones(built):
    summary, db, out = built
    payload = json.loads(out.read_text())

    with LabelStore(db) as store:
        clusters = store.clusters(summary["atlas_version"])

    assert len(payload["areas"]) == len(clusters)
    for area, cluster in zip(payload["areas"], clusters):
        assert area["cluster"] == cluster.cluster
        assert area["name"] == cluster.name  # None stays None, not ""
        assert area["n_members"] == cluster.n_members
        # The centroid is what an area is *drawn* at, so the asset has to
        # carry it: the idle brain shows areas without holding the member
        # nodes they were computed from.
        assert tuple(area["centroid"]) == pytest.approx(cluster.centroid)
        assert area["spread"] == pytest.approx(cluster.spread)
