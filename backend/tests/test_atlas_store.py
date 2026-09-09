"""The atlas tables in the label store: positions, clusters, the record.

No umap here — this is the storage contract the build writes into and the
service reads out of, tested with hand-written rows.
"""

from __future__ import annotations

import pytest

from app.labels import AtlasRecord, ClusterRow, LabelStore, LayoutRow

VERSION = "test-atlas-v1"


@pytest.fixture
def store(tmp_path):
    with LabelStore(tmp_path / "labels.db") as s:
        yield s


# -- positions -----------------------------------------------------------


def test_positions_round_trip(store):
    rows = [
        LayoutRow(layer=0, feature=1, x=0.1, y=-0.2, z=0.3, cluster=4),
        LayoutRow(layer=20, feature=6631, x=-0.5, y=0.25, z=0.75, cluster=7),
    ]
    assert store.put_layout(VERSION, rows) == 2

    got = store.layout(VERSION, [(0, 1), (20, 6631)])
    assert got[(0, 1)] == rows[0]
    assert got[(20, 6631)] == rows[1]


def test_an_unplaced_pair_is_absent_not_zeroed(store):
    """The whole reason `layout` returns a dict: a caller must be able to tell
    a feature the atlas cannot place from one it places at the origin."""
    store.put_layout(VERSION, [LayoutRow(layer=0, feature=1, x=0.0, y=0.0, z=0.0)])

    got = store.layout(VERSION, [(0, 1), (0, 2)])
    assert (0, 1) in got  # really is at the origin
    assert (0, 2) not in got  # has no position at all


def test_no_cluster_is_stored_as_minus_one(store):
    """HDBSCAN's noise label is a real answer, distinct from 'not clustered'."""
    store.put_layout(VERSION, [LayoutRow(layer=3, feature=9, x=0.0, y=0.0, z=0.0)])
    assert store.layout(VERSION, [(3, 9)])[(3, 9)].cluster == -1


def test_positions_are_scoped_by_atlas_version(store):
    """Two atlases coexist, so one can be compared against another."""
    store.put_layout(VERSION, [LayoutRow(layer=0, feature=1, x=1.0, y=0.0, z=0.0)])
    store.put_layout("other-v2", [LayoutRow(layer=0, feature=1, x=-1.0, y=0.0, z=0.0)])

    assert store.layout(VERSION, [(0, 1)])[(0, 1)].x == 1.0
    assert store.layout("other-v2", [(0, 1)])[(0, 1)].x == -1.0


def test_put_layout_replaces_a_position_in_place(store):
    store.put_layout(VERSION, [LayoutRow(layer=0, feature=1, x=1.0, y=0.0, z=0.0)])
    store.put_layout(VERSION, [LayoutRow(layer=0, feature=1, x=2.0, y=0.0, z=0.0, cluster=5)])

    row = store.layout(VERSION, [(0, 1)])[(0, 1)]
    assert (row.x, row.cluster) == (2.0, 5)


def test_layout_reads_more_pairs_than_one_sqlite_batch(store):
    """`layout` chunks its IN clause; the chunking must not drop pairs."""
    rows = [
        LayoutRow(layer=layer, feature=feature, x=float(layer), y=float(feature), z=0.0)
        for layer in range(4)
        for feature in range(300)
    ]
    store.put_layout(VERSION, rows)

    pairs = [(r.layer, r.feature) for r in rows]
    got = store.layout(VERSION, pairs)
    assert len(got) == len(rows) == 1200


def test_layout_of_nothing_is_empty(store):
    assert store.layout(VERSION, []) == {}
    assert store.put_layout(VERSION, []) == 0


# -- the record ----------------------------------------------------------


def _record(**kwargs) -> AtlasRecord:
    defaults = dict(
        atlas_version=VERSION,
        release="gemma-scope-2b-pt-res-canonical",
        width="16k",
        seed=0,
        params={"n_neighbors": 15, "min_dist": 0.1},
        metrics={"knn_preservation": 0.42, "knn_k": 20},
        positions_sha256="a" * 64,
        n_features=1200,
        n_clusters=17,
    )
    return AtlasRecord(**{**defaults, **kwargs})


def test_the_record_round_trips_including_its_metrics(store):
    store.put_atlas_record(_record())
    got = store.atlas_record(VERSION)

    assert got.positions_sha256 == "a" * 64
    assert got.metrics["knn_preservation"] == 0.42
    assert got.metrics["knn_k"] == 20
    assert got.params["n_neighbors"] == 15
    assert got.built_at, "built_at is filled in when not supplied"


def test_a_missing_atlas_reads_as_none_not_as_an_empty_one(store):
    """"no atlas" and "an atlas with no features" are different answers."""
    assert store.atlas_record("never-built") is None
    assert store.atlas_record() is None

    store.put_atlas_record(_record(n_features=0, n_clusters=0))
    empty = store.atlas_record()
    assert empty is not None and empty.n_features == 0


def test_atlas_record_defaults_to_the_most_recent(store):
    store.put_atlas_record(_record(atlas_version="v1", built_at="2026-01-01T00:00:00+00:00"))
    store.put_atlas_record(_record(atlas_version="v2", built_at="2026-06-01T00:00:00+00:00"))
    assert store.atlas_record().atlas_version == "v2"


def test_a_rebuilt_atlas_replaces_its_own_record(store):
    store.put_atlas_record(_record(positions_sha256="b" * 64))
    store.put_atlas_record(_record(positions_sha256="c" * 64))
    assert store.atlas_record(VERSION).positions_sha256 == "c" * 64


# -- clusters ------------------------------------------------------------


def test_a_named_cluster_records_where_its_name_came_from(store):
    store.put_clusters(
        VERSION,
        [
            ClusterRow(
                cluster=7,
                n_members=214,
                centroid=(0.1, 0.2, 0.3),
                spread=0.35,
                name="bridge and crossing",
                name_source=(20, 1370),
                coherence=0.71,
                baseline_coherence=0.08,
                explainers="gpt-4o-mini:180, gemini-2.5-flash-lite:34",
            )
        ],
    )

    (row,) = store.clusters(VERSION)
    assert row.name == "bridge and crossing"
    assert row.name_source == (20, 1370)
    assert row.coherence == 0.71
    assert row.baseline_coherence == 0.08
    assert "gemini-2.5-flash-lite" in row.explainers


def test_an_unnamed_cluster_still_records_the_coherence_that_failed(store):
    """Below threshold: unnamed for a stated reason, not for no reason."""
    store.put_clusters(
        VERSION,
        [ClusterRow(cluster=3, n_members=40, centroid=(0, 0, 0), spread=0.9, coherence=0.05)],
    )

    (row,) = store.clusters(VERSION)
    assert row.name is None
    assert row.name_source is None
    assert row.coherence == 0.05


def test_measured_but_low_is_distinguishable_from_never_measured(store):
    """`coherence is None` means there were no embeddings to measure with;
    `coherence == 0.05` means we measured and it was bad."""
    store.put_clusters(
        VERSION,
        [
            ClusterRow(cluster=1, n_members=10, centroid=(0, 0, 0), spread=0.1, coherence=0.05),
            ClusterRow(cluster=2, n_members=10, centroid=(0, 0, 0), spread=0.1, coherence=None),
        ],
    )

    by_id = {row.cluster: row for row in store.clusters(VERSION)}
    assert by_id[1].coherence == 0.05
    assert by_id[2].coherence is None
    assert by_id[1].name is None and by_id[2].name is None


def test_clusters_of_an_unknown_atlas_are_empty(store):
    assert store.clusters("never-built") == []
