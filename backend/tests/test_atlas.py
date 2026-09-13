"""The atlas's geometry, measurements, naming gate and identity.

No umap and no SAEs here: this is the logic worth testing on a dozen
hand-written vectors. The build script that wires it to 425,984 real ones is
exercised end to end in test_atlas_build.py against a fake SAE.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app import atlas

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


# --------------------------------------------------------------------------
# the shell warp
# --------------------------------------------------------------------------


def _reference_points() -> np.ndarray:
    """Points spanning the shaping's every branch: the poles, the temporal
    band, the crown, the underside floor, and the midline."""
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.05, 1.0, size=(400, 1))
    corners = np.array(
        [
            [0.0, 0.0, 1.0],  # frontal pole
            [0.0, 0.0, -1.0],  # occipital pole
            [0.0, 1.0, 0.0],  # crown
            [0.0, -1.0, 0.0],  # underside, hits the y < -0.3 floor
            [1.0, -0.25, 0.0],  # temporal band
            [-1.0, -0.25, 0.0],
            [0.0, 0.5, 0.0],  # midline near the crown
            [0.0, 0.0, 0.0],  # centre
        ]
    )
    return np.vstack([directions * radii, corners]) * atlas.SHELL_RADIUS * atlas.DEFAULT_FILL





def test_the_dropped_guards_really_are_no_ops():
    """The port drops two `if` guards the TypeScript has, on the grounds that
    the clamped factors make them redundant. Asserted, not assumed."""
    # z well inside both guarded ranges: frontal needs z > 0.15/1.18, occipital
    # needs z < -0.1/1.18, so a band around zero triggers neither.
    points = np.array([[0.5, 0.2, 0.0], [-0.5, -0.2, 0.05], [0.3, 0.0, -0.05]])
    warped = atlas.shape_ellipsoid(points)
    # The only factors that may have acted are temporal/crown/floor, none of
    # which touch z — so z is exactly the plain 1.18 scaling.
    np.testing.assert_allclose(warped[:, 2], points[:, 2] * 1.18, rtol=0, atol=1e-15)


def test_shape_ellipsoid_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        atlas.shape_ellipsoid(np.zeros((4, 2)))


# --------------------------------------------------------------------------
# normalisation and placement
# --------------------------------------------------------------------------


def test_to_unit_ball_fits_inside_the_unit_ball():
    rng = np.random.default_rng(3)
    blob = rng.normal(size=(2000, 3)) * 40 + 1000  # arbitrary scale and offset
    out = atlas.to_unit_ball(blob)
    assert np.linalg.norm(out, axis=1).max() <= 1.0 + 1e-9


def test_to_unit_ball_clips_outliers_rather_than_shrinking_everything():
    """One stranded point must not shrink the cloud it was stranded from."""
    cloud = np.random.default_rng(4).normal(size=(1000, 3))
    with_outlier = np.vstack([cloud, [[500.0, 500.0, 500.0]]])

    without = atlas.to_unit_ball(cloud)
    with_ = atlas.to_unit_ball(with_outlier)[:-1]

    # The bulk keeps its extent; the outlier sits on the surface.
    assert np.linalg.norm(with_, axis=1).mean() > 0.5 * np.linalg.norm(without, axis=1).mean()
    assert np.linalg.norm(atlas.to_unit_ball(with_outlier)[-1]) == pytest.approx(1.0)


def test_to_unit_ball_handles_a_degenerate_projection():
    """Every point identical is a bad projection, not a crash."""
    out = atlas.to_unit_ball(np.ones((10, 3)))
    assert np.all(out == 0.0)


def test_placed_positions_land_inside_the_rendered_shell():
    rng = np.random.default_rng(11)
    blob = rng.normal(size=(5000, 3)) * 7.5
    placed = atlas.place_in_shell(blob)

    atlas.assert_finite(placed)
    assert atlas.within_shell(placed), "a node was placed outside the shell's bounds"


def test_the_bound_holds_over_the_whole_solid_ball():
    """Regression, from a real build failing.

    The bound was first estimated by warping the sphere's *surface* on a coarse
    angular grid. Nodes fill the volume, and the shaping is not monotonic in
    radius, so that under-reported the x extent by ~7e-3 — and the first pilot
    build over 98,304 real features tripped the containment check for a reason
    that was entirely about the estimate.
    """
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(200_000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = atlas.SHELL_RADIUS * atlas.DEFAULT_FILL * rng.random((200_000, 1)) ** (1 / 3)

    interior = atlas.shape_ellipsoid(directions * radii)
    assert atlas.within_shell(interior), "an interior point fell outside the bound"


def test_the_bound_is_tight_enough_to_mean_something():
    """A bound with a generous slack would check nothing. Refining the grid
    must not move it by more than the slack allows."""
    coarse_low, coarse_high = atlas.shell_bounds()

    original, atlas._BOUND_GRID = atlas._BOUND_GRID, 1500
    atlas._BOUNDS_CACHE.clear()
    try:
        fine_low, fine_high = atlas.shell_bounds()
    finally:
        atlas._BOUND_GRID = original
        atlas._BOUNDS_CACHE.clear()

    assert np.abs(fine_high - coarse_high).max() < atlas.BOUND_SLACK
    assert np.abs(fine_low - coarse_low).max() < atlas.BOUND_SLACK


def test_the_z_bound_is_exact():
    """z_final = 1.18 * z0 is the shaping's only z operation, so this extent
    needs no search and the grid must not be what determines it."""
    _, high = atlas.shell_bounds()
    assert high[2] == pytest.approx(1.18 * atlas.SHELL_RADIUS * atlas.DEFAULT_FILL)


def test_the_bound_is_deterministic():
    """A grid, not a sample — so `within_shell` cannot give different answers
    in different processes."""
    atlas._BOUNDS_CACHE.clear()
    first = atlas.shell_bounds()
    atlas._BOUNDS_CACHE.clear()
    second = atlas.shell_bounds()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_within_shell_rejects_a_position_outside_it():
    low, high = atlas.shell_bounds()
    assert not atlas.within_shell(np.array([[high[0] + 1.0, 0.0, 0.0]]))
    assert not atlas.within_shell(np.array([[0.0, low[1] - 1.0, 0.0]]))
    assert atlas.within_shell(np.zeros((1, 3)))


def test_the_cloud_is_kept_clear_of_the_mesh_surface():
    """The cloud is deliberately inset from the surface, where the mesh's gyri
    and longitudinal fissure displace vertices inward by up to ~0.14.

    Compared against the shaped mesh at `fill=1.0` rather than against a sphere
    of SHELL_RADIUS: the shaping scales z by 1.18, so the brain is longer
    front-to-back than 1.4 and a sphere is the wrong reference.
    """
    inner_low, inner_high = atlas.shell_bounds(fill=atlas.DEFAULT_FILL)
    mesh_low, mesh_high = atlas.shell_bounds(fill=1.0)

    assert (inner_high < mesh_high).all(), "the cloud reaches the mesh's own extent"
    assert (inner_low > mesh_low).all()

    # And the inset is worth more than the mesh's own displacement amplitude.
    assert (mesh_high - inner_high).min() > 0.05


def test_assert_finite_names_the_damage():
    bad = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="1 of 2"):
        atlas.assert_finite(bad)


# --------------------------------------------------------------------------
# kNN preservation
# --------------------------------------------------------------------------


def test_an_identity_layout_preserves_everything():
    """3D data laid out as itself: the measurement's upper bound."""
    rng = np.random.default_rng(5)
    points = rng.normal(size=(300, 3))
    # Cosine in high-dim against euclidean in low-dim, so this is exercised on
    # unit-norm points where the two orderings agree.
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    got = atlas.knn_preservation(points, points, k=10, sample=100, seed=0)
    assert got["knn_preservation"] == pytest.approx(1.0)
    assert got["knn_k"] == 10
    assert got["knn_sample"] == 100


def test_a_scrambled_layout_preserves_almost_nothing():
    rng = np.random.default_rng(6)
    high = rng.normal(size=(600, 32))
    low = rng.normal(size=(600, 3))  # unrelated to `high`

    got = atlas.knn_preservation(high, low, k=10, sample=200, seed=0)
    assert got["knn_preservation"] < 0.1


def test_knn_preservation_records_how_it_was_measured():
    rng = np.random.default_rng(8)
    got = atlas.knn_preservation(
        rng.normal(size=(200, 16)), rng.normal(size=(200, 3)), k=7, sample=50
    )
    assert got["knn_k"] == 7
    assert got["knn_sample"] == 50
    assert got["knn_high_metric"] == "cosine"
    assert got["knn_low_metric"] == "euclidean"
    assert "knn_preservation_std" in got


def test_knn_preservation_is_deterministic_under_a_seed():
    rng = np.random.default_rng(9)
    high, low = rng.normal(size=(300, 16)), rng.normal(size=(300, 3))
    a = atlas.knn_preservation(high, low, k=5, sample=80, seed=42)
    b = atlas.knn_preservation(high, low, k=5, sample=80, seed=42)
    assert a == b


def test_knn_preservation_refuses_mismatched_inputs():
    with pytest.raises(ValueError, match="rows against"):
        atlas.knn_preservation(np.zeros((10, 4)), np.zeros((9, 3)))


def test_knn_preservation_of_a_single_point_is_zero_not_an_error():
    got = atlas.knn_preservation(np.zeros((1, 4)), np.zeros((1, 3)))
    assert got["knn_preservation"] == 0.0
    assert got["knn_k"] == 0


def test_knn_k_is_capped_by_the_population():
    got = atlas.knn_preservation(np.eye(4), np.eye(4)[:, :3], k=50, sample=4)
    assert got["knn_k"] == 3


# --------------------------------------------------------------------------
# coherence, naming, explainers
# --------------------------------------------------------------------------


def test_mean_pairwise_cosine_matches_the_naive_computation():
    """The O(n) identity is exact, and it is compared against a threshold, so
    an approximation would not do."""
    rng = np.random.default_rng(12)
    v = rng.normal(size=(40, 8))
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)

    naive = np.array(
        [unit[i] @ unit[j] for i in range(len(unit)) for j in range(len(unit)) if i != j]
    ).mean()
    assert atlas.mean_pairwise_cosine(v) == pytest.approx(naive)


def test_mean_pairwise_cosine_of_identical_vectors_is_one():
    assert atlas.mean_pairwise_cosine(np.tile([1.0, 2.0, 3.0], (5, 1))) == pytest.approx(1.0)


def test_mean_pairwise_cosine_edge_cases():
    assert atlas.mean_pairwise_cosine(np.zeros((0, 3))) == 0.0
    assert atlas.mean_pairwise_cosine(np.ones((1, 3))) == 1.0


def test_the_gate_is_relative_to_the_baseline_not_absolute():
    """Regression from the pilot build.

    An absolute threshold of 0.25 named clusters whose coherence was 0.281
    against a baseline of 0.232 — members agreeing 0.049 more than a random
    group of the same size, dressed up as a name. Embeddings from one explainer
    share a similarity floor, so an absolute test measures the floor.
    """
    barely = atlas.Coherence(value=0.281, baseline=0.232, n_measured=200)
    assert barely.margin == pytest.approx(0.049)
    assert not barely.earns_a_name()  # would have passed an absolute 0.25

    genuinely = atlas.Coherence(value=0.62, baseline=0.23, n_measured=200)
    assert genuinely.earns_a_name()


def test_a_high_absolute_coherence_still_fails_if_the_baseline_is_high():
    """The whole point: 0.9 is not impressive when a random group scores 0.88."""
    assert not atlas.Coherence(value=0.90, baseline=0.88, n_measured=50).earns_a_name()


def test_no_baseline_means_no_name():
    """A margin over a baseline that was never measured is not zero, it is
    undefined — treating it as zero would readmit the absolute test."""
    unmeasurable = atlas.Coherence(value=0.99, baseline=None, n_measured=50)
    assert unmeasurable.margin is None
    assert not unmeasurable.earns_a_name()


def test_the_margin_is_configurable_and_reported():
    coherence = atlas.Coherence(value=0.40, baseline=0.20, n_measured=50)
    assert coherence.earns_a_name(margin=0.15)
    assert not coherence.earns_a_name(margin=0.25)


def test_a_tight_group_beats_its_own_random_baseline():
    rng = np.random.default_rng(13)
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    members = [direction + rng.normal(scale=0.05, size=4) for _ in range(20)]
    pool = [rng.normal(size=4) for _ in range(200)]

    got = atlas.measure_coherence(members, pool, seed=0)
    assert got.measured
    assert got.value > 0.9
    assert got.baseline is not None and got.value > got.baseline


def test_a_scattered_group_does_not_beat_its_baseline_by_much():
    rng = np.random.default_rng(14)
    members = [rng.normal(size=8) for _ in range(30)]
    pool = [rng.normal(size=8) for _ in range(300)]

    got = atlas.measure_coherence(members, pool, seed=0)
    assert abs(got.value - got.baseline) < 0.25


def test_no_embeddings_reads_as_unmeasured_not_as_zero():
    """"we never measured" and "we measured and it was bad" are different
    facts, and both end in an unnamed cluster."""
    got = atlas.measure_coherence([], [], seed=0)
    assert not got.measured
    assert got.value is None and got.baseline is None


def test_one_member_is_unmeasurable():
    got = atlas.measure_coherence([np.ones(4)], [np.ones(4)] * 5)
    assert not got.measured
    assert got.n_measured == 1


def test_a_baseline_needs_a_pool_at_least_as_big_as_the_group():
    got = atlas.measure_coherence([np.ones(4), np.zeros(4) + 0.5], [np.ones(4)])
    assert got.measured
    assert got.baseline is None


def test_medoid_is_a_real_member_nearest_the_group_direction():
    embeddings = {
        ("a"): np.array([1.0, 0.0]),
        ("b"): np.array([0.98, 0.2]),
        ("c"): np.array([-1.0, 0.0]),
    }
    # a and b point together; c opposes. The mean direction sits near a/b.
    assert atlas.medoid(["a", "b", "c"], embeddings) in {"a", "b"}


def test_medoid_of_members_without_embeddings_is_none():
    assert atlas.medoid(["x", "y"], {}) is None


def test_medoid_ignores_members_it_has_no_embedding_for():
    embeddings = {"a": np.array([1.0, 0.0])}
    assert atlas.medoid(["a", "missing"], embeddings) == "a"


def test_explainer_mix_is_ordered_by_count():
    explainers = {1: "gpt-4o-mini", 2: "gpt-4o-mini", 3: "gemini-2.5-flash-lite"}
    assert atlas.explainer_mix([1, 2, 3], explainers) == "gpt-4o-mini:2, gemini-2.5-flash-lite:1"


def test_explainer_mix_records_an_unknown_explainer_rather_than_dropping_it():
    assert atlas.explainer_mix([1, 2], {1: "gpt-4o-mini"}) == "?:1, gpt-4o-mini:1"


def test_cluster_geometry_reports_a_centroid_and_a_spread():
    positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    centre, spread = atlas.cluster_geometry(positions)
    assert centre == pytest.approx((0.0, 0.0, 0.0))
    assert spread == pytest.approx(1.0)


def test_cluster_geometry_of_nothing_is_the_origin():
    assert atlas.cluster_geometry(np.zeros((0, 3))) == ((0.0, 0.0, 0.0), 0.0)


# --------------------------------------------------------------------------
# identity and determinism
# --------------------------------------------------------------------------


def test_the_same_positions_hash_the_same():
    a = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert atlas.positions_hash(a) == atlas.positions_hash(a.copy())


def test_different_positions_hash_differently():
    a = np.array([[0.1, 0.2, 0.3]])
    b = np.array([[0.1, 0.2, 0.3001]])
    assert atlas.positions_hash(a) != atlas.positions_hash(b)


def test_the_hash_tolerates_float_noise_below_its_rounding():
    """An identical rebuild differing in the last bits of a float must not
    report itself as a different atlas — that would be a false alarm about the
    one property this hash exists to confirm."""
    a = np.array([[0.1, 0.2, 0.3]])
    b = a + 1e-12
    assert atlas.positions_hash(a) == atlas.positions_hash(b)


def test_the_version_changes_with_the_seed():
    params = {"n_neighbors": 15}
    assert atlas.atlas_version("r", "16k", 0, params) != atlas.atlas_version("r", "16k", 1, params)


def test_the_version_changes_with_the_parameters():
    assert atlas.atlas_version("r", "16k", 0, {"n_neighbors": 15}) != atlas.atlas_version(
        "r", "16k", 0, {"n_neighbors": 30}
    )


def test_the_version_is_stable_for_the_same_inputs():
    args = ("r", "16k", 3, {"n_neighbors": 15, "min_dist": 0.1})
    assert atlas.atlas_version(*args) == atlas.atlas_version(*args)


def test_the_version_names_the_width_and_seed_readably():
    version = atlas.atlas_version("r", "16k", 7, {})
    assert version.startswith("16k-s7-")


# --------------------------------------------------------------------------
# the idle subsample
# --------------------------------------------------------------------------


def test_the_subsample_is_a_deterministic_strict_subset():
    a = atlas.subsample_indices(10_000, 500, seed=1)
    b = atlas.subsample_indices(10_000, 500, seed=1)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 500
    assert len(set(a.tolist())) == 500
    assert a.max() < 10_000


def test_asking_for_more_than_exists_returns_everything():
    np.testing.assert_array_equal(atlas.subsample_indices(50, 500), np.arange(50))


def test_a_different_seed_draws_a_different_sample():
    a = atlas.subsample_indices(10_000, 200, seed=1)
    b = atlas.subsample_indices(10_000, 200, seed=2)
    assert not np.array_equal(a, b)


def test_quantisation_error_is_far_below_a_pixel():
    rng = np.random.default_rng(15)
    positions = atlas.place_in_shell(rng.normal(size=(3000, 3)))
    quantised, extent = atlas.quantise_positions(positions)

    assert quantised.dtype == np.int16
    error = atlas.relative_error(positions, quantised, extent)
    # A shell radius of 1.4 across ~600 device pixels is ~0.0023 units/pixel.
    assert error < 1e-4


def test_quantisation_of_nothing_is_not_a_division_by_zero():
    quantised, extent = atlas.quantise_positions(np.zeros((0, 3)))
    assert quantised.shape == (0, 3)
    assert extent > 0


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------


def test_cluster_summary_counts_noise_rather_than_sweeping_it_up():
    labels = np.array([0, 0, 0, 1, 1, -1, -1, -1, -1])
    got = atlas.summarise_clusters(labels)
    assert got["n_clusters"] == 2
    assert got["n_noise"] == 4
    assert got["noise_fraction"] == pytest.approx(4 / 9)
    assert got["cluster_size_min"] == 2
    assert got["cluster_size_max"] == 3


def test_cluster_summary_of_all_noise():
    got = atlas.summarise_clusters(np.array([-1, -1]))
    assert got["n_clusters"] == 0
    assert got["noise_fraction"] == 1.0


def test_an_absent_metric_says_so_rather_than_printing_zero():
    assert atlas.fmt_metric(None) == "not measured"
    assert atlas.fmt_metric(0.0) == "0.000"


def test_size_and_percent_formatting():
    assert atlas.sizeof_fmt(512) == "512B"
    assert atlas.sizeof_fmt(2048) == "2.0KB"
    assert atlas.percent(1, 4) == "25.0%"
    assert atlas.percent(0, 0) == "0%"
