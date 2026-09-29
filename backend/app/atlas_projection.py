"""The two heavy steps of an atlas build: projection to 3D, and clustering.

Shared by `scripts/build_feature_atlas.py` (Gemma Scope) and
`app.dictionary_atlas.build` (local dictionaries). umap and scikit-learn are
imported inside the functions, never at module scope, so nothing that merely
imports this module pulls in numba — see `tests/test_atlas_deps.py`.
"""

from __future__ import annotations

import time

import numpy as np

from . import atlas

# Dimensions to reduce to before UMAP; 0 disables the step, which is the
# default, and the default is measured rather than conventional.
#
# Reducing first is the standard recipe and it is wrong for this data. SAE
# decoder directions are a near-orthogonal dictionary, so their spectrum is
# almost flat: on the six pilot layers, PCA to 50 dimensions retained **7.8%**
# of the variance and PCA to 200 retained 20.3%. The first pilot build ran with
# PCA-50 and scored a kNN preservation of 0.044 with two clusters over 98,304
# features — the projection had nothing left to preserve, because 92% of the
# signal was discarded before UMAP saw it.
#
# The structure is really there: a feature's nearest neighbour in the full
# 2304-d space sits at a cosine of +0.43 on average (p99 +0.90) against a
# random-pair scale of 1/sqrt(2304) = 0.02. UMAP's own neighbour search handles
# 2304 dimensions directly, so it is given them directly.
DEFAULT_PCA_DIM = 0

# UMAP's own parameters. `n_neighbors` is how much of the neighbourhood the
# projection tries to keep — larger trades local fidelity for global shape, and
# local fidelity is the only thing this atlas claims.
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1

# HDBSCAN's floor on a cluster. At 425,984 features this yields clusters big
# enough to draw as an area and to measure a coherence over.
DEFAULT_MIN_CLUSTER_SIZE = 200


def project(
    directions: np.ndarray,
    pca_dim: int = DEFAULT_PCA_DIM,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """[N, d_model] decoder directions -> [N, 3].

    Cosine throughout: `W_dec` rows are read for where they point, and their
    norms carry a different fact (roughly, how much the feature writes), which
    is not what a position should encode.

    `random_state` is set, which makes UMAP single-threaded and slower. That is
    the right trade for an artifact whose whole value is being the same map
    every time — see the determinism check on the record.
    """
    import umap
    from sklearn.decomposition import PCA

    x = np.asarray(directions, dtype=np.float32)
    t0 = time.time()

    # Off by default; see DEFAULT_PCA_DIM for the measurement that settled it.
    if pca_dim and pca_dim > 0 and min(x.shape) > 1:
        n_components = min(pca_dim, min(x.shape) - 1)
        if n_components < x.shape[1]:
            reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
            x = reducer.fit_transform(x)
            if verbose:
                kept = float(reducer.explained_variance_ratio_.sum())
                print(f"PCA -> {x.shape[1]}d in {time.time() - t0:.0f}s (kept {kept:.1%} of variance)")

    # UMAP needs more points than neighbours; a small slice (a test, or
    # --layers 0) must not fail on the parameter rather than on the data.
    neighbors = max(2, min(n_neighbors, len(x) - 1))

    t1 = time.time()
    embedding = umap.UMAP(
        n_components=3,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=verbose,
    ).fit_transform(x)
    if verbose:
        print(f"UMAP -> 3d in {time.time() - t1:.0f}s")

    return np.asarray(embedding, dtype=np.float64)


def project_pca(directions: np.ndarray, seed: int = 0) -> np.ndarray:
    """[N, d_model] -> [N, 3] by PCA on unit-normalised rows.

    The fast option: seconds instead of minutes, at the cost of keeping only
    the three widest directions of variance. Rows are normalised first so it
    reads direction, like UMAP's cosine metric does.
    """
    from sklearn.decomposition import PCA

    x = np.asarray(directions, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return PCA(n_components=3, random_state=seed).fit_transform(x)


def cluster(
    positions: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    verbose: bool = True,
) -> np.ndarray:
    """Density-based clusters over the placed positions.

    HDBSCAN rather than k-means: the number of natural groupings is not known
    ahead of time, and "this feature belongs to no cluster" (-1) is a real
    answer that k-means cannot give. Clustered in the 3D layout rather than in
    the original space so the areas drawn on screen are the areas that were
    measured — a cluster that is only a cluster in 2304 dimensions would look
    like scattered noise to a viewer, which would make its name a lie.
    """
    from sklearn.cluster import HDBSCAN

    size = max(2, min(min_cluster_size, max(2, len(positions) // 2)))
    t0 = time.time()
    labels = HDBSCAN(min_cluster_size=size).fit_predict(np.asarray(positions, dtype=np.float64))
    if verbose:
        summary = atlas.summarise_clusters(labels)
        print(
            f"HDBSCAN -> {summary['n_clusters']} clusters, "
            f"{summary['n_noise']:,} unclustered "
            f"({atlas.percent(summary['n_noise'], len(labels))}) in {time.time() - t0:.0f}s"
        )
    return np.asarray(labels)
