"""Build the feature atlas: a fixed position and a cluster for every SAE feature.

    python scripts/build_feature_atlas.py                    # all 26 layers, ~7.9GB of SAEs
    python scripts/build_feature_atlas.py --layers 0-5       # a smaller slice
    python scripts/build_feature_atlas.py --dry-run          # measure, print, write nothing

Projects one representation of every feature down to three dimensions and warps
the result into the rendered brain's own volume. Writes the positions, the
clusters and the atlas record into the label DB beside the explanations, so one
SQLite file answers both "where is this feature" and "what does it mean".

Two sources, and an atlas records which it used (see SOURCES):

    --source labels    explanation embeddings. The default, and the only source
                       that yields areas: 0.303 kNN preservation, 32 clusters,
                       27 of them earning a name.
    --source decoder   every layer's `W_dec` rows, all in the one
                       residual-stream basis the layers share. The model's own
                       geometry: 0.154 preservation, and a continuum with no
                       areas in it at all.

This is the only module in the project that imports umap; see
`tests/test_atlas_deps.py`, which keeps it that way. The logic worth testing on
a dozen vectors instead of 425,984 lives in `app/atlas.py`.

What the build measures, and why each number is here:

    knn_preservation      how much of the 2304-d neighbourhood structure
                          survived the projection. The headline. 3 dimensions
                          out of 2304 lose a great deal; this says how much.
    cluster coherence     whether a cluster's members' labels agree, against a
                          shuffled baseline of the same size. Gates naming.
    cross_source_ari      whether the two sources carve the space the same
                          way. Not a gate; two independent representations
                          agreeing is evidence the areas are real.
    explainer_ami         how far the clustering is predicted by which
                          explainer wrote the labels. The hazard for a
                          text-derived layout. Measured 0.026 on the pilot,
                          which is what reversed the original refusal to lay
                          features out by label text.
    positions_sha256      the determinism check. A rebuild from the same seed
                          must reproduce it.

On what a label-derived layout may claim: it maps *descriptions of* features,
not the model's computation — near means an explainer described two features
similarly. Neuronpedia's export uses `gemini-2.5-flash-lite` on layers
16/18/20/22/24 and `gpt-4o-mini` on the other 21, so the worry was that such a
layout would separate those five layers for a reason about the labeller. That
worry is measured (`explainer_ami`), not assumed, and on the pilot it came back
at 0.026 — no meaningful influence. The explainer mix is still recorded per
cluster, because a measurement that stops being taken stops being true.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import atlas  # noqa: E402
from app.labels import (  # noqa: E402
    DEFAULT_DB_PATH,
    AtlasRecord,
    ClusterRow,
    LabelStore,
    LayoutRow,
)
from app.sae_cache import DEFAULT_WIDTH, RELEASE, get_sae, pick_device  # noqa: E402

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

# How many features the cross-source comparison uses. Clustering the full set
# twice doubles the build for a diagnostic, and a sample answers the question.
DEFAULT_CROSS_SAMPLE = 20_000

DEFAULT_SUBSAMPLE = 20_000


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


# The two representations an atlas can be projected from. They answer
# different questions and neither subsumes the other, so the atlas records
# which one it came from rather than leaving it to be inferred.
#
#   decoder  W_dec rows: the model's own geometry. Near means two features
#            write similarly into the residual stream. Measured on the pilot:
#            kNN preservation 0.154, and no areas at all — HDBSCAN calls 99.1%
#            of the 2304-d space noise and a forced partition earns 0 names.
#   labels   explanation embeddings: descriptions of features. Near means an
#            explainer described them similarly. Measured: kNN preservation
#            0.303, 32 clusters, 27 of them earning a name, and an explainer
#            influence (AMI) of 0.026 — the confound this was feared for does
#            not materialise. This is the default because it is the only one
#            that yields areas.
SOURCES = ("labels", "decoder")
DEFAULT_SOURCE = "labels"


def load_decoder_directions(
    layers: list[int],
    width: str = DEFAULT_WIDTH,
    device: str | None = None,
    sae_provider=None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Every layer's `W_dec` rows, stacked, with the key for each row.

    Returned as one [n_layers * d_sae, d_model] array because that is the whole
    premise: the rows share a basis, so they are projected together and a
    feature from layer 12 can land beside one from layer 20.
    """
    provider = sae_provider or (lambda layer: get_sae(layer, width, device or pick_device()))

    blocks: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    t0 = time.time()

    for i, layer in enumerate(layers):
        sae = provider(layer)
        w_dec = np.asarray(
            sae.W_dec.detach().cpu().numpy() if hasattr(sae.W_dec, "detach") else sae.W_dec,
            dtype=np.float32,
        )
        blocks.append(w_dec)
        keys.extend((layer, feature) for feature in range(w_dec.shape[0]))
        if verbose:
            print(
                f"\rW_dec {i + 1}/{len(layers)} layers, {sum(b.shape[0] for b in blocks):,} "
                f"features ({time.time() - t0:.0f}s)",
                end="",
                flush=True,
            )

    if verbose:
        print()
    if not blocks:
        raise ValueError("no layers requested")

    return np.vstack(blocks), keys


def load_label_embeddings(
    layers: list[int],
    width: str = DEFAULT_WIDTH,
    store: LabelStore | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    verbose: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Every feature's explanation embedding, stacked, with the key for each row.

    Only features that *have* an embedding are returned, and the keys say which
    — a feature Neuronpedia never explained gets no position from this source
    rather than a position derived from nothing. ~0.3% of features are in that
    position, plus everything not covered by an `--embeddings` import.
    """
    owned = store is None
    store = store or LabelStore(db_path, width=width)
    try:
        keys: list[tuple[int, int]] = []
        vectors: list[np.ndarray] = []
        for layer in layers:
            found = store.embeddings(layer, range(WIDTH_D_SAE.get(width, 16384)))
            for feature in sorted(found):
                keys.append((layer, feature))
                vectors.append(found[feature])
            if verbose:
                print(
                    f"\rembeddings {len(keys):,} features over "
                    f"{layers.index(layer) + 1}/{len(layers)} layers",
                    end="",
                    flush=True,
                )
    finally:
        if owned:
            store.close()

    if verbose:
        print()
    if not vectors:
        raise ValueError(
            "no explanation embeddings in the label store — import them with "
            "scripts/import_neuronpedia.py --embeddings"
        )
    return np.stack(vectors).astype(np.float32), keys


# Feature counts per SAE width, for enumerating candidate indices without
# loading an SAE. Mirrors service/app.py's table of the same name.
WIDTH_D_SAE = {"16k": 16384, "65k": 65536, "262k": 262144}


# --------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# labels: naming, coherence, the cross-source check
# --------------------------------------------------------------------------


def load_label_side_tables(
    store: LabelStore, keys: list[tuple[int, int]], verbose: bool = True
) -> tuple[dict, dict, dict]:
    """(embedding, text, explainer) for the keys the store knows about.

    All three come back keyed by (layer, feature) and all three may be sparse:
    ~0.3% of features have no explanation at all, and embeddings are only
    present if the export was imported with `--embeddings`.
    """
    by_layer: dict[int, list[int]] = {}
    for layer, feature in keys:
        by_layer.setdefault(layer, []).append(feature)

    embeddings: dict[tuple[int, int], np.ndarray] = {}
    texts: dict[tuple[int, int], str] = {}
    explainers: dict[tuple[int, int], str] = {}

    for layer, features in sorted(by_layer.items()):
        for feature, vector in store.embeddings(layer, features).items():
            embeddings[(layer, feature)] = vector
        for feature, label in store.get_many(layer, features).items():
            if label.text:
                texts[(layer, feature)] = label.text
            if label.explainer:
                explainers[(layer, feature)] = label.explainer

    if verbose:
        print(
            f"labels: {len(texts):,} texts, {len(embeddings):,} embeddings "
            f"for {len(keys):,} features"
        )
    return embeddings, texts, explainers


def name_clusters(
    labels: np.ndarray,
    positions: np.ndarray,
    keys: list[tuple[int, int]],
    embeddings: dict,
    texts: dict,
    explainers: dict,
    margin: float = atlas.DEFAULT_COHERENCE_MARGIN,
    seed: int = 0,
) -> list[ClusterRow]:
    """One ClusterRow per cluster, named only where the naming is earned.

    A cluster of decoder-similar features is not guaranteed to be
    label-coherent, so the name is gated on a measurement rather than assumed —
    and the gate is how far the members beat a random group of their own size,
    not an absolute coherence. See `atlas.DEFAULT_COHERENCE_MARGIN` for the
    pilot numbers that settled that.

    Three outcomes, all distinguishable downstream: named; unnamed *with* a
    coherence and a baseline, meaning it was measured and did not clear the
    margin; and unnamed with `coherence=None`, meaning there were no embeddings
    and naming was never attempted.
    """
    pool = list(embeddings.values())
    rows: list[ClusterRow] = []

    for cluster_id in sorted({int(v) for v in labels.tolist()} - {-1}):
        member_mask = labels == cluster_id
        member_keys = [keys[i] for i in np.flatnonzero(member_mask)]
        centroid, spread = atlas.cluster_geometry(positions[member_mask])

        member_embeddings = [embeddings[k] for k in member_keys if k in embeddings]
        coherence = atlas.measure_coherence(member_embeddings, pool, seed=seed)

        name: str | None = None
        name_source: tuple[int, int] | None = None
        if coherence.earns_a_name(margin):
            pick = atlas.medoid(member_keys, embeddings)
            if pick is not None and pick in texts:
                name = texts[pick]
                name_source = pick

        rows.append(
            ClusterRow(
                cluster=cluster_id,
                n_members=int(member_mask.sum()),
                centroid=centroid,
                spread=spread,
                name=name,
                name_source=name_source,
                coherence=coherence.value,
                baseline_coherence=coherence.baseline,
                explainers=atlas.explainer_mix(member_keys, explainers),
            )
        )

    return rows


def cross_source_agreement(
    labels: np.ndarray,
    keys: list[tuple[int, int]],
    embeddings: dict,
    seed: int = 0,
    sample: int = DEFAULT_CROSS_SAMPLE,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> dict:
    """Do explanation embeddings carve the space the way decoder directions do?

    Clusters a sample of the same features by their label embeddings and
    reports the adjusted Rand index against the shipped clustering. Never a
    gate: the positions and clusters that ship are always the decoder ones. A
    low score is recorded as a low score.
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score

    have = [i for i, key in enumerate(keys) if key in embeddings]
    if len(have) < max(4, min_cluster_size):
        return {
            "cross_source_ari": None,
            "cross_source_sample": len(have),
            "cross_source_note": "too few explanation embeddings to compare",
        }

    rng = np.random.default_rng(seed)
    if len(have) > sample:
        have = sorted(rng.choice(have, size=sample, replace=False).tolist())

    matrix = np.stack([np.asarray(embeddings[keys[i]], dtype=np.float32) for i in have])
    n_components = min(DEFAULT_PCA_DIM, min(matrix.shape) - 1)
    if n_components >= 2:
        matrix = PCA(n_components=n_components, random_state=seed).fit_transform(matrix)

    size = max(2, min(min_cluster_size, max(2, len(matrix) // 2)))
    text_labels = HDBSCAN(min_cluster_size=size).fit_predict(matrix)

    return {
        "cross_source_ari": float(adjusted_rand_score(labels[have], text_labels)),
        "cross_source_sample": len(have),
        "cross_source_text_clusters": int(len({int(v) for v in text_labels.tolist()} - {-1})),
    }


def explainer_influence(labels: np.ndarray, keys: list, explainers: dict) -> dict:
    """How far a cluster assignment is predicted by which explainer wrote the
    members' labels.

    The standing hazard for a text-derived layout: Neuronpedia's export uses
    `gemini-2.5-flash-lite` on five of gemma's 26 layers and `gpt-4o-mini` on
    the other 21, and the two write in visibly different styles — so a layout
    over label text could separate features by writing style rather than by
    meaning. Adjusted mutual information because the two labellings have
    different cardinalities: 0 means the clustering carries no information
    about the explainer, 1 that it carries nothing else.

    Measured on the pilot at 0.026, which is what reversed this project's
    original refusal to lay features out by label text at all. Reported either
    way; never a gate.
    """
    from sklearn.metrics import adjusted_mutual_info_score

    clustered = labels != -1
    if clustered.sum() < 2:
        return {"explainer_ami": None, "explainer_ami_note": "nothing clustered"}

    who = np.array([explainers.get(keys[i], "?") for i in np.flatnonzero(clustered)])
    if len({*who.tolist()}) < 2:
        return {
            "explainer_ami": None,
            "explainer_ami_note": f"only one explainer present ({who[0] if len(who) else '?'})",
        }

    return {
        "explainer_ami": float(adjusted_mutual_info_score(who, labels[clustered])),
        "explainer_ami_sample": int(clustered.sum()),
    }


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def build(
    layers: list[int],
    source: str = DEFAULT_SOURCE,
    width: str = DEFAULT_WIDTH,
    seed: int = 0,
    pca_dim: int = DEFAULT_PCA_DIM,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    fill: float = atlas.DEFAULT_FILL,
    knn_k: int = atlas.DEFAULT_KNN_K,
    knn_sample: int = atlas.DEFAULT_KNN_SAMPLE,
    coherence_margin: float = atlas.DEFAULT_COHERENCE_MARGIN,
    cross_sample: int = DEFAULT_CROSS_SAMPLE,
    subsample: int = DEFAULT_SUBSAMPLE,
    db_path: Path | str = DEFAULT_DB_PATH,
    device: str | None = None,
    sae_provider=None,
    store: LabelStore | None = None,
    subsample_out: Path | str | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """Build one atlas from one source, measure it, and write it beside the labels.

    Returns the summary that gets printed and recorded. `dry_run` measures and
    reports without writing, which is how the numbers are inspected before an
    atlas is committed to.

    `source` is recorded on the atlas rather than inferred from its positions,
    because the two sources answer different questions and are not
    interchangeable — see SOURCES.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")

    owned = store is None
    store = store or LabelStore(db_path, width=width)

    if source == "decoder":
        vectors, keys = load_decoder_directions(
            layers, width, device, sae_provider, verbose=verbose
        )
    else:
        vectors, keys = load_label_embeddings(layers, width, store=store, verbose=verbose)

    projection = project(vectors, pca_dim, n_neighbors, min_dist, seed, verbose=verbose)

    positions = atlas.place_in_shell(projection, fill=fill)
    atlas.assert_finite(positions)
    if not atlas.within_shell(positions, fill=fill):
        raise ValueError("placed a node outside the shell's bounds")

    metrics = atlas.knn_preservation(vectors, positions, k=knn_k, sample=knn_sample, seed=seed)
    metrics["source"] = source
    metrics["source_dim"] = int(vectors.shape[1])

    cluster_labels = cluster(positions, min_cluster_size, verbose=verbose)
    metrics.update(atlas.summarise_clusters(cluster_labels))

    try:
        embeddings, texts, explainers = load_label_side_tables(store, keys, verbose=verbose)

        clusters = name_clusters(
            cluster_labels,
            positions,
            keys,
            embeddings,
            texts,
            explainers,
            margin=coherence_margin,
            seed=seed,
        )
        metrics.update(
            cross_source_agreement(
                cluster_labels, keys, embeddings, seed, cross_sample, min_cluster_size
            )
        )
        # The standing hazard for a text-derived layout, measured either way so
        # the decoder atlas has a comparable figure to be read against.
        metrics.update(explainer_influence(cluster_labels, keys, explainers))

        named = [c for c in clusters if c.name is not None]
        measured = [c for c in clusters if c.coherence is not None]
        metrics["n_named_clusters"] = len(named)
        metrics["n_unnamed_clusters"] = len(clusters) - len(named)
        metrics["naming_attempted"] = bool(embeddings)
        if measured:
            metrics["cluster_coherence_mean"] = float(
                np.mean([c.coherence for c in measured])
            )
            baselines = [c.baseline_coherence for c in measured if c.baseline_coherence is not None]
            if baselines:
                metrics["cluster_coherence_baseline_mean"] = float(np.mean(baselines))

        params = {
            "source": source,
            "pca_dim": pca_dim,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": "cosine",
            "min_cluster_size": min_cluster_size,
            "fill": fill,
            "coherence_margin": coherence_margin,
            "layers": ",".join(str(layer) for layer in layers),
            "shell_radius": atlas.SHELL_RADIUS,
        }
        version = atlas.atlas_version(RELEASE, width, seed, params)
        digest = atlas.positions_hash(positions)

        record = AtlasRecord(
            atlas_version=version,
            release=RELEASE,
            width=width,
            seed=seed,
            params=params,
            metrics=metrics,
            positions_sha256=digest,
            n_features=len(keys),
            n_clusters=int(metrics["n_clusters"]),
        )

        if not dry_run:
            store.put_layout(
                version,
                [
                    LayoutRow(
                        layer=layer,
                        feature=feature,
                        x=float(positions[i, 0]),
                        y=float(positions[i, 1]),
                        z=float(positions[i, 2]),
                        cluster=int(cluster_labels[i]),
                    )
                    for i, (layer, feature) in enumerate(keys)
                ],
            )
            store.put_clusters(version, clusters)
            store.put_atlas_record(record)
    finally:
        if owned:
            store.close()

    subsample_info = None
    if subsample_out is not None and not dry_run:
        subsample_info = write_subsample(
            Path(subsample_out),
            version,
            keys,
            positions,
            cluster_labels,
            clusters,
            subsample,
            seed,
            source=source,
            metrics=metrics,
        )

    summary = {
        "atlas_version": version,
        "positions_sha256": digest,
        "n_features": len(keys),
        "metrics": metrics,
        "params": params,
        "dry_run": dry_run,
        "subsample": subsample_info,
    }
    if verbose:
        report(summary, clusters)
    return summary


def write_subsample(
    path: Path,
    version: str,
    keys: list[tuple[int, int]],
    positions: np.ndarray,
    cluster_labels: np.ndarray,
    clusters: list[ClusterRow],
    size: int,
    seed: int,
    source: str = DEFAULT_SOURCE,
    metrics: dict | None = None,
) -> dict:
    """The asset the idle brain renders: a uniform sample, labelled as one.

    Quantised to int16 over the cloud's own extent — the error is far below a
    pixel and it halves the bytes. Explicitly a *sample*: it exists so the
    brain has structure before a prompt is sent, and it is never the set a
    trace's activations are read against.
    """
    picks = atlas.subsample_indices(len(keys), size, seed=seed)
    sampled = positions[picks]
    quantised, extent = atlas.quantise_positions(sampled)

    metrics = metrics or {}
    payload = {
        "atlas_version": version,
        # The source is in the asset, not only in the DB, because the view has
        # to state what "near" means and the two sources mean different things.
        "source": source,
        # Carried so the legend can publish the layout's own fidelity rather
        # than asking the reader to take the picture on trust.
        "knn_preservation": metrics.get("knn_preservation"),
        "knn_k": metrics.get("knn_k"),
        "explainer_ami": metrics.get("explainer_ami"),
        "note": (
            "A uniform sample of the atlas, for rendering the brain before a "
            "trace exists. Not the set a trace's features are placed against."
        ),
        "extent": extent,
        "quantisation": "int16, position = value / 32767 * extent",
        "n_sampled": int(len(picks)),
        "n_total": len(keys),
        "max_quantisation_error": atlas.relative_error(sampled, quantised, extent),
        "nodes": {
            "layer": [int(keys[i][0]) for i in picks],
            "feature": [int(keys[i][1]) for i in picks],
            "cluster": [int(cluster_labels[i]) for i in picks],
            "xyz": quantised.reshape(-1).tolist(),
        },
        "areas": [
            {
                "cluster": c.cluster,
                "name": c.name,
                # The medoid whose label became the name, so the view can show
                # which member is speaking for the cluster.
                "name_source": list(c.name_source) if c.name_source else None,
                "n_members": c.n_members,
                "centroid": list(c.centroid),
                "spread": c.spread,
                "coherence": c.coherence,
                "baseline_coherence": c.baseline_coherence,
                "explainers": c.explainers,
            }
            for c in clusters
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "n_sampled": int(len(picks)),
        "max_quantisation_error": payload["max_quantisation_error"],
    }


def rewrite_subsample(
    path: Path,
    db_path: str,
    atlas_version: str | None = None,
    source: str = DEFAULT_SOURCE,
    size: int = DEFAULT_SUBSAMPLE,
    seed: int = 0,
) -> dict:
    """Re-emit the idle asset from an atlas already in the DB.

    The asset is derived output — positions, clusters and metrics all live in
    the label store once a build has run — so regenerating it must not cost a
    rebuild. UMAP is the expensive, stochastic step and rerunning it to change
    a serialised field would risk the one property the atlas is required to
    have: that a feature is placed identically everywhere it is drawn.

    Same sample as the build would draw, because `subsample_indices` is seeded
    and the row order is the layout table's own.
    """
    with LabelStore(db_path) as store:
        record = store.atlas_record(atlas_version, source=None if atlas_version else source)
        if record is None:
            raise SystemExit(
                f"no atlas in {db_path}"
                + (f" for version {atlas_version}" if atlas_version else f" from source {source}")
            )
        version = record.atlas_version
        rows = store.layout_all(version)
        clusters = store.clusters(version)

    if not rows:
        raise SystemExit(f"atlas {version} has no layout rows")

    keys = [(r.layer, r.feature) for r in rows]
    # float64, not float32: the store round-trips doubles, and narrowing here
    # would shift the quantisation extent in the last decimals and re-emit an
    # asset whose nodes differ from the build's for no reason.
    positions = np.array([[r.x, r.y, r.z] for r in rows], dtype=np.float64)
    cluster_labels = np.array([r.cluster for r in rows], dtype=np.int32)

    return write_subsample(
        path,
        version,
        keys,
        positions,
        cluster_labels,
        clusters,
        size,
        seed,
        source=str(record.params.get("source", source)),
        metrics=record.metrics,
    )


def report(summary: dict, clusters: list[ClusterRow]) -> None:
    """Print the numbers that say whether this atlas is worth drawing on."""
    m = summary["metrics"]
    print()
    print(
        f"atlas {summary['atlas_version']}  ({summary['n_features']:,} features, "
        f"source={m.get('source')} {m.get('source_dim')}d)"
    )
    print(f"  sha256              {summary['positions_sha256'][:16]}…")
    print(
        f"  kNN preservation    {atlas.fmt_metric(m.get('knn_preservation'))} "
        f"(k={m.get('knn_k')}, {m.get('knn_sample'):,} sampled, "
        f"±{atlas.fmt_metric(m.get('knn_preservation_std'))})"
    )
    print(
        f"  clusters            {m.get('n_clusters')} "
        f"({m.get('n_named_clusters')} named, {m.get('n_unnamed_clusters')} unnamed)"
    )
    print(
        f"  unclustered         {m.get('n_noise'):,} "
        f"({atlas.percent(int(m.get('n_noise', 0)), summary['n_features'])})"
    )
    if m.get("naming_attempted"):
        coherence = m.get("cluster_coherence_mean")
        baseline = m.get("cluster_coherence_baseline_mean")
        gap = None if coherence is None or baseline is None else coherence - baseline
        print(
            f"  cluster coherence   {atlas.fmt_metric(coherence)} "
            f"vs baseline {atlas.fmt_metric(baseline)} "
            f"(margin {atlas.fmt_metric(gap)}, need {summary['params']['coherence_margin']:.3f})"
        )
    else:
        print("  cluster coherence   not measured — no explanation embeddings in the label DB")
        print("                      (import them with import_neuronpedia.py --embeddings)")
    print(
        f"  cross-source ARI    {atlas.fmt_metric(m.get('cross_source_ari'))} "
        f"({m.get('cross_source_sample', 0):,} features)"
    )
    print(
        f"  explainer influence {atlas.fmt_metric(m.get('explainer_ami'))} "
        f"(0 = areas are meaning, 1 = areas are the labeller)"
        + (f" — {m['explainer_ami_note']}" if m.get("explainer_ami_note") else "")
    )

    named = [c for c in clusters if c.name][:10]
    if named:
        print("\n  largest named areas:")
        for c in sorted(named, key=lambda c: -c.n_members):
            gap = (
                None
                if c.coherence is None or c.baseline_coherence is None
                else c.coherence - c.baseline_coherence
            )
            print(
                f"    {c.n_members:>6,}  coh {atlas.fmt_metric(c.coherence)} "
                f"(+{atlas.fmt_metric(gap)})  "
                f"L{c.name_source[0]}#{c.name_source[1]}  {c.name[:52]}"
            )

    if summary.get("subsample"):
        s = summary["subsample"]
        print(
            f"\n  idle subsample      {s['n_sampled']:,} nodes, "
            f"{atlas.sizeof_fmt(s['bytes'])} -> {s['path']}"
        )
    if summary["dry_run"]:
        print("\n  dry run — nothing written")


def parse_layers(spec: str, default_n: int = 26) -> list[int]:
    """"0-25", "0,4,8", "20" -> a sorted list of distinct layers."""
    if not spec:
        return list(range(default_n))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"no layers in {spec!r}")
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", default="", help='e.g. "0-25" or "0,4,8" (default: all 26)')
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        choices=SOURCES,
        help="what to project: explanation embeddings (default) or SAE decoder directions",
    )
    parser.add_argument("--width", default=DEFAULT_WIDTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM)
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_N_NEIGHBORS)
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--fill", type=float, default=atlas.DEFAULT_FILL)
    parser.add_argument("--knn-k", type=int, default=atlas.DEFAULT_KNN_K)
    parser.add_argument("--knn-sample", type=int, default=atlas.DEFAULT_KNN_SAMPLE)
    parser.add_argument(
        "--coherence-margin",
        type=float,
        default=atlas.DEFAULT_COHERENCE_MARGIN,
        help="how far a cluster's label coherence must beat its own random baseline",
    )
    parser.add_argument("--cross-sample", type=int, default=DEFAULT_CROSS_SAMPLE)
    parser.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--subsample-out",
        default=str(Path(__file__).resolve().parents[2] / "frontend" / "public" / "atlas-idle.json"),
    )
    parser.add_argument("--dry-run", action="store_true", help="measure and print, write nothing")
    parser.add_argument(
        "--asset-only",
        action="store_true",
        help="re-emit the idle asset from the atlas already in the DB, without rebuilding",
    )
    parser.add_argument(
        "--atlas-version", default="", help="with --asset-only: which atlas (default: latest)"
    )
    args = parser.parse_args()

    if args.asset_only:
        info = rewrite_subsample(
            Path(args.subsample_out),
            db_path=args.db,
            atlas_version=args.atlas_version or None,
            source=args.source,
            size=args.subsample,
            seed=args.seed,
        )
        print(
            f"rewrote {info['path']} — {info['n_sampled']:,} nodes, "
            f"{info['bytes'] / 1024:.0f}KB, no rebuild"
        )
        return

    layers = parse_layers(args.layers)
    note = (
        f"the first run downloads ~{0.302 * len(layers):.1f}GB of SAEs"
        if args.source == "decoder"
        else "reading explanation embeddings from the label DB"
    )
    print(
        f"building the {args.source} atlas over {len(layers)} layers at "
        f"{args.width} (seed {args.seed}); {note}"
    )

    build(
        layers=layers,
        source=args.source,
        width=args.width,
        seed=args.seed,
        pca_dim=args.pca_dim,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_cluster_size=args.min_cluster_size,
        fill=args.fill,
        knn_k=args.knn_k,
        knn_sample=args.knn_sample,
        coherence_margin=args.coherence_margin,
        cross_sample=args.cross_sample,
        subsample=args.subsample,
        db_path=args.db,
        device=args.device,
        subsample_out=None if args.dry_run else args.subsample_out,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
