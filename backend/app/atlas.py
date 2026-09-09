"""The feature atlas: one fixed position per SAE feature, and its diagnostics.

A trace tells you which features fired. It does not tell you where to *draw*
them, and a position has to come from somewhere that means something. Here it
comes from the features' own decoder directions: every layer's SAE decodes into
the same residual-stream basis, so `W_dec` rows from all 26 layers live in one
comparable vector space, and features that write in similar directions land
near each other.

What this module does *not* do is load SAEs or run UMAP — that is
`scripts/build_feature_atlas.py`, which is the only thing in the project that
imports a projection library. Everything here is the part worth testing on
twelve hand-written vectors instead of 425,984 real ones: the shell warp, the
measurements, the naming gate, and the hashing.

Three things are load-bearing and easy to get wrong:

**2304 -> 3 is a brutal reduction, so how much survived is measured.**
`knn_preservation` is this module's peer to the SAE pass's
`explained_variance` and the lens's `final_layer_agreement`. If it comes back
at 0.31, 0.31 is what gets recorded. A layout nobody measured is decoration.

**Positions come from the model; names come from the labels.** Never the other
way round. Neuronpedia's export uses `gemini-2.5-flash-lite` on layers 16, 18,
20, 22 and 24 and `gpt-4o-mini` on the other 21, so a layout built from label
text would show those five layers as their own region for a reason that has
nothing to do with gemma — see `labels.py`'s module docstring, which called
this out before there was an atlas to get it wrong. Label embeddings are used
to *name* and to *check* clusters, where that split is visible and recorded
rather than baked into the geometry.

**A name has to be earned.** A cluster of decoder-similar features is not
guaranteed to be label-coherent. So coherence is measured against a shuffled
baseline of the same size, and a cluster below threshold stays unnamed with its
measured coherence recorded. A synthesised summary over an incoherent cluster
is exactly the kind of plausible invention the rest of this codebase refuses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# The rendered shell's radius, and the shaping applied to it. Both must match
# frontend/src/components/Brain.tsx: the atlas's positions are consumed in that
# mesh's own coordinate space, so a divergence here puts nodes outside the
# brain.
SHELL_RADIUS = 1.4

# How much of the shell's radius the node cloud is allowed to fill. The mesh
# displaces its own vertices inward and outward by up to ~0.14 for gyri and the
# longitudinal fissure, so a cloud filling the full radius would poke through
# the surface in the grooves.
DEFAULT_FILL = 0.9

# Neighbourhood size for the preservation measurement. 20 is small enough to be
# about local structure -- which is the only thing UMAP claims to preserve --
# and large enough not to be noise.
DEFAULT_KNN_K = 20
DEFAULT_KNN_SAMPLE = 4000

# How far a cluster's label coherence must exceed its own random baseline
# before it earns a name.
#
# A *margin over the baseline*, not an absolute coherence, and that distinction
# was learned from real data. The first calibration gated on an absolute 0.25;
# the pilot build over 98,304 features measured a mean cluster coherence of
# 0.281 against a random baseline of 0.232 and named 14 clusters that beat
# chance by 0.049. The names read plausibly, which is what made it dangerous.
# Explanation embeddings from one explainer share a similarity floor well above
# zero, so an absolute threshold measures that floor rather than any agreement.
DEFAULT_COHERENCE_MARGIN = 0.15


# --------------------------------------------------------------------------
# geometry: UMAP's arbitrary blob -> the rendered brain's volume
# --------------------------------------------------------------------------


def shape_ellipsoid(points: np.ndarray) -> np.ndarray:
    """Port of `shapeEllipsoid` in Brain.tsx, vectorised over [N, 3].

    Same operations in the same order, because the two have to agree: the mesh
    shapes its surface with this and the atlas places nodes inside it with
    this, so any divergence shows up as nodes outside the brain.

    The TypeScript guards two of these blocks on `v.z > 0.15` / `v.z < -0.1`.
    Both guards are redundant and are dropped here: each block's strength is
    driven by a value clamped to [0, 1] that is exactly 0 outside the guarded
    range, so the factors collapse to 1 and the block is a no-op. Verified
    against the original in test_atlas.py rather than argued.

    Input is expected in the mesh's own scale (radius up to SHELL_RADIUS).
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected [N, 3], got {p.shape}")

    x = p[:, 0] * 0.86
    y = p[:, 1] * 0.66
    z = p[:, 2] * 1.18

    # Frontal lobe: the anterior pole is narrower and rounder than the back, so
    # width and height pinch in as z increases.
    t = np.clip((z - 0.15) / 0.85, 0.0, 1.0)
    x *= 1 - t * t * 0.36
    y *= 1 - t * t * 0.14

    # Occipital lobe: fuller through the back, then pinching at the pole.
    t = np.clip((-z - 0.1) / 0.9, 0.0, 1.0)
    x *= 1 + np.sin(t * np.pi) * 0.09 * (1 - t)
    pinch = 1 - t**4 * 0.22
    x *= pinch
    y *= pinch

    # Temporal lobes: bulge out and droop down along the lower sides. Reads the
    # already-scaled y, as the original does.
    temporal = np.exp(-(((y + 0.16) / 0.26) ** 2)) * np.exp(-((z / 0.65) ** 2))
    x *= 1 + temporal * 0.24
    y -= temporal * 0.15

    # Twin-hemisphere crown: each hemisphere domes up and away from the midline.
    crown = np.exp(-(((y - 0.32) / 0.34) ** 2))
    lift = np.exp(-(((np.abs(x) - 0.3) / 0.22) ** 2))
    y += crown * lift * 0.06

    # Flatten the underside.
    y = np.where(y < -0.3, -0.3 + (y + 0.3) * 0.45, y)

    return np.stack([x, y, z], axis=1)


def to_unit_ball(points: np.ndarray) -> np.ndarray:
    """Centre a projection on its own median and scale it into the unit ball.

    Median rather than mean, and the 99.5th percentile radius rather than the
    maximum, because UMAP output routinely strands a handful of points far from
    everything else. Scaling by the true maximum would shrink the entire cloud
    to accommodate them. The stragglers are clipped to the surface instead,
    which is honest for a layout that never claimed its distances were metric.
    """
    p = np.asarray(points, dtype=np.float64)
    centred = p - np.median(p, axis=0)
    radii = np.linalg.norm(centred, axis=1)

    scale = float(np.percentile(radii, 99.5)) if len(radii) else 0.0
    if scale <= 0:
        # Every point is at the centre: a degenerate projection, not an error.
        return np.zeros_like(centred)

    scaled = centred / scale
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    over = norms[:, 0] > 1.0
    scaled[over] /= norms[over]
    return scaled


def place_in_shell(projection: np.ndarray, fill: float = DEFAULT_FILL) -> np.ndarray:
    """A 3D projection -> positions inside the rendered brain.

    The whole geometric pipeline: normalise into the unit ball, scale to the
    portion of the shell radius the cloud is allowed to occupy, then warp
    through the same shaping the mesh uses.
    """
    return shape_ellipsoid(to_unit_ball(projection) * SHELL_RADIUS * fill)


# Slack on the bound below, in shell units. The x and y extents come from a
# grid search over a smooth function, so they are close but not exact; the grid
# spacing is ~2e-3 and the residual error measured against a finer grid is well
# under 1e-3. Sized to cover that and nothing more — a generous slack would
# turn the containment check into a check of nothing.
BOUND_SLACK = 2e-3

# Grid resolution for the bound. 900 puts ~636k points in the disc and, times
# the |x| fractions below, ~20M evaluations — a second or so, and cached.
_BOUND_GRID = 900
_BOUND_X_FRACTIONS = (1.0, 0.97, 0.93, 0.88, 0.82, 0.75, 0.66, 0.55, 0.42, 0.28, 0.14, 0.0)


def _bounds_uncached(fill: float) -> tuple[tuple, tuple]:
    radius = SHELL_RADIUS * fill

    # z is exact and needs no search: the shaping's only z operation is the
    # opening `v.z *= 1.18`, so z_final = 1.18 * z0 and |z0| <= radius.
    z_extent = 1.18 * radius

    # x needs no interior search either. Reading the shaping through, x_final
    # is 0.86 * x0 times a factor that depends only on y0 and z0 — the frontal
    # taper, the occipital pinch and the temporal bulge are all functions of y
    # and z, and the crown and floor blocks write to y alone. So for any
    # (y0, z0) the extreme x comes from the largest available |x0|, which is
    # the sphere. y *does* need the interior: its crown lift reads |x|, so the
    # extreme can sit at an intermediate radius. Hence the fractions.
    grid = np.linspace(-radius, radius, _BOUND_GRID)
    y0, z0 = np.meshgrid(grid, grid, indexing="ij")
    remainder = radius**2 - y0**2 - z0**2
    inside = remainder >= 0
    y0, z0, remainder = y0[inside], z0[inside], remainder[inside]
    x_max = np.sqrt(remainder)

    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for fraction in _BOUND_X_FRACTIONS:
        for sign in (1.0, -1.0):
            warped = shape_ellipsoid(
                np.stack([sign * x_max * fraction, y0, z0], axis=1)
            )
            low = np.minimum(low, warped.min(axis=0))
            high = np.maximum(high, warped.max(axis=0))

    low[2], high[2] = -z_extent, z_extent
    return tuple(low.tolist()), tuple(high.tolist())


_BOUNDS_CACHE: dict[float, tuple[tuple, tuple]] = {}


def shell_bounds(fill: float = DEFAULT_FILL) -> tuple[np.ndarray, np.ndarray]:
    """The axis-aligned box `place_in_shell` can put a point in.

    Found by grid search over the *solid* ball rather than its surface. Nodes
    fill the volume and the shaping is not monotonic in radius, so an interior
    point can land further out along an axis than any surface point — which is
    not hypothetical: an earlier surface-only estimate under-reported the x
    extent by 7e-3 and the first real build tripped over it.

    Note what actually guarantees a node is inside the brain, which is not this
    function. `place_in_shell` clips its input to a ball of radius
    `SHELL_RADIUS * fill` and applies one fixed map, so every output lies in
    the image of that ball — the brain, shrunk by `fill`, by construction. This
    box is a numerical net over that argument, and it is deterministic: a grid,
    not a sample, so it returns the same number in every process.
    """
    if fill not in _BOUNDS_CACHE:
        _BOUNDS_CACHE[fill] = _bounds_uncached(fill)
    low, high = _BOUNDS_CACHE[fill]
    return np.array(low), np.array(high)


# --------------------------------------------------------------------------
# the measurement that says how much to trust the layout
# --------------------------------------------------------------------------


def _cosine_neighbours(matrix: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest rows of `matrix` to each query row, by cosine.

    Cosine because these are directions in the residual stream: `W_dec` rows
    are read for where they point, and their norms carry a different fact.
    """
    import torch

    m = torch.as_tensor(matrix, dtype=torch.float32)
    m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q = torch.as_tensor(queries, dtype=torch.float32)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)

    if torch.cuda.is_available():
        m, q = m.cuda(), q.cuda()

    out = np.empty((q.shape[0], k), dtype=np.int64)
    # Chunked over queries: the full similarity matrix for a real atlas would
    # be 4000 x 425,984 floats, which is fine, but the chunking keeps this
    # bounded when the sample is raised.
    step = max(1, 8_000_000 // max(m.shape[0], 1))
    for start in range(0, q.shape[0], step):
        stop = min(start + step, q.shape[0])
        sim = q[start:stop] @ m.T
        out[start:stop] = sim.topk(k, dim=1).indices.cpu().numpy()
    return out


def _euclidean_neighbours(matrix: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest rows by Euclidean distance.

    Euclidean in the layout, because that is the distance a viewer's eye
    actually reads off the screen — regardless of the metric the projection was
    optimised under.
    """
    import torch

    m = torch.as_tensor(matrix, dtype=torch.float32)
    q = torch.as_tensor(queries, dtype=torch.float32)
    if torch.cuda.is_available():
        m, q = m.cuda(), q.cuda()

    out = np.empty((q.shape[0], k), dtype=np.int64)
    step = max(1, 8_000_000 // max(m.shape[0], 1))
    for start in range(0, q.shape[0], step):
        stop = min(start + step, q.shape[0])
        d = torch.cdist(q[start:stop], m)
        out[start:stop] = (-d).topk(k, dim=1).indices.cpu().numpy()
    return out


def knn_preservation(
    high: np.ndarray,
    low: np.ndarray,
    k: int = DEFAULT_KNN_K,
    sample: int = DEFAULT_KNN_SAMPLE,
    seed: int = 0,
) -> dict:
    """How much of the original neighbourhood structure the layout kept.

    For a random sample of features, take the `k` nearest in the full decoder
    space and the `k` nearest in the 3D layout, and report the mean fraction
    that appear in both. 1.0 would mean the projection preserved every local
    neighbourhood exactly; 0.0 that it kept none of them.

    This is the atlas's headline number and it is reported as measured. Three
    dimensions out of 2304 will lose a great deal — the point is that the
    amount is a measurement and not an assumption.
    """
    high = np.asarray(high)
    low = np.asarray(low)
    if len(high) != len(low):
        raise ValueError(f"{len(high)} high-dim rows against {len(low)} low-dim rows")

    n = len(high)
    if n <= 1:
        return {
            "knn_preservation": 0.0,
            "knn_k": 0,
            "knn_sample": 0,
            "knn_high_metric": "cosine",
            "knn_low_metric": "euclidean",
        }

    # k+1 fetched, self dropped: a point is always its own nearest neighbour.
    k_eff = min(k, n - 1)
    rng = np.random.default_rng(seed)
    n_sample = min(sample, n)
    picks = rng.choice(n, size=n_sample, replace=False)

    high_nn = _cosine_neighbours(high, high[picks], k_eff + 1)
    low_nn = _euclidean_neighbours(low, low[picks], k_eff + 1)

    overlaps = np.empty(n_sample, dtype=np.float64)
    for row, index in enumerate(picks):
        a = set(high_nn[row].tolist()) - {int(index)}
        b = set(low_nn[row].tolist()) - {int(index)}
        # Trim any k+1'th survivor so both sets are the same size.
        overlaps[row] = len(set(list(a)[:k_eff]) & set(list(b)[:k_eff])) / k_eff

    return {
        "knn_preservation": float(overlaps.mean()),
        "knn_preservation_std": float(overlaps.std()),
        "knn_k": int(k_eff),
        "knn_sample": int(n_sample),
        "knn_high_metric": "cosine",
        "knn_low_metric": "euclidean",
    }


# --------------------------------------------------------------------------
# clusters: coherence, naming, and what the members were explained by
# --------------------------------------------------------------------------


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    """Mean cosine over all distinct pairs, in O(n) rather than O(n^2).

    For unit vectors, sum over ordered pairs i != j of u_i . u_j is
    ||sum u||^2 - n, so the mean needs one sum and one norm. Exact, not an
    approximation — which matters because this is compared against a baseline
    and a threshold.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if len(v) < 2:
        return 1.0 if len(v) == 1 else 0.0
    unit = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    total = unit.sum(axis=0)
    n = len(unit)
    return float((total @ total - n) / (n * (n - 1)))


@dataclass(frozen=True)
class Coherence:
    """A cluster's label agreement, and what that agreement is worth.

    `value` alone says nothing: a set of embeddings from one model has a
    baseline similarity well above zero, so a raw 0.3 could be excellent or
    could be chance. `baseline` is the same statistic over a random group of
    the same size, which is what makes `value` readable.
    """

    value: float | None
    baseline: float | None
    n_measured: int

    @property
    def measured(self) -> bool:
        return self.value is not None

    @property
    def margin(self) -> float | None:
        """How far the group beat a random group of its own size, or None.

        None whenever either half is missing — there is no such thing as a
        margin over a baseline that was never measured, and treating one as
        zero would silently readmit the absolute test this replaced.
        """
        if self.value is None or self.baseline is None:
            return None
        return self.value - self.baseline

    def earns_a_name(self, margin: float = DEFAULT_COHERENCE_MARGIN) -> bool:
        """Whether this group's labels agree enough to name it after one of them."""
        actual = self.margin
        return actual is not None and actual >= margin


def measure_coherence(
    member_embeddings: list[np.ndarray],
    pool_embeddings: list[np.ndarray],
    seed: int = 0,
) -> Coherence:
    """Mean pairwise cosine among members, against a same-size random group.

    Returns an unmeasured `Coherence` when there are no embeddings, which is a
    different fact from a low one: it means naming was never attempted, not
    that it was attempted and failed. Both end in an unnamed cluster and a
    consumer has to be able to tell them apart.
    """
    if len(member_embeddings) < 2:
        return Coherence(value=None, baseline=None, n_measured=len(member_embeddings))

    value = mean_pairwise_cosine(np.stack(member_embeddings))

    baseline = None
    if len(pool_embeddings) >= len(member_embeddings):
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(pool_embeddings), size=len(member_embeddings), replace=False)
        baseline = mean_pairwise_cosine(np.stack([pool_embeddings[i] for i in picks]))

    return Coherence(value=value, baseline=baseline, n_measured=len(member_embeddings))


def medoid(keys: list, embeddings: dict) -> object | None:
    """The member whose label embedding is closest to the group's mean direction.

    A real member, not a synthetic centroid — so a cluster's name is something
    a specific feature actually says, and the interface can show which one.
    """
    present = [key for key in keys if key in embeddings]
    if not present:
        return None

    stacked = np.stack([np.asarray(embeddings[key], dtype=np.float64) for key in present])
    unit = stacked / np.clip(np.linalg.norm(stacked, axis=1, keepdims=True), 1e-12, None)
    centre = unit.mean(axis=0)
    centre = centre / max(float(np.linalg.norm(centre)), 1e-12)
    return present[int(np.argmax(unit @ centre))]


def explainer_mix(keys: list, explainers: dict) -> str:
    """"gpt-4o-mini:180, gemini-2.5-flash-lite:34", commonest first.

    Recorded per cluster because a cluster whose labels come predominantly from
    one explainer, in a model that uses several, may be a cluster of writing
    style rather than of meaning. Formatted the way LabelsPass records its own
    mix, for the same reason: `params` values are scalars.
    """
    counts: dict[str, int] = {}
    for key in keys:
        name = explainers.get(key) or "?"
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def cluster_geometry(positions: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """A cluster's centroid and how far its members spread from it.

    `spread` is the mean distance to the centroid, which is what an area marker
    is sized by. Deliberately not presented as a radius: the layout's distances
    are not metric, so this sizes a glow and measures nothing.
    """
    p = np.asarray(positions, dtype=np.float64)
    if len(p) == 0:
        return (0.0, 0.0, 0.0), 0.0
    centre = p.mean(axis=0)
    spread = float(np.linalg.norm(p - centre, axis=1).mean())
    return (float(centre[0]), float(centre[1]), float(centre[2])), spread


# --------------------------------------------------------------------------
# identity: a version that changes when the layout does
# --------------------------------------------------------------------------


def positions_hash(positions: np.ndarray) -> str:
    """A content hash over the positions, for the determinism check.

    Rounded to 6 decimals before hashing. Without that, an identical rebuild
    could differ in the last bits of a float and report itself as a different
    atlas — a false alarm about the one property this hash exists to confirm.
    """
    rounded = np.round(np.asarray(positions, dtype=np.float64), 6)
    return hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()


def atlas_version(release: str, width: str, seed: int, params: dict) -> str:
    """A stable id for one (release, width, seed, parameters) combination.

    Everything that changes the layout is in the digest, so a rebuild under
    different parameters cannot overwrite the previous atlas under its
    identity — two atlases coexist and one can be compared against the other.
    """
    digest = hashlib.sha256(
        json.dumps(
            {"release": release, "width": width, "seed": seed, "params": params},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    return f"{width}-s{seed}-{digest[:10]}"


# --------------------------------------------------------------------------
# the subsample the idle brain renders
# --------------------------------------------------------------------------


def subsample_indices(n: int, size: int, seed: int = 0) -> np.ndarray:
    """A uniform sample of node indices, deterministic under `seed`.

    The brain has to have structure before a prompt is sent, and shipping all
    425,984 positions to place the ~6,000 a trace actually reports would be
    seventy times more bytes than the job needs. So the idle view gets a
    uniform sample, drawn once at build time and labelled as a sample.
    """
    if size >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def quantise_positions(positions: np.ndarray, extent: float | None = None) -> tuple[np.ndarray, float]:
    """Positions -> int16, plus the scale needed to read them back.

    The idle asset is thousands of coordinates whose only job is to be a dim
    dot in a 3D scene; int16 over the cloud's own extent is ~0.00005 of the
    shell radius per step, far finer than a pixel, at half the bytes of
    float32.
    """
    p = np.asarray(positions, dtype=np.float64)
    if extent is None:
        extent = float(np.abs(p).max()) if p.size else 1.0
    extent = max(extent, 1e-9)
    scaled = np.clip(np.round(p / extent * 32767.0), -32767, 32767)
    return scaled.astype(np.int16), extent


def relative_error(original: np.ndarray, quantised: np.ndarray, extent: float) -> float:
    """Worst per-coordinate error introduced by `quantise_positions`, in shell units."""
    restored = np.asarray(quantised, dtype=np.float64) / 32767.0 * extent
    if restored.size == 0:
        return 0.0
    return float(np.abs(restored - np.asarray(original, dtype=np.float64)).max())


def sizeof_fmt(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes}B"
    for unit, limit in (("KB", 1024**2), ("MB", 1024**3)):
        if n_bytes < limit:
            return f"{n_bytes / (limit / 1024):.1f}{unit}"
    return f"{n_bytes / 1024**3:.2f}GB"


def fmt_metric(value: float | None) -> str:
    """A measurement, or an explicit statement that there is none."""
    return "not measured" if value is None else f"{value:.3f}"


def assert_finite(positions: np.ndarray) -> None:
    """A NaN in a position is silent in a renderer and fatal to the view."""
    p = np.asarray(positions)
    if not np.isfinite(p).all():
        bad = int((~np.isfinite(p)).any(axis=1).sum())
        raise ValueError(f"{bad} of {len(p)} positions are not finite")


def within_shell(
    positions: np.ndarray, fill: float = DEFAULT_FILL, slack: float = BOUND_SLACK
) -> bool:
    """Whether every position is inside the box `place_in_shell` can produce.

    `slack` defaults to `BOUND_SLACK` because the bound is sampled; see
    `shell_bounds` for what actually guarantees containment.
    """
    low, high = shell_bounds(fill)
    p = np.asarray(positions, dtype=np.float64)
    if p.size == 0:
        return True
    return bool((p >= low - slack).all() and (p <= high + slack).all())


def summarise_clusters(labels: np.ndarray) -> dict:
    """Cluster-count facts, with noise counted rather than swept up.

    HDBSCAN's -1 is a real answer — this feature belongs to no cluster — so the
    fraction of features that got one is part of the atlas's record, not a
    detail of the build.
    """
    labels = np.asarray(labels)
    unique = sorted({int(v) for v in labels.tolist()} - {-1})
    n_noise = int((labels == -1).sum())
    sizes = [int((labels == cluster).sum()) for cluster in unique]
    return {
        "n_clusters": len(unique),
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / len(labels)) if len(labels) else 0.0,
        "cluster_size_min": float(min(sizes)) if sizes else 0.0,
        "cluster_size_median": float(np.median(sizes)) if sizes else 0.0,
        "cluster_size_max": float(max(sizes)) if sizes else 0.0,
    }


def percent(part: int, whole: int) -> str:
    return "0%" if whole == 0 else f"{100.0 * part / whole:.1f}%"

