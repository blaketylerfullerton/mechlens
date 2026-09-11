"""Phase 3: what Neuronpedia says each SAE feature means.

A trace's top features are integers. This module turns them into text:

    from app.labels import LabelStore
    with LabelStore() as store:
        store.get(layer=20, feature=12082)
        # FeatureLabel(text='References to dogs as pets...', explainer='gpt-4o-mini')

The lookup table is SQLite, populated in bulk from Neuronpedia's S3 export
(scripts/import_neuronpedia.py). It is *not* per-trace: every gemma-2-2b/16k
trace draws on the same ~426k explanations, so the DB lives beside the code and
the trace JSON carries only the handful of labels it actually needs.

Three states, not two
---------------------
Coverage is not total — about 0.3% of features have no explanation at all. So a
row's absence and a row with no text mean different things:

    no row            never looked up
    row, text NULL    looked up, Neuronpedia has nothing
    row, text set     labelled

Collapsing those two loses the ability to tell "unlabelled" from "unvisited",
and the API fallback then re-requests every blank feature on every run — 1.2MB
a time, to re-learn nothing.

The explainer is not uniform
----------------------------
Neuronpedia's export carries exactly one explanation per feature, but not from
one explainer. For gemma-2-2b at 16k, layers 16, 18, 20, 22 and 24 were
re-explained with gemini-2.5-flash-lite (typeName np_acts-logits-general) while
the other 21 layers still carry gpt-4o-mini (oai_token-act-pair). The two
prompt styles produce systematically different descriptions, so any analysis
that pools label text across layers — clustering, embedding, a UMAP layout —
will see those five layers as their own region for reasons that have nothing to
do with the model. Hence `explainer` on every label: the split cannot be fixed
here, but it must stay visible.

Scores are not a selection criterion
------------------------------------
The export carries none, and sampling the live API found one score across 21
explanations. Anything ranking labels by score is ranking by NULL. The
preference ladder below is only for the API fallback, where a feature can come
back with several explanations and one has to be chosen deterministically.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .sae_cache import DEFAULT_WIDTH, neuronpedia_id
from .schema import FeatureLabel

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "neuronpedia.db"

SITE = "https://www.neuronpedia.org"
API = SITE + "/api/feature/{model_id}/{source_set}/{feature}"
URL_TEMPLATE = SITE + "/{model_id}/{source_set}/{feature}"

# Only consulted when the live API returns several explanations for one
# feature. Ordered so the choice is deterministic and reproducible rather than
# "whatever the API listed first", which is not stable.
EXPLAINER_PREFERENCE: tuple[tuple[str, str], ...] = (
    ("gpt-4o-mini", "oai_token-act-pair"),  # the export's majority explainer
    ("gemini-2.5-flash-lite", "np_acts-logits-general"),  # layers 16/18/20/22/24
    ("gemini-2.0-flash", "np_acts-logits-general"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    source_set       TEXT    NOT NULL,
    feature          INTEGER NOT NULL,
    text             TEXT,             -- NULL: looked up, no explanation exists
    explainer        TEXT,
    explanation_type TEXT,
    score            REAL,
    embedding        BLOB,             -- float32 vector of the text, or NULL
    fetched_at       TEXT    NOT NULL,
    PRIMARY KEY (source_set, feature)
) WITHOUT ROWID;

-- The feature atlas: one position per (layer, feature), keyed by layer rather
-- than by source_set. A position comes from the SAE's own decoder direction,
-- which is a property of the release and the width -- not of Neuronpedia's
-- naming -- so `atlas_version` is what scopes a row, and several atlases can
-- sit side by side while one is compared against another.
CREATE TABLE IF NOT EXISTS atlas_layout (
    atlas_version TEXT    NOT NULL,
    layer         INTEGER NOT NULL,
    feature       INTEGER NOT NULL,
    x             REAL    NOT NULL,
    y             REAL    NOT NULL,
    z             REAL    NOT NULL,
    -- HDBSCAN's noise label (-1) is a real answer: this feature belongs to no
    -- cluster. Stored as -1 rather than NULL so "no cluster" and "not yet
    -- clustered" stay distinguishable.
    cluster       INTEGER NOT NULL DEFAULT -1,
    PRIMARY KEY (atlas_version, layer, feature)
) WITHOUT ROWID;

-- One row per atlas: how it was built, and every number that says how much to
-- trust it. `metrics_json` holds the measurements (kNN preservation and its
-- neighbourhood size, cross-source agreement, ...) so a new diagnostic does
-- not need a migration; `positions_sha256` is the determinism check.
CREATE TABLE IF NOT EXISTS atlas_record (
    atlas_version    TEXT NOT NULL PRIMARY KEY,
    release          TEXT NOT NULL,
    width            TEXT NOT NULL,
    seed             INTEGER NOT NULL,
    params_json      TEXT NOT NULL,
    metrics_json     TEXT NOT NULL,
    positions_sha256 TEXT NOT NULL,
    n_features       INTEGER NOT NULL,
    n_clusters       INTEGER NOT NULL,
    built_at         TEXT NOT NULL
) WITHOUT ROWID;

-- One row per cluster. A name is present only when the build measured its
-- members' label coherence above threshold; below it, `name` is NULL and
-- `coherence` still records what was measured, so an unnamed cluster is
-- unnamed for a stated reason rather than for no reason.
CREATE TABLE IF NOT EXISTS atlas_cluster (
    atlas_version     TEXT    NOT NULL,
    cluster           INTEGER NOT NULL,
    name              TEXT,             -- NULL: coherence below threshold
    name_source_layer INTEGER,          -- the medoid member the name came from
    name_source_feature INTEGER,
    coherence         REAL,             -- NULL: no embeddings to measure with
    baseline_coherence REAL,
    n_members         INTEGER NOT NULL,
    explainers        TEXT    NOT NULL DEFAULT '',
    centroid_x        REAL    NOT NULL,
    centroid_y        REAL    NOT NULL,
    centroid_z        REAL    NOT NULL,
    spread            REAL    NOT NULL,
    PRIMARY KEY (atlas_version, cluster)
) WITHOUT ROWID;
"""


def feature_url(layer: int, feature: int, width: str = DEFAULT_WIDTH) -> str:
    """The page a user clicks through to for raw activating examples."""
    model_id, source_set = neuronpedia_id(layer, width)
    return URL_TEMPLATE.format(model_id=model_id, source_set=source_set, feature=feature)


def source_set_template(n_layers: int = 26, width: str = DEFAULT_WIDTH) -> str | None:
    """"{layer}-gemmascope-res-16k", if every layer really follows that shape.

    Lets a trace record the whole per-layer mapping as one string instead of 26
    entries. Returns None rather than guessing if any layer breaks the pattern,
    because a frontend building URLs off a wrong template links every feature
    to the wrong page.
    """
    template: str | None = None
    for layer in range(n_layers):
        _, source_set = neuronpedia_id(layer, width)
        candidate = source_set.replace(str(layer), "{layer}", 1)
        if template is None:
            template = candidate
        elif template != candidate:
            return None
    return template


def url_template(layer: int = 0, width: str = DEFAULT_WIDTH) -> str:
    """The template itself, for a PassRecord to record once per trace.

    `{source_set}` still carries the layer, so a consumer needs the per-layer
    ids too — but recording the shape here means a frontend never hardcodes a
    neuronpedia.org URL of its own.
    """
    return URL_TEMPLATE


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelRow:
    """One (source_set, feature) row on its way into the DB.

    `text=None` is a real, storable answer: it records that we asked and
    Neuronpedia had nothing.
    """

    source_set: str
    feature: int
    text: str | None
    explainer: str | None = None
    explanation_type: str | None = None
    score: float | None = None
    embedding: np.ndarray | None = None

    def as_params(self) -> tuple:
        blob = None
        if self.embedding is not None:
            blob = np.asarray(self.embedding, dtype=np.float32).tobytes()
        return (
            self.source_set,
            self.feature,
            self.text,
            self.explainer,
            self.explanation_type,
            self.score,
            blob,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


@dataclass(frozen=True)
class LayoutRow:
    """One feature's place in an atlas."""

    layer: int
    feature: int
    x: float
    y: float
    z: float
    # -1 means "belongs to no cluster" — HDBSCAN's noise label, a real answer.
    cluster: int = -1


@dataclass(frozen=True)
class ClusterRow:
    """One atlas cluster, and whether its name was earned.

    `name is None` with a non-None `coherence` is the interesting case: the
    build measured the members' label agreement and it came in below the
    threshold, so the cluster is deliberately unnamed. `coherence is None`
    means there were no explanation embeddings to measure with at all. The two
    are different facts and a consumer has to be able to tell them apart.
    """

    cluster: int
    n_members: int
    centroid: tuple[float, float, float]
    spread: float
    name: str | None = None
    name_source: tuple[int, int] | None = None  # (layer, feature) of the medoid
    coherence: float | None = None
    baseline_coherence: float | None = None
    explainers: str = ""


@dataclass(frozen=True)
class AtlasRecord:
    """How an atlas was built, and how much of the original geometry survived.

    `metrics` is the honest headline: `knn_preservation` is this table's peer
    to the SAE pass's `explained_variance` and the lens's
    `final_layer_agreement` — 3 dimensions out of 2304 lose a great deal, and
    the point is that how much is measured rather than assumed.
    """

    atlas_version: str
    release: str
    width: str
    seed: int
    params: dict
    metrics: dict
    positions_sha256: str
    n_features: int
    n_clusters: int
    built_at: str = ""


def pick_explanation(explanations: list[dict]) -> dict | None:
    """Choose one of an API response's explanations, deterministically.

    Preference order first, then — among equally preferred ones — the highest
    score, then the oldest, which is the only remaining stable tiebreak. Score
    never crosses explainer boundaries: a `recall_alt` value from one explainer
    is not comparable to another type's, so ranking across them would be
    arithmetic on unlike units.
    """
    if not explanations:
        return None

    def rank(e: dict) -> tuple:
        pair = (e.get("explanationModelName"), e.get("typeName"))
        try:
            preference = EXPLAINER_PREFERENCE.index(pair)
        except ValueError:
            preference = len(EXPLAINER_PREFERENCE)  # unknown explainers go last
        scores = e.get("scores") or []
        best = max((s.get("value") or 0.0 for s in scores), default=0.0)
        return (preference, -best, e.get("createdAt") or "")

    return min(explanations, key=rank)


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


class LabelStore:
    """The SQLite lookup table, plus an optional live-API fallback.

    Offline by default. `fetch_missing=True` opts into hitting Neuronpedia for
    rows the DB has never seen — useful for a handful of features, ruinous for
    a whole trace, so `max_fetches` caps it rather than trusting the caller to
    notice.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_DB_PATH,
        width: str = DEFAULT_WIDTH,
        fetch_missing: bool = False,
        max_fetches: int = 200,
        timeout_s: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.fetch_missing = fetch_missing
        self.max_fetches = max_fetches
        self.timeout_s = timeout_s
        self.fetched = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> LabelStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- reading -----------------------------------------------------------

    def get(self, layer: int, feature: int) -> FeatureLabel | None:
        """One label, or None if unlabelled (whether or not we have looked)."""
        return self.get_many(layer, [feature]).get(feature)

    def get_many(self, layer: int, features: Iterable[int]) -> dict[int, FeatureLabel]:
        """Labels for many features of one layer, in one query.

        Only labelled features appear in the result — an unexplained feature is
        absent rather than present-and-empty, so callers can just do a dict
        lookup and get None.
        """
        wanted = list(dict.fromkeys(int(f) for f in features))  # dedupe, keep order
        if not wanted:
            return {}

        _, source_set = neuronpedia_id(layer, self.width)
        found: dict[int, FeatureLabel] = {}
        seen: set[int] = set()

        # Chunked: SQLite caps host parameters (999 on older builds) and a
        # trace can want thousands of features from a single layer.
        for chunk in _chunked(wanted, 500):
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT feature, text, explainer, explanation_type, score "
                f"FROM labels WHERE source_set = ? AND feature IN ({placeholders})",
                (source_set, *chunk),
            ).fetchall()
            for row in rows:
                seen.add(row["feature"])
                if row["text"]:
                    found[row["feature"]] = FeatureLabel(
                        text=row["text"],
                        explainer=row["explainer"],
                        explanation_type=row["explanation_type"],
                        score=row["score"],
                    )

        if self.fetch_missing:
            for feature in (f for f in wanted if f not in seen):
                label = self._fetch(layer, source_set, feature)
                if label is not None:
                    found[feature] = label

        return found

    def embeddings(self, layer: int, features: Iterable[int]) -> dict[int, np.ndarray]:
        """Stored explanation embeddings, for laying features out.

        Read the module docstring before pooling these across layers: the text
        they embed comes from two different explainers depending on the layer.
        """
        _, source_set = neuronpedia_id(layer, self.width)
        out: dict[int, np.ndarray] = {}
        for chunk in _chunked(list(features), 500):
            placeholders = ",".join("?" * len(chunk))
            for row in self.conn.execute(
                f"SELECT feature, embedding FROM labels "
                f"WHERE source_set = ? AND embedding IS NOT NULL "
                f"AND feature IN ({placeholders})",
                (source_set, *chunk),
            ):
                out[row["feature"]] = np.frombuffer(row["embedding"], dtype=np.float32)
        return out

    # -- writing -----------------------------------------------------------

    def upsert(self, rows: Iterable[LabelRow]) -> int:
        """Bulk insert-or-replace. The import path; also how fetches are cached."""
        params = [row.as_params() for row in rows]
        if not params:
            return 0
        self.conn.executemany(
            "INSERT INTO labels "
            "(source_set, feature, text, explainer, explanation_type, score, embedding, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_set, feature) DO UPDATE SET "
            "  text = excluded.text, explainer = excluded.explainer, "
            "  explanation_type = excluded.explanation_type, score = excluded.score, "
            "  embedding = CASE WHEN excluded.embedding IS NOT NULL THEN excluded.embedding "
            "    WHEN labels.text IS excluded.text AND labels.explainer IS excluded.explainer "
            "      AND labels.explanation_type IS excluded.explanation_type THEN labels.embedding "
            "    ELSE NULL END, "
            "  fetched_at = excluded.fetched_at",
            params,
        )
        self.conn.commit()
        return len(params)

    # -- the atlas ---------------------------------------------------------
    #
    # Kept on the label store rather than in a store of its own because the two
    # are looked up together on every request that draws a feature: a node
    # needs a position and the text that says what it is. One SQLite file, one
    # connection, one thing to build and one thing to delete.

    def publish_atlas(self, record: AtlasRecord, rows: list[LayoutRow], clusters: list[ClusterRow]) -> None:
        """Publish a complete immutable artifact, or leave the database unchanged."""
        from dataclasses import asdict

        if len(rows) != record.n_features or len(clusters) != record.n_clusters:
            raise ValueError("atlas counts do not match its rows")
        if len({(r.layer, r.feature) for r in rows}) != len(rows):
            raise ValueError("duplicate atlas feature identities")
        if len({c.cluster for c in clusters}) != len(clusters):
            raise ValueError("duplicate atlas clusters")
        rows = sorted(rows, key=lambda r: (r.layer, r.feature))
        clusters = sorted(clusters, key=lambda c: c.cluster)
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            old = self.atlas_record(record.atlas_version)
            if old is not None:
                before, after = asdict(old), asdict(record)
                before.pop("built_at")
                after.pop("built_at")
                if (before != after or self.layout_all(record.atlas_version) != rows
                        or self.clusters(record.atlas_version) != clusters):
                    raise ValueError("refusing to overwrite an existing atlas identity")
                return
            # Clean up orphan rows from older, interrupted non-atomic builds.
            for table in ("atlas_layout", "atlas_cluster"):
                self.conn.execute(f"DELETE FROM {table} WHERE atlas_version = ?", (record.atlas_version,))
            self.put_layout(record.atlas_version, rows, commit=False)
            self.put_clusters(record.atlas_version, clusters, commit=False)
            self.put_atlas_record(record, commit=False)

    def put_layout(self, atlas_version: str, rows: Iterable[LayoutRow], *, commit: bool = True) -> int:
        """Bulk insert-or-replace positions for one atlas."""
        params = [
            (atlas_version, r.layer, r.feature, r.x, r.y, r.z, r.cluster) for r in rows
        ]
        if not params:
            return 0
        self.conn.executemany(
            "INSERT INTO atlas_layout "
            "(atlas_version, layer, feature, x, y, z, cluster) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(atlas_version, layer, feature) DO UPDATE SET "
            "  x = excluded.x, y = excluded.y, z = excluded.z, "
            "  cluster = excluded.cluster",
            params,
        )
        if commit:
            self.conn.commit()
        return len(params)

    def layout(
        self, atlas_version: str, pairs: Iterable[tuple[int, int]]
    ) -> dict[tuple[int, int], LayoutRow]:
        """Positions for the (layer, feature) pairs asked for, and only those.

        A pair with no row is simply absent from the result. That is the whole
        point: a caller has to be able to tell a feature the atlas cannot place
        from one it places at the origin, and inventing a position for the
        former is exactly the failure this shape prevents.
        """
        out: dict[tuple[int, int], LayoutRow] = {}
        wanted = list(pairs)
        for chunk in _chunked(wanted, 400):
            placeholders = ",".join("(?, ?)" for _ in chunk)
            flat = [value for pair in chunk for value in pair]
            for row in self.conn.execute(
                f"SELECT layer, feature, x, y, z, cluster FROM atlas_layout "
                f"WHERE atlas_version = ? AND (layer, feature) IN ({placeholders})",
                (atlas_version, *flat),
            ):
                out[(row["layer"], row["feature"])] = LayoutRow(
                    layer=row["layer"],
                    feature=row["feature"],
                    x=row["x"],
                    y=row["y"],
                    z=row["z"],
                    cluster=row["cluster"],
                )
        return out

    def layout_all(self, atlas_version: str) -> list[LayoutRow]:
        """Every position in one atlas, ordered by (layer, feature).

        The order is part of the contract, not incidental: a client sampling
        this list has to get the same nodes on every request, and callers hash
        and subsample it positionally. Unlike `layout`, this is for serving a
        whole atlas rather than resolving the features one trace happens to
        report.
        """
        return [
            LayoutRow(
                layer=row["layer"],
                feature=row["feature"],
                x=row["x"],
                y=row["y"],
                z=row["z"],
                cluster=row["cluster"],
            )
            for row in self.conn.execute(
                "SELECT layer, feature, x, y, z, cluster FROM atlas_layout "
                "WHERE atlas_version = ? ORDER BY layer, feature",
                (atlas_version,),
            )
        ]

    def put_atlas_record(self, record: AtlasRecord, *, commit: bool = True) -> None:
        self.conn.execute(
            "INSERT INTO atlas_record "
            "(atlas_version, release, width, seed, params_json, metrics_json, "
            " positions_sha256, n_features, n_clusters, built_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(atlas_version) DO UPDATE SET "
            "  release = excluded.release, width = excluded.width, "
            "  seed = excluded.seed, params_json = excluded.params_json, "
            "  metrics_json = excluded.metrics_json, "
            "  positions_sha256 = excluded.positions_sha256, "
            "  n_features = excluded.n_features, n_clusters = excluded.n_clusters, "
            "  built_at = excluded.built_at",
            (
                record.atlas_version,
                record.release,
                record.width,
                record.seed,
                json.dumps(record.params, sort_keys=True),
                json.dumps(record.metrics, sort_keys=True),
                record.positions_sha256,
                record.n_features,
                record.n_clusters,
                record.built_at
                or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        if commit:
            self.conn.commit()

    def atlas_record(
        self, atlas_version: str | None = None, source: str | None = None,
        release: str | None = None, width: str | None = None
    ) -> AtlasRecord | None:
        """One atlas's record, or the most recently built one. None if there is
        no atlas — which is a different answer from an atlas with no features,
        and callers are expected to distinguish them.

        `source` narrows "most recently built" to one source representation,
        and a caller that shows a layout to a human should pass it. Several
        atlases coexist here by design — the spec requires both sources to be
        retained under distinct versions, neither overwriting the other — so
        plain recency picks whichever build happened to run last, which is not
        a statement about which layout anyone wants to look at. Ignored when
        `atlas_version` names one outright.

        Builds predating the `source` param recorded none, so they never match
        a source filter rather than being guessed at.
        """
        if atlas_version is not None:
            row = self.conn.execute(
                "SELECT * FROM atlas_record WHERE atlas_version = ?", (atlas_version,)
            ).fetchone()
        else:
            filters, values = [], []
            for column, value in (("json_extract(params_json, '$.source')", source),
                                  ("release", release), ("width", width)):
                if value is not None:
                    filters.append(f"{column} = ?")
                    values.append(value)
            where = " WHERE " + " AND ".join(filters) if filters else ""
            row = self.conn.execute(
                "SELECT * FROM atlas_record" + where + " ORDER BY built_at DESC LIMIT 1", values
            ).fetchone()
        if row is None:
            return None
        return AtlasRecord(
            atlas_version=row["atlas_version"],
            release=row["release"],
            width=row["width"],
            seed=row["seed"],
            params=json.loads(row["params_json"]),
            metrics=json.loads(row["metrics_json"]),
            positions_sha256=row["positions_sha256"],
            n_features=row["n_features"],
            n_clusters=row["n_clusters"],
            built_at=row["built_at"],
        )

    def put_clusters(self, atlas_version: str, rows: Iterable[ClusterRow], *, commit: bool = True) -> int:
        params = [
            (
                atlas_version,
                r.cluster,
                r.name,
                r.name_source[0] if r.name_source else None,
                r.name_source[1] if r.name_source else None,
                r.coherence,
                r.baseline_coherence,
                r.n_members,
                r.explainers,
                r.centroid[0],
                r.centroid[1],
                r.centroid[2],
                r.spread,
            )
            for r in rows
        ]
        if not params:
            return 0
        self.conn.executemany(
            "INSERT INTO atlas_cluster "
            "(atlas_version, cluster, name, name_source_layer, name_source_feature, "
            " coherence, baseline_coherence, n_members, explainers, "
            " centroid_x, centroid_y, centroid_z, spread) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(atlas_version, cluster) DO UPDATE SET "
            "  name = excluded.name, name_source_layer = excluded.name_source_layer, "
            "  name_source_feature = excluded.name_source_feature, "
            "  coherence = excluded.coherence, "
            "  baseline_coherence = excluded.baseline_coherence, "
            "  n_members = excluded.n_members, explainers = excluded.explainers, "
            "  centroid_x = excluded.centroid_x, centroid_y = excluded.centroid_y, "
            "  centroid_z = excluded.centroid_z, spread = excluded.spread",
            params,
        )
        if commit:
            self.conn.commit()
        return len(params)

    def clusters(self, atlas_version: str) -> list[ClusterRow]:
        return [
            ClusterRow(
                cluster=row["cluster"],
                n_members=row["n_members"],
                centroid=(row["centroid_x"], row["centroid_y"], row["centroid_z"]),
                spread=row["spread"],
                name=row["name"],
                name_source=(
                    (row["name_source_layer"], row["name_source_feature"])
                    if row["name_source_layer"] is not None
                    else None
                ),
                coherence=row["coherence"],
                baseline_coherence=row["baseline_coherence"],
                explainers=row["explainers"],
            )
            for row in self.conn.execute(
                "SELECT * FROM atlas_cluster WHERE atlas_version = ? ORDER BY cluster",
                (atlas_version,),
            )
        ]

    # -- the API fallback --------------------------------------------------

    def _fetch(self, layer: int, source_set: str, feature: int) -> FeatureLabel | None:
        """Ask Neuronpedia about one feature, and remember the answer either way.

        A miss is written to the DB as a text-NULL row. That is the whole point
        of the third state: without it this method re-runs forever on the ~0.3%
        of features nobody has explained.
        """
        if self.fetched >= self.max_fetches:
            return None

        import requests  # local: the offline path should not need it

        model_id, _ = neuronpedia_id(layer, self.width)
        url = API.format(model_id=model_id, source_set=source_set, feature=feature)
        self.fetched += 1
        try:
            response = requests.get(url, timeout=self.timeout_s)
            response.raise_for_status()
            chosen = pick_explanation(response.json().get("explanations") or [])
        except Exception as exc:  # noqa: BLE001 — a lookup must not kill a pass
            # Deliberately *not* cached: a timeout is not evidence that the
            # feature is unexplained, and writing a NULL row here would make a
            # network blip permanent.
            print(f"  neuronpedia lookup failed for {source_set}:{feature} — {exc}")
            return None

        row = LabelRow(
            source_set=source_set,
            feature=feature,
            text=(chosen or {}).get("description"),
            explainer=(chosen or {}).get("explanationModelName"),
            explanation_type=(chosen or {}).get("typeName"),
        )
        self.upsert([row])
        if not row.text:
            return None
        return FeatureLabel(
            text=row.text, explainer=row.explainer, explanation_type=row.explanation_type
        )

    # -- introspection -----------------------------------------------------

    def stats(self, source_set: str | None = None) -> dict[str, int]:
        """Row counts, split by the three states. What `--labels` reports."""
        where, params = ("WHERE source_set = ?", (source_set,)) if source_set else ("", ())
        row = self.conn.execute(
            f"SELECT COUNT(*) AS looked_up, "
            f"       SUM(text IS NOT NULL) AS labelled, "
            f"       SUM(embedding IS NOT NULL) AS with_embedding "
            f"FROM labels {where}",
            params,
        ).fetchone()
        looked_up = row["looked_up"] or 0
        labelled = row["labelled"] or 0
        return {
            "looked_up": looked_up,
            "labelled": labelled,
            "unexplained": looked_up - labelled,
            "with_embedding": row["with_embedding"] or 0,
        }

    def source_sets(self) -> list[str]:
        return [
            r["source_set"]
            for r in self.conn.execute(
                "SELECT DISTINCT source_set FROM labels ORDER BY source_set"
            )
        ]


def _chunked(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------

_default: LabelStore | None = None


def get_label(layer: int, feature_idx: int, width: str = DEFAULT_WIDTH) -> FeatureLabel | None:
    """One-off lookup against the default DB, for a REPL or a notebook.

    A pass or a server should build its own `LabelStore` — this one holds a
    module-level connection open for the life of the process, which is wrong
    for anything that wants to control when the DB is closed.
    """
    global _default
    if _default is None or _default.width != width:
        _default = LabelStore(width=width)
    return _default.get(layer, feature_idx)
