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
            "  embedding = COALESCE(excluded.embedding, labels.embedding), "
            "  fetched_at = excluded.fetched_at",
            params,
        )
        self.conn.commit()
        return len(params)

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
