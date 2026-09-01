"""Phase 3: the label store.

Everything here runs offline. The store's whole reason for existing is that a
trace wants thousands of labels and must not go to the network for them, so a
test suite that quietly hits neuronpedia.org would be testing the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.labels import EXPLAINER_PREFERENCE, LabelRow, LabelStore, feature_url, pick_explanation

LAYER = 20
SOURCE_SET = "20-gemmascope-res-16k"


@pytest.fixture
def store(tmp_path):
    with LabelStore(tmp_path / "labels.db") as s:
        yield s


def test_url_is_derived_from_the_saelens_mapping():
    assert feature_url(20, 12082) == (
        "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/12082"
    )


def test_roundtrip(store):
    store.upsert(
        [
            LabelRow(
                source_set=SOURCE_SET,
                feature=12082,
                text="references to dogs as pets",
                explainer="gpt-4o-mini",
                explanation_type="oai_token-act-pair",
            )
        ]
    )
    label = store.get(LAYER, 12082)
    assert label.text == "references to dogs as pets"
    assert label.explainer == "gpt-4o-mini"


def test_the_three_states_are_distinguishable(store):
    """A feature nobody explained must not look like one nobody looked up.

    Both come back as None from get(), which is right for a caller rendering a
    label — but the DB has to keep them apart, or the API fallback re-requests
    the unexplained ones forever.
    """
    store.upsert([LabelRow(source_set=SOURCE_SET, feature=1, text="something")])
    store.upsert([LabelRow(source_set=SOURCE_SET, feature=2, text=None)])  # looked up, empty

    assert store.get(LAYER, 1) is not None
    assert store.get(LAYER, 2) is None  # unexplained
    assert store.get(LAYER, 3) is None  # never looked up

    stats = store.stats()
    assert stats == {"looked_up": 2, "labelled": 1, "unexplained": 1, "with_embedding": 0}


def test_get_many_returns_only_labelled_features(store):
    store.upsert(
        [
            LabelRow(source_set=SOURCE_SET, feature=1, text="a"),
            LabelRow(source_set=SOURCE_SET, feature=2, text=None),
            LabelRow(source_set=SOURCE_SET, feature=3, text="c"),
        ]
    )
    assert set(store.get_many(LAYER, [1, 2, 3, 4])) == {1, 3}


def test_get_many_handles_more_features_than_sqlite_takes_parameters(store):
    """A trace asks for thousands of features from one layer; SQLite caps
    host parameters at 999 on older builds, so the query has to chunk."""
    features = list(range(2000))
    store.upsert([LabelRow(source_set=SOURCE_SET, feature=f, text=f"f{f}") for f in features])
    assert len(store.get_many(LAYER, features)) == 2000


def test_offline_by_default(store, monkeypatch):
    """The default store must never reach the network, even on a total miss."""
    monkeypatch.setattr(
        "app.labels.LabelStore._fetch",
        lambda *a, **k: pytest.fail("offline store attempted a network fetch"),
    )
    assert store.get(LAYER, 999) is None


def test_upsert_replaces_text_but_keeps_an_existing_embedding(store):
    vector = np.arange(4, dtype=np.float32)
    store.upsert([LabelRow(source_set=SOURCE_SET, feature=7, text="old", embedding=vector)])
    store.upsert([LabelRow(source_set=SOURCE_SET, feature=7, text="new")])

    assert store.get(LAYER, 7).text == "new"
    # Re-importing explanations should not silently drop the vectors that a
    # separate, much more expensive import pass put there.
    assert np.array_equal(store.embeddings(LAYER, [7])[7], vector)


def test_embeddings_round_trip_as_float32(store):
    vector = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    store.upsert([LabelRow(source_set=SOURCE_SET, feature=9, text="x", embedding=vector)])
    assert np.array_equal(store.embeddings(LAYER, [9])[9], vector)


# --------------------------------------------------------------------------
# choosing among several explanations (the API fallback's problem)
# --------------------------------------------------------------------------


def test_pick_explanation_prefers_the_configured_explainer():
    chosen = pick_explanation(
        [
            {"explanationModelName": "gemini-2.0-flash", "typeName": "np_acts-logits-general", "description": "B"},
            {"explanationModelName": "gpt-4o-mini", "typeName": "oai_token-act-pair", "description": "A"},
        ]
    )
    assert chosen["description"] == "A"


def test_pick_explanation_ranks_unknown_explainers_last():
    chosen = pick_explanation(
        [
            {"explanationModelName": "some-new-model", "typeName": "whatever", "description": "new"},
            {"explanationModelName": "gemini-2.0-flash", "typeName": "np_acts-logits-general", "description": "known"},
        ]
    )
    assert chosen["description"] == "known"


def test_pick_explanation_uses_score_only_within_one_explainer():
    """Scores are per (type, scorer) and are mostly absent, so they can break a
    tie between two labels from the same explainer but must never outrank the
    preference order."""
    preferred, fallback = EXPLAINER_PREFERENCE[0], EXPLAINER_PREFERENCE[1]
    chosen = pick_explanation(
        [
            {
                "explanationModelName": fallback[0],
                "typeName": fallback[1],
                "description": "scored high but wrong explainer",
                "scores": [{"value": 1.0}],
            },
            {
                "explanationModelName": preferred[0],
                "typeName": preferred[1],
                "description": "unscored but preferred",
            },
        ]
    )
    assert chosen["description"] == "unscored but preferred"


def test_pick_explanation_on_an_empty_list():
    assert pick_explanation([]) is None
