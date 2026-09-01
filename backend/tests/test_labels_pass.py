"""Phase 3: the labels enrichment pass. Offline — the store is injected."""

from __future__ import annotations

import pytest

from app.labels import LabelRow, LabelStore
from app.passes import apply
from app.passes.labels import LabelsPass, _features_by_layer
from app.schema import Feature, label_key

from factories import make_result

SOURCE_SET = "{}-gemmascope-res-16k"


@pytest.fixture
def store(tmp_path):
    """A DB holding labels for features 100-103 of layers 0-2."""
    with LabelStore(tmp_path / "labels.db") as s:
        s.upsert(
            [
                LabelRow(
                    source_set=SOURCE_SET.format(layer),
                    feature=feature,
                    text=f"label L{layer}F{feature}",
                    explainer="gpt-4o-mini",
                    explanation_type="oai_token-act-pair",
                )
                for layer in range(3)
                for feature in (100, 101, 102)
            ]
        )
        yield s


@pytest.fixture
def trace():
    """A synthetic trace with SAE features already on it."""
    t = make_result().trace
    for step in t.steps:
        for state in step.layers:
            state.features = [
                Feature(index=100, activation=5.0),
                Feature(index=101, activation=2.0),
            ]
            state.l0 = 2
    return t


def test_labels_land_in_the_side_table_not_on_features(trace, store):
    apply(LabelsPass(store=store, verbose=False), trace, None)

    assert trace.labels[label_key(1, 100)].text == "label L1F100"
    assert trace.label(2, 101).text == "label L2F101"
    # Feature itself stays a bare (index, activation) pair.
    assert not hasattr(trace.steps[0].layers[0].features[0], "label")


def test_a_repeated_feature_is_looked_up_and_stored_once(trace, store):
    """4 tokens x 3 layers x 2 features = 24 entries, 6 distinct pairs."""
    n_entries = sum(len(s.features) for step in trace.steps for s in step.layers)
    assert n_entries == 24

    record = apply(LabelsPass(store=store, verbose=False), trace, None)
    assert record.stats["features_wanted"] == 6
    assert len(trace.labels) == 6


def test_unlabelled_features_are_simply_absent(trace, store):
    for step in trace.steps:
        step.layers[0].features.append(Feature(index=999, activation=1.0))

    record = apply(LabelsPass(store=store, verbose=False), trace, None)

    assert label_key(0, 999) not in trace.labels
    assert record.stats["features_wanted"] == 7
    assert record.stats["features_labelled"] == 6
    assert record.stats["coverage"] == pytest.approx(6 / 7)


def test_the_record_carries_enough_to_rebuild_every_url(trace, store):
    record = apply(LabelsPass(store=store, verbose=False), trace, None)

    url = record.params["url_template"].format(
        model_id=record.params["neuronpedia_model"],
        source_set=record.params["source_set_template"].format(layer=20),
        feature=12082,
    )
    assert url == "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/12082"


def test_the_record_names_the_explainer_mix(trace, store):
    """The export mixes explainers across layers, so a trace has to say which
    ones its labels came from — see phase3.md."""
    record = apply(LabelsPass(store=store, verbose=False), trace, None)
    assert record.params["explainers"] == "gpt-4o-mini:6"


def test_labelling_an_unencoded_trace_fails_loudly(store):
    """No SAE pass means no features; silently writing an empty label map
    would look like a trace whose features are all unexplained."""
    bare = make_result().trace
    with pytest.raises(ValueError, match="run the SAE pass first"):
        LabelsPass(store=store, verbose=False).run(bare, None)


def test_features_by_layer_dedupes_and_sorts(trace):
    by_layer = _features_by_layer(trace)
    assert by_layer == {0: [100, 101], 1: [100, 101], 2: [100, 101]}


def test_rerunning_the_pass_replaces_its_record(trace, store):
    apply(LabelsPass(store=store, verbose=False), trace, None)
    apply(LabelsPass(store=store, verbose=False), trace, None)
    assert [p.name for p in trace.passes] == ["labels"]
