"""Regressions for mismatched and stale scientific measurements."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.labels import AtlasRecord, LabelRow, LabelStore, LayoutRow
from app.passes import apply
from app.passes.attribution import AttributionPass
from app.passes.labels import LabelsPass
from app.passes.layout import LayoutPass
from app.passes.sae import SAEPass
from app.sae_cache import RELEASE
from app.schema import Feature, FeatureLabel, NodePosition, PassRecord, SteeringInfo
from factories import make_result


def featured(width="16k"):
    result = make_result()
    result.trace.passes = [PassRecord(name="sae", params={"release": RELEASE, "width": width})]
    result.trace.steps[1].layers[0].features = [Feature(index=7, activation=2)]
    return result


def record(version="v1", width="16k"):
    return AtlasRecord(version, RELEASE, width, 0, {"source": "labels"}, {}, "hash", 1, 0)


def test_labels_reject_wrong_dictionary(tmp_path):
    result = featured("65k")
    with LabelStore(tmp_path / "labels.db") as store:
        with pytest.raises(ValueError, match="width"):
            apply(LabelsPass(store=store, verbose=False), result.trace, None)


def test_labels_infer_width_from_features(tmp_path):
    result = featured("65k")
    with LabelStore(tmp_path / "labels.db", width="65k") as store:
        store.upsert([LabelRow("0-gemmascope-res-65k", 7, "correct label")])
        p = apply(LabelsPass(store=store, verbose=False), result.trace, None)
        assert p.params["width"] == "65k"
        assert result.trace.label(0, 7).text == "correct label"


def test_layout_requires_provenance(tmp_path):
    result = featured()
    result.trace.passes = []
    with LabelStore(tmp_path / "db") as store:
        with pytest.raises(ValueError, match="provenance"):
            apply(LayoutPass(store=store, verbose=False), result.trace, None)


def test_pinned_wrong_atlas_rejected_and_previous_trace_preserved(tmp_path):
    result = featured()
    result.trace.layout["0/7"] = NodePosition(x=1, y=2, z=3)
    before = result.trace.model_dump()
    with LabelStore(tmp_path / "db") as store:
        store.publish_atlas(record(width="65k"), [LayoutRow(0, 7, 9, 9, 9)], [])
        with pytest.raises(ValueError, match="dictionary"):
            apply(LayoutPass(store=store, atlas_version="v1", verbose=False), result.trace, None)
    assert result.trace.model_dump() == before


def test_latest_atlas_selection_filters_width(tmp_path):
    with LabelStore(tmp_path / "db") as store:
        store.put_atlas_record(replace(record("right"), built_at="2020"))
        store.put_atlas_record(replace(record("wrong", "65k"), built_at="2030"))
        assert store.atlas_record(source="labels", release=RELEASE, width="16k").atlas_version == "right"


def test_missing_atlas_clears_old_positions(tmp_path):
    result = featured()
    result.trace.layout["0/7"] = NodePosition(x=1, y=2, z=3)
    with LabelStore(tmp_path / "db") as store:
        p = apply(LayoutPass(store=store, verbose=False), result.trace, None)
    assert not result.trace.layout
    assert p.params["atlas_available"] is False


def test_smaller_atlas_does_not_retain_old_positions(tmp_path):
    result = featured()
    result.trace.steps[1].layers[0].features.append(Feature(index=8, activation=1))
    result.trace.layout["0/8"] = NodePosition(x=9, y=9, z=9)
    with LabelStore(tmp_path / "db") as store:
        store.publish_atlas(record(), [LayoutRow(0, 7, 1, 2, 3)], [])
        apply(LayoutPass(store=store, verbose=False), result.trace, None)
    assert set(result.trace.layout) == {"0/7"}


def test_failed_pass_is_atomic():
    result = featured()
    before = result.trace.model_dump()
    class Failing:
        name = "sae"
        def run(self, trace, residuals):
            trace.steps[0].layers[0].features = [Feature(index=99, activation=99)]
            raise RuntimeError("failed midway")
    with pytest.raises(RuntimeError):
        apply(Failing(), result.trace, result.residuals)
    assert result.trace.model_dump() == before


def test_reencoding_subset_invalidates_old_layers_labels_and_layout():
    result = featured()
    result.trace.labels["0/7"] = FeatureLabel(text="old")
    result.trace.layout["0/7"] = NodePosition(x=1, y=2, z=3)
    result.trace.passes += [PassRecord(name="labels"), PassRecord(name="layout")]
    class SAE:
        def encode(self, x):
            return torch.ones(len(x), 3)
        def decode(self, a):
            return torch.zeros(len(a), result.trace.d_model)
    apply(SAEPass(layers=[1], saes={1: SAE()}, device="cpu", hook="hook_resid_post", verbose=False),
          result.trace, result.residuals)
    assert result.trace.steps[1].layers[0].features == []
    assert result.trace.steps[1].layers[0].l0 is None
    assert result.trace.steps[1].layers[1].features
    assert not result.trace.labels and not result.trace.layout
    assert [p.name for p in result.trace.passes] == ["sae"]


def test_sae_rejects_real_metadata_for_another_model():
    result = make_result()
    sae = SimpleNamespace(cfg=SimpleNamespace(metadata=SimpleNamespace(model_name="other-model")))
    with pytest.raises(ValueError, match="SAE expects"):
        apply(SAEPass(layers=[0], saes={0: sae}, hook="hook_resid_post"), result.trace, result.residuals)


def test_steered_attribution_is_rejected_before_model_load():
    result = featured()
    result.trace.steering = SteeringInfo(layer=1, feature_idx=0, coefficient=2)
    with pytest.raises(ValueError, match="steered traces"):
        apply(AttributionPass(), result.trace, result.residuals)
    assert result.trace.pass_record("attribution") is None


def test_same_text_refresh_keeps_embedding(tmp_path):
    with LabelStore(tmp_path / "db") as store:
        store.upsert([LabelRow("0-gemmascope-res-16k", 7, "same", embedding=np.ones(3))])
        store.upsert([LabelRow("0-gemmascope-res-16k", 7, "same")])
        np.testing.assert_array_equal(store.embeddings(0, [7])[7], np.ones(3))


def test_atlas_publication_is_immutable(tmp_path):
    with LabelStore(tmp_path / "db") as store:
        rows = [LayoutRow(0, 7, 1, 2, 3)]
        store.publish_atlas(record(), rows, [])
        store.publish_atlas(record(), rows, [])  # exact retry is harmless
        with pytest.raises(ValueError, match="overwrite"):
            store.publish_atlas(record(), [LayoutRow(0, 7, 9, 9, 9)], [])
        assert store.layout_all("v1") == rows


def test_failed_publication_rolls_back_rows(tmp_path, monkeypatch):
    with LabelStore(tmp_path / "db") as store:
        def fail(*args, **kwargs):
            raise RuntimeError("interrupted after layout")
        monkeypatch.setattr(store, "put_clusters", fail)
        with pytest.raises(RuntimeError):
            store.publish_atlas(record(), [LayoutRow(0, 7, 1, 2, 3)], [])
        assert store.atlas_record("v1") is None
        assert store.layout_all("v1") == []
