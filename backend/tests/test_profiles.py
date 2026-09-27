"""The model registry: every model-specific fact in one table.

Fast tests read SAELens' bundled registry (no network). The last one runs a
real gpt2-small trace through its SAEs and skips unless MECHLENS_SLOW=1.
"""

from __future__ import annotations

import os

import pytest
import torch
from app import profiles
from app.profiles import (
    GEMMA_SCOPE,
    GPT2_RES_JB,
    get_profile,
    reads_layer,
    release_for,
    same_model,
)
from app.sae_cache import center_input, expects_centered


def test_every_name_a_user_might_type_finds_the_same_profile():
    gpt2 = get_profile("gpt2")
    assert get_profile("gpt2-small") is gpt2
    assert get_profile("openai-community/gpt2") is gpt2
    assert get_profile("google/gemma-2-2b").name == "gemma-2-2b"
    with pytest.raises(KeyError, match="unknown model"):
        get_profile("not-a-model")


def test_default_model_follows_the_environment(monkeypatch):
    monkeypatch.setenv("MECHLENS_MODEL", "gemma-2-2b")
    assert get_profile().name == "gemma-2-2b"
    monkeypatch.delenv("MECHLENS_MODEL")
    assert get_profile().name == profiles.DEFAULT_MODEL


def test_saelens_and_transformerlens_spellings_are_the_same_model():
    assert same_model("gpt2-small", "gpt2")
    assert not same_model("gpt2", "gemma-2-2b")
    assert not same_model("unknown-a", "unknown-b")


def test_a_model_without_a_release_has_none():
    assert release_for("tiny-stories-1M") is None
    assert release_for("unknown") is None
    assert release_for("gpt2") is GPT2_RES_JB


def test_gemma_scope_ids_are_unchanged():
    assert GEMMA_SCOPE.sae_id(20) == "layer_20/width_16k/canonical"
    assert GEMMA_SCOPE.neuronpedia_id(20) == ("gemma-2-2b", "20-gemmascope-res-16k")
    assert GEMMA_SCOPE.sae_id(20, "65k") == "layer_20/width_65k/canonical"
    with pytest.raises(ValueError, match="no width"):
        GEMMA_SCOPE.sae_id(20, "24k")


def test_resid_pre_releases_map_onto_trace_layers_one_block_later():
    """Trace layer L is blocks.L.hook_resid_post, which is blocks.{L+1}.hook_resid_pre."""
    assert GPT2_RES_JB.sae_id(0) == "blocks.1.hook_resid_pre"
    assert GPT2_RES_JB.sae_id(10) == "blocks.11.hook_resid_pre"
    # No block 12 exists; the last layer has its own resid_post SAE.
    assert GPT2_RES_JB.sae_id(11) == "blocks.11.hook_resid_post"
    assert GPT2_RES_JB.neuronpedia_id(0) == ("gpt2-small", "1-res-jb")
    assert GPT2_RES_JB.neuronpedia_id(11) == ("gpt2-small", "12-res-jb")
    assert all(reads_layer(f"blocks.{i}.hook_resid_pre" if i < 11 else "blocks.11.hook_resid_post", i - (i < 11))
               for i in range(1, 12))


def test_reads_layer_rejects_other_sites():
    assert reads_layer("blocks.5.hook_resid_post", 5)
    assert reads_layer("blocks.6.hook_resid_pre", 5)
    assert not reads_layer("blocks.5.hook_resid_pre", 5)
    assert not reads_layer("blocks.5.hook_mlp_out", 5)
    assert not reads_layer(None, 5)


def test_centering_applies_only_to_saes_trained_on_a_centered_model():
    class Meta:
        model_from_pretrained_kwargs = {"center_writing_weights": True}

    class Cfg:
        metadata = Meta()

    class Centered:
        cfg = Cfg()

    x = torch.randn(3, 8) + 5.0
    assert expects_centered(Centered())
    assert torch.allclose(center_input(Centered(), x).mean(-1), torch.zeros(3), atol=1e-5)
    assert center_input(object(), x) is x


@pytest.mark.skipif(not os.getenv("MECHLENS_SLOW"), reason="downloads gpt2 and three of its SAEs (~0.5GB)")
def test_real_gpt2_trace_encodes_cleanly():
    """The whole point of the registry: a second model works with no code of its own."""
    from app.capture import generate_trace
    from app.model_cache import get_model
    from app.passes import apply
    from app.passes.sae import SAEPass

    model = get_model("gpt2-small")
    result = generate_trace(model, "The Golden Gate Bridge is in", max_new_tokens=4)
    record = apply(SAEPass(layers=[0, 5, 11], hook=result.hook, verbose=False), result.trace, result.residuals)
    assert record.params["release"] == "gpt2-small-res-jb"
    assert record.params["centered"] is True
    assert min(record.stats["explained_variance_by_layer"]) > 0.85
    assert all(5 < l0 < 200 for l0 in record.stats["l0_by_layer"])
