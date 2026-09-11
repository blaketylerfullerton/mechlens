"""Fixed-prefix interventions through a real, random tiny transformer; no download."""
from types import SimpleNamespace

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from app.experiments import measure_feature


@pytest.fixture
def model():
    cfg = HookedTransformerConfig(n_layers=2, d_model=8, n_ctx=16, d_head=4,
                                  n_heads=2, d_mlp=16, d_vocab=20, act_fn="relu",
                                  normalization_type=None, device="cpu", seed=42,
                                  model_name="synthetic")
    model = HookedTransformer(cfg)
    model.to_tokens = lambda prompt: torch.tensor([[1, 2, 3]])
    return model


class SAE:
    # Positive activation makes the ablation test non-vacuous. Its one
    # feature is an arbitrary direction; reconstruction error is preserved.
    W_dec = torch.ones(1, 8)
    def encode(self, x):
        return x.new_full((len(x), 1), 2.0)


def run(model, **kwargs):
    return measure_feature(model, SAE(), "fixed prefix", 0, 0, 4, [-2, 2], **kwargs)


def test_zero_control_and_ablation_match_direct_forward(model):
    report = run(model)
    baseline, zero, ablation, negative, positive = report["measurements"]
    assert zero["delta_probability"] == 0
    assert report["zero_control_max_error"] == 0
    assert ablation["target_probability"] == negative["target_probability"]
    def remove(resid, hook):
        output = resid.clone()
        output[:, -1] -= 2
        return output
    with torch.no_grad(), model.hooks(fwd_hooks=[("blocks.0.hook_resid_post", remove)]):
        expected = model(model.to_tokens("fixed prefix"))[0, -1].float().softmax(-1)[4]
    assert ablation["target_probability"] == pytest.approx(float(expected), abs=1e-7)
    assert positive["target_probability"] != baseline["target_probability"]
    assert report["token_ids"] == [1, 2, 3]


def test_experiment_leaves_model_unmodified(model):
    with torch.no_grad():
        before = model(model.to_tokens("x")).clone()
    run(model)
    with torch.no_grad():
        after = model(model.to_tokens("x"))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert all(not h.fwd_hooks for h in model.hook_dict.values())


def test_hook_removed_even_if_encoding_fails(model):
    class BrokenSAE(SAE):
        def encode(self, x):
            raise RuntimeError("encoder failed")
    with pytest.raises(RuntimeError):
        measure_feature(model, BrokenSAE(), "x", 0, 0, 4, [1])
    assert all(not h.fwd_hooks for h in model.hook_dict.values())


def test_nonfinite_strength_rejected(model):
    with pytest.raises(ValueError, match="finite"):
        measure_feature(model, SAE(), "x", 0, 0, 4, [float("inf")])


def test_wrong_sae_model_rejected(model):
    sae = SAE()
    sae.cfg = SimpleNamespace(metadata=SimpleNamespace(model_name="other"))
    with pytest.raises(ValueError, match="identities"):
        measure_feature(model, sae, "x", 0, 0, 4, [1])


def test_explicit_position_is_recorded(model):
    report = run(model, position=1)
    assert report["position"] == 1
    with pytest.raises(ValueError, match="position"):
        run(model, position=99)


def test_cli_saves_examples_and_controls(model, tmp_path, monkeypatch):
    import json
    from app import cli, sae_cache
    monkeypatch.setattr(cli, "get_model", lambda: model)
    monkeypatch.setattr(sae_cache, "get_sae", lambda *args: SAE())
    output = tmp_path / "results" / "experiment.json"
    args = cli.build_parser().parse_args([
        "experiment", "--prompt", "related", "--control-prompt", "unrelated",
        "--layer", "0", "--feature-idx", "0", "--target-token-id", "4",
        "--coefficients", "-2", "2", "--out", str(output),
    ])
    args.func(args)
    reports = json.loads(output.read_text())["experiments"]
    assert [r["role"] for r in reports] == ["example", "control"]
    assert all(len(r["measurements"]) == 5 for r in reports)
    assert all(r["zero_control_max_error"] == 0 for r in reports)
