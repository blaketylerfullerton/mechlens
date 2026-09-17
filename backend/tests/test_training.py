"""Offline integration tests run actual SAELens optimization against a tiny transformer."""
import threading
from types import MethodType

import pytest
import torch
from fastapi.testclient import TestClient
from transformer_lens import HookedTransformer, HookedTransformerConfig

from app.training.store import RunStore
from app.training.runner import TrainingConfig, train, load_artifact, Cancelled


@pytest.fixture
def model():
    torch.manual_seed(7)
    model = HookedTransformer(HookedTransformerConfig(n_layers=1, d_model=16,
        n_ctx=32, d_head=8, n_heads=2, d_mlp=32, d_vocab=64,
        act_fn="relu", normalization_type="LN", device="cpu", model_name="training-test"))
    # Deterministic offline tokenizer; optimization and activation capture remain real.
    model.to_tokens = MethodType(lambda self, text, **kw: torch.tensor([[1] + [2 + ord(c) % 62 for c in text]]), model)
    return model


@pytest.fixture
def config():
    return TrainingConfig(layer=0, features=32, training_tokens=64, batch_size=16,
        context_size=16, dataset="text", training_text="abcdef " * 32,
        evaluation_text="a different evaluation document " * 3)


def test_real_training_roundtrip_and_frozen_model(tmp_path, model, config, monkeypatch):
    from app.training import runner
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    # The first checkpoint is pruned once a later one lands, so its weights are
    # read as it is written rather than from disk afterwards.
    first_step = []
    original = runner.checkpoint
    def recording(store, run_id, sae, trainer, manifest):
        if not first_step:
            first_step.append(sae.W_enc.detach().clone())
        return original(store, run_id, sae, trainer, manifest)
    monkeypatch.setattr(runner, "checkpoint", recording)
    train(store, run["id"], model)
    saved = store.get(run["id"])
    assert saved["status"] == "completed"
    assert saved["tokens"] == config.training_tokens
    assert saved["validation"]["model_agreement"]["status"] == "not_run"
    assert saved["validation"]["checkpoint_agreement"]["status"] == "passed"
    assert saved["validation"]["activation_identity"]["status"] == "passed"
    assert saved["evaluation"]["identity_substitution"]["status"] == "passed"
    assert saved["evaluation"]["ablation_loss"] > 0
    assert saved["evaluation"]["tokens"] > 0
    assert saved["evaluation"]["reconstruction_loss"] > 0
    assert [p.requires_grad for p in model.parameters()] == flags
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert store.metrics(run["id"])[-1]["tokens"] == 64
    sae, manifest = load_artifact(store, run["id"], model)
    assert sae.encode(torch.randn(3, 16)).shape == (3, 32)
    assert manifest["model"] == "training-test"
    assert len(manifest["artifact_id"]) == 64
    # The trained weights differ from the first-step checkpoint, and only the
    # newest checkpoint is kept.
    assert not torch.equal(first_step[0], sae.W_enc)
    assert [p.name for p in (tmp_path / run["id"]).glob("checkpoint-*")] == [
        f"checkpoint-{config.training_tokens}"
    ]
    # Content identity is enforced at the loading boundary.
    path = tmp_path / manifest["path"] / "cfg.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="content"):
        load_artifact(store, run["id"], model)


def test_feature_examples_are_bounded_and_checkpoint_scoped(tmp_path, model, config):
    from app.training.examples import collect_feature_examples, save_feature_examples
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    train(store, run["id"], model)
    sae, manifest = load_artifact(store, run["id"], model)
    report = collect_feature_examples(model, sae, manifest, config, [0, 1],
        max_examples=2, max_sequences=3)
    assert report["artifact_id"] == manifest["artifact_id"]
    assert report["layer"] == 0
    assert report["sequences_scanned"] == 3
    assert set(report["examples"]) == {"0", "1"}
    for rows in report["examples"].values():
        assert len(rows) <= 2
        assert all(row["activation"] > 0 and row["token_ids"] and row["context"] for row in rows)
    path = save_feature_examples(tmp_path, run["id"], report)
    assert path.is_file()
    assert path.name.startswith("examples-")


def test_cancel_during_training_saves_checkpoint(tmp_path, model, config, monkeypatch):
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    original = store.metric
    def metric(*args):
        original(*args)
        store.cancel(run["id"])
    monkeypatch.setattr(store, "metric", metric)
    with pytest.raises(Cancelled):
        train(store, run["id"], model)
    assert store.get(run["id"])["checkpoint"]["tokens"] == 16
    assert all(p.requires_grad for p in model.parameters())


def test_store_recovery_cursor_and_terminal_cancel(tmp_path):
    store = RunStore(tmp_path)
    active = store.create({})
    completed = store.create({})
    store.update(completed["id"], status="completed")
    store.metric(active["id"], {"tokens": 1})
    store.metric(active["id"], {"tokens": 2})
    reopened = RunStore(tmp_path)
    reopened.recover()
    assert reopened.get(active["id"])["status"] == "interrupted"
    assert reopened.cancel(completed["id"])["status"] == "completed"
    assert [m["tokens"] for m in reopened.metrics(active["id"], after=1)] == [2]
    with pytest.raises(KeyError):
        reopened.get("missing")
    job = reopened.create_interp_job(active["id"], "artifact", [3, 7], "http://127.0.0.1:8080", "local-model")
    assert reopened.list_interp_jobs()[0] == job


def test_dataset_exhaustion_preserves_checkpoint(tmp_path, model, config):
    config.training_tokens = 1024
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    with pytest.raises(ValueError, match="exhausted"):
        train(store, run["id"], model)
    assert store.get(run["id"])["checkpoint"] is not None


def test_resume_matches_uninterrupted_training(tmp_path, model, config, monkeypatch):
    store = RunStore(tmp_path)
    complete = store.create(config.model_dump())
    train(store, complete["id"], model)
    expected, _ = load_artifact(store, complete["id"], model)
    interrupted = store.create(config.model_dump())
    original = store.metric
    def metric(run_id, values):
        original(run_id, values)
        store.cancel(run_id)
    monkeypatch.setattr(store, "metric", metric)
    with pytest.raises(Cancelled):
        train(store, interrupted["id"], model)
    store.update(interrupted["id"], status="cancelled")
    monkeypatch.setattr(store, "metric", original)
    store.resume(interrupted["id"])
    with pytest.raises(ValueError, match="Only stopped"):
        store.resume(interrupted["id"])
    train(store, interrupted["id"], model, resume=True)
    resumed, _ = load_artifact(store, interrupted["id"], model)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in expected.state_dict().items())


def test_api_queue_cancellation_validation_and_lock(tmp_path, model, config, monkeypatch):
    from fastapi import FastAPI
    from app.training.api import router
    from app.service import jobs
    from app.training import api
    callbacks = []
    monkeypatch.setattr(jobs, "submit", lambda fn: callbacks.append(fn) or "job")
    lock = threading.Lock()
    store = RunStore(tmp_path)
    app = FastAPI()
    app.include_router(router(lambda: model, lock, store))
    with TestClient(app) as client:
        response = client.post('/training/runs', json=config.model_dump())
        assert response.status_code == 202
        run_id = response.json()["id"]
        assert "training_text" not in response.json()["config"]
        assert client.post(f'/training/runs/{run_id}/cancel').json()["status"] == "cancelled"
        callbacks.pop()(None)
        assert store.get(run_id)["checkpoint"] is None
        bad = dict(config.model_dump(), layer=2)
        assert client.post('/training/runs', json=bad).status_code == 422
        assert client.get('/training/runs/nope').status_code == 404
        assert client.post(f'/training/runs/{run_id}/trace', json={"prompt": "hello"}).status_code == 422
        def fail(store, run_id, model):
            assert lock.locked()
            raise RuntimeError("worker failure")
        monkeypatch.setattr(api, "train", fail)
        second = client.post('/training/runs', json=config.model_dump()).json()
        callbacks.pop()(None)
        assert store.get(second["id"])["status"] == "failed"
        assert not lock.locked()


def test_custom_sae_identity_rejects_gemma_labels(tmp_path, model, config):
    from app.passes import apply
    from app.passes.sae import SAEPass
    from app.identity import feature_identity
    from factories import make_result
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    train(store, run["id"], model)
    sae, manifest = load_artifact(store, run["id"], model)
    result = make_result(n_layers=1, d_model=16)
    result.trace.model = model.cfg.model_name
    apply(SAEPass(saes={0: sae}, layers=[0], artifact_id=manifest["artifact_id"],
        feature_count=32, width="32", hook="hook_resid_post", verbose=False), result.trace, result.residuals)
    assert result.trace.steps[1].layers[0].l0 is not None
    assert result.trace.pass_record("sae").params["release"].startswith("local/")
    with pytest.raises(ValueError, match="unsupported SAE identity"):
        feature_identity(result.trace)


def test_model_switch_serializes_and_releases_old_cache(tmp_path, model, monkeypatch):
    from app.service.app import create_app
    from app.service import jobs
    from app import model_cache
    from app.service.jobs import JobRecord
    callbacks = []
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "submit", lambda fn: callbacks.append(fn) or "switch-job")
    monkeypatch.setattr(model_cache, "get_model", lambda name: model)
    app = create_app(model=model, training_dir=tmp_path)
    with TestClient(app) as client:
        assert client.post('/training/model', json={"repository": "unsupported/model"}).status_code == 422
        jobs.JOBS["busy"] = JobRecord(id="busy", status="running")
        assert client.post('/training/model', json={"repository": "openai-community/gpt2"}).status_code == 409
        jobs.JOBS.clear()
        assert client.post('/training/model', json={"repository": "openai-community/gpt2"}).status_code == 202
        assert client.get('/health').json()["status"] == "loading"
        assert client.post('/trace', json={"prompt": "hello", "max_tokens": 2}).status_code == 503
        callbacks.pop()(None)
        assert client.get('/health').json()["status"] == "ready"


def test_evaluation_identity_sae_and_zero_ablation(model):
    from app.training.runner import evaluate
    class IdentitySAE:
        def eval(self):
            return self
        def encode(self, x):
            return x
        def decode(self, x):
            return x
    sequences = [model.to_tokens("abcdef"), model.to_tokens("a longer document")]
    result = evaluate(model, IdentitySAE(), sequences, "blocks.0.hook_resid_post", lambda: None, limit=64)
    assert result["mse"] == 0
    assert result["explained_variance"] == 1
    assert result["loss_increase"] == 0
    assert result["identity_substitution"]["status"] == "passed"
    assert result["sequences"] == 2
    assert result["tokens"] == sum(t.shape[1] - 1 for t in sequences)
    assert result["sample_limit_reached"] is False
    if result["ablation_loss"] > result["baseline_loss"] + 1e-8:
        assert result["loss_recovered"] == pytest.approx(1)
    else:
        assert result["loss_recovered"] is None


def test_evaluation_zero_sae_matches_ablation_and_sample_limit(model):
    from app.training.runner import evaluate
    class ZeroSAE:
        def eval(self):
            return self
        def encode(self, x):
            return torch.zeros_like(x)
        def decode(self, x):
            return x
    tokens = model.to_tokens("abcdef")
    result = evaluate(model, ZeroSAE(), [tokens, tokens], "blocks.0.hook_resid_post", lambda: None, limit=1)
    assert result["reconstruction_loss"] == result["ablation_loss"]
    assert result["l0"] == 0
    assert result["sample_limit_reached"] is True
    assert result["sequences"] == 1


def test_checkpoint_agreement_detects_changes():
    from app.training.validation import checkpoint_agreement
    class SAE:
        def __init__(self, offset):
            self.offset = offset
        def encode(self, x):
            return x + self.offset
        def decode(self, x):
            return x
    x = torch.ones(2, 4)
    assert checkpoint_agreement(SAE(0), SAE(0), x)["status"] == "passed"
    assert checkpoint_agreement(SAE(0), SAE(1), x)["status"] == "failed"
    assert checkpoint_agreement(SAE(0), SAE(float('nan')), x)["status"] == "failed"


def test_huggingface_agreement_offline(model, monkeypatch):
    from app.training.validation import huggingface_agreement
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import loading_from_pretrained
    from types import SimpleNamespace
    assert huggingface_agreement(model, lambda: None)["status"] == "not_run"
    model.cfg.model_revision = "test-revision"
    monkeypatch.setattr(loading_from_pretrained, "get_official_model_name", lambda name: "test/model")
    mismatch = False
    class Reference:
        def cpu(self):
            return self
        def eval(self):
            return self
        def __call__(self, input_ids):
            logits = model(input_ids, return_type="logits")
            if mismatch:
                logits = torch.zeros_like(logits)
                logits[..., 0] = 100
            return SimpleNamespace(logits=logits)
    # Fixture's tokenizer ignores prepend_bos; match its explicit no-BOS behavior here.
    tokenizer = SimpleNamespace(encode=lambda text, **kw: model.to_tokens(text)[0].tolist())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: Reference())
    monkeypatch.setattr('app.training.validation.PROMPTS', ("abc", "def"))
    assert huggingface_agreement(model, lambda: None)["status"] == "passed"
    mismatch = True
    assert huggingface_agreement(model, lambda: None)["status"] == "failed"


def test_report_api_redacts_corpus_and_supports_legacy_runs(tmp_path, model, config):
    from fastapi import FastAPI
    from app.training.api import router
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    app = FastAPI()
    app.include_router(router(lambda: model, threading.Lock(), store))
    with TestClient(app) as client:
        response = client.get(f'/training/runs/{run["id"]}/report')
        assert response.status_code == 200
        assert response.json()["correctness"] is None
        assert "training_text" not in response.json()["config"]
        assert "evaluation_text" not in response.json()["config"]
        assert "attachment" in response.headers["content-disposition"]
        assert client.get('/training/runs/missing/report').status_code == 404


def test_validation_progress_is_persisted_before_completion(tmp_path, model, config, monkeypatch):
    config.compare_huggingface = True
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    snapshots = []
    update = store.update
    def record(run_id, **fields):
        result = update(run_id, **fields)
        snapshots.append(store.get(run_id))
        return result
    monkeypatch.setattr(store, "update", record)
    def compare(model, check):
        current = store.get(run["id"])
        assert current["validation"]["model_agreement"]["status"] == "running"
        assert current["validation"]["checkpoint_agreement"]["status"] == "pending"
        return {"status": "passed"}
    monkeypatch.setattr('app.training.validation.huggingface_agreement', compare)
    train(store, run["id"], model)
    evaluating = [s for s in snapshots if s["phase"] == "evaluating"]
    assert evaluating[0]["validation_progress"]["completed"] == 0
    assert evaluating[0]["validation"]["checkpoint_agreement"]["status"] == "running"
    partial = next(s for s in evaluating if s["validation_progress"]["completed"] == 1)
    assert partial["status"] == "running"
    assert partial["validation"]["checkpoint_agreement"]["status"] == "passed"
    assert partial["validation"]["identity_substitution"]["status"] == "running"
    final = store.get(run["id"])
    assert final["validation_progress"]["completed"] == final["evaluation"]["sequences"]
    assert final["validation"]["identity_substitution"]["status"] == "passed"


def test_training_options_identify_only_the_ready_model(tmp_path, model):
    from fastapi import FastAPI, HTTPException
    from app.training.api import router

    loading = False

    def get_model():
        if loading:
            raise HTTPException(503, "Model is loading")
        return model

    app = FastAPI()
    app.include_router(router(get_model, threading.Lock(), RunStore(tmp_path)))
    with TestClient(app) as client:
        # An injected/unsupported model must not be mistaken for the default.
        assert client.get('/training/options').json()['model_repository'] is None
        model.cfg.model_name = 'gpt2'
        ready = client.get('/training/options').json()
        assert ready['model_repository'] == 'openai-community/gpt2'
        assert ready['readiness'] == 'ready'
        loading = True
        preparing = client.get('/training/options').json()
        assert preparing['model_repository'] is None
        assert preparing['readiness'] != 'ready'
