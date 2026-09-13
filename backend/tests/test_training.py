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


def test_real_training_roundtrip_and_frozen_model(tmp_path, model, config):
    store = RunStore(tmp_path)
    run = store.create(config.model_dump())
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    train(store, run["id"], model)
    saved = store.get(run["id"])
    assert saved["status"] == "completed"
    assert saved["tokens"] == config.training_tokens
    assert saved["evaluation"]["tokens"] > 0
    assert saved["evaluation"]["reconstruction_loss"] > 0
    assert [p.requires_grad for p in model.parameters()] == flags
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert store.metrics(run["id"])[-1]["tokens"] == 64
    sae, manifest = load_artifact(store, run["id"], model)
    assert sae.encode(torch.randn(3, 16)).shape == (3, 32)
    assert manifest["model"] == "training-test"
    assert len(manifest["artifact_id"]) == 64
    # The trained weights differ from the first-step checkpoint.
    from sae_lens import SAE
    first = SAE.load_from_disk(tmp_path / run["id"] / "checkpoint-16")
    assert not torch.equal(first.W_enc, sae.W_enc)
    # Content identity is enforced at the loading boundary.
    path = tmp_path / manifest["path"] / "cfg.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="content"):
        load_artifact(store, run["id"], model)


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
