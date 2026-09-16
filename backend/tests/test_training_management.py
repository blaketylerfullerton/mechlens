"""Training queue and deletion checks use temporary stores, never real weights."""
from types import SimpleNamespace
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.service import jobs
from app.service.jobs import JobRecord
from app.training.api import router
from app.training.store import RunStore


def client_for(tmp_path, monkeypatch):
    callbacks = []
    monkeypatch.setattr(jobs, 'JOBS', {})

    def submit(fn):
        job_id = f'job-{len(callbacks)}'
        callbacks.append(fn)
        jobs.JOBS[job_id] = JobRecord(id=job_id)
        return job_id

    monkeypatch.setattr(jobs, 'submit', submit)
    model = SimpleNamespace(cfg=SimpleNamespace(model_name='gpt2', n_layers=3, n_ctx=128,
                                               normalization_type='LN', device='cpu'))
    store = RunStore(tmp_path)
    app = FastAPI()
    app.include_router(router(lambda: model, threading.Lock(), store))
    return TestClient(app), store, callbacks


def test_batch_validation_queue_and_stop(tmp_path, monkeypatch):
    client, store, callbacks = client_for(tmp_path, monkeypatch)
    body = {'config': {'layer': 0}, 'layers': [0, 3], 'model_repository': 'openai-community/gpt2'}
    assert client.post('/training/batches', json=body).status_code == 422
    assert store.list() == []
    body['layers'] = [2, 0, 1]
    result = client.post('/training/batches', json=body)
    assert result.status_code == 202
    batch = result.json()
    assert [run['config']['layer'] for run in batch['runs']] == [0, 1, 2]
    assert len(callbacks) == 3
    assert all(job.training_run_id for job in jobs.JOBS.values())
    stopped = client.post(f"/training/batches/{batch['batch_id']}/cancel").json()
    assert all(run['status'] == 'cancelled' for run in stopped)
    for callback in callbacks:
        callback(None)
    assert all(run['checkpoint'] is None for run in store.list())


def test_deletion_protects_dependencies_and_other_files(tmp_path, monkeypatch):
    client, store, _ = client_for(tmp_path, monkeypatch)
    run = store.create({'layer': 0, 'features': 4096})
    directory = tmp_path / run['id']
    directory.mkdir()
    (directory / 'weights').write_bytes(b'12345')
    outside = tmp_path / 'keep-model'
    outside.write_text('shared model')
    assert client.delete(f"/training/runs/{run['id']}").status_code == 409
    store.update(run['id'], status='completed')
    jobs.JOBS['using'] = JobRecord(id='using', training_run_id=run['id'])
    assert client.delete(f"/training/runs/{run['id']}").status_code == 409
    assert (directory / 'weights').exists()
    jobs.JOBS['using'].training_run_id = 'unrelated-run'
    assert client.get('/training/storage').json()['total_bytes'] == 5
    assert client.delete(f"/training/runs/{run['id']}").status_code == 200
    assert not directory.exists()
    assert outside.read_text() == 'shared model'
    assert store.list() == []
    assert client.delete(f"/training/runs/{run['id']}").status_code == 404


def test_checkpoint_retention_preserves_latest_and_examples(tmp_path):
    store = RunStore(tmp_path)
    run = store.create({})
    directory = tmp_path / run['id']
    directory.mkdir()
    for name in ('checkpoint-16', 'checkpoint-32'):
        (directory / name).mkdir()
        (directory / name / 'weights').write_bytes(b'weights')
    (directory / 'examples.json').write_text('{}')
    store.prune_checkpoints(run['id'], f"{run['id']}/checkpoint-32")
    assert not (directory / 'checkpoint-16').exists()
    assert (directory / 'checkpoint-32' / 'weights').exists()
    assert (directory / 'examples.json').exists()


def test_models_reports_cache_without_touching_the_hub(tmp_path, monkeypatch):
    client, _, _ = client_for(tmp_path, monkeypatch)
    import huggingface_hub

    repos = [SimpleNamespace(repo_id='google/gemma-2-2b', repo_type='model', size_on_disk=10),
             SimpleNamespace(repo_id='gpt2', repo_type='model', size_on_disk=5),
             SimpleNamespace(repo_id='roneneldan/TinyStories', repo_type='dataset', size_on_disk=7)]
    monkeypatch.setattr(huggingface_hub, 'scan_cache_dir', lambda: SimpleNamespace(repos=repos))

    body = client.get('/training/models').json()
    sizes = {record['repository']: record for record in body['models']}
    assert sizes['google/gemma-2-2b']['downloaded'] and sizes['google/gemma-2-2b']['bytes'] == 10
    # Stored under its bare name on the hub, listed under the canonical one here.
    assert sizes['openai-community/gpt2']['bytes'] == 5
    # The dataset of the same family is not a model download.
    assert sizes['roneneldan/TinyStories-1M']['downloaded'] is False
    assert sizes['openai-community/gpt2']['loaded'] is True  # the fixture model is gpt2
    assert body['total_bytes'] == 15 and body['scan_error'] is None


def test_models_survives_an_unreadable_cache(tmp_path, monkeypatch):
    client, _, _ = client_for(tmp_path, monkeypatch)
    import huggingface_hub

    def explode():
        raise OSError('cache is gone')

    monkeypatch.setattr(huggingface_hub, 'scan_cache_dir', explode)
    body = client.get('/training/models').json()
    assert 'cache is gone' in body['scan_error']
    assert all(record['downloaded'] is False for record in body['models'])
