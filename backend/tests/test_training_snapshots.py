"""Cloud snapshots are immutable copies and cannot race active run operations."""
import io
import json
import zipfile

from app.service import jobs
from app.service.jobs import JobRecord
from test_training_management import client_for


def completed(store):
    run = store.create({'layer': 0, 'features': 8})
    folder = store.root / run['id'] / 'checkpoint-64'
    folder.mkdir(parents=True)
    (folder / 'cfg.json').write_text('{}')
    (folder / 'sae_weights.safetensors').write_bytes(b'weights')
    store.update(run['id'], status='completed', checkpoint={'artifact_id': 'a' * 64,
        'path': str(folder.relative_to(store.root))})
    return run['id']


def test_snapshot_survives_source_mutation_and_retry(tmp_path, monkeypatch):
    client, store, _ = client_for(tmp_path, monkeypatch)
    run_id = completed(store)
    base = f'/training/runs/{run_id}'
    key = client.get(base + '/snapshot').json()['snapshot_key']
    first = client.get(base + '/exports/' + key)
    assert first.status_code == 200
    assert zipfile.ZipFile(io.BytesIO(first.content)).read('import.json')
    store.update(run_id, examples={'collected': True})
    new_key = client.get(base + '/snapshot').json()['snapshot_key']
    assert key != new_key
    assert client.get(base + '/exports/' + key).content == first.content
    second = client.get(base + '/exports/' + new_key)
    meta = json.loads(zipfile.ZipFile(io.BytesIO(second.content)).read('import.json'))
    assert meta['run']['examples'] == {'collected': True}
    assert client.delete(base + '/exports/' + key).status_code == 200
    assert client.get(base + '/exports/' + key).status_code == 409


def test_snapshot_rejects_active_jobs_and_unfinished_training(tmp_path, monkeypatch):
    client, store, _ = client_for(tmp_path, monkeypatch)
    run_id = completed(store)
    base = f'/training/runs/{run_id}'
    key = client.get(base + '/snapshot').json()['snapshot_key']
    jobs.JOBS['active'] = JobRecord(id='active', training_run_id=run_id)
    assert client.get(base + '/snapshot').status_code == 409
    assert client.get(base + '/exports/' + key).status_code == 409
    jobs.JOBS.clear()
    store.update(run_id, status='running')
    assert client.get(base + '/snapshot').status_code == 422
    assert client.get(base + '/exports/' + key).status_code == 409


def test_interpretation_changes_snapshot_without_changing_weights(tmp_path, monkeypatch):
    client, store, _ = client_for(tmp_path, monkeypatch)
    run_id = completed(store)
    base = f'/training/runs/{run_id}/snapshot'
    first = client.get(base).json()
    job = store.create_interp_job(run_id, 'a' * 64, [1], 'local', 'test')
    store.update_interp_job(job['id'], status='completed', records=[{'label': 'new label'}])
    second = client.get(base).json()
    assert first['artifact_id'] == second['artifact_id']
    assert first['snapshot_key'] != second['snapshot_key']


def test_partial_result_can_be_exported_without_model_work(tmp_path, monkeypatch):
    client, store, _ = client_for(tmp_path, monkeypatch)
    run_id = completed(store)
    store.update(run_id, status='interrupted')
    key = client.get(f'/training/runs/{run_id}/snapshot').json()['snapshot_key']
    assert client.get(f'/training/runs/{run_id}/exports/{key}').status_code == 200
    assert client.get(f'/training/runs/{run_id}/exports/not-a-key').status_code == 409
