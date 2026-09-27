"""`mechlens push`: pick a stopped run, export it, upload it through serve's pairing."""
import json
import sqlite3
import zipfile

import httpx
import pytest

from app import push
from app.cloud_link import session_file

RUN, BUSY = 'a' * 32, 'b' * 32


@pytest.fixture
def training(tmp_path):
    root = tmp_path / 'training'
    (root / RUN / 'checkpoint-64').mkdir(parents=True)
    (root / RUN / 'checkpoint-64' / 'weights').write_bytes(b'saved weights')
    with sqlite3.connect(root / 'runs.sqlite3') as db:
        db.execute('CREATE TABLE runs(id TEXT, body TEXT)')
        db.execute('CREATE TABLE metrics(run_id TEXT, seq INTEGER, body TEXT)')
        db.execute('CREATE TABLE interp_jobs(id TEXT, body TEXT)')
        for run in (dict(id=RUN, status='completed', config={'layer': 3, 'features': 16384}, checkpoint={'path': RUN + '/checkpoint-64'}),
                    dict(id=BUSY, status='running', config={'layer': 4}, checkpoint=None)):
            db.execute('INSERT INTO runs VALUES (?,?)', (run['id'], json.dumps(run)))
    return root


def pair(training_dir):
    session_file().parent.mkdir(parents=True, exist_ok=True)
    session_file().write_text(json.dumps({'cloud_url': 'https://cloud.example', 'session_token': 'tok',
                                          'training_dir': str(training_dir)}))


def test_only_stopped_runs_with_a_dictionary_are_offered(training):
    assert [run['id'] for run in push.saved_runs(training)] == [RUN]
    assert 'layer 3' in push.describe(push.saved_runs(training)[0])
    assert '16,384 features' in push.describe(push.saved_runs(training)[0])


def test_push_without_a_paired_serve_says_what_to_do(training, capsys):
    with pytest.raises(SystemExit):
        push.main([])
    assert 'mechlens serve --cloud' in capsys.readouterr().err


def test_picked_run_is_exported_and_uploaded_with_the_session(training, monkeypatch, capsys):
    pair(training)
    seen = {}

    def cloud(request):
        seen['auth'] = request.headers['authorization']
        seen['path'] = request.url.path
        body = request.read()
        with zipfile.ZipFile(__import__('io').BytesIO(body)) as archive:
            seen['run'] = json.loads(archive.read('import.json'))['run']['id']
        return httpx.Response(200, json={'already_imported': False, 'dictionary': {}})

    real_upload = push.upload
    monkeypatch.setattr(push, 'upload', lambda session, path: real_upload(session, path, httpx.Client(transport=httpx.MockTransport(cloud))))
    monkeypatch.setattr(push, 'choose', lambda labels: [0])
    push.main([])
    assert seen == {'auth': 'Bearer tok', 'path': '/api/gpu/dictionaries/import', 'run': RUN}
    assert 'pushed' in capsys.readouterr().out
    assert not list(training.glob('.push-*'))  # the temporary ZIP is cleaned up


def test_expired_pairing_is_explained(training, tmp_path):
    pair(training)
    zip_path = tmp_path / 'x.zip'
    zip_path.write_bytes(b'zip')
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={'detail': 'no'})))
    with pytest.raises(push.PushError, match='pair again'):
        push.upload(push.load_session(), zip_path, client)


def test_numbered_fallback_when_not_a_terminal(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda prompt: '2')
    assert push.choose_by_number(['one', 'two']) == [1]
    monkeypatch.setattr('builtins.input', lambda prompt: '1,3-4')
    assert push.choose_by_number(['a', 'b', 'c', 'd']) == [0, 2, 3]
    monkeypatch.setattr('builtins.input', lambda prompt: 'all')
    assert push.choose_by_number(['a', 'b']) == [0, 1]
    monkeypatch.setattr('builtins.input', lambda prompt: '')
    assert push.choose_by_number(['one']) == []


def test_all_pushes_every_saved_run_one_zip_each(training, monkeypatch, capsys):
    other = 'c' * 32
    (training / other / 'checkpoint-8').mkdir(parents=True)
    (training / other / 'checkpoint-8' / 'weights').write_bytes(b'more weights')
    with sqlite3.connect(training / 'runs.sqlite3') as db:
        run = dict(id=other, status='completed', config={'layer': 5}, checkpoint={'path': other + '/checkpoint-8'})
        db.execute('INSERT INTO runs VALUES (?,?)', (other, json.dumps(run)))
    pair(training)
    sent = []

    def cloud(request):
        with zipfile.ZipFile(__import__('io').BytesIO(request.read())) as archive:
            sent.append(json.loads(archive.read('import.json'))['run']['id'])
        return httpx.Response(200, json={'already_imported': False, 'dictionary': {}})

    real_upload = push.upload
    monkeypatch.setattr(push, 'upload', lambda session, path: real_upload(session, path, httpx.Client(transport=httpx.MockTransport(cloud))))
    push.main(['--all'])
    assert sorted(sent) == sorted([RUN, other])
    assert 'Done' in capsys.readouterr().out
