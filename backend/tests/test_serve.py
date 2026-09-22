"""Installed CLI and remote boundary checks, without downloading a model."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.command import main


def test_help_is_lightweight():
    result = subprocess.run([sys.executable, '-c', '''
import sys
from app.command import main
try:
    main(['serve', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules
assert 'app.service.app' not in sys.modules
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--token-file' in result.stdout


@pytest.mark.parametrize('args', [['--host', '0.0.0.0'], ['--port', '0'], ['--port', '65536']])
def test_invalid_or_unauthenticated_remote_binding(args, monkeypatch):
    monkeypatch.delenv('MECHLENS_API_TOKEN', raising=False)
    with pytest.raises(SystemExit) as exc:
        main(['serve', *args])
    assert exc.value.code == 2


def test_serve_configures_one_worker_and_reads_token_file(tmp_path, monkeypatch, capsys):
    token_file = tmp_path / 'token'
    token_file.write_text('test-secret\n')
    monkeypatch.delenv('MECHLENS_API_TOKEN', raising=False)
    monkeypatch.delenv('MECHLENS_DATA_DIR', raising=False)
    run = Mock()
    monkeypatch.setattr('uvicorn.run', run)
    main(['serve', '--host', '0.0.0.0', '--port', '8123', '--token-file', str(token_file),
          '--data-dir', str(tmp_path / 'data')])
    assert os.environ['MECHLENS_API_TOKEN'] == 'test-secret'
    assert Path(os.environ['MECHLENS_DATA_DIR']) == tmp_path / 'data'
    assert (tmp_path / 'data').is_dir()
    run.assert_called_once_with('app.service.app:app', host='0.0.0.0', port=8123,
                                workers=1, proxy_headers=False)
    assert 'test-secret' not in capsys.readouterr().out


def test_empty_token_file_rejected(tmp_path):
    token_file = tmp_path / 'token'
    token_file.write_text('\n')
    with pytest.raises(SystemExit):
        main(['serve', '--token-file', str(token_file)])


@pytest.mark.parametrize('path', ['/health', '/stats', '/training/runs', '/circuits/analyses', '/docs', '/openapi.json'])
def test_auth_covers_entire_service(path, tmp_path, monkeypatch):
    from app.service.app import create_app
    monkeypatch.setenv('MECHLENS_API_TOKEN', 'private-test-token')
    app = create_app(model=object(), training_dir=tmp_path / 'training', circuits_dir=tmp_path / 'circuits')
    client = TestClient(app)
    for headers in [{}, {'Authorization': 'Bearer wrong'}, {'Authorization': 'Basic private-test-token'}]:
        response = client.get(path, headers=headers)
        assert response.status_code == 401
        assert response.headers['www-authenticate'] == 'Bearer'
    assert client.get('/health', headers={'Authorization': 'Bearer private-test-token'}).json()['status'] == 'ready'
    assert client.post('/trace', json={}).status_code == 401


def test_auth_rejects_duplicate_headers(tmp_path):
    from app.service.app import create_app
    app = create_app(model=object(), api_token='secret', training_dir=tmp_path/'training', circuits_dir=tmp_path/'circuits')
    client = TestClient(app)
    assert client.get('/health', headers=[('Authorization', 'Bearer secret'), ('Authorization', 'Bearer wrong')]).status_code == 401


def test_no_token_preserves_local_access(tmp_path, monkeypatch):
    from app.service.app import create_app
    monkeypatch.delenv('MECHLENS_API_TOKEN', raising=False)
    client = TestClient(create_app(model=object(), training_dir=tmp_path/'training', circuits_dir=tmp_path/'circuits'))
    assert client.get('/health').status_code == 200
