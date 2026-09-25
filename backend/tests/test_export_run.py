import json
import sqlite3
import zipfile

import pytest
from app.export_run import export_run

RUN = 'a' * 32


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'training'
    current = root / RUN / 'checkpoint-64'
    current.mkdir(parents=True)
    (current / 'weights').write_bytes(b'saved weights')
    old = root / RUN / 'checkpoint-32'
    old.mkdir()
    (old / 'weights').write_bytes(b'obsolete weights')
    (root / RUN / 'labels.json').write_text('{}')
    with sqlite3.connect(root / 'runs.sqlite3') as db:
        db.execute('CREATE TABLE runs(id TEXT, body TEXT)')
        db.execute('CREATE TABLE metrics(run_id TEXT, seq INTEGER, body TEXT)')
        db.execute('CREATE TABLE interp_jobs(id TEXT, body TEXT)')
        run = dict(id=RUN, status='completed', config={'layer': 1}, checkpoint={'path': RUN + '/checkpoint-64'})
        db.execute('INSERT INTO runs VALUES (?,?)', (RUN, json.dumps(run)))
        db.execute('INSERT INTO metrics VALUES (?,?,?)', (RUN, 1, '{"seq":1,"loss":0.2}'))
    return root, tmp_path / 'export.zip'


def test_export_preserves_source_and_selects_current_checkpoint(source):
    root, output = source
    export_run(root, RUN, output)
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read('import.json'))
        assert metadata['metrics'][0]['loss'] == .2
        assert {entry['path'] for entry in metadata['files']} == {RUN + '/checkpoint-64/weights', RUN + '/labels.json'}
        assert archive.read(RUN + '/checkpoint-64/weights') == b'saved weights'
    assert (root / RUN / 'checkpoint-32/weights').read_bytes() == b'obsolete weights'
    with pytest.raises(ValueError, match='already exists'):
        export_run(root, RUN, output)


def test_symlink_is_not_exported(source):
    root, output = source
    (root / RUN / 'outside').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='symlinks'):
        export_run(root, RUN, output)
    assert not output.exists()
    assert not list(output.parent.glob('*.partial'))


def test_active_run_is_not_exported(source):
    root, output = source
    with sqlite3.connect(root / 'runs.sqlite3') as db:
        run = json.loads(db.execute('SELECT body FROM runs').fetchone()[0])
        run['status'] = 'running'
        db.execute('UPDATE runs SET body=?', (json.dumps(run),))
    with pytest.raises(ValueError, match='Stop the run'):
        export_run(root, RUN, output)


def test_output_must_be_outside_run(source):
    root, _ = source
    with pytest.raises(ValueError, match='outside'):
        export_run(root, RUN, root / RUN / 'recursive.zip')
