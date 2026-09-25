"""Portable run export. Uses only the standard library; never loads a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import zipfile


def export_run(root: Path, run_id: str, output: Path):
    root = root.resolve()
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise ValueError('Invalid run ID')
    with sqlite3.connect(f'{root.as_uri()}/runs.sqlite3?mode=ro', uri=True) as db:
        db.execute('BEGIN')
        row = db.execute('SELECT body FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('Run not found')
        run = json.loads(row[0])
        if run['status'] not in {'completed', 'failed', 'cancelled', 'interrupted'}:
            raise ValueError('Stop the run before exporting it')
        if not run.get('checkpoint'):
            raise ValueError('This run has no saved dictionary checkpoint')
        metrics = [json.loads(row[0]) for row in db.execute('SELECT body FROM metrics WHERE run_id=? ORDER BY seq', (run_id,))]
        interp_jobs = [json.loads(row[0]) for row in db.execute('SELECT body FROM interp_jobs') if json.loads(row[0]).get('run_id') == run_id]
    directory = root / run_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Run directory is unavailable or is a symlink')
    checkpoint = root / run['checkpoint']['path']
    if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(directory):
        raise ValueError('Checkpoint is outside the run directory')
    if output.resolve().is_relative_to(directory):
        raise ValueError('Save the export outside the run directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Output already exists; choose a new filename')
    # Preserve the whole run configuration, including user-supplied training text.
    # Include current checkpoint and all examples/labels; omit obsolete checkpoints.
    entries = []
    temp = tempfile.NamedTemporaryFile(dir=output.parent, suffix='.partial', delete=False)
    temp.close()
    try:
        with zipfile.ZipFile(temp.name, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for folder, dirs, names in os.walk(directory, followlinks=False):
                folder = Path(folder)
                if any((folder / name).is_symlink() for name in dirs + names):
                    raise ValueError('Run contains symlinks; refusing to export external files')
                dirs[:] = sorted(name for name in dirs if not name.startswith('.') and
                                  (not name.startswith('checkpoint-') or folder / name == checkpoint))
                for name in sorted(names):
                    path = folder / name
                    if not path.is_file() or name.startswith('.'):
                        continue
                    relative = path.relative_to(root).as_posix()
                    digest = hashlib.sha256()
                    size = 0
                    with path.open('rb') as source, archive.open(relative, 'w', force_zip64=True) as target:
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            digest.update(chunk); size += len(chunk); target.write(chunk)
                    entries.append(dict(path=relative, bytes=size, sha256=digest.hexdigest()))
            metadata = dict(format='mechlens-run', version=1, run=run, metrics=metrics,
                            interp_jobs=interp_jobs, files=entries)
            archive.writestr('import.json', json.dumps(metadata, allow_nan=False))
        # Atomic publication without overwriting an existing export.
        os.link(temp.name, output)
    finally:
        os.unlink(temp.name)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export a stopped training run for Mechlens Cloud')
    repo_data = Path(__file__).resolve().parents[1] / 'data'
    default_data = repo_data if repo_data.is_dir() else Path.home() / '.local/share/mechlens'
    parser.add_argument('--training-dir', type=Path, default=Path(os.environ.get('MECHLENS_TRAINING_DIR', str(Path(os.environ.get('MECHLENS_DATA_DIR', str(default_data))) / 'training'))))
    parser.add_argument('--list', action='store_true', help='list saved runs without loading a model')
    parser.add_argument('--run', help='ID of the stopped training run')
    parser.add_argument('--output', type=Path, help='new ZIP filename')
    args = parser.parse_args(argv)
    try:
        if args.list:
            with sqlite3.connect(f'{args.training_dir.resolve().as_uri()}/runs.sqlite3?mode=ro', uri=True) as db:
                for row in db.execute('SELECT body FROM runs ORDER BY rowid DESC'):
                    run = json.loads(row[0])
                    print(f"{run['id']}  {run['status']}  layer {run['config']['layer']}  {'checkpoint saved' if run.get('checkpoint') else 'no checkpoint'}")
            return
        if not args.run or not args.output:
            parser.error('use --list or provide both --run and --output')
        path = export_run(args.training_dir, args.run, args.output)
        print(f'Exported {path} ({path.stat().st_size:,} bytes)')
        print('Includes training text, examples, labels, and checkpoint state. Import it in Cloud → Dictionaries.')
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'Export failed: {exc}\n')
