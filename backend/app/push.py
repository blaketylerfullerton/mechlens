"""`mechlens push`: pick a saved dictionary and send it to your Cloud workspace.

Uploads through the pairing of a running `mechlens serve --cloud`, so there is
nothing extra to sign in to. Uses only the standard library and httpx; never
loads a model.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

import httpx

from .cloud_link import session_file
from .export_run import export_run

STOPPED = {'completed', 'failed', 'cancelled', 'interrupted'}
MAX_CHOICES = 20
NOT_PAIRED = ('No paired GPU found. Start `mechlens serve --cloud https://your-cloud-url`, '
              'enter the code in your workspace, then run mechlens push again.')


class PushError(Exception):
    pass


def load_session() -> dict:
    try:
        session = json.loads(session_file().read_text())
    except (OSError, ValueError):
        raise PushError(NOT_PAIRED) from None
    if not isinstance(session, dict) or not session.get('cloud_url') or not session.get('session_token'):
        raise PushError(NOT_PAIRED)
    return session


def saved_runs(training_dir: Path) -> list[dict]:
    """Runs that can be pushed: stopped and with a saved checkpoint, newest first."""
    try:
        with sqlite3.connect(f'{training_dir.resolve().as_uri()}/runs.sqlite3?mode=ro', uri=True) as db:
            runs = [json.loads(row[0]) for row in db.execute('SELECT body FROM runs ORDER BY rowid DESC')]
    except sqlite3.Error:
        raise PushError(f'No training runs found in {training_dir}') from None
    return [run for run in runs if run.get('status') in STOPPED and run.get('checkpoint')]


def ago(seconds: float) -> str:
    for unit, size in (('d', 86400), ('h', 3600), ('m', 60)):
        if seconds >= size:
            return f'{int(seconds // size)}{unit} ago'
    return 'just now'


def describe(run: dict) -> str:
    config = run.get('config') or {}
    when = run.get('updated_at') or run.get('created_at')
    parts = [f"layer {config.get('layer', '?')}",
             f"{config['features']:,} features" if isinstance(config.get('features'), int) else None,
             run['status'], ago(time.time() - when) if isinstance(when, (int, float)) else None, run['id'][:8]]
    return '  ·  '.join(part for part in parts if part)


def choose(labels: list[str]) -> int | None:
    """Arrow keys (or j/k, Tab) to move, Enter to pick, q or Esc to cancel."""
    try:
        import termios
        import tty
    except ImportError:
        termios = None
    if termios is None or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return choose_by_number(labels)
    fd, index = sys.stdin.fileno(), 0
    saved = termios.tcgetattr(fd)

    def draw(first: bool):
        out = '' if first else f'\x1b[{len(labels)}A'
        for i, label in enumerate(labels):
            out += '\r\x1b[2K' + (f'\x1b[1m❯ {label}\x1b[0m' if i == index else f'  {label}') + '\n'
        sys.stdout.write(out)
        sys.stdout.flush()

    print('Which dictionary do you want to push?  (↑/↓ to move, Enter to push, q to cancel)')
    try:
        tty.setcbreak(fd)  # keys arrive one at a time; Ctrl+C still works
        sys.stdout.write('\x1b[?25l')  # hide the cursor while the list is up
        draw(True)
        while True:
            key = os.read(fd, 3)
            if key in (b'\x1b[A', b'k'):
                index = (index - 1) % len(labels)
            elif key in (b'\x1b[B', b'j', b'\t'):
                index = (index + 1) % len(labels)
            elif key in (b'\n', b'\r'):
                return index
            elif key in (b'q', b'\x1b'):
                return None
            draw(False)
    finally:
        sys.stdout.write('\x1b[?25h')
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def choose_by_number(labels: list[str]) -> int | None:
    for n, label in enumerate(labels, 1):
        print(f'  {n}. {label}')
    answer = input('Push which one? (number, blank to cancel) ').strip()
    if not answer:
        return None
    if not answer.isdigit() or not 1 <= int(answer) <= len(labels):
        raise PushError(f'Pick a number from 1 to {len(labels)}')
    return int(answer) - 1


def upload(session: dict, path: Path, client: httpx.Client | None = None) -> dict:
    total = path.stat().st_size
    show = sys.stderr.isatty()

    def chunks():
        sent = 0
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                sent += len(chunk)
                if show:
                    print(f'\rUploading {sent / 1e6:,.0f} / {total / 1e6:,.0f} MB', end='', file=sys.stderr, flush=True)
                yield chunk
        if show:
            print(file=sys.stderr)

    url = session['cloud_url'].rstrip('/') + '/api/gpu/dictionaries/import'
    client = client or httpx.Client(timeout=httpx.Timeout(600, connect=30), trust_env=True)
    try:
        response = client.post(url, content=chunks(), headers={
            'Authorization': 'Bearer ' + session['session_token'], 'Content-Type': 'application/zip',
            'Content-Length': str(total)})
    except httpx.HTTPError as exc:
        raise PushError(f'Could not reach {url}: {exc}. Nothing was imported; run mechlens push again.') from None
    if response.status_code == 401:
        raise PushError('The cloud no longer knows this GPU. Restart mechlens serve --cloud, pair again, then retry.')
    if response.status_code == 404:
        raise PushError('This Cloud server does not support mechlens push yet. Deploy the latest Cloud backend.')
    if not response.is_success:
        try:
            detail = response.json().get('detail')
        except ValueError:
            detail = None
        raise PushError(detail if isinstance(detail, str) else f'Cloud answered {response.status_code}')
    return response.json()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='mechlens push', description=__doc__.splitlines()[0].strip('`'))
    parser.add_argument('--run', help='push this run ID without asking')
    parser.add_argument('--training-dir', type=Path, help='defaults to the one mechlens serve is using')
    args = parser.parse_args(argv)
    try:
        session = load_session()
        training_dir = args.training_dir or (Path(session['training_dir']) if session.get('training_dir') else None)
        if training_dir is None:
            raise PushError('Pass --training-dir; the running mechlens serve did not say where its runs are.')
        runs = saved_runs(training_dir)
        if args.run:
            run = next((r for r in runs if r['id'] == args.run), None)
            if run is None:
                raise PushError('That run is not stopped with a saved dictionary (or does not exist).')
        else:
            if not runs:
                raise PushError('No stopped runs with a saved dictionary yet. Train one first.')
            if len(runs) > MAX_CHOICES:
                print(f'Showing the newest {MAX_CHOICES} of {len(runs)}; use --run ID for older ones.')
                runs = runs[:MAX_CHOICES]
            picked = choose([describe(run) for run in runs])
            if picked is None:
                print('Cancelled. Nothing was pushed.')
                return
            run = runs[picked]
        print(f'Packing {describe(run)} ...')
        # Beside the runs, so a big export stays on the same disk and never in /tmp's RAM.
        with tempfile.TemporaryDirectory(dir=training_dir, prefix='.push-') as folder:
            path = export_run(training_dir, run['id'], Path(folder) / f"mechlens-{run['id']}.zip")
            result = upload(session, path)
        where = session['cloud_url'].rstrip('/') + '/app/dictionaries'
        if result.get('already_imported'):
            print(f'Already in your workspace; nothing new was uploaded. {where}')
        else:
            print(f'Pushed. It is in Cloud → Dictionaries: {where}')
    except (PushError, ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'Push failed: {exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nCancelled. Nothing was pushed.\n')
