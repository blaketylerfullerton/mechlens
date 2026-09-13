"""Submit a bounded benchmark through the running backend's shared compute queue.

From backend/: python -m app.training.benchmark --tokens 32768 --layer 12
"""
import argparse
import json
import time
from pathlib import Path

import requests

from .runner import TrainingConfig
from .store import TERMINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--tokens', type=int, default=32768)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--features', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output', type=Path, default=Path('data/training-benchmark.json'))
    parser.add_argument('--evaluation-sequences', type=int, default=64)
    parser.add_argument('--compare-huggingface', action='store_true',
                        help='Load an additional CPU reference model for correctness checks')
    args = parser.parse_args()
    config = TrainingConfig(layer=args.layer, features=args.features,
        training_tokens=args.tokens, batch_size=args.batch_size,
        evaluation_sequences=args.evaluation_sequences, compare_huggingface=args.compare_huggingface)
    base = args.api.rstrip('/')
    def get(path):
        response = requests.get(base + path, timeout=30)
        response.raise_for_status()
        return response.json()
    options = get('/training/options')
    if options['readiness'] != 'ready':
        raise SystemExit('Load a supported model in the Training page before benchmarking.')
    response = requests.post(base + '/training/runs', json=config.model_dump(), timeout=30)
    response.raise_for_status()
    run_id = response.json()['id']
    print(f'Benchmark run: {run_id}', flush=True)
    start = time.monotonic()
    try:
        while True:
            run = get(f'/training/runs/{run_id}')
            if run['status'] in TERMINAL:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        requests.post(base + f'/training/runs/{run_id}/cancel', timeout=30).raise_for_status()
        raise
    report = {'validation_report': get(f'/training/runs/{run_id}/report'), 'run': run, 'metrics': get(f'/training/runs/{run_id}/metrics'),
              'wall_seconds': time.monotonic() - start, 'backend': options,
              'resources': get('/stats')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved {run["status"]} benchmark: {args.output.resolve()}')
    if run['status'] != 'completed':
        raise SystemExit(run.get('error') or run['status'])


if __name__ == '__main__':
    main()
