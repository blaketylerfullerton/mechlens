#!/usr/bin/env bash
# Starts backend + frontend together, locally, no Docker.
#   ./dev.sh
# Ctrl+C stops both.
set -e
cd "$(dirname "$0")"

cleanup() {
  echo "stopping..."
  kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT INT TERM

source .venv/bin/activate

(cd backend && python -m uvicorn app.main:app --reload --port 8000) &
(cd frontend && npm run dev) &

wait
