#!/usr/bin/env bash
# Starts the backend, waits for it to come up, then starts the frontend.
#   ./dev.sh
# Ctrl+C tears both down.
set -e
cd "$(dirname "$0")"

BACKEND_PORT=8000
BACKEND_URL="http://localhost:${BACKEND_PORT}/docs"

backend_pid=""
frontend_pid=""

cleanup() {
  echo
  echo "stopping..."
  [ -n "$frontend_pid" ] && kill "$frontend_pid" 2>/dev/null
  [ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

source .venv/bin/activate

echo "starting backend..."
(cd backend && exec python -m uvicorn app.main:app --reload --port "$BACKEND_PORT") &
backend_pid=$!

echo "waiting for backend on :${BACKEND_PORT}..."
until curl -sf "$BACKEND_URL" -o /dev/null 2>/dev/null; do
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    echo "backend failed to start" >&2
    exit 1
  fi
  sleep 0.5
done
echo "backend ready"

echo "starting frontend..."
(cd frontend && exec npm run dev) &
frontend_pid=$!

wait -n "$backend_pid" "$frontend_pid"
