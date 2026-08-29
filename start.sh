#!/usr/bin/env bash
# One-command start for the KMRL Document Intelligence platform.
#
#   ./start.sh          install (first run), build the UI, serve on :8000
#   ./start.sh --dev    run the API on :8000 and the Vite dev server on :5173
#   ./start.sh --seed   start, then load the demo document library
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
MODE="${1:-}"

echo "→ Installing Python dependencies"
python3 -m pip install --quiet -r "$ROOT/backend/requirements.txt"

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "→ Installing frontend dependencies"
  (cd "$ROOT/frontend" && npm install --no-audit --no-fund)
fi

if [ "$MODE" = "--dev" ]; then
  echo "→ API   http://127.0.0.1:$PORT"
  echo "→ UI    http://127.0.0.1:5173  (hot reload, proxies /api to the backend)"
  (cd "$ROOT/backend" && python3 -m uvicorn app.main:app --reload --port "$PORT") &
  API_PID=$!
  trap 'kill $API_PID 2>/dev/null || true' EXIT
  (cd "$ROOT/frontend" && npm run dev)
  exit 0
fi

echo "→ Building the frontend"
(cd "$ROOT/frontend" && npm run build)

echo "→ Serving on http://127.0.0.1:$PORT"
(cd "$ROOT/backend" && python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT") &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

if [ "$MODE" = "--seed" ]; then
  sleep 4
  (cd "$ROOT/backend" && python3 scripts/seed_demo.py --url "http://127.0.0.1:$PORT")
fi

wait $API_PID
