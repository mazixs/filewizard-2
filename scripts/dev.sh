#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Development launcher for FileWizard
# Runs uvicorn with auto-reload + huey consumer in the foreground.
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Auto-activate venv if present
if [[ -d ".venv" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
elif [[ -d "venv" && -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv"
fi

# Load .env if it exists
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Loaded environment from .env"
fi

# Defaults
export LOCAL_ONLY="${LOCAL_ONLY:-True}"
export SECRET_KEY="${SECRET_KEY:-dev-secret-key}"
export UPLOADS_DIR="${UPLOADS_DIR:-./uploads}"
export PROCESSED_DIR="${PROCESSED_DIR:-./processed}"
export CHUNK_TMP_DIR="${CHUNK_TMP_DIR:-./uploads/tmp}"
export ENV="${ENV:-dev}"

mkdir -p "$UPLOADS_DIR" "$PROCESSED_DIR" "$CHUNK_TMP_DIR"

echo "============================================================"
echo "Starting FileWizard (DEVELOPMENT mode)"
echo "============================================================"
echo "  API:     http://${BIND:-0.0.0.0:8000}"
echo "  Reload:  enabled"
echo "============================================================"

if ! command -v uvicorn &>/dev/null; then
    echo "ERROR: uvicorn not found. Run ./scripts/setup.sh first."
    exit 1
fi
if ! command -v huey_consumer &>/dev/null; then
    echo "ERROR: huey_consumer not found. Run ./scripts/setup.sh first."
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "$UVICORN_PID" 2>/dev/null; then
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
    fi
    if [[ -n "${HUEY_PID:-}" ]] && kill -0 "$HUEY_PID" 2>/dev/null; then
        kill -TERM "$HUEY_PID" 2>/dev/null || true
        wait "$HUEY_PID" 2>/dev/null || true
    fi
    echo "Goodbye."
    exit 0
}
trap cleanup INT TERM

# Start Uvicorn with reload
uvicorn main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --reload \
    --reload-dir app \
    &
UVICORN_PID=$!
echo "Uvicorn started (PID: $UVICORN_PID) → http://${HOST:-0.0.0.0}:${PORT:-8000}"

# Start Huey consumer
huey_consumer main.huey -w 2 &
HUEY_PID=$!
echo "Huey consumer started (PID: $HUEY_PID)"

# Wait for either child to exit
wait -n
EXIT_CODE=$?

echo ""
echo "A child process exited with code $EXIT_CODE. Stopping the other..."
cleanup
