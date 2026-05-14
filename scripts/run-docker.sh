#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Docker launcher for FileWizard
# Builds and runs the application via docker-compose.
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TARGET="${1:-small-final}"
PORT="${2:-6969}"

echo "============================================================"
echo "Starting FileWizard via Docker Compose"
echo "============================================================"
echo "  Target: $TARGET"
echo "  Port:   $PORT"
echo "============================================================"

# Validate target
case "$TARGET" in
    cuda-final|full-final|small-final)
        ;;
    *)
        echo "ERROR: Unknown target '$TARGET'"
        echo "Valid targets: cuda-final, full-final, small-final"
        exit 1
        ;;
esac

# Ensure data directories exist
mkdir -p config uploads_data processed_data

# Export for docker-compose.yml substitution
export TARGET PORT

# Build and start
if command -v docker-compose &>/dev/null; then
    docker-compose build
    docker-compose up -d
else
    docker compose build
    docker compose up -d
fi

echo ""
echo "FileWizard is starting..."
echo "  Logs:   docker compose logs -f  (or docker-compose logs -f)"
echo "  Health: curl http://localhost:$PORT/health"
echo ""
echo "To stop: docker compose down"
