#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Setup local development environment for FileWizard
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "Setting up FileWizard local dev environment..."

# Create .env from example if missing
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        echo "WARNING: .env.example not found. Creating minimal .env"
        cat > .env << 'EOF'
LOCAL_ONLY=True
SECRET_KEY=
UPLOADS_DIR=./uploads
PROCESSED_DIR=./processed
OMP_NUM_THREADS=1
EOF
    fi
fi

# Create virtual environment if missing
VENV_NAME=".venv"
if [[ ! -d "$VENV_NAME" ]]; then
    echo "Creating virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
fi

# Activate venv
source "$VENV_NAME/bin/activate"

# Upgrade core tools
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
if [[ -f requirements_small.txt ]]; then
    echo "Installing dependencies from requirements_small.txt..."
    pip install -r requirements_small.txt
else
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# Create required directories
mkdir -p uploads processed uploads/tmp config

echo ""
echo "Setup complete! To start the application locally, run:"
echo "  ./run.sh"
echo ""
echo "Or for development mode (uvicorn reload):"
echo "  ./scripts/dev.sh"
