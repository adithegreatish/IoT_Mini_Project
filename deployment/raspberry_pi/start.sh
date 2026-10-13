#!/usr/bin/env bash
# ==============================================================================
# Smart Parking & HSRP: Application Launch Script for Raspberry Pi 5
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=================================================================="
echo " Starting Smart Parking & HSRP System (Raspberry Pi Edition)"
echo " Project Root: ${REPO_ROOT}"
echo " Configuration: configs/raspberry_pi.yaml"
echo "=================================================================="

# Check virtual environment
VENV_DIR="${REPO_ROOT}/.venv-pi"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[ERROR] Virtual environment not found at ${VENV_DIR}."
    echo "Please run the installer first: bash deployment/raspberry_pi/install.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Ensure root in sys.path
export SMART_PARKING_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

# Run FastAPI / Uvicorn service
exec python app/main.py --config configs/raspberry_pi.yaml "$@"
