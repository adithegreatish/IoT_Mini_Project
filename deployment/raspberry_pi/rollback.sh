#!/usr/bin/env bash
# ==============================================================================
# Smart Parking & HSRP: Model Rollback Script
# Reverts active occupancy model to the validated baseline checkpoint.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"

echo "=================================================================="
echo " Smart Parking Model Rollback Utility"
echo "=================================================================="

BASELINE_ONNX="${MODELS_DIR}/occupancy_classifier_baseline_cnrpark.onnx"
BASELINE_PT="${MODELS_DIR}/occupancy_classifier_baseline_cnrpark.pt"
ACTIVE_PT="${MODELS_DIR}/occupancy_classifier.pt"
ACTIVE_ONNX="${MODELS_DIR}/occupancy_classifier.onnx"

if [ ! -f "${BASELINE_ONNX}" ] && [ ! -f "${BASELINE_PT}" ]; then
    echo "[ERROR] Neither baseline ONNX (${BASELINE_ONNX}) nor baseline checkpoint (${BASELINE_PT}) found."
    exit 1
fi

echo "Creating backup of current active models..."
if [ -f "${ACTIVE_PT}" ]; then
    cp "${ACTIVE_PT}" "${ACTIVE_PT}.backup_$(date +%Y%m%d_%H%M%S)" || true
fi
if [ -f "${ACTIVE_ONNX}" ]; then
    cp "${ACTIVE_ONNX}" "${ACTIVE_ONNX}.backup_$(date +%Y%m%d_%H%M%S)" || true
fi

if [ -f "${BASELINE_ONNX}" ]; then
    echo "Restoring baseline ONNX model directly (ARM64 CPU compatible, no PyTorch needed)..."
    cp "${BASELINE_ONNX}" "${ACTIVE_ONNX}"
    if [ -f "${BASELINE_PT}" ]; then
        cp "${BASELINE_PT}" "${ACTIVE_PT}" || true
    fi
    echo "[OK] Successfully restored baseline ONNX occupancy model!"
else
    echo "Baseline ONNX not found. Attempting PyTorch re-export..."
    cp "${BASELINE_PT}" "${ACTIVE_PT}"
    VENV_DIR="${REPO_ROOT}/.venv-pi"
    if [ -d "${VENV_DIR}" ]; then
        source "${VENV_DIR}/bin/activate"
    fi

    python -c "
import torch
from pathlib import Path
from scripts.train_and_evaluate_all_approaches import LightweightOccupancyNet
from app.infrastructure.paths import paths

model = LightweightOccupancyNet()
model.load_state_dict(torch.load('${BASELINE_PT}', weights_only=True))
model.eval()

dummy_input = torch.randn(1, 3, 150, 150)
torch.onnx.export(
    model,
    dummy_input,
    str(paths.resolve('models/occupancy_classifier.onnx')),
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
)
print('Successfully restored and re-exported baseline occupancy model!')
"
fi

echo "=================================================================="
echo " Rollback completed successfully."
echo " Restart the application using: bash deployment/raspberry_pi/start.sh"
echo "=================================================================="
