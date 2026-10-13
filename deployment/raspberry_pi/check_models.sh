#!/usr/bin/env bash
# ==============================================================================
# Model Integrity and Checksum Verification Script for Raspberry Pi 5
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=================================================================="
echo " Smart Parking & HSRP: Model Integrity & Checksum Verification"
echo " Target Root: ${REPO_ROOT}"
echo "=================================================================="

MODELS_DIR="${REPO_ROOT}/models"
FAILED=0

# Expected SHA-256 Checksums
EXPECTED_OCC_SHA="e42e18e40b9337fde295c2e269845bfb9cb3df54d341971a03c6ae8b5d80105f"
EXPECTED_VEH_SHA="8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc"
EXPECTED_PLT_SHA="efddfbc0d555a374389cb7847db06b776a22fed2b706eac56ce51596bae5cea4"
EXPECTED_BASE_ONNX_SHA="835781c13fb70a5e80505ab29c324a37aceac5803ec218a5ed249ef789e3e8bc"
EXPECTED_BASE_PT_SHA="5040dc0cfc9edda3c3f698444bb8e9ed039d20750f96683f34b7f255a6b20884"

verify_model() {
    local file_path="$1"
    local expected_hash="$2"
    local model_name="$3"
    local is_required="$4"

    if [ ! -f "${file_path}" ]; then
        if [ "${is_required}" = "true" ]; then
            echo -e "\033[31m[FAIL]\033[0m Missing REQUIRED model: ${model_name} (${file_path})"
            FAILED=$((FAILED + 1))
        else
            echo -e "\033[33m[WARN]\033[0m Optional model not found: ${model_name} (${file_path})"
        fi
        return
    fi

    local actual_hash
    actual_hash="$(sha256sum "${file_path}" | awk '{print $1}')"

    if [ "${actual_hash}" = "${expected_hash}" ]; then
        local size_mb
        size_mb="$(du -h "${file_path}" | awk '{print $1}')"
        echo -e "\033[32m[PASS]\033[0m ${model_name} verified (${size_mb}, SHA-256: ${actual_hash:0:16}...)"
    else
        echo -e "\033[31m[FAIL]\033[0m Checksum MISMATCH for ${model_name}!"
        if head -n 1 "${file_path}" 2>/dev/null | grep -q "^version https://git-lfs"; then
            echo "       >>> File is a Git LFS pointer text file! Real binary weights were not downloaded."
            echo "       >>> Fix by running: 'sudo apt install -y git-lfs && git lfs pull'"
        fi
        echo "       Expected: ${expected_hash}"
        echo "       Actual:   ${actual_hash}"
        FAILED=$((FAILED + 1))
    fi
}

# Auto-pull Git LFS if pointer files are detected
if [ -f "${MODELS_DIR}/occupancy_classifier.onnx" ] && head -n 1 "${MODELS_DIR}/occupancy_classifier.onnx" 2>/dev/null | grep -q "^version https://git-lfs"; then
    echo -e "\033[33m[INFO] Git LFS pointer files detected in models/. Attempting 'git lfs pull'...\033[0m"
    if command -v git-lfs &> /dev/null; then
        (cd "${REPO_ROOT}" && git lfs install && git lfs pull) || true
    else
        echo -e "\033[33m[WARN] git-lfs is not installed yet. Real model binaries are missing.\033[0m"
    fi
fi

echo ""
echo "Verifying ONNX Deployment Models..."
verify_model "${MODELS_DIR}/occupancy_classifier.onnx" "${EXPECTED_OCC_SHA}" "Parking Occupancy Net (Winning Approach D)" "true"
verify_model "${MODELS_DIR}/vehicle_detector.onnx" "${EXPECTED_VEH_SHA}" "Vehicle Detector (YOLOv8n)" "true"
verify_model "${MODELS_DIR}/plate_detector.onnx" "${EXPECTED_PLT_SHA}" "Plate Detector (YOLOv8n-Tiny)" "false"

echo ""
echo "Verifying Rollback Checkpoints..."
verify_model "${MODELS_DIR}/occupancy_classifier_baseline_cnrpark.onnx" "${EXPECTED_BASE_ONNX_SHA}" "Rollback Baseline ONNX (ARM64 CPU)" "false"
verify_model "${MODELS_DIR}/occupancy_classifier_baseline_cnrpark.pt" "${EXPECTED_BASE_PT_SHA}" "Rollback Baseline PyTorch Checkpoint" "false"

echo ""
if [ ${FAILED} -eq 0 ]; then
    echo -e "\033[32m>>> ALL DEPLOYMENT MODELS VERIFIED SUCCESSFULLY! <<<\033[0m"
    exit 0
else
    echo -e "\033[31m>>> VERIFICATION FAILED: ${FAILED} error(s) detected. Please re-copy model files. <<<\033[0m"
    exit 1
fi
