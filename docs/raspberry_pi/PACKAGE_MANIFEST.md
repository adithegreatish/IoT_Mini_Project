# Raspberry Pi 5 Standalone Deployment Package Manifest

**Archive Path:** `dist/smart_parking_raspberry_pi_deployment.zip` (and `dist/raspberry_pi.zip`)  
**Unpacked Staging Directory:** `dist/raspberry_pi/`  
**Package Size:** 24.65 MB (compressed) / 28.66 MB (uncompressed)  
**Total Included Files:** 96 files  
**Target Hardware:** Raspberry Pi 5 (8 GB RAM, ARM64 Cortex-A76)  
**Target OS:** Raspberry Pi OS 64-bit (Debian 12 Bookworm aarch64)  
**Packaging Date:** 2026-10-12  

---

## 1. Package Purpose & Design Rules

This deployment package contains exclusively the files genuinely required to run, configure, calibrate, monitor, and maintain the Smart Parking & Indian HSRP Management System on Raspberry Pi hardware.

### Exclusions (Keeps Archive Under 25 MB):
- **`.venv` & `.venv-pi`**: Excluded. Linux virtual environment `.venv-pi` is created natively on the Pi by `install.sh`.
- **`.git` & `.pytest_cache`**: Excluded.
- **`data/raw/` & `data/processed/`**: Excluded. Eliminates multi-gigabyte training images and dataset archives.
- **`models/training_runs/`**: Excluded. Eliminates heavy training checkpoints, runs, and logs.
- **`yolov8n.pt`**: Excluded. Production YOLO models are deployed as ONNX files.
- **`*.bak` & `__pycache__`**: Excluded.

---

## 2. Directory Structure in Archive

```text
smart_parking_raspberry_pi_deployment.zip
├── LICENSES.md
├── README.md
├── pyproject.toml
├── requirements-pi.txt
├── requirements-windows.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   ├── camera/
│   ├── config/
│   ├── database/
│   │   └── schema.sql
│   ├── detection/
│   ├── diagnostics/
│   ├── infrastructure/
│   ├── ocr/
│   ├── parking/
│   ├── plates/
│   └── services/
├── configs/
│   ├── datasets.yaml
│   ├── deployment.yaml
│   ├── evaluation.yaml
│   ├── raspberry_pi.yaml
│   ├── training.yaml
│   └── windows.yaml
├── data/
│   ├── README.md
│   └── calibration.json
├── deployment/
│   └── raspberry_pi/
│       ├── check_models.sh
│       ├── install.sh
│       ├── kiosk.sh
│       ├── rollback.sh
│       ├── smart-parking.service
│       └── start.sh
├── docs/
│   └── raspberry_pi/
│       ├── ACCEPTANCE_CHECKLIST.md
│       ├── FINAL_DEPLOYMENT_READINESS_REPORT.md
│       ├── PERFORMANCE_REPORT.md
│       ├── README.md
│       ├── TROUBLESHOOTING.md
│       ├── model_selection_report.md
│       └── project_readiness_report.md
├── models/
│   ├── README.md
│   ├── manifest.json
│   ├── occupancy_classifier.onnx
│   ├── occupancy_classifier.pt
│   ├── occupancy_classifier_baseline_cnrpark.onnx
│   ├── occupancy_classifier_baseline_cnrpark.pt
│   ├── plate_detector.onnx
│   └── vehicle_detector.onnx
├── scripts/
│   ├── benchmark_inference.py
│   ├── export_deployment_models.py
│   ├── preflight_raspberry_pi.py
│   └── train_and_evaluate_all_approaches.py
├── tests/
│   ├── integration/
│   └── unit/
└── web/
    ├── static/
    │   ├── css/styles.css
    │   └── js/app.js
    └── templates/
        └── index.html
```

---

## 3. Included Model Manifest & Checksums

| File Path | Architecture | Format | Size | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- |
| `models/occupancy_classifier.onnx` | Winning Approach D | ONNX | 0.95 MB | `e42e18e40b9337fde295c2e269845bfb9cb3df54d341971a03c6ae8b5d80105f` |
| `models/vehicle_detector.onnx` | YOLOv8n ($640 \times 640$) | ONNX | 12.18 MB | `8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc` |
| `models/plate_detector.onnx` | YOLOv8n-Tiny ($640 \times 640$) | ONNX | 12.18 MB | `efddfbc0d555a374389cb7847db06b776a22fed2b706eac56ce51596bae5cea4` |
| `models/occupancy_classifier_baseline_cnrpark.onnx` | Baseline Approach A (Rollback) | ONNX | 0.95 MB | `835781c13fb70a5e80505ab29c324a37aceac5803ec218a5ed249ef789e3e8bc` |
| `models/occupancy_classifier_baseline_cnrpark.pt` | Baseline Approach A (PyTorch) | PyTorch | 0.96 MB | `5040dc0cfc9edda3c3f698444bb8e9ed039d20750f96683f34b7f255a6b20884` |
| `models/occupancy_classifier.pt` | Approach D (PyTorch Weights) | PyTorch | 0.96 MB | `77f10bfe8ff8df37ba42a4b868e8e78553f1917f6e2b978b2762a632b56e6d1f` |

---

## 4. Rollback Compatibility Verification

The rollback script `deployment/raspberry_pi/rollback.sh` was inspected and verified:
- **Compatible Model Format:** The script checks for and directly restores `models/occupancy_classifier_baseline_cnrpark.onnx`.
- **Pure CPU Operation:** Restoring this baseline ONNX file requires zero PyTorch libraries, enabling immediate emergency rollback on Raspberry Pi OS without installing PyTorch.
- **Integrity Validation:** Both baseline ONNX and baseline PyTorch checkpoint files are included and verified by `deployment/raspberry_pi/check_models.sh`.

---

## 5. Verification & Integrity Checklist

- [x] All 4 deployment ONNX sessions load cleanly with `onnxruntime` CPU provider.
- [x] All model SHA-256 hashes match `models/manifest.json`.
- [x] All shell scripts in `deployment/raspberry_pi/` normalized to Unix LF line endings.
- [x] SQLite schema verified: tables `parking_slots`, `slot_reservations`, `occupancy_events` create successfully.
- [x] Preflight diagnostic script `scripts/preflight_raspberry_pi.py` executes inside unzipped package.
- [x] All 13 readiness tests pass inside unzipped package.
- [x] Zero raw datasets, `.venv`, or temporary caches present in archive.
