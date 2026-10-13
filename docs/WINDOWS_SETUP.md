# Windows Development Setup & Execution Guide

## 1. Prerequisites
- **Operating System:** Windows 10 or 11 (64-bit)
- **Python Version:** Python 3.10 to 3.12 (standard official installer from python.org)
- **GPU (Recommended for Training):** NVIDIA GPU with CUDA 12.x support (RTX series or GTX 1660+) or CPU fallback.
- **Hardware:** Standard USB 2.0/3.0 port connected to a Zebronics USB webcam (or any standard UVC webcam).

---

## 2. Centralized Absolute Path Configuration (`SMART_PARKING_ROOT`)

The project uses a single source of truth for all filesystem paths (`app.infrastructure.paths.PathManager`).
All datasets, models, checkpoints, databases, logs, and reports resolve to absolute paths derived from this root.

### Setting `SMART_PARKING_ROOT` in PowerShell:
```powershell
# Set for current session:
$env:SMART_PARKING_ROOT = "C:\Users\adity\Downloads\IoT"

# Or persist permanently across sessions:
[System.Environment]::SetEnvironmentVariable("SMART_PARKING_ROOT", "C:\Users\adity\Downloads\IoT", "User")
```

If `$env:SMART_PARKING_ROOT` is not configured, the system uses a deterministic repository-structure discovery mechanism (locating `pyproject.toml` and marker directories) ensuring complete independence from the terminal's Current Working Directory (CWD).

---

## 3. Environment Installation

```powershell
# 1. Create a virtual environment
py -3.12 -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install CUDA-enabled PyTorch (for NVIDIA GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Install project dependencies
pip install -r requirements-windows.txt
```

---

## 4. End-to-End Workflow Commands

### A. Acquire & Prepare Datasets
Downloads genuine COCO128 vehicle dataset and Indian license plate datasets, validates bounding box annotations, and generates absolute path configurations (`data/vehicles.yaml`, `data/plates.yaml`, `data/splits/test_manifest.json`):
```powershell
python scripts/acquire_datasets.py
```

### B. Train Vehicle Detector
Trains a YOLOv8 detector on vehicle classes (`car`, `motorcycle`, `bus`, `truck`) using GPU acceleration, validates checkpoints, and saves `models/vehicle_detector.pt`:
```powershell
python scripts/train_vehicle_detector.py --epochs 15 --batch 16
```

### C. Train License Plate Detector
Trains a high-resolution plate localization model on 1,812 annotated plate samples with real bounding box coordinates:
```powershell
python scripts/train_plate_detector.py --epochs 5 --batch 16
```

### D. Export Models & Verify ONNX Numerical Parity
Exports PyTorch weights to ONNX format and verifies that maximum tensor error is within 0.01 tolerance on test images:
```powershell
python scripts/export_deployment_models.py
```

### E. Run Held-Out Test Evaluation
Evaluates vehicle detector, plate detector, and EasyOCR on the held-out test splits, recording mAP50, mAP50-95, precision, recall, and real ONNX runtime latencies:
```powershell
python scripts/evaluate_models.py
```

### F. Run the Full Test Suite
Runs all 46 unit and integration tests from any directory:
```powershell
python -m pytest
```

### G. Launch the Application
Starts the FastAPI server with live camera capture, real-time ONNX vehicle detection, and web dashboard:
```powershell
python -m app.main --config configs/windows.yaml
```
Open your browser at `http://127.0.0.1:8000` to view the technical dashboard.
