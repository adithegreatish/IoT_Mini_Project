# Smart Parking Management & Indian HSRP-Style Registration-Plate Recognition

A modular, robust, edge-friendly computer vision prototype designed for **Windows 10/11 development** and target deployment on a **Raspberry Pi 5 (8 GB RAM)** with a standard **Zebronics USB webcam** and HDMI local display.

---

## Key Features

1. **High-Reliability Occupancy Detection:** Real-time state machine classifying parking bays into `FREE`, `OCCUPIED`, or `UNKNOWN`. Plate detection and OCR are supplementary capabilities and **never** a single point of failure for occupancy.
2. **Interactive Web-Based Calibration:** Visual GUI to capture camera reference frames, draw, edit, move, validate, and persist polygon parking slots with coordinate normalization across arbitrary camera resolutions.
3. **Optional Plate Detection & HSRP-Style Cue Analysis:** Detects license plate crops, checks image quality, and inspects visible cues (blue IND strip, aspect ratio, fonts) providing cautious classifications (`HSRP_STYLE_CONSISTENT`, `NON_HSRP_OR_UNCERTAIN`, `INSUFFICIENT_IMAGE_QUALITY`).
4. **Offline & Self-Contained:** Built with zero runtime cloud or CDN dependencies; bundled frontend assets, local SQLite persistence with audit logging, and full offline testing modes.
5. **Multi-Level Automated Diagnostics:** Includes startup health checks, in-app one-click self-tests with automated JSON/Markdown reports, and a comprehensive pytest suite with synthetic camera mocks.
6. **Cross-Platform Parity:** Clean separation of concerns with an ONNX-runtime backend optimized for both x86_64 Windows and ARM64 Raspberry Pi 5.

---

## Quick Start (Windows Development)

### 1. Prerequisites & Environment Setup
Requires Python 3.10 to 3.12:
```powershell
# Create and activate virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Optional: set project root (auto-discovered if omitted)
$env:SMART_PARKING_ROOT = "C:\Users\adity\Downloads\IoT"

# Install Windows dependencies with GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-windows.txt
```

### 2. Dataset Acquisition & Model Pipeline
```powershell
# Acquire real COCO and Indian license plate datasets
python scripts/acquire_datasets.py

# Train vehicle detector (YOLOv8) & license plate detector
python scripts/train_vehicle_detector.py --epochs 15
python scripts/train_plate_detector.py --epochs 5

# Export deployment ONNX models & verify numerical parity
python scripts/export_deployment_models.py

# Evaluate models on held-out test datasets
python scripts/evaluate_models.py
```

### 3. Launch the Application
```powershell
python -m app.main --config configs/windows.yaml
```
Open your browser at `http://127.0.0.1:8000` to access the technical dashboard.

### 4. Run Developer Test Suite
```powershell
python -m pytest
```

---

## Raspberry Pi 5 Deployment

For complete instructions regarding ARM64 setup, systemd services, and HDMI kiosk mode, see [`docs/RASPBERRY_PI_SETUP.md`](docs/RASPBERRY_PI_SETUP.md).

---

## Documentation Index

- [Architecture & Layer Boundaries](docs/ARCHITECTURE.md)
- [Camera Calibration & Slot Setup](docs/CAMERA_CALIBRATION.md)
- [Testing & In-App Diagnostics](docs/TESTING_DIAGNOSTICS.md)
- [Windows Setup Guide](docs/WINDOWS_SETUP.md)
- [Raspberry Pi 5 Setup Guide](docs/RASPBERRY_PI_SETUP.md)
- [Datasets & Annotation Schema](docs/DATASETS.md)
- [Model Training & Evaluation](docs/TRAINING.md)
- [Privacy & Data Retention](docs/PRIVACY.md)
- [Security Considerations](docs/SECURITY.md)
- [Acceptance Gates & Testing](docs/ACCEPTANCE_TESTS.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

---

## Licence

MIT License. See [`LICENSES.md`](LICENSES.md) for full details and third-party notices.
