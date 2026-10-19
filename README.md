# Smart Parking Management & Indian HSRP-Style Recognition System

**Release Status:** **FINAL RELEASE — READY FOR DEMONSTRATION (College Submission Freeze)**  
**Target Environments:** Windows 10/11 (Development & Evaluation) and Raspberry Pi 4 / Debian Linux (Edge Deployment)  
**Primary Detector:** Frozen Production ONNX YOLOv8-Nano Baseline (`models/vehicle_detector.onnx`, SHA-256: `8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc`)  
**Automated Verification:** 114 Passed / 0 Failed (100% Pass Rate)

---

## Authoritative Documentation & Release Deliverables

- [Final Release & Verification Audit Report](docs/FINAL_RELEASE_REPORT.md) — Comprehensive release verdict, bug catalog, and model provenance.
- [Production Deployment & Operation Guide](docs/DEPLOYMENT_GUIDE.md) — Step-by-step setup, startup, shutdown, backup, and live demonstration script.
- [Security Audit & Vulnerability Mitigation](docs/SECURITY_AUDIT.md) — Access control, CORS, HTTP security headers, and path traversal defenses.
- [Acceptance Test Matrix & Execution Results](docs/ACCEPTANCE_TEST_RESULTS.md) — Full test run results across all 114 test cases.
- [Known Limitations & Operational Boundaries](docs/KNOWN_LIMITATIONS.md) — Transparent disclosure of AI, optical, and hardware boundaries.

---

## Core System Architecture

1. **Fail-Closed Occupancy Engine:** Spatial polygon IoU + hybrid ground-contact containment with 3-frame temporal hysteresis. Stale camera signals or inference exceptions automatically transition bays to `UNKNOWN`, **never falsely to `FREE`**.
2. **ACID Virtual Parking Allocation:** Transactional SQLite `BEGIN IMMEDIATE` locks preventing double-allocation and enforcing single-active-reservation per vehicle plate.
3. **Interactive Web Calibration CAD:** Real-time polygon drawing, vertex dragging, Excel-style spreadsheet grid editing, and persistence with normalized coordinates.
4. **Advisory Indian HSRP & OCR Analysis:** Visual cue inspection (aspect ratio, blue IND strip, contrast) and format-normalized Indian registration plate recognition with privacy masking.
5. **Completely Local & Offline:** Zero external cloud dependencies; bundled vanilla frontend, local SQLite WAL database, and ONNX Runtime CPU execution.

---

## Quick Start (Windows Demonstration)

### 1. Prerequisites & Virtual Environment
Requires Python 3.10 to 3.12:
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt

# Copy environment template
Copy-Item .env.example .env
```

### 2. Run Release Acceptance Tests
```powershell
# Run the complete test suite (114 tests)
python -m pytest tests/ -v
```

### 3. Launch the Application
```powershell
python -m app.main --config configs/windows.yaml
```

Open your browser at `http://127.0.0.1:8000` to interact with the technical dashboard.

### 4. Clean Shutdown
Press `Ctrl+C` in the terminal to gracefully flush database write-ahead logs, stop camera threads, and exit.

---

## 5-Minute Live Demonstration Walkthrough

1. **Dashboard Overview:** Navigate to `http://localhost:8000`. Observe live video preview with detection overlays and real-time state counters.
2. **Virtual Allocation:** Click the **Virtual Allocation** tab. Select an available bay, enter vehicle plate (e.g. `MH12AB1234`), and click **Allot Bay**. Observe instant state transition to `RESERVED`. Submit again to observe duplicate allocation rejection (HTTP 409).
3. **Calibration CAD Editor:** Switch to **Parking Calibration**. Click a bay to highlight in the Excel grid. Drag vertices to recalibrate.
4. **Vehicle & Plate Events:** Switch to **Vehicle & Plate Events**. Inspect arrival logs, masked plates (`MH12****34`), and advisory HSRP cue analysis reports.
5. **System Diagnostics:** Switch to **System Diagnostics**. Run self-diagnostics to verify sub-20ms inference latency, CPU footprint, and WAL database health.

---

## Licence
MIT License. See [`LICENSES.md`](LICENSES.md) for full details.
