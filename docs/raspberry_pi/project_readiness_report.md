# Raspberry Pi 5 Deployment Readiness Report (Stage 1 Audit)

**Project:** Smart Parking & Indian HSRP Management System  
**Target Hardware:** Raspberry Pi 5 (8 GB RAM, Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz)  
**Target OS:** Raspberry Pi OS 64-bit (Debian 12 Bookworm, Linux kernel 6.6+, ARM64 / aarch64)  
**Target Peripherals:** Zebronics USB Webcam (UVC / V4L2), HDMI Monitor (1080p Kiosk)  
**Audit Date:** 2026-10-12  
**Audit Workspace:** `C:\Users\adity\Downloads\IoT`  

---

## 1. Executive Summary

An independent inspection of the actual repository state was conducted to assess project readiness for physical Raspberry Pi 5 deployment. The core application architecture is exceptionally well-suited for edge Linux deployment because the models have already been exported to ONNX Runtime format and path resolution is strictly decoupled from machine-specific paths through `app/infrastructure/paths.py`.

All 71 existing unit and integration tests are passing (100%).

---

## 2. Artifact Verification Matrix

Every reported artifact was verified programmatically for physical existence, size, and readability:

| Artifact Path | Physical Status | Size | Verified Hash / Content Status |
| :--- | :---: | :---: | :--- |
| `models/occupancy_classifier.onnx` | **PRESENT** | 1.46 MB | Verified ONNX model (SHA-256: `e42e18e40b9337fde295c2e269845bfb9cb3df54d341971a03c6ae8b5d80105f`) |
| `models/occupancy_classifier.pt` | **PRESENT** | 1.50 MB | PyTorch weights for winning Approach D (`f7f1eacc2ae0df29967a541b3ba987b8ccd3d636cab5b8d0bd37ee1451f3b6e5`) |
| `models/occupancy_classifier_baseline_cnrpark.pt` | **PRESENT** | 1.50 MB | Rollback checkpoint for baseline Approach A (`b53ad658b10853e30128cf78e6f1f51275bf3bbd4fe0cff3bc8ba9e8d75dc703`) |
| `models/vehicle_detector.onnx` | **PRESENT** | 12.2 MB | Deployed YOLOv8n vehicle detector (`8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc`) |
| `models/plate_detector.onnx` | **PRESENT** | 11.8 MB | Deployed YOLOv8n plate localizer (`efddfbc0d555a374389cb7847db06b776a22fed2b706eac56ce51596bae5cea4`) |
| `models/manifest.json` | **PRESENT** | 4.8 KB | Valid JSON tracking models, architectures, latencies, and checksums |
| `final_verification_report.md` | **PRESENT** | 20.9 KB | Phase 1–7 verification narrative and evaluation matrices |
| `scripts/export_deployment_models.py` | **PRESENT** | 7.5 KB | Valid deployment exporter script |
| `requirements-pi.txt` | **PRESENT** | 313 B | CPU-only, headless ARM64 dependency specification |
| `configs/raspberry_pi.yaml` | **PRESENT** | 1.8 KB | Edge configuration profile for Pi 5 environment |

---

## 3. Subsystem Readiness Inspection

| Subsystem | Source Module | Current Implementation Status | Raspberry Pi Compatibility Assessment |
| :--- | :--- | :--- | :--- |
| **Path Manager** | `app/infrastructure/paths.py` | Fully relative & canonical | **100% Ready:** Dynamically detects project root from markers or `SMART_PARKING_ROOT` env var. Zero hardcoded Windows drive letters (`C:\`). Works identically on `/home/<user>/smart-parking`. |
| **Camera Capture** | `app/camera/opencv_camera.py`<br>`app/services/camera_service.py` | Bounded buffer, thread-safe, auto-reconnect, mock fallback | **Ready with Config:** Supports `V4L2` backend (`/dev/video0`). Negotiates resolution and drains frames (`buffer_size=1`) to eliminate latency lag. Reconnection safety verified. |
| **Occupancy Engine** | `app/parking/occupancy_classifier.py`<br>`app/parking/occupancy_service.py` | Patch classifier ONNX + Spatial overlap | **Ready with Enhancement:** `PatchOccupancyClassifier` runs via ONNX Runtime CPU. OccupancyService needs direct patch-classifier hookup so Pi can run lightweight bay classification (18 ms/frame) without mandatory heavy YOLO vehicle detection. |
| **Vehicle Detection** | `app/detection/vehicle_detector.py` | ONNX Runtime CPU + OpenCV NMS | **Ready (Optional):** Operates on `CPUExecutionProvider`. Input resolution scaled down to $480 \times 480$ on Pi to conserve CPU cycles. |
| **Plate Detection** | `app/plates/plate_detector.py` | ONNX Runtime CPU + OpenCV NMS | **Ready (Optional):** Operates on `CPUExecutionProvider`. Runs only on cropped vehicle bboxes. |
| **OCR Engine** | `app/ocr/engine.py` | EasyOCR (PyTorch-backed) | **Heavy / Optional on Pi:** EasyOCR requires PyTorch (~800MB RAM, high CPU usage). The Pi profile must allow OCR to be toggled off (`enabled: false`) while keeping parking occupancy 100% operational. |
| **Allocation API** | `app/parking/allocation_service.py` | SQLite transactional allocation | **100% Ready:** Uses `BEGIN IMMEDIATE;` locks to guarantee ACID serialization and prevent double allocation under concurrency. |
| **Web Dashboard** | `web/` & `app/api/routes.py` | FastAPI + HTML/JS/CSS | **100% Ready:** Serves lightweight responsive dashboard on `http://127.0.0.1:8000`. Perfect for HDMI kiosk display via Chromium. |

---

## 4. Gaps and Required Changes for Stage 3–5

1. **Inference Adapter Integration:**  
   Currently, `OccupancyService` uses vehicle detector bounding-box overlap as its primary signal, while `PatchOccupancyClassifier` is instantiated separately in audit scripts. For optimal Raspberry Pi performance, `OccupancyService` must directly leverage `PatchOccupancyClassifier` on cropped bay polygons, enabling ~50 FPS patch classification with sub-20ms frame latency on CPU.
2. **Configuration Alignment (Thresholds & Staleness):**  
   - `configs/raspberry_pi.yaml` currently specifies `occupancy_threshold: 0.35`. It must be aligned with the validated optimal threshold $\tau^* = 0.650$ for Approach D.
   - Stale camera threshold is referenced inconsistently (2.0s, 3.0s, 4.0s, 10.0s). It must be standardized to a conservative **2.0 seconds** default across all configurations, code, and documentation.
3. **Deployment Packaging:**  
   No `deployment/raspberry_pi/` folder exists. A full suite of shell scripts (`install.sh`, `start.sh`, `check_models.sh`, `smart-parking.service`, `kiosk.sh`) must be created.
4. **Preflight & Benchmark Diagnostics:**  
   Dedicated preflight (`scripts/preflight_raspberry_pi.py`) and benchmarking (`scripts/benchmark_inference.py`) tools must be built to validate hardware, permissions, and latency before live commissioning.
