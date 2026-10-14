# Vehicle Detection Baseline Audit & Provenance Verification

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-14  
**Audit Author:** Computer Vision & MLOps Research Lab  
**Evaluation Environment:** Windows 11 | Python 3.12.10 | NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.4) | ONNX Runtime 1.31.0  

---

## 1. Executive Summary & Verification Finding

An empirical verification audit was conducted on the smart parking vehicle detector artifacts. Every metric recorded in project history was re-evaluated against the active checkpoint, model architecture, test dataset, and runtime configuration.

### Metric Verification Status: **100% REPRODUCED**
The active model weights `models/vehicle_detector.pt` and deployment ONNX artifact `models/vehicle_detector.onnx` replicate the historical expanded-data metrics **to four decimal places** on the independent held-out test split:
- **Precision:** `62.24%` (0.622449)
- **Recall:** `54.59%` (0.545858)
- **mAP@50:** `59.50%` (0.595006)
- **mAP@50–95:** `37.97%` (0.379748)

---

## 2. Checkpoint & Deployment Artifact Inventory

| Property | Value |
|---|---|
| **Canonical PyTorch Checkpoint** | `models/vehicle_detector.pt` |
| **PyTorch File Size** | 6,506,595 bytes (6.21 MB) |
| **PyTorch Checkpoint SHA-256** | `0f8c08dc8f4aff7f0b52582f38677c016b8caedf73ab4b39c182046a086e6089` |
| **Canonical Deployment ONNX** | `models/vehicle_detector.onnx` |
| **ONNX File Size** | 12,776,170 bytes (12.18 MB) |
| **ONNX Checkpoint SHA-256** | `8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc` |
| **Training Run Source Checkpoint** | `models/training_runs/vehicle/weights/best.pt` |
| **Source Checkpoint SHA-256 Match** | **Identical** (`0f8c08dc8f4aff7f0b52582f38677c016b8caedf73ab4b39c182046a086e6089`) |
| **Last-Epoch Checkpoint** | `models/training_runs/vehicle/weights/last.pt` (SHA-256: `ced1e32410bdf1893...`) |

### Model Architecture
- **Base Architecture:** YOLOv8-Nano (`yolov8n.pt` base weights)
- **Fused Layer Count:** 73 layers
- **Total Parameters:** 3,133,436
- **GFLOPs:** 8.6 GFLOPs at 640x640 resolution
- **Input Dimensions:** 640 x 640 x 3 (B, C, H, W = 1, 3, 640, 640)
- **Color Normalization:** RGB, scaled by `1/255.0` to `[0.0, 1.0]`

---

## 3. Dataset Manifest & Label Mapping

- **Dataset Configuration:** `data/vehicles.yaml`
- **Class Mapping:**
  ```yaml
  names:
    0: car
    1: motorcycle
    2: bus
    3: truck
  ```
- **Provenance & Splits Composition:**
  - **Source:** COCO 2017 (val2017 filtered subset + targeted train2017 high-recall vehicle expansion)
  - **Training Split (`train`):** 1,657 images | 6,870 instances (3,580 cars, 1,284 trucks, 1,240 motorcycles, 766 buses)
  - **Validation Split (`val`):** 119 images | 436 instances (286 cars, 79 trucks, 38 motorcycles, 33 buses)
  - **Independent Held-Out Test Split (`test`):** 120 images | 509 instances (335 cars, 60 trucks, 59 motorcycles, 55 buses)
  - **Split Manifest:** Tracked in `data/splits/test_manifest.json`
  - **Partition Leakage:** 0 duplicates across partitions (verified by SHA-256 image hashes)

---

## 4. Training & Validation Configuration

Recorded from `models/training_runs/vehicle/args.yaml` and `models/training_runs/vehicle/training_run_metadata.json`:
- **Pretrained Weights:** `yolov8n.pt`
- **Target Epochs:** 35 epochs
- **Best Validation Epoch:** Epoch 32 (`mAP50 = 0.5680`, `mAP50-95 = 0.3881`)
- **Batch Size:** 16
- **Input Size (`imgsz`):** 640
- **Optimizer:** `auto` (AdamW with initial lr=0.002, final lr factor=0.01)
- **Seed:** 42 (deterministic)
- **Workers:** 0 (Windows multiprocessing safety)
- **Mixed Precision (`amp`):** Enabled (FP16 on CUDA)
- **Training Duration:** 974.54 seconds (~16.2 minutes on RTX 4070 GPU)

---

## 5. Quantitative Evaluation Metrics

### 5.1 Progression: Earlier Baseline vs Current Expanded Model
Evaluated on the exact same 120-image held-out test benchmark (`data/processed/vehicles/images/test`):

| Metric | Earlier Baseline (`e0bd835b...`) | Current Active Model (`0f8c08dc...`) | Absolute Delta | Relative Gain |
|---|---:|---:|---:|---:|
| **Precision** | `51.29%` | **62.24%** | `+10.95%` | `+21.35%` |
| **Recall** | `43.84%` | **54.59%** | `+10.75%` | `+24.52%` |
| **mAP@50** | `45.74%` | **59.50%** | `+13.76%` | `+30.08%` |
| **mAP@50–95** | `28.08%` | **37.97%** | `+9.89%` | `+35.22%` |
| **Training Corpus** | 3,003 instances | 6,870 instances | `+3,867 instances` | `+128.8%` |

### 5.2 Per-Class Breakdown (Current Active Model)

| Class | Ground-Truth Instances | Precision | Recall | AP@50 | AP@50–95 |
|---|---:|---:|---:|---:|---:|
| **Car** | 335 | **69.80%** | 55.52% | 60.38% | 37.17% |
| **Motorcycle** | 59 | **76.48%** | 59.32% | 66.98% | 35.85% |
| **Bus** | 55 | **67.97%** | **61.82%** | **71.92%** | **54.49%** |
| **Truck** | 60 | **34.73%** | 41.68% | 38.73% | 24.38% |
| **Overall** | **509** | **62.24%** | **54.59%** | **59.50%** | **37.97%** |

---

## 6. Confusion Matrix & Error Analysis

Inspection of `models/training_runs/vehicle/confusion_matrix.png` and `vehicle_detector_diagnostics.json` reveals the primary error patterns:

### Primary Error Modalities
1. **Truck vs. Car Confusion:**
   - 12 out of 79 validation trucks were predicted as `car`.
   - Pickup trucks, delivery vans, and large SUVs share severe visual ambiguity in COCO-derived bounding box annotations.
   - At high confidence thresholds (`conf >= 0.40`), truck recall drops to ~41% because truck predictions register lower confidence than passenger cars.
2. **Truck Misses (Background False Negatives):**
   - 23 validation trucks missed completely (occluded or small/distant in background).
3. **Car Under-Recall on Dense Parking:**
   - Occluded vehicles overlapping bay boundaries are occasionally missed when confidence falls below 0.40.
4. **False Positives on Signs & Structural Elements:**
   - Rectangular reflective structures, metallic signs, and sharp ground shadows occasionally trigger false detections.

### Representative Error Samples (Recorded in Diagnostic Registry)
- `000000055150.jpg`: Ground-truth truck predicted as `car` (conf=0.27, IoU=0.823).
- `000000097679.jpg`: Ground-truth truck predicted as `car` (conf=0.81, IoU=0.870).
- `000000136715.jpg`: Metallic barrier false positive as `truck` (conf=0.49).
- `000000210394.jpg`: Truck predicted as `bus` (conf=0.61, IoU=0.965).
- `000000350003.jpg`: Complex background clutter false positive (conf=0.52).

---

## 7. Inference Latency & System Performance

### Windows Development Workstation (Intel i7 / RTX 4070 Laptop GPU)
- **PyTorch GPU Inference:** 3.9 – 4.4 ms per frame
- **Preprocess (resize + normalize):** 1.2 ms
- **Postprocess + NMS:** 1.0 ms
- **Total GPU Pipeline Latency:** ~6.6 ms (~151 FPS theoretical)
- **ONNXRuntime CPU Latency:** Mean 42.35 ms (p50: 40.37 ms, p95: 51.45 ms) at 640x640

### Raspberry Pi 5 Target (8GB RAM, Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz)
- **Inference Engine:** ONNX Runtime (`CPUExecutionProvider`, 2 intra-op threads)
- **Measured ONNX Latency (640x640):** 78.4 ms (~12.7 FPS maximum single-thread inference)
- **Operating Architecture:** Decoupled detection loop (250 ms interval = 4 FPS) alongside smooth 30 FPS camera preview. This prevents CPU thermal throttling on the Pi 5.
- **Process RSS Memory:** ~180 – 240 MB (well within the 1024 MB conservative budget).

---

## 8. Identified Reproducibility & Pipeline Flaws (Resolved in Current Upgrade)

1. **Destructive Training Overwrite:**
   - Previous `scripts/train_vehicle_detector.py` directly copied `best.pt` over `models/vehicle_detector.pt` and `models/vehicle_detector.onnx`, violating model-promotion safety.
2. **Hardcoded Machine Paths:**
   - `args.yaml` contained absolute Windows developer paths (`C:\Users\adity\Downloads\IoT\...`).
3. **Absence of Staged Candidate Registry:**
   - No separate candidate quarantine: new models were immediately deployed regardless of regressions on truck precision or deployment latency.
4. **Lack of Automated Rollback:**
   - If an experimental training run regressed, manual file renaming was required.

---

## 9. Baseline Lockdown & Preservation Guarantee

The active baseline model:
- **PyTorch:** `models/vehicle_detector.pt` (SHA-256: `0f8c08dc8f4aff7f0b52582f38677c016b8caedf73ab4b39c182046a086e6089`)
- **ONNX:** `models/vehicle_detector.onnx` (SHA-256: `8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc`)

is formally designated as **`baseline-v1.0.0`** in the model registry. All subsequent experimental training runs will output to dedicated candidate directories (`models/experiments/<exp_id>`) and will be subject to strict promotion gates before any deployment artifact is updated.
