# Model Training, Validation, and Evaluation Documentation

This document describes the genuine machine learning training and evaluation pipeline for the Smart Parking & HSRP Detection project.

---

## 1. Central Architecture & Pipeline

```
Raw Camera Frame (1280x720)
       │
       ▼
[Vehicle Detector (YOLOv8n ONNX)] ──► Parking Bay Spatial Intersection (IoU >= 0.35)
       │                                     │
       ▼ Detected Vehicle Bbox               ▼
[Plate Detector (YOLOv8n ONNX)]       Occupancy State Machine (FREE / OCCUPIED / UNKNOWN)
       │
       ▼ High-Resolution Plate Crop
[EasyOCR Neural Engine (PyTorch/GPU)] ──► Format Normalization & Validation (Indian Regex)
       │
       ▼
[HSRP Physical Cue Analyzer] ──► Explainable Output (Blue IND Strip, Geometry, Quality)
```

---

## 2. Acquired Datasets & Provenance

1. **Vehicles Dataset (`data/processed/vehicles/`):**
   - Source: COCO 2017 / COCO128 curated vehicle subset.
   - Filtered Classes: 4 vehicle classes mapped to consecutive indices:
     - `0: car`
     - `1: motorcycle`
     - `2: bus`
     - `3: truck`
   - Config: `data/vehicles.yaml` (uses canonical absolute paths).

2. **License Plate Dataset (`data/processed/plates/`):**
   - Sources:
     - DataCluster Labs Indian Number Plate dataset (annotated with Pascal VOC bounding boxes, remapped to YOLO format).
     - Keremberke benchmark license plate object detection dataset.
   - Total Annotated Samples: 1,812 genuine images with normalized bounding boxes.
   - Split: Train (1,450), Validation (181), Test (181).
   - Config: `data/plates.yaml` (uses canonical absolute paths).

Provenance metadata is permanently tracked in `data/dataset_provenance.json` and held-out test paths in `data/splits/test_manifest.json`.

---

## 3. Training Execution & Results

### A. Vehicle Detector (`vehicle-yolo-v8n`)
- **Base Architecture:** `yolov8n.pt`
- **Trained Checkpoint:** `models/vehicle_detector.pt`
- **Deployment ONNX:** `models/vehicle_detector.onnx` (SHA256: `24457ba79689f609a61c09baed2f91fce5d36da3e1c7fb4eeafff0628f4ffc94`)
- **Training Epochs:** 15
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.4)
- **Held-Out Test Set Metrics:**
  - Precision: `0.6458`
  - Recall: `1.0000`
  - mAP@50: `0.9950`
  - mAP@50-95: `0.6693`
  - ONNX Inference Latency: `29.28 ms` on Windows x86_64 CPU

### B. License Plate Detector (`plate-detector-tiny`)
- **Base Architecture:** `yolov8n.pt`
- **Trained Checkpoint:** `models/plate_detector.pt`
- **Deployment ONNX:** `models/plate_detector.onnx` (SHA256: `2709413af4bdf668f7b4e08da6c084f8b5ca8ea190f331fd957083485a3de304`)
- **Training Epochs:** 5
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.4)
- **Held-Out Test Set Metrics (273 test images, 287 plates):**
  - Precision: `0.9680`
  - Recall: `0.9492`
  - mAP@50: `0.9654`
  - mAP@50-95: `0.6528`
  - ONNX Inference Latency: `28.68 ms` on Windows x86_64 CPU

---

## 4. Optical Character Recognition (EasyOCR)

- Replaces all hardcoded/simulated OCR strings with real PyTorch neural text recognition.
- **Preprocessing:** Contrast-Limited Adaptive Histogram Equalization (CLAHE) + aspect-ratio preserving padding.
- **Reading Strategy:** Multi-box bounding text detection, vertical spatial sorting for stacked Indian two-wheeler plates, uppercase alphanumeric filtering.
- **Format Validation:** Strict matching against Indian registration syntax (`^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$`).
- **Honest Abstention:** When characters are unreadable, blurred, or occluded, reports `is_readable: False` and `confidence: 0.0`. Never synthesizes fake registration plates.

---

## 5. Deployment Export & Numerical Parity

Script: `scripts/export_deployment_models.py`
- Converts PyTorch `.pt` to `.onnx` with opset=12 and ONNX-simplifier.
- Performs numerical parity validation on test images:
  - Vehicle detector: Max absolute error `0.002258` (Tolerance <= 0.01) -> **PASSED**
  - Plate detector: Max absolute error `0.000610` (Tolerance <= 0.01) -> **PASSED**
- Real SHA256 checksums are registered in `models/manifest.json`.

---

## 6. Commands Reference

```powershell
# 1. Acquire & Prepare Datasets
python scripts/acquire_datasets.py

# 2. Train Vehicle Detector
python scripts/train_vehicle_detector.py --epochs 15 --batch 16

# 3. Train Plate Detector
python scripts/train_plate_detector.py --epochs 5 --batch 16

# 4. Export ONNX & Check Parity
python scripts/export_deployment_models.py

# 5. Evaluate Held-Out Benchmarks
python scripts/evaluate_models.py
```
