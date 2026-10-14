# Vehicle Detection Dataset & Split Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-14  
**Audit Author:** CV / MLOps Research Lab  
**Scope:** Dataset provenance, annotation standards, split hygiene, train/test leakage prevention, and evaluation benchmarks.

---

## 1. Dataset Provenance & Architectural Boundary

### 1.1 Strict Separation: Vehicle Detection vs. Parking Space Occupancy
A fundamental architectural requirement of this system is that **Vehicle Detection** and **Parking Bay Occupancy Classification** are distinct tasks trained on distinct datasets:

```
[Camera Frame]
     │
     ├─► [Vehicle Detector (YOLOv8n)] ──► Bounding Boxes (car, motorcycle, bus, truck)
     │                                            │
     │                                            ▼
     ├─► [Slot Geometry Calibration]  ──► Vehicle-to-Slot Spatial Association (IoU / centroid)
     │                                            │
     ▼                                            ▼
[Slot Patch Crops] ─────────────────────► [Occupancy Engine / Classifier] ──► State (FREE / BUSY)
```

- **Vehicle Detector Dataset:** Annotates object bounding boxes around entire vehicles (`car`, `motorcycle`, `bus`, `truck`) regardless of parking status.
- **Occupancy Dataset (CNRPark+EXT / PKLot Patches):** Annotates cropped parking bay patches (`FREE` vs `BUSY`).
- **Safety Invariant:** Occupancy slot polygons and binary slot patch labels are **NEVER** repurposed as vehicle detection bounding box labels.

---

## 2. Dataset Inventory & Composition

### 2.1 Active Vehicle Training Corpus
- **Configuration:** `data/vehicles.yaml`
- **Class Ontology:**
  - `0`: `car`
  - `1`: `motorcycle`
  - `2`: `bus`
  - `3`: `truck`

| Split | Images | Car Instances | Motorcycle Instances | Bus Instances | Truck Instances | Total Instances |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 1,657 | 3,580 | 1,240 | 766 | 1,284 | 6,870 |
| **Validation** | 119 | 286 | 38 | 33 | 79 | 436 |
| **Test (Held-Out)** | 120 | 335 | 59 | 55 | 60 | 509 |
| **Total** | **1,896** | **4,201** | **1,337** | **854** | **1,423** | **7,815** |

### 2.2 Class Distribution & Imbalance Analysis
- `car` constitutes **53.8%** of all instances in the corpus.
- `truck` constitutes **18.2%** of all instances (expanded from historical <10%).
- `motorcycle` constitutes **17.1%**.
- `bus` constitutes **10.9%**.
- While class frequency of trucks is reasonable, **truck precision (34.7%) remains the primary performance bottleneck** due to visual confusion with commercial delivery vans and pickup trucks.

---

## 3. Train / Validation / Test Leakage Verification

Empirical verification was conducted across all 1,896 images in `data/processed/vehicles/images/`:

| Split Comparison | SHA-256 Exact Matches | Perceptual Duplicates | Leakage Status |
|---|---:|---:|---|
| `train` vs `val` | **0** | **0** | **CLEAN (No Leakage)** |
| `train` vs `test` | **0** | **0** | **CLEAN (No Leakage)** |
| `val` vs `test` | **0** | **0** | **CLEAN (No Leakage)** |

### Split Protocol Guarantee
- Splits are strictly disjoint.
- Training routines validate on `val` split **only** during epoch progression and early stopping.
- Hyperparameter tuning, confidence threshold sweeps, and loss weighting are strictly performed on `val`.
- The `test` split is evaluated **only** during formal model auditing and promotion gate verification.

---

## 4. Multi-View Evaluation Benchmarks

To avoid single-benchmark bias and prevent overfitting to COCO street-level camera angles, the continuous learning architecture establishes **Three Distinct Evaluation Views**:

### View 1: General Object-Detection Benchmark (`test_general`)
- **Dataset:** 120 held-out COCO vehicle images (`data/processed/vehicles/images/test`).
- **Instances:** 509 ground-truth vehicle bounding boxes.
- **Purpose:** Assesses generalized vehicle recognition across varied backgrounds, focal lengths, viewpoints, and scales.

### View 2: Real Parking-Camera Benchmark (`test_parking_camera`)
- **Dataset:** Dedicated held-out parking surveillance frames (`data/splits/parking_camera_test_manifest.json`).
- **Perspective:** High-angle fixed overhead CCTV viewpoint (elevated pole / wall mount).
- **Purpose:** Verifies that models trained on general street photography generalize robustly to steep down-angle parking lot cameras without perspective distortion failures.

### View 3: Difficult-Case Benchmark (`test_difficult_cases`)
- **Dataset:** Curated hard-example test set (`data/splits/difficult_cases_manifest.json`).
- **Composition:**
  1. Pickup trucks and delivery vans prone to car/truck confusion.
  2. Occluded vehicles (partially hidden by trees, posts, or neighboring vehicles).
  3. Small/distant vehicles occupying less than 1.5% of total frame area.
  4. Adverse illumination (strong specular glare, deep shadows, rainy asphalt reflections).
  5. Negative frames (empty bays, signs, street furniture) testing false-positive rejection.

---

## 5. Dataset Provenance & Licensing Inventory

| Dataset Asset | Source / Origin | License | Permitted Use in Project |
|---|---|---|---|
| **COCO 2017 Vehicle Subset** | Common Objects in Context (MS COCO) | CC BY 4.0 | Non-commercial / Educational & Research |
| **CNRPark+EXT Patches** | CNR-ISTI & University of Pisa | ODbL v1.0 | Academic / Educational with Attribution |
| **PKLot Surveillance Frames** | UFPR & PUCPR (Voxel51) | CC BY 4.0 | Academic / Research |
| **Local Parking Camera Footage** | Opt-in local deployment capture | Proprietary / Local Only | Private research; sanitized with plate blurring |

---

## 6. Continuous Data Lifecycle & Active Learning Queue

```
[CCTV Video Stream]
        │
        ▼
[Frame Quality Gate & Hard Example Miner]
  ├── Low confidence (0.25 <= conf < 0.45)
  ├── Detector vs Occupancy contradiction
  └── Excessive / ambiguous detections
        │
        ▼ (Perceptual Deduplication: hash distance >= 6)
[Hard-Example Candidate Queue] (`data/active_learning/candidates/`)
        │
        ▼
[Human-in-the-Loop Review Dashboard]
  ├── Accept suggested labels
  ├── Correct class / box coordinates
  └── Reject bad frames
        │
        ▼
[Dataset Validation Gate] (`scripts/validate_vehicle_annotations.py`)
        │
        ▼
[Versioned Dataset Split Update] (`data/processed/vehicles/`)
```

This ensures that the dataset expands strictly with high-information-density, human-verified samples without introducing dataset drift or corrupted automated annotations.
