# Dataset Inventory, Provenance & Acquisition Guide

This document provides complete provenance, licensing, local disk layouts, and automated reproduction instructions for all research datasets used across the Smart Parking and Indian HSRP system.

---

## 1. Dataset Repository Policy

Due to GitHub repository size limits (recommended < 1–5 GB per repository, strict 100 MB per-file threshold), raw and intermediate image datasets (~4.5 GB total) are maintained locally in the research environment and excluded from Git commits via `.gitignore`.

To ensure 100% scientific and operational reproducibility without committing massive image archives:
1. **Manifests & Splits:** All deterministic split definitions, bounding box annotations, and ground truth files are tracked in version control under `data/splits/` and `data/annotations/`.
2. **Deterministic Pipelines:** Python acquisition scripts in `scripts/` automate download, checksum validation, polygon parsing, patch cropping, and YOLO format conversion from official public sources.
3. **Integrity Audits:** Checksum manifests and geometry audits are stored in `data/dataset_inventory.json` and `models/*audit.json`.

---

## 2. Comprehensive Dataset Inventory

| Dataset | Primary Source / Authority | Local Storage Path | Uncompressed / Archive Size | License / Terms | Repository Tracking |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **COCO 2017 (Vehicles)** | [cocodataset.org](https://cocodataset.org) | `data/raw/coco2017/`<br>`data/processed/vehicles/` | `val2017.zip` (778 MB)<br>`instances_train2017.json` (448 MB)<br>`annotations_trainval2017.zip` (241 MB) | CC BY 4.0 | Manifests & splits tracked in Git; raw archives kept locally |
| **PKLot** | UFPR Research / Voxel51 | `data/raw/pklot/`<br>`data/processed/parking_slots/` | Full frames & annotations (~1.8 GB)<br>`pucpr_spots.json` (514 MB)<br>`ufpr05_spots.json` (114 MB)<br>`ufpr04_spots.json` (83 MB) | CC BY 4.0 / Academic Non-commercial | Polygons & calibrations in `data/calibration.json`; raw images kept locally |
| **CNRPark+EXT** | CNR-ISTI Pisa | `data/raw/cnrpark/`<br>`data/processed/parking_occupancy/` | `CNR-EXT-Patches-150x150.zip` (429 MB)<br>`CNRPark-Patches-150x150.zip` (35 MB)<br>`splits.zip` (26 MB) | CC BY-NC-SA 4.0 / ODbL v1.0 | Splits & manifests in `data/splits/`; extracted patches kept locally |
| **Indian License Plates** | Kaggle / Open Communities | `data/raw/plates/indian/`<br>`data/processed/plates/` | 237 curated images + labels (~50 MB)<br>`valid_benchmark.zip` (43 MB) | Open Database License (ODbL) | Ground truth in `data/annotations/indian_plates_ground_truth.json` |
| **COCO128 (Vehicles)** | Ultralytics | `data/raw/vehicles/` | `coco128.zip` (6.7 MB) | AGPL-3.0 / CC BY 4.0 | Baseline sanity checks; kept locally |

---

## 3. Blocked / Inaccessible Datasets Audit

During the dataset review and candidate evaluation phase, the following candidate datasets were formally audited and marked blocked:

1. **BDD100K (Berkeley DeepDrive):**
   - *Status:* `BLOCKED_BY_ACCESS_CONTROLS`
   - *Reason:* Berkeley portal (`bdd-data.berkeley.edu`) requires interactive researcher login and multi-step manual authentication. Direct unauthenticated automated programmatic downloads return HTTP 403 Forbidden.
2. **UA-DETRAC:**
   - *Status:* `BLOCKED_BY_LICENSE_CONTROLS`
   - *Reason:* Demands university/institutional credential verification and written academic license agreement on the Albany/RIT server.

---

## 4. Local Reproduction & Acquisition Commands

All datasets can be downloaded and prepared from scratch using the automated scripts included in `scripts/`:

### A. Vehicles Dataset (COCO 2017 Target Classes)
Downloads COCO 2017 images containing cars, motorcycles, buses, and trucks, converts Pascal/COCO annotations to YOLO format, and builds leakage-free train/val/test splits:
```bash
python scripts/acquire_coco_vehicles.py
# Or general acquisition engine:
python scripts/acquire_datasets.py --dataset vehicles
```

### B. Parking Occupancy Patches (CNRPark+EXT)
Acquires the patch archives, uncompresses 150x150 slot crops under various weather conditions (Sunny, Overcast, Rainy), and constructs manifest records:
```bash
python scripts/acquire_cnr_ext_archive.py
python scripts/prepare_parking_dataset.py
```

### C. Parking Geometry & Baselines (PKLot)
Extracts slot boundary polygons, validates clockwise convexity, denormalizes slot polygons for cameras (PUCPR, UFPR04, UFPR05), and generates calibration mappings:
```bash
python scripts/acquire_pklot_patches.py
python scripts/acquire_pklot_coco_annotations.py
python scripts/audit_pklot_and_geometry.py
```

### D. Indian License Plates & Benchmark Ground Truth
Prepares normalized YOLO annotations and creates the ground truth evaluation matrix:
```bash
python scripts/prepare_plate_dataset.py
python scripts/create_indian_plates_ground_truth.py
```

### E. Full Automated Integrity Audit
Verifies that all local dataset files match their expected cryptographic SHA-256 hashes:
```bash
python scripts/audit_dataset_integrity.py
```

---

## 5. Dataset Manifests Included in Git

The following core metadata files are tracked directly in Git to ensure full reproducibility:
- `data/calibration.json`: Canonical 28-slot parking polygon coordinates with entry/exit regions.
- `data/dataset_inventory.json`: Instance counts, class distributions, and zero-leakage verification stats.
- `data/dataset_provenance.json`: Lineage and attribution records for all evaluated assets.
- `data/annotations/indian_plates_ground_truth.json`: Indian plate character annotations and bounding boxes.
- `data/splits/parking_occupancy_manifest.json`: Slot-level training/validation splits.
- `data/splits/leave_one_lot_out_occupancy_manifest.json`: Multi-lot cross-validation split definitions.
- `data/splits/cross_dataset_occupancy_manifest.json`: Domain generalization split mappings.
- `data/splits/cnrpark_full_manifest.json`: Full patch-level CNRPark+EXT inventory (tracked via Git LFS).
