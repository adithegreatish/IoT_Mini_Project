# Multi-Dataset Split Audit & Leakage Verification Report

**Document Version:** 2.0.0  
**Audit Date:** 2026-10-14 14:10:00  
**Author:** CV & MLOps Research Engineering  
**Scope:** Evaluation protocol leakage correction, independent Indian CCTV held-out benchmark establishment, multi-source balanced training partition audit, and zero-leakage governance.

---

## 1. Executive Summary & Leakage Diagnosis

An exhaustive audit of training, validation, and test manifests was conducted following experiment `exp_india_cctv_10ep`.

### 1.1 Diagnosis of Prior Evaluation Protocol Leakage
- **Mechanism:** In `exp_india_cctv_10ep`, `data/processed/uvh26_subset/images/val` (200 images) was designated as the validation set in `data/vehicles_india_cctv.yaml` (used for checkpointing during training). Simultaneously, the candidate was evaluated on `data/splits/test_uvh26_indian_traffic_manifest.json`, which referenced the identical 200 validation images.
- **Correction Action:** Those 200 images are formally reclassified as **Historical Training-Time Validation Data** and are **never** treated as an untouched final test benchmark.
- **Original Manifest Integrity:** The original manifest `data/splits/test_uvh26_indian_traffic_manifest.json` was kept strictly unmodified to maintain historical provenance.

### 1.2 Independent Indian CCTV Benchmark Establishment
A project-specific, untouched Indian CCTV held-out evaluation set was established:
- **Manifest:** `data/splits/test_indian_cctv_heldout_manifest.json`
- **YAML Config:** `data/vehicles_indian_cctv_heldout.yaml`
- **Location:** `data/processed/uvh26_heldout_test/images/` and `labels/`
- **Sample Count:** 200 images, **2,102 annotated vehicle instances** (586 cars, 1,214 motorcycles, 141 buses, 161 trucks).
- **Temporal / Sequence Buffer:** Selected exclusively from unused images in `UVH-26-MV-Val.json` with an enforced **frame ID buffer $>100$** from all 1,000 training and 200 previous validation frames.
- **Deduplication Check:** 0 SHA-256 collisions and 0 perceptual dHash near-duplicates (Hamming distance $\ge 8$) against all training and validation data.
- **Camera Station Provenance Limitation:** In the official UVH-26 release, IISc AIM omitted camera station IDs from image records. Therefore, while frame-level and annotation-level disjointness is 100% verified, camera pole independence across Bengaluru Safe City intersections cannot be formally proven. This limitation is explicitly stated without claiming unverified camera independence.

---

## 2. Training Corpus Source Mixture & Balance (Arm C)

For Arm C (`exp_balanced_multisource_10ep`), a source-aware balanced training corpus was assembled to preserve horizontal street-level vehicle profiles while improving Indian CCTV and small/distant truck detection:

| Source Component | Dataset Domain | Sampling Policy | Effective Samples | Share (%) | Purpose |
|---|---|---|---:|---:|---|
| **COCO General Traffic** | Street photography | 100% of base training set | 1,657 | 51.80% | Preserve baseline street perspective & vehicle geometry. |
| **COCO Motorcycle Reinforcement** | Horizontal street views | 1x duplicate of 497 motorcycle images | 497 | 15.54% | Counteract overhead CCTV gradient bias on two-wheelers. |
| **UVH-26 Indian CCTV (Stratified)** | Bengaluru Safe City (1080p) | 566 truck images + 134 bus images | 700 | 21.88% | Enhance commercial vehicle recall under CCTV angles without motorcycle flooding. |
| **VisDrone Supplement** | Aerial drone surveillance | 100% of acquired frames | 345 | 10.78% | Small/distant vehicles (<1.5% frame) and steep overhead angles. |
| **Total Combined Train (Arm C)** | **Balanced Multi-Source** | **Documented Source-Aware Policy** | **3,199** | **100.0%** | Multi-domain balanced training mixture. |

### Class Instance Counts in Arm C Training Mix:
- **Car:** 15,702 instances
- **Motorcycle:** 10,171 instances (reinforced with 2,480 horizontal COCO annotations)
- **Bus:** 1,621 instances
- **Truck:** 2,831 instances (1,284 COCO + 750 UVH-26 commercial + 592 VisDrone small aerial)

---

## 3. Split Verification & Zero-Leakage Audit Matrix

| Verification Check | Target Partitions | Result | Audit Outcome |
|---|---|:---:|---|
| **Exact SHA-256 Collisions** | Train (3,199) vs Held-Out Indian CCTV (200) | **0** | **PASSED (Strict Disjointness)** |
| **Exact SHA-256 Collisions** | Train (3,199) vs Frozen General COCO Test (120) | **0** | **PASSED (Strict Disjointness)** |
| **Exact SHA-256 Collisions** | Train (3,199) vs Real Parking Camera Test (30) | **0** | **PASSED (Strict Disjointness)** |
| **Exact SHA-256 Collisions** | Train (3,199) vs Difficult Cases Test (87) | **0** | **PASSED (Strict Disjointness)** |
| **Perceptual dHash Near-Duplicates** | Train vs All Test Partitions (Hamming < 8) | **0** | **PASSED (Zero Near-Duplicates)** |
| **Temporal Frame Buffer** | UVH-26 Train vs Held-Out Indian CCTV Test | **$> 100$ frames** | **PASSED (No Adjacent Video Frames)** |
| **Frozen Benchmark Preservation** | General COCO Test (`data/processed/vehicles/images/test`) | **Bit-for-bit** | **PASSED (Hash Verified: Untouched)** |

---

## 4. Authoritative Multi-View Evaluation Benchmarks

Four strictly isolated evaluation benchmarks are maintained:

1. **View 1: Frozen General Object-Detection Benchmark (`test_general`)**
   - Path: `data/splits/test_general_manifest.json` | 120 images (509 instances)
   - Role: Authoritative promotion gate benchmark. Never modified.
2. **View 2: Real Parking-Camera Benchmark (`test_parking_camera`)**
   - Path: `data/splits/test_parking_camera_manifest.json` | 30 images
   - Role: Real overhead pole-mounted surveillance footage (PKLot PUC-PR lot) assessing false positive resistance.
3. **View 3: Difficult-Case Benchmark (`test_difficult_cases`)**
   - Path: `data/splits/test_difficult_cases_manifest.json` | 87 images
   - Role: Hard negatives, occlusions, glare, and vehicle scale extremes.
4. **View 4: Untouched Independent Indian CCTV Benchmark (`test_indian_cctv_heldout`)**
   - Path: `data/splits/test_indian_cctv_heldout_manifest.json` | 200 images (2,102 instances)
   - Role: Untouched Indian CCTV test set never seen during training or checkpoint selection.
