# Vehicle Detection Multi-Dataset Acquisition Report

**Document Version:** 1.0.0  
**Acquisition Date:** 2026-10-14 12:56:05  
**Author:** Computer Vision Research & Infrastructure Engineering  
**Disk Budget Allocated:** 20.0 GB (Verified Available: ~336 GB)  
**Primary Objective:** Acquire, audit, govern, and standardize a multi-source vehicle detection corpus for YOLOv8-Nano to improve detection of Indian traffic, heavy trucks, motorcycles, small distant vehicles, and overhead CCTV perspectives.

---

## 1. Executive Summary & Acquisition Inventory

In strict accordance with the dataset priority protocol, all candidate external datasets were audited for accessibility, licensing, annotation schema, access controls, and storage requirements before downloading.

### Acquisition Status Overview

| Priority | Dataset | Source / Host | License | Access Status | Acquired Images | Acquired Instances | Storage Used |
|---|---|---|---|---|---:|---:|---:|
| **1 (Highest)** | **UVH-26** | AIM @ IISc (Hugging Face) | CC BY 4.0 | **ACQUIRED & VERIFIED** | **1200** | **12847** | ~631.8 MB |
| **2** | **IDD Detection** | IIIT Hyderabad (INSAAN) | Academic Research | **BLOCKED BY ACCESS CONTROLS** | 0 | 0 | 0 MB |
| **3** | **BDD100K** | UC Berkeley (bdd-data) | BDD Research Terms | **BLOCKED BY ACCESS CONTROLS** | 0 | 0 | 0 MB |
| **4** | **Open Images V7** | Google Cloud Storage | CC BY 4.0 / CC BY 2.0 | **METADATA AUDITED** | Supplementary | Supplementary | <10 MB |
| **5 (Ablation)** | **VisDrone2019-DET** | Tianjin University / AISKYEYE | Academic Non-Commercial | **ACQUIRED & VERIFIED** | **345** | **13006** | ~78 MB |
| **Baseline** | **COCO 2017 Vehicles** | COCO Consortium | CC BY 4.0 | **PRESERVED ACTIVE BASELINE** | **1,896** | **7,815** | ~350 MB |

---

## 2. Source-by-Source Governance & Audit Details

### 2.1 Dataset A — UVH-26 (Highest Priority)
- **Official Portal:** [https://huggingface.co/datasets/iisc-aim/UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26)
- **Official Announcement:** [IISc AIM Urban Vision Hackathon](https://www.iisc.ac.in/events/aim-iisc-announces-public-release-of-uvh-26-dataset-and-vision-models-for-indian-urban-traffic/)
- **Technical Report:** Akash Sharma, Chinmay Mhatre, Sankalp Gawali, et al., *The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic*, IISc Technical Report, arXiv:2511.02563, Nov 2025.
- **License & Attribution:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Requires formal citation and attribution in project documentation.
- **Native Specifications:** 26,646 high-resolution (1080p) frames sampled from ~2,800 Bengaluru *Safe City* CCTV surveillance cameras, with ~1.8 million consensus bounding boxes across 14 India-specific classes.
- **Acquisition Outcome:**
  - Full COCO JSON metadata acquired: `UVH-26-MV-Train.json` (60.7 MB, 21,349 images) and `UVH-26-MV-Val.json` (15.25 MB, 5,297 images).
  - Global file index constructed across all chunk subfolders (`000` to `004`): 26,646 files mapped in 11.25s.
  - Stratified training subset acquired: **1000 images** with **10764 ground-truth instances**.
  - Genuinely held-out Indian traffic CCTV validation benchmark: **200 images** with **2083 instances**.
  - Conversion: Decoded 1080p PNG frames converted to high-quality JPEG (quality=92) to eliminate excessive disk bloat while retaining fine vehicle features.
  - Class coverage:
    - Cars (Hatchback/Sedan/SUV/MUV): `2965` train instances
    - Motorcycles (Two-wheelers): `6343` train instances
    - Buses (Bus/Mini-bus): `706` train instances
    - Trucks (Heavy commercial trucks): `750` train instances

### 2.2 Dataset B — India Driving Dataset (IDD Detection)
- **Official Portal:** [https://insaan.iiit.ac.in/datasets/](https://insaan.iiit.ac.in/datasets/) | [https://idd.insaan.iiit.ac.in/dataset/details/](https://idd.insaan.iiit.ac.in/dataset/details/)
- **Access Audit Finding:** **BLOCKED BY ACCESS CONTROLS / AUTHENTICATION GATE**.
- **Governance Rationale:** The official download endpoint `https://idd.insaan.iiit.ac.in/dataset/download/` requires manual user registration and interactive login credentials. Direct unauthenticated calls to Hugging Face mirrors (`AutoNUE/IDD`, `idd`) return `HTTP 401: Unauthorized`.
- **Operating Policy Compliance:** In strict adherence to project safety rules ("Do not scrape unauthorized mirror sites or bypass gated dataset access. If a dataset cannot be acquired within the available space or access permissions, continue with the successfully acquired datasets and record the limitation"), the limitation is formally documented and IDD was excluded from automated pipelines until institutional credentials are provided.

### 2.3 Dataset C — BDD100K
- **Official Portal:** [https://www.bdd100k.com/](https://www.bdd100k.com/)
- **Access Audit Finding:** **BLOCKED BY ACCESS CONTROLS / TLS VERIFICATION FAILURE**.
- **Governance Rationale:** BDD100K requires user registration and research account approval via `bdd-data.berkeley.edu` (HTTP 403 on direct queries). The mirror `https://dl.yf.io/bdd100k/` fails SSL certificate validation (`certificate verify failed: Hostname mismatch`).
- **Operating Policy Compliance:** Project policy strictly forbids disabling TLS certificate verification or utilizing unauthorized third-party mirrors. Documented as access-controlled.

### 2.4 Dataset D — Open Images V7
- **Official Documentation:** [https://storage.googleapis.com/openimages/web/download_v7.html](https://storage.googleapis.com/openimages/web/download_v7.html)
- **Status:** Evaluated and metadata cataloged. Used as supplementary benchmark ontology.

### 2.5 Dataset E — VisDrone (Hard-Case Surveillance Supplement)
- **Official Repository:** [https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- **Archive SHA-256:** `abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9`
- **Acquisition Outcome:** Acquired official release archive (`VisDrone2019-DET-val.zip`, 77.86 MB). Converted **345 representative surveillance frames** (13006 instances) featuring small, occluded, and steep downward-angle vehicles for targeted hard-case ablation.

### 2.6 Baseline Corpus Preservation
- **Existing Baseline:** 1,896 COCO-derived vehicle images (`1,657` train, `119` val, `120` held-out test).
- **Preservation Invariant:** The 120-image held-out test partition (`data/processed/vehicles/images/test`) and parking-camera benchmark remain **100% frozen and unmodified**.

---

## 3. Storage & Integrity Audit

- **Total Disk Space Available:** >330 GB (exceeds 20 GB allocation).
- **Downloaded Archives Verified:** SHA-256 integrity checks computed and validated.
- **Duplicate Prevention:** Per-file existence and hash checks ensure rerunning the acquisition pipeline executes idempotently without redundant network transfers.

---

## 4. Next Phase Action Items

1. Establish canonical class-mapping policy (`docs/vehicle_detection/CLASS_MAPPING_POLICY.md`).
2. Construct leakage-safe multi-dataset training partitions and audit duplicate hashes (`docs/vehicle_detection/MULTI_DATASET_SPLIT_AUDIT.md`).
3. Execute controlled 45-epoch comparative training experiments across Baseline, India CCTV, Multi-Source, and Truck-Focus configurations.
