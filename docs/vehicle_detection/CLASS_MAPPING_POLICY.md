# Canonical Class-Mapping Policy & Taxonomy Specification

**Document Version:** 1.0.0  
**Effective Date:** 2026-10-14  
**Author:** Computer Vision Research & Data Governance  
**Scope:** Multi-dataset class harmonization for YOLOv8-Nano vehicle detection across COCO 2017, UVH-26, VisDrone2019-DET, IDD, and Open Images.

---

## 1. Architectural Taxonomy Baseline

The production smart parking system utilizes a canonical **4-Class Vehicle Detection Ontology**:

| Canonical Class ID | Canonical Class Name | Canonical Definition | Deployment Domain |
|:---:|:---:|---|---|
| **0** | **`car`** | Standard 4-wheeled passenger motor vehicles, including sedans, hatchbacks, SUVs, station wagons, and passenger minivans designed primarily for personal/family transit. | Passenger bay occupancy association, general parking monitoring. |
| **1** | **`motorcycle`** | Motorized two-wheeled vehicles, including motorcycles, scooters, mopeds, and motorbikes with petrol or electric drivetrains. | Dedicated two-wheeler bays, shared aisle monitoring. |
| **2** | **`bus`** | High-capacity passenger transit vehicles, including city transit buses, coaches, intercity buses, and passenger minibuses. | Commercial/bus bay monitoring, oversize lane enforcement. |
| **3** | **`truck`** | Heavy commercial freight vehicles, box trucks, flatbeds, cargo haulers, dump trucks, and articulated lorries with dedicated cargo beds/trailers. | Loading zones, commercial delivery bays, oversize bays. |

> [!IMPORTANT]
> **Taxonomy Safety Invariant:**  
> Production class IDs (`0: car`, `1: motorcycle`, `2: bus`, `3: truck`) are **strictly frozen**. Downstream parking slot allocation algorithms, spatial IoU matching, database schemas (`parking_events.sqlite`), and Raspberry Pi runtime logic depend on these exact numerical IDs. No additional classes may be silently injected into production models without an explicit versioned migration.

---

## 2. Multi-Dataset Class-Mapping Matrix

The following table documents the exhaustive, audited mapping from each external source taxonomy to the canonical 4-class ontology:

| Source Dataset | Source Class Name | Source Class ID | Canonical Class | Canonical ID | Mapping Rationale | Ambiguity / Exclusion Handling | Verification Status |
|---|---|:---:|---|:---:|---|---|:---:|
| **COCO 2017** | `car` | 3 | `car` | 0 | Direct semantic equivalence for standard passenger vehicles. | None. | Verified |
| **COCO 2017** | `motorcycle` | 4 | `motorcycle` | 1 | Direct semantic equivalence for motorized two-wheelers. | `bicycle` (ID 2) explicitly excluded. | Verified |
| **COCO 2017** | `bus` | 6 | `bus` | 2 | Direct semantic equivalence for transit buses. | None. | Verified |
| **COCO 2017** | `truck` | 8 | `truck` | 3 | Direct semantic equivalence for freight haulers. | Pickup trucks occasionally cross-confused in source. | Verified |
| **UVH-26** | `Hatchback` | 1 | `car` | 0 | Standard compact 2-box passenger automobile. | None. | Verified |
| **UVH-26** | `Sedan` | 2 | `car` | 0 | Standard 3-box saloon passenger automobile. | None. | Verified |
| **UVH-26** | `SUV` | 3 | `car` | 0 | Sport utility vehicle; passenger carriage geometry. | High ground clearance, mapped to car. | Verified |
| **UVH-26** | `MUV` | 4 | `car` | 0 | Multi-utility vehicle (e.g. Innova, Ertiga); passenger transit. | Van-like profile, treated as passenger car. | Verified |
| **UVH-26** | `Two-wheeler` | 8 | `motorcycle` | 1 | Represents motorized scooters, mopeds, motorcycles in Indian CCTV. | Non-motorized `Bicycle` (ID 12) strictly excluded. | Verified |
| **UVH-26** | `Bus` | 5 | `bus` | 2 | Standard heavy passenger transit bus. | None. | Verified |
| **UVH-26** | `Mini-bus` | 10 | `bus` | 2 | Small passenger transit bus (e.g., school/feeder bus). | Shared passenger transit geometry. | Verified |
| **UVH-26** | `Truck` | 6 | `truck` | 3 | Heavy commercial freight carrier / cargo lorry. | Direct match to canonical truck. | Verified |
| **UVH-26** | `Three-wheeler` | 7 | *EXCLUDED* | - | Auto-rickshaws feature unique 3-wheel delta geometry. | **Excluded / Filtered**: Frames dominated by auto-rickshaws excluded to avoid label noise. | Audit Rule A |
| **UVH-26** | `LCV` | 9 | *EXCLUDED* | - | Light commercial vehicles (e.g. Tata Ace, Bolero Maxi Truck). | Ambiguous boundary between pickup and truck. Excluded from target subset. | Audit Rule B |
| **UVH-26** | `Tempo-traveller`| 11 | *EXCLUDED* | - | Van-based high-roof passenger van. | Excluded to prevent bus vs car confusion. | Audit Rule B |
| **UVH-26** | `Bicycle` | 12 | *EXCLUDED* | - | Human-powered pedal cycle. | Excluded to protect motorcycle precision. | Audit Rule C |
| **UVH-26** | `Van` | 13 | *EXCLUDED* | - | Utility delivery / cargo van (e.g., Maruti Omni/Eeco). | Excluded from training set to reduce car/truck noise. | Audit Rule B |
| **UVH-26** | `Others` | 14 | *EXCLUDED* | - | Miscellaneous non-standard vehicles (tractors, carts). | Excluded. | Audit Rule D |
| **VisDrone** | `car` | 4 | `car` | 0 | Standard passenger car viewed from overhead drone. | Small distant bounding boxes preserved. | Verified |
| **VisDrone** | `van` | 5 | `car` | 0 | Passenger minivan in Asian traffic. | Mapped to car per VisDrone benchmark convention. | Verified |
| **VisDrone** | `motor` | 10 | `motorcycle` | 1 | Motorized scooter and motorbike. | Distinct from bicycle (ID 3) and tricycle (ID 7). | Verified |
| **VisDrone** | `bus` | 9 | `bus` | 2 | Transit passenger bus. | Steep angle variation. | Verified |
| **VisDrone** | `truck` | 6 | `truck` | 3 | Commercial lorry / flatbed / box truck. | Small/distant truck examples. | Verified |
| **VisDrone** | `pedestrian` | 1 | *EXCLUDED* | - | Human pedestrians on streets. | Excluded. | Non-vehicle |
| **VisDrone** | `people` | 2 | *EXCLUDED* | - | Crowds / standing humans. | Excluded. | Non-vehicle |
| **VisDrone** | `bicycle` | 3 | *EXCLUDED* | - | Pedal bicycle. | Excluded. | Non-vehicle |
| **VisDrone** | `tricycle` | 7 | *EXCLUDED* | - | Human/cargo pedal tricycles. | Excluded. | Non-vehicle |
| **VisDrone** | `awning-tricycle`| 8 | *EXCLUDED* | - | Covered tricycles. | Excluded. | Non-vehicle |
| **VisDrone** | `others` | 11 | *EXCLUDED* | - | Unknown objects / ignored regions (`score=0`). | Excluded via VisDrone score filtering. | Verified |

---

## 3. Ambiguity & Unmapped Vehicle Handling Policy

A common failure mode in multi-dataset expansion is allowing prominent unmapped vehicles (e.g., auto-rickshaws, cargo vans, pickup trucks) to remain unannotated in training images. The detector treats unannotated objects as **negative background**, penalizing true vehicle proposals and causing severe label noise.

### Strict Governance Rules

1. **Audit Rule A (Three-Wheeler / Auto-Rickshaw Isolation):**
   - Auto-rickshaws (`UVH-26: Three-wheeler`) possess a distinct compact triangular footprint, three wheels, open canvas sides, and high front windshields.
   - They cannot be cleanly mapped to `car` (destroys passenger car aspect ratio priors) or `motorcycle` (lacks two-wheel inline kinematics).
   - *Policy:* Any UVH-26 image where auto-rickshaws occupy >10% of total image area is **excluded from training**. In candidate training images with minor distant auto-rickshaws, their bboxes are omitted only if distant (<15 px height), preventing significant background gradients.

2. **Audit Rule B (LCV, Van, and Pickup Truck Ambiguity):**
   - Light Commercial Vehicles (`LCV`, e.g., Tata Ace / Mahindra Bolero Pick-up) are the single largest source of truck precision degradation in Indian traffic.
   - Annotators frequently label them inconsistently (some as `car`, others as `truck`).
   - *Policy:* Canonical `truck` is strictly reserved for heavy commercial freight lorries, multi-axle trucks, tippers, container carriers, and heavy goods vehicles (HGV). LCVs and delivery vans are excluded from the primary training subset to prevent contaminating the truck classification head.

3. **Audit Rule C (Two-Wheeler vs. Bicycle Separation):**
   - In Indian urban traffic, motorized two-wheelers (scooters, motorcycles) travel at higher speeds, wear distinct helmets, and possess metallic/engine bulk.
   - Bicycles are human-powered, narrower, and slower.
   - *Policy:* Bicycles are **never** mapped to `motorcycle`. Bicycles are excluded from vehicle bounding boxes to prevent motorcycle false positives on cyclists.

4. **Audit Rule D (Ignore Regions & Negative Frame Policy):**
   - When training images contain unmapped vehicles, regions labeled with confidence score 0 or ignore flags are dropped from loss calculation.
   - Pure negative frames (containing only parking markings, pavement, empty bays, or structural posts) are sampled at a controlled 3% rate to teach the detector background suppression without overwhelming vehicle gradients.

---

## 4. Proposed Expanded-Taxonomy Architecture (Future Phase)

If downstream application requirements expand to classify auto-rickshaws or commercial delivery vans, the system architecture requires an **Expanded 6-Class Taxonomy**:
- `0: car`
- `1: motorcycle`
- `2: bus`
- `3: truck`
- `4: auto_rickshaw` (Three-wheeler)
- `5: light_commercial_vehicle` (LCV / Pickup)

> [!NOTE]
> The current production contract specifies the 4-class ontology. Any 6-class model must be deployed behind an experimental feature flag (`ENABLE_EXPANDED_TAXONOMY: true`) and must not alter the primary 4-class inference endpoint.
