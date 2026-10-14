# Experiment: Truck-Focused & Multi-Source Detection Improvement (Analysis & Decision)

**Document Version:** 2.0.0  
**Experiment Identifiers:** `exp_india_cctv_10ep` (Arm B), `exp_balanced_multisource_10ep` (Arm C), `exp_truck_focus_45ep` (Arm D Planned)  
**Date:** 2026-10-14  
**Author:** Computer Vision Research & Data Governance  
**Hardware Accelerator:** NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.4, FP16 AMP)  
**Base Architecture:** YOLOv8-Nano (`yolov8n.pt` base weights)  

---

## 1. Executive Summary & Gating Decision

A primary objective was to investigate whether extending training to 45 epochs with multi-source data (COCO + UVH-26 + VisDrone) resolves commercial truck recall and precision weaknesses.

### Empirical Decision: **DO NOT PROCEED TO 45-EPOCH TRAINING ON ARM C**
- **Intermediate 10-Epoch Empirical Evidence:** Evaluating the 10-epoch controlled Arm C (`exp_balanced_multisource_10ep`) revealed that incorporating VisDrone's aerial imagery caused severe cross-scale anchor interference on the compact YOLOv8-Nano backbone (3.1M parameters).
- **Metric Degradation:** View 1 General COCO mAP@50 dropped from `0.5930` (baseline) to `0.4578` (-13.52%), and truck AP@50 collapsed to `0.2303` on general photography.
- **Scientific Rationale:** Extending an arm with fundamental scale and feature conflict to 45 epochs would exacerbate anchor degradation and waste compute without addressing the root cause.
- **Recommended Path Forward:** Deploy Arm B (`exp_india_cctv_10ep`) for elevated CCTV and parking camera environments, and pursue neck-only fine-tuning on street-level commercial trucks for general deployment.

---

## 2. Truck Detection Root-Cause & Cross-Arm Evaluation Matrix

Commercial truck detection was evaluated across four strictly isolated test sets using 101-point PR curve integration and operational fixed-threshold metrics at `conf=0.40`:

### Table 1: Commercial Truck Detection Performance Across Experiment Arms

| Metric / Benchmark View | Arm A: Baseline (`v1.0.0`) | Arm B: India CCTV (`exp_india_cctv_10ep`) | Arm C: Balanced Mix (`exp_balanced_multisource_10ep`) | Operational Analysis |
|---|---:|---:|---:|---|
| **View 1 (General COCO): Truck AP@50** | **`0.3893`** | `0.3772` (-1.21%) | `0.2303` (-15.90%) | Arm B preserves horizontal truck AP; Arm C suffers scale interference |
| View 1 (General COCO): Truck AP@50–95 | `0.2430` | **`0.2456` (+0.26%)** | `0.1087` | Arm B slightly improves horizontal bounding-box tightness |
| View 1 (General COCO): Truck Operational P | **`0.3510`** | `0.3148` | `0.2385` | Baseline and Arm B comparable on street trucks |
| View 1 (General COCO): Truck Operational R | `0.4000` | `0.3667` | `0.3000` | Moderate recall on horizontal trailers |
| **View 4 (Untouched Indian CCTV): Truck AP@50** | `0.0952` | **`0.5271` (+43.19%)** | `0.2865` (+19.13%) | **Arm B achieves 5.5x gain on Indian CCTV trucks** |
| View 4 (Untouched Indian CCTV): Truck AP@50–95| `0.0531` | **`0.3842` (+33.11%)** | `0.1768` | Arm B delivers 7.2x gain in strict localization |
| View 4 (Indian CCTV): Operational Precision | `0.1788` | **`0.6842` (+50.54%)** | `0.4412` | **Arm B cuts truck false alarms by 79.6%** |
| View 4 (Indian CCTV): Operational Recall | `0.1988` | **`0.4037` (+20.49%)** | `0.0932` | Arm B doubles detection recall on commercial trucks |
| View 4 (Indian CCTV): **Truck F1 Score** | `0.1883` | **`0.5078` (+31.95%)** | `0.1539` | Transformative gain for CCTV monitoring |
| **View 3 (Difficult Cases): Truck AP@50** | **`0.3810`** | `0.3755` | `0.3257` | Arm B maintains resilience in difficult scenes |
| **View 2 (Real Parking Camera): Truck FP Count**| **4** | **4** | **0** | Baseline & Arm B stable on empty lot barriers |

---

## 3. Analysis of Truck Detection Failure Modes & Mitigations

### 3.1 Perspective Distortion & Foreshortening
- **Baseline Failure:** Street photography portrays trucks with prominent horizontal grilles, vertical cabs, and side wheel assemblies (aspect ratio ~1.20). Pole-mounted surveillance cameras view roofs and flatbeds from 30°–60° elevation (aspect ratio ~0.98). Baseline achieved only `9.52%` AP@50 on CCTV trucks.
- **Arm B Resolution:** Exposure to 750 high-angle commercial truck annotations from UVH-26 enabled Arm B to learn roof containers, corrugated cargo bodies, and flatbeds, driving CCTV truck AP@50 to `52.71%`.

### 3.2 False Alarms on Infrastructure & Shadows
- **Baseline Failure:** Roadside concrete barriers, overpass shadows, and corrugated warehouse walls triggered 147 false truck detections in 200 CCTV frames.
- **Arm B Resolution:** False alarms dropped from 147 to 30 (**79.6% reduction**), elevating operational precision from `17.88%` to `68.42%`.

### 3.3 Large Foreground vs. Small/Distant Trucks (The VisDrone Trade-off)
- Adding VisDrone to train small/distant trucks (<0.6% frame area) degraded YOLOv8-Nano's feature representations across all vehicle classes. A 3.1M-parameter model does not have sufficient capacity to simultaneously encode high-altitude aerial blobs and 1080p Safe City commercial trucks.
- **Conclusion:** Small/distant vehicle detection requires higher-capacity backbones (YOLOv8-Small or YOLOv8-Medium) or a decoupled multi-scale tiling pipeline rather than mixed training into a nano detector.

---

## 4. Promotion Gate Result & Governance Record

```
[Candidate Model: exp_balanced_multisource_10ep]
  Baseline mAP@50:         0.5950
  Candidate mAP@50:        0.4578 (Delta: -0.1372) -> FAILED
  Class Regression Bounds: Car (-0.0846), Motorcycle (-0.2008), Bus (-0.1065), Truck (-0.1570) -> FAILED
  Promotion Status:        BLOCKED BY GATE CHECKS

[Candidate Model: exp_india_cctv_10ep]
  Baseline mAP@50:         0.5950
  Candidate mAP@50:        0.5722 (Delta: -0.0228) -> FAILED
  Class Regression Bounds: Motorcycle dropped by 7.45% on COCO -> FAILED
  Promotion Status:        BLOCKED FROM GENERAL PRODUCTION; RETAINED AS SPECIALIZED CCTV CANDIDATE

[Active Production Deployment]
  Model:                   vehicle-yolo-v8n v1.0.0 (models/vehicle_detector.onnx)
  SHA-256:                 8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc
  Deployment Invariant:    UNTOUCHED & PRESERVED
```

---

## 5. Evidence-Based Recommendation for Next Experiment

Rather than launching arbitrary retraining:
1. **Domain-Specialized Routing:** Utilize Candidate B (`exp_india_cctv_10ep`) directly for CCTV-based parking and traffic monitoring installations where elevation angle is fixed and two-wheelers are dense.
2. **Backbone-Frozen Feature Fusion:** If a single unified checkpoint is mandatory, freeze YOLOv8-Nano backbone stages 0–4 and fine-tune only neck and detection heads on COCO + UVH-26 trucks/buses with high-weight loss on horizontal street motorcycles, omitting VisDrone entirely.
