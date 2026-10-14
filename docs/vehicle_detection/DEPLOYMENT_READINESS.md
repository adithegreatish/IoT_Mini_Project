# Vehicle Detector Deployment Readiness Report

**Experiment ID:** `exp_20261014_001738_balanced_smoke`  
**Report Generated:** 2026-10-14T00:19:33.153188  
**Evaluator:** CV / MLOps Continuous Learning Pipeline  
**Target Deployment Target:** Raspberry Pi 5 (8 GB RAM, ARM64 Cortex-A76)  

---

## 1. Executive Summary & Readiness Verdict

### Overall Deployment Status: **REJECTED / ATTENTION REQUIRED**

| Check | Requirement | Result | Status |
|---|---|---|---|
| **Dataset & Annotation Validation** | Zero broken files, valid boxes | PASSED | PASS |
| **Split Hygiene & Leakage** | 0 partition collisions | ZERO LEAKAGE | PASS |
| **Aggregate mAP@50 Gain** | Delta >= 0.00% (non-regression) | -0.4966 | REGRESSION |
| **mAP@50-95 Retention** | Delta >= -0.03 | -0.3363 | REGRESSION |
| **Critical Class Stability** | No class drops > 5% | 8 regressions | WARNING |
| **ONNX Numerical Parity** | Output diff < 0.05 | Verified | PASS |
| **Deployment File Size** | < 25 MB ONNX | 12.18 MB | PASS |

---

## 2. Comparative Evaluation Matrix: Baseline vs Candidate

| Metric | Active Baseline | Candidate (exp_20261014_001738_balanced_smoke) | Delta | Verdict |
|---|---:|---:|---:|---|
| **mAP@50** | `0.5950` | `0.0984` | `-0.4966` | **REGRESSED** |
| **mAP@50-95** | `0.3797` | `0.0434` | `-0.3363` | **REGRESSED** |
| **Precision** | `0.6224` | `0.1436` | `-0.4788` | **REGRESSED** |
| **Recall** | `0.5459` | `0.2231` | `-0.3228` | **REGRESSED** |

### Per-Class Performance Breakdown (AP@50)
| Class | Baseline AP@50 | Candidate AP@50 | Delta AP@50 | Status |
|---|---:|---:|---:|---|
| **Car** | `0.6038` | `0.1959` | `-0.4079` | **REGRESSED** |
| **Motorcycle** | `0.6698` | `0.0655` | `-0.6043` | **REGRESSED** |
| **Bus** | `0.7192` | `0.0804` | `-0.6388` | **REGRESSED** |
| **Truck** | `0.3873` | `0.0519` | `-0.3354` | **REGRESSED** |

---

## 3. Edge Latency & Hardware Efficiency

- **Deployment ONNX File:** `models\experiments\exp_20261014_001738_balanced_smoke\weights\best.onnx`
- **ONNX Checksum:** `0928321cc66abf4facb9f206f8c0f517e57a992c340d6f4f10ec5a3fea24002d`
- **Model Resolution:** `640x640`
- **Inference Latency (Windows CPU 2-threads):** `61.63 ms` (~`16.23 FPS`)
- **Estimated Pi 5 Latency:** ~`110.9 ms` (Well within decoupled 4 FPS background inference cadence)

---

## 4. Promotion Recommendation & Gate Status

**RECOMMENDATION:** Candidate `exp_20261014_001738_balanced_smoke` does NOT satisfy all automated promotion gates.
Blocking reasons:
- [!] Candidate mAP@50 (0.0984) does not satisfy required improvement against baseline (0.5950, delta: -0.4966).
- [!] Candidate mAP@50-95 (0.0434) regresses by more than 3% vs baseline (0.3797).
- [!] Class 'car' regressed by 0.4079 (baseline AP50=0.6038, candidate AP50=0.1959, max allowed regression=0.0500).
- [!] Class 'motorcycle' regressed by 0.6043 (baseline AP50=0.6698, candidate AP50=0.0655, max allowed regression=0.0500).
- [!] Class 'bus' regressed by 0.6388 (baseline AP50=0.7192, candidate AP50=0.0804, max allowed regression=0.0500).
- [!] Class 'truck' regressed by 0.3354 (baseline AP50=0.3873, candidate AP50=0.0519, max allowed regression=0.0500).

The production model remains untouched. Review error galleries in `data/test_reports/error_gallery/` before deciding whether to iterate further or manually force promotion with `--force`.
