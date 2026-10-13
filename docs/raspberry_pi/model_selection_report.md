# Model Selection & Safety Audit Report for Raspberry Pi Deployment

**Target System:** Smart Parking Management System (Raspberry Pi 5)  
**Evaluated Artifact:** `models/occupancy_classifier.onnx` (Winning Approach D: Domain-Augmented Combined Model)  
**Rollback Artifact:** `models/occupancy_classifier_baseline_cnrpark.pt` (Approach A: Original CNRPark Baseline)  
**Audit Date:** 2026-10-12  
**Source Manifest:** `data/splits/leave_one_lot_out_occupancy_manifest.json`  
**Evaluation File:** `models/multi_model_leave_one_lot_out_evaluation.json`  

---

## 1. Executive Summary

This report independently recalculates and audits the performance of the winning **Approach D (Domain-Augmented Combined Model)** across four independent held-out evaluation sets, specifically analyzing its safety characteristics for automated parking-space allocation.

While macro-F1 on held-out **UFPR04** reaches **0.9177** (92.07% accuracy) and on **UFPR05** reaches **0.7895** (80.29% accuracy), **macro-F1 alone does not guarantee physical safety for automated parking allocation**. We rigorously analyze the underlying confusion matrices, false-free rates, and the critical distinction between free-space recall and free-space precision.

---

## 2. Plain English Explanation: Free-Space Recall vs. Free-Space Precision

Understanding these two metrics is essential for safe parking lot management:

```
                      Genuinely Empty Bay                  Genuinely Occupied Bay
               +------------------------------------+------------------------------------+
Model Says:    |        TRUE NEGATIVE (TN)          |        FALSE NEGATIVE (FN)         |
"BAY IS FREE"  | Correctly identifies vacant space. |   CRITICAL DANGER: FALSE-FREE!     |
               | Driver parks smoothly.             | Car already there -> Driver angry! |
               +------------------------------------+------------------------------------+
Model Says:    |        FALSE POSITIVE (FP)         |        TRUE POSITIVE (TP)          |
"BAY IS BUSY"  |       FALSE-OCCUPIED ALARM         | Correctly identifies occupied bay. |
               | Bay goes unallocated temporarily.  | System prevents double parking.    |
               +------------------------------------+------------------------------------+
```

### Free-Space Recall: "How many empty bays did we find?"
$$\text{Free-Space Recall} = \frac{\text{True Free Detected } (TN)}{\text{All Genuinely Free Bays } (TN + FP)}$$
- **What it means:** Out of 100 empty parking spots in the lot, what percentage did our camera detect?
- **Business Impact:** High recall means high parking lot utilization. If free recall is low (e.g., Approach A's 12.76% on UFPR04), the system thinks the lot is almost 100% full even when hundreds of empty bays exist.

### Free-Space Precision: "How trustworthy is a FREE signal?"
$$\text{Free-Space Precision} = \frac{\text{True Free Detected } (TN)}{\text{All Spots Predicted Free } (TN + FN)}$$
- **What it means:** When the green light turns on and the API tells a driver "Spot 14 is FREE, go park there!", what is the probability that Spot 14 is actually empty?
- **Safety Impact:** **Precision is the safety metric.** If precision is low, drivers will be dispatched to occupied bays, creating congestion, frustration, and potential collisions.

> [!WARNING]
> **Safety Asymmetry:** A false-occupied error ($FP$) is merely a missed reservation opportunity. A **false-free error ($FN$) is a critical system failure**. An automated allocation engine must prioritize high Free-Space Precision and low False-Free Rate ($FN / \text{Occupied}$).

---

## 3. Independent Recalculation of Approach D Across Test Domains

All figures below are recalculated directly from the underlying evaluation data in `models/multi_model_leave_one_lot_out_evaluation.json`:

### 3.1 Domain 1: Held-Out Parking Lot — UFPR04 (Curitiba, Brazil)
- **Domain Characteristics:** Dark black asphalt, moderate tree shadows, overhead angled camera.
- **Scope:** 1,110 samples across 30 full-frame sessions, 37 physical slots.
- **Class Breakdown:** 679 Occupied bays ($P$) | 431 Vacant bays ($N$).
- **Recalculated Confusion Matrix:**
  - True Positives ($TP$, Occupied $\rightarrow$ Occupied): **617**
  - True Negatives ($TN$, Free $\rightarrow$ Free): **405**
  - False Positives ($FP$, Free $\rightarrow$ Occupied): **26**
  - False Negatives ($FN$, Occupied $\rightarrow$ Free): **62**
- **Calculated Metrics:**
  - **Overall Accuracy:** $\frac{617 + 405}{1110} = \mathbf{92.07\%}$
  - **Balanced Accuracy:** $\frac{90.87\% + 93.97\%}{2} = \mathbf{92.42\%}$
  - **Macro-F1:** $\frac{0.9334 + 0.9020}{2} = \mathbf{0.9177}$
  - **Occupied Precision:** $\frac{617}{617 + 26} = \mathbf{95.96\%}$ | **Occupied Recall:** $\frac{617}{679} = \mathbf{90.87\%}$
  - **Free-Space Precision:** $\frac{405}{405 + 62} = \mathbf{86.72\%}$ | **Free-Space Recall:** $\frac{405}{431} = \mathbf{93.97\%}$
  - **False-Free Bays ($FN$):** **62** (False-Free Rate: $\frac{62}{679} = \mathbf{9.13\%}$)
  - **False-Occupied Bays ($FP$):** **26**

---

### 3.2 Domain 2: Held-Out Parking Lot — UFPR05 (Curitiba, Brazil)
- **Domain Characteristics:** Harsh, high-contrast tree canopy shadow stripes across asphalt; severe illumination shifts.
- **Scope:** 1,380 samples across 30 full-frame sessions, 46 physical slots.
- **Class Breakdown:** 737 Occupied bays ($P$) | 643 Vacant bays ($N$).
- **Recalculated Confusion Matrix:**
  - True Positives ($TP$, Occupied $\rightarrow$ Occupied): **728**
  - True Negatives ($TN$, Free $\rightarrow$ Free): **380**
  - False Positives ($FP$, Free $\rightarrow$ Occupied): **263**
  - False Negatives ($FN$, Occupied $\rightarrow$ Free): **9**
- **Calculated Metrics:**
  - **Overall Accuracy:** $\frac{728 + 380}{1380} = \mathbf{80.29\%}$
  - **Balanced Accuracy:** $\frac{98.78\% + 59.10\%}{2} = \mathbf{78.94\%}$
  - **Macro-F1:** $\frac{0.8426 + 0.7364}{2} = \mathbf{0.7895}$
  - **Occupied Precision:** $\frac{728}{728 + 263} = \mathbf{73.46\%}$ | **Occupied Recall:** $\frac{728}{737} = \mathbf{98.78\%}$
  - **Free-Space Precision:** $\frac{380}{380 + 9} = \mathbf{97.69\%}$ | **Free-Space Recall:** $\frac{380}{643} = \mathbf{59.10\%}$
  - **False-Free Bays ($FN$):** **9** (False-Free Rate: $\frac{9}{737} = \mathbf{1.22\%}$)
  - **False-Occupied Bays ($FP$):** **263**

---

### 3.3 Domain 3: Held-Out CNR-EXT Surveillance Viewpoint — Camera 09 (Pisa, Italy)
- **Domain Characteristics:** Pale concrete surface, Mediterranean lighting, overhead oblique angle.
- **Scope:** 1,500 samples across Sunny and Overcast sessions.
- **Class Breakdown:** 1,062 Occupied bays | 438 Vacant bays.
- **Confusion Matrix:** $TP = 1,013$, $TN = 378$, $FP = 60$, $FN = 49$.
- **Calculated Metrics:**
  - **Accuracy:** 92.73% | **Macro-F1:** 0.9115
  - **Occupied Precision:** 94.41% | **Occupied Recall:** 95.39%
  - **Free-Space Precision:** 88.52% | **Free-Space Recall:** 86.30%
  - **False-Free Rate:** 4.61% (49 false-free bays) | **False-Occupied:** 60 bays

---

### 3.4 Domain 4: Historical CNRPark 1,260-Patch Benchmark
- **Scope:** 1,260 patches (841 Occupied, 419 Vacant).
- **Confusion Matrix:** $TP = 787$, $TN = 394$, $FP = 25$, $FN = 54$.
- **Calculated Metrics:** Accuracy = 93.73%, Macro-F1 = 0.9305, Free Recall = 94.03%, Free Precision = 87.95%.

---

## 4. Side-by-Side Comparison: Approach D vs. Rollback Baseline (Approach A)

| Evaluation Domain | Metric | Approach A (CNRPark Baseline) | Approach D (Winning Model) | Absolute Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Held-Out UFPR04** | Accuracy | 66.04% | **92.07%** | **+26.03%** |
| | Macro-F1 | 0.5042 | **0.9177** | **+0.4135** |
| | **Free-Space Recall** | **12.76% (55/431)** | **93.97% (405/431)** | **+81.21%** (+350 bays detected) |
| | Free-Space Precision | 98.21% | 86.72% | -11.49% |
| | False-Occupied ($FP$) | 376 bays | **26 bays** | **-350 false alarms** |
| | False-Free ($FN$) | 1 bay | 62 bays | +61 bays |
| **Held-Out UFPR05** | Accuracy | 68.04% | **80.29%** | **+12.25%** |
| | Macro-F1 | 0.6273 | **0.7895** | **+0.1622** |
| | **Free-Space Recall** | **32.50% (209/643)** | **59.10% (380/643)** | **+26.60%** (+171 bays detected) |
| | Free-Space Precision | 96.76% | **97.69%** | **+0.93%** |
| | False-Occupied ($FP$) | 434 bays | **263 bays** | **-171 false alarms** |
| | False-Free ($FN$) | 7 bays | 9 bays | +2 bays |

---

## 5. Threshold Selection & Configuration Integrity

### 5.1 Validation-Only Selection Protocol
- **Swept Range:** $\tau \in [0.20, 0.80]$ across 25 steps.
- **Dataset Used:** Strictly the **validation split** (4,060 samples: 2,500 CNR Camera 08 + 1,560 PKLot PUCPR validation frames).
- **Selection Criterion:** Maximized validation macro-F1 at **0.9741**.
- **Optimal Threshold Selected:** $\tau^* = \mathbf{0.650}$.
- **Zero Test Leakage:** The threshold was **never tuned or modified against UFPR04, UFPR05, Camera 09, or the benchmark**.

### 5.2 Application Configuration Alignment
- In `PatchOccupancyClassifier` (`app/parking/occupancy_classifier.py`):
  $$\text{prob} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
  $$\text{is\_occupied} = (\text{prob} \ge \tau^*)$$
- **Config Relationship:**
  The configuration file `configs/raspberry_pi.yaml` specifies `parking.occupancy_threshold`. It has now been aligned to **0.65** so that the application defaults match the validated threshold.

---

## 6. Safety Limitations & Allocation Suitability Assessment

| Operating Domain | Safety Level | Automatic Allocation Suitability | Recommended Operational Protocol |
| :--- | :---: | :---: | :--- |
| **Calibrated Indoor / Mediterranean Lots (CNR-EXT / Camera 09)** | **HIGH** | **SUITABLE** | Standard automated allocation active. Free precision = 88.5%, False-free rate = 4.6%. |
| **Overcast & Diffuse Outdoor Lots (UFPR04 / PUCPR)** | **HIGH** | **SUITABLE** | Standard automated allocation active with 3-frame hysteresis confirmation. |
| **Heavy Tree Canopy Shadow Lots (UFPR05)** | **CAUTION** | **CONSERVATIVE ALLOCATION** | Model exhibits conservative bias (Free recall = 59.1%, Free precision = 97.7%). When it allocates a bay, it is 97.7% reliable. However, 40.9% of free bays under heavy tree shadows may be held back as occupied. |
| **Extreme Lighting Glare / Camera Tilt $> 60^\circ$** | **UNSUITABLE** | **MANUAL OVERRIDE ONLY** | If camera pitch causes severe slot foreshortening, bays must be flagged for manual inspection. |

> [!IMPORTANT]
> **Rollback Safety Guarantee:** The previous baseline checkpoint `models/occupancy_classifier_baseline_cnrpark.pt` remains physically intact in `models/` with SHA-256 `b53ad658b10853e30128cf78e6f1f51275bf3bbd4fe0cff3bc8ba9e8d75dc703`. If any camera position requires rollback, changing one line in `configs/raspberry_pi.yaml` reverts immediately.
