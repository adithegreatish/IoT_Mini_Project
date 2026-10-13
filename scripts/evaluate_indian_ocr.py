"""
Evaluation of OCR Pipeline on Ground-Truth Labeled Indian License Plates
Computes:
- Localization rate
- Readable-output rate
- Exact-match transcription accuracy
- Character Error Rate (CER) via Levenshtein edit distance
- Valid Indian format rate
- Honest abstention rate on illegible samples
- Confidence distribution (p50, p95, mean)
- Latency per recognized crop (mean, p95)
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.config.schema import OCRConfig, PlatesConfig
from app.plates.plate_detector import PlateDetector
from app.ocr.engine import OCREngine

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_parking.ocr_eval")


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def run_indian_ocr_benchmark(gt_manifest_path: Path) -> Dict:
    gt_samples = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    detector = PlateDetector(PlatesConfig(runtime="onnx"))
    ocr = OCREngine(OCRConfig(engine="easyocr", confidence_threshold=0.3))

    total_samples = len(gt_samples)
    plates_detected = 0
    readable_count = 0
    valid_format_count = 0
    exact_match_count = 0
    total_cer_sum = 0.0
    legible_eval_count = 0
    correct_abstentions = 0
    illegible_total = 0

    latencies = []
    confidences = []
    eval_details = []

    for item in gt_samples:
        img_rel = item["image"]
        gt_plate = item.get("ground_truth_plate")
        is_legible = item.get("is_legible", True)

        img_path = paths.resolve(img_rel)
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        plates = detector.detect_plates(img)
        if not plates:
            if not is_legible:
                correct_abstentions += 1
                illegible_total += 1
            eval_details.append({
                "image": Path(img_rel).name,
                "detected": False,
                "is_legible": is_legible,
                "ground_truth": gt_plate,
                "predicted": None,
                "status": "CORRECT_ABSTENTION" if not is_legible else "LOCALIZATION_FAILURE"
            })
            continue

        plates_detected += 1
        top_plate = plates[0]

        t0 = time.perf_counter()
        ocr_res = ocr.recognize(top_plate.crop)
        ocr_lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(ocr_lat_ms)

        if ocr_res.is_readable:
            readable_count += 1
            confidences.append(ocr_res.confidence)

        if ocr_res.is_valid_format:
            valid_format_count += 1

        pred_clean = ocr_res.normalized_text.replace(" ", "").upper()

        if is_legible and gt_plate:
            legible_eval_count += 1
            gt_clean = gt_plate.replace(" ", "").upper()
            is_exact = (pred_clean == gt_clean)
            if is_exact:
                exact_match_count += 1

            edit_dist = levenshtein_distance(gt_clean, pred_clean)
            sample_cer = edit_dist / max(1, len(gt_clean))
            total_cer_sum += sample_cer

            eval_details.append({
                "image": Path(img_rel).name,
                "detected": True,
                "is_legible": True,
                "ground_truth": gt_plate,
                "raw_text": ocr_res.raw_text,
                "predicted_normalized": ocr_res.normalized_text,
                "is_exact_match": is_exact,
                "cer": round(sample_cer, 3),
                "confidence": ocr_res.confidence,
                "is_valid_format": ocr_res.is_valid_format,
                "latency_ms": round(ocr_lat_ms, 2)
            })
        else:
            illegible_total += 1
            # For illegible samples, honest unreadable or format rejection counts as correct abstention
            if not ocr_res.is_readable or not ocr_res.is_valid_format:
                correct_abstentions += 1
            eval_details.append({
                "image": Path(img_rel).name,
                "detected": True,
                "is_legible": False,
                "ground_truth": None,
                "raw_text": ocr_res.raw_text,
                "predicted_normalized": ocr_res.normalized_text,
                "is_valid_format": ocr_res.is_valid_format,
                "confidence": ocr_res.confidence,
                "status": "CORRECT_ABSTENTION" if not ocr_res.is_valid_format else "FALSE_ACCEPT"
            })

    exact_acc = (exact_match_count / max(1, legible_eval_count)) if legible_eval_count else 0.0
    mean_cer = (total_cer_sum / max(1, legible_eval_count)) if legible_eval_count else 0.0
    abstention_rate = (correct_abstentions / max(1, illegible_total)) if illegible_total else 1.0

    return {
        "total_evaluated_samples": total_samples,
        "legible_ground_truth_samples": legible_eval_count,
        "illegible_abstention_samples": illegible_total,
        "plate_detection_count": plates_detected,
        "plate_detection_rate": round(plates_detected / max(1, total_samples), 4),
        "readable_crop_count": readable_count,
        "readable_rate": round(readable_count / max(1, plates_detected), 4) if plates_detected else 0.0,
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": round(exact_acc, 4),
        "character_error_rate": round(mean_cer, 4),
        "valid_format_count": valid_format_count,
        "valid_format_rate": round(valid_format_count / max(1, plates_detected), 4) if plates_detected else 0.0,
        "abstention_success_rate": round(abstention_rate, 4),
        "latency": {
            "mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "p50_ms": round(float(np.median(latencies)), 2) if latencies else 0.0,
            "p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        },
        "confidence": {
            "mean": round(float(np.mean(confidences)), 3) if confidences else 0.0,
            "p50": round(float(np.median(confidences)), 3) if confidences else 0.0,
            "p95": round(float(np.percentile(confidences, 95)), 3) if confidences else 0.0,
        },
        "sample_evaluations": eval_details
    }


def main():
    gt_path = paths.resolve("data/annotations/indian_plates_ground_truth.json")
    if not gt_path.exists():
        logger.error(f"Ground-truth file not found: {gt_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GENUINE INDIAN NUMBER-PLATE OCR BENCHMARK")
    print("=" * 70)
    report = run_indian_ocr_benchmark(gt_path)

    print(f"Total Evaluated Samples:     {report['total_evaluated_samples']}")
    print(f"Legible Samples Evaluated:   {report['legible_ground_truth_samples']}")
    print(f"Illegible/Blur Samples:      {report['illegible_abstention_samples']}")
    print(f"Plate Detection Rate:        {report['plate_detection_count']}/{report['total_evaluated_samples']} ({report['plate_detection_rate']*100:.1f}%)")
    print(f"Readable Crop Rate:          {report['readable_crop_count']}/{report['plate_detection_count']} ({report['readable_rate']*100:.1f}%)")
    print(f"Valid Indian Format Rate:    {report['valid_format_count']}/{report['plate_detection_count']} ({report['valid_format_rate']*100:.1f}%)")
    print(f"Exact-Match Accuracy:        {report['exact_match_count']}/{report['legible_ground_truth_samples']} ({report['exact_match_accuracy']*100:.1f}%)")
    print(f"Character Error Rate (CER):  {report['character_error_rate']*100:.2f}%")
    print(f"Abstention Success Rate:     {report['abstention_success_rate']*100:.1f}%")
    print(f"Mean OCR Latency:            {report['latency']['mean_ms']} ms (p95: {report['latency']['p95_ms']} ms)")
    print(f"Confidence (Mean / p50):     {report['confidence']['mean']} / {report['confidence']['p50']}")
    print("=" * 70)

    # Save report
    out_dir = paths.resolve("data/test_reports/evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "indian_ocr_benchmark_results.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[+] Benchmark results saved to {out_json}")


if __name__ == "__main__":
    main()
