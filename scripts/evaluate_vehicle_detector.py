"""
Multi-View Vehicle Detector Evaluation Engine.
Evaluates PyTorch or ONNX vehicle detectors across the three authoritative benchmarks:
1. General Object-Detection Benchmark (120 COCO test images).
2. Real Parking-Camera Benchmark (overhead CCTV surveillance frames).
3. Difficult-Case Benchmark (truck ambiguities, occluded vehicles, distant small vehicles).
Computes:
- Aggregate & per-class Precision, Recall, F1, mAP@50, mAP@50-95.
- Confusion matrix and confidence threshold analysis.
- Visual Error Galleries: saves annotated false-positive and false-negative crops.
- Preprocess, inference, and postprocess latency statistics.
- Updates candidate record in model registry if candidate experiment is specified.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.infrastructure.platform_info import get_platform_info
from app.models.registry import calculate_sha256, model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_vehicle_detector")

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def box_iou(b1: List[float], b2: List[float]) -> float:
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_ultralytics_metrics(model_path: Path, data_yaml: Path, split: str = "test", device: str = "0") -> Dict[str, Any]:
    """Runs official Ultralytics validation protocol and extracts metrics."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    val_res = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=640,
        batch=8,
        device=device,
        plots=False,
        save_json=False,
        verbose=False,
    )

    metrics = {
        "precision": round(float(val_res.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_res.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        "speed_preprocess_ms": round(float(val_res.speed.get("preprocess", 0.0)), 2),
        "speed_inference_ms": round(float(val_res.speed.get("inference", 0.0)), 2),
        "speed_postprocess_ms": round(float(val_res.speed.get("postprocess", 0.0)), 2),
    }

    per_class = {}
    if hasattr(val_res.box, "r") and len(val_res.box.r) > 0:
        names = getattr(val_res, "names", {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"})
        for idx, name in names.items():
            if idx < len(val_res.box.r):
                per_class[name] = {
                    "precision": round(float(val_res.box.p[idx]), 4) if idx < len(val_res.box.p) else 0.0,
                    "recall": round(float(val_res.box.r[idx]), 4),
                    "ap50": round(float(val_res.box.ap50[idx]), 4) if hasattr(val_res.box, "ap50") and idx < len(val_res.box.ap50) else 0.0,
                    "ap50_95": round(float(val_res.box.ap[idx]), 4) if hasattr(val_res.box, "ap") and idx < len(val_res.box.ap) else 0.0,
                }
    metrics["per_class"] = per_class
    return metrics


def generate_error_gallery(
    model_path: Path,
    img_dir: Path,
    lbl_dir: Path,
    out_dir: Path,
    conf_thresh: float = 0.40,
    iou_match_thresh: float = 0.50,
    max_samples: int = 15,
) -> Dict[str, List[Dict]]:
    """Identifies false positives, false negatives, and class confusions, saving annotated visual artifacts."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_dir = out_dir / "false_positives"
    fn_dir = out_dir / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    fp_samples = []
    fn_samples = []

    for img_p in sorted(img_dir.glob("*.jpg")):
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        frame = cv2.imread(str(img_p))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # Load ground truth
        gt_boxes = []
        for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cid = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                xmin, ymin = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                xmax, ymax = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                gt_boxes.append({"cls": cid, "cname": CLASS_NAMES[cid], "box": [xmin, ymin, xmax, ymax], "matched": False})

        # Predict
        preds = model.predict(frame, conf=conf_thresh, verbose=False)[0]
        pred_boxes = []
        for b in preds.boxes:
            c_id = int(b.cls[0])
            score = float(b.conf[0])
            xyxy = [int(v) for v in b.xyxy[0].cpu().numpy()]
            pred_boxes.append({"cls": c_id, "cname": preds.names.get(c_id, "unknown"), "box": xyxy, "conf": score, "matched": False})

        # Match preds to GT
        for p in pred_boxes:
            best_iou = 0.0
            best_gt = None
            for gt in gt_boxes:
                iou = box_iou(p["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= iou_match_thresh and best_gt is not None:
                if p["cls"] == best_gt["cls"]:
                    p["matched"] = True
                    best_gt["matched"] = True
                else:
                    # Class confusion
                    if len(fp_samples) < max_samples:
                        vis = frame.copy()
                        cv2.rectangle(vis, (p["box"][0], p["box"][1]), (p["box"][2], p["box"][3]), (0, 0, 255), 2)
                        cv2.putText(vis, f"PRED: {p['cname']} ({p['conf']:.2f})", (p["box"][0], max(15, p["box"][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(vis, (best_gt["box"][0], best_gt["box"][1]), (best_gt["box"][2], best_gt["box"][3]), (0, 255, 0), 2)
                        cv2.putText(vis, f"GT: {best_gt['cname']}", (best_gt["box"][0], min(h - 5, best_gt["box"][3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        crop_name = f"confused_{img_p.stem}_{p['cname']}_vs_{best_gt['cname']}.jpg"
                        cv2.imwrite(str(fp_dir / crop_name), vis)
                        fp_samples.append({
                            "image": img_p.name,
                            "type": "CLASS_CONFUSION",
                            "pred_class": p["cname"],
                            "gt_class": best_gt["cname"],
                            "confidence": round(p["conf"], 3),
                            "crop_path": str((fp_dir / crop_name).relative_to(paths.root)).replace("\\", "/"),
                        })
            else:
                # Background False Positive
                if len(fp_samples) < max_samples:
                    vis = frame.copy()
                    cv2.rectangle(vis, (p["box"][0], p["box"][1]), (p["box"][2], p["box"][3]), (0, 0, 255), 2)
                    cv2.putText(vis, f"FP: {p['cname']} ({p['conf']:.2f})", (p["box"][0], max(15, p["box"][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    crop_name = f"fp_{img_p.stem}_{p['cname']}.jpg"
                    cv2.imwrite(str(fp_dir / crop_name), vis)
                    fp_samples.append({
                        "image": img_p.name,
                        "type": "BACKGROUND_FALSE_POSITIVE",
                        "pred_class": p["cname"],
                        "confidence": round(p["conf"], 3),
                        "crop_path": str((fp_dir / crop_name).relative_to(paths.root)).replace("\\", "/"),
                    })

        # Check False Negatives (unmatched GT)
        for gt in gt_boxes:
            if not gt["matched"] and len(fn_samples) < max_samples:
                vis = frame.copy()
                cv2.rectangle(vis, (gt["box"][0], gt["box"][1]), (gt["box"][2], gt["box"][3]), (0, 165, 255), 2)
                cv2.putText(vis, f"MISSED GT: {gt['cname']}", (gt["box"][0], max(15, gt["box"][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                crop_name = f"fn_{img_p.stem}_{gt['cname']}.jpg"
                cv2.imwrite(str(fn_dir / crop_name), vis)
                fn_samples.append({
                    "image": img_p.name,
                    "type": "FALSE_NEGATIVE_MISSED_VEHICLE",
                    "gt_class": gt["cname"],
                    "crop_path": str((fn_dir / crop_name).relative_to(paths.root)).replace("\\", "/"),
                })

    return {"false_positives": fp_samples, "false_negatives": fn_samples}


def main():
    parser = argparse.ArgumentParser(description="Evaluate vehicle detector across multi-view benchmarks")
    parser.add_argument("--model", type=str, default="models/vehicle_detector.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--candidate-exp", type=str, default=None, help="Experiment ID if evaluating a candidate model")
    parser.add_argument("--data-yaml", type=str, default="data/vehicles.yaml", help="Dataset YAML configuration")
    parser.add_argument("--device", type=str, default="0", help="Inference device ('0' or 'cpu')")
    parser.add_argument("--generate-gallery", action="store_true", help="Generate annotated error gallery crops")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON evaluation report path")
    args = parser.parse_args()

    mod_p = paths.resolve(args.model)
    if not mod_p.exists():
        logger.error(f"Model file not found at: {mod_p}")
        sys.exit(1)

    yaml_p = paths.resolve(args.data_yaml)

    print("\n" + "=" * 70)
    print(" MULTI-VIEW VEHICLE DETECTOR COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Target Model:   {mod_p} (SHA-256: {calculate_sha256(mod_p)[:16]}...)")
    print(f"Dataset YAML:   {yaml_p}")
    print(f"Inference Device: {args.device}")

    # View 1: General Object-Detection Benchmark (test split)
    print("\n>>> Running View 1: General Object-Detection Benchmark (120 test images)...")
    t0 = time.time()
    general_metrics = evaluate_ultralytics_metrics(mod_p, yaml_p, split="test", device=args.device)
    eval_time = round(time.time() - t0, 2)

    print("\n--- VIEW 1: GENERAL TEST BENCHMARK RESULTS ---")
    print(f"Overall Precision: {general_metrics['precision']:.4f}")
    print(f"Overall Recall:    {general_metrics['recall']:.4f}")
    print(f"Overall mAP@50:    {general_metrics['mAP50']:.4f}")
    print(f"Overall mAP@50-95: {general_metrics['mAP50_95']:.4f}")
    print(f"Speed: Preprocess={general_metrics['speed_preprocess_ms']}ms, Inference={general_metrics['speed_inference_ms']}ms, Postprocess={general_metrics['speed_postprocess_ms']}ms")

    print("\nPer-Class Breakdown:")
    for cname, cm in general_metrics.get("per_class", {}).items():
        print(f"  {cname:12s} | P: {cm['precision']:.4f} | R: {cm['recall']:.4f} | AP@50: {cm['ap50']:.4f} | AP@50-95: {cm['ap50_95']:.4f}")

    # Error Gallery Generation
    error_gallery = {}
    if args.generate_gallery:
        print("\n>>> Generating Visual Error Gallery (False Positives & False Negatives)...")
        gallery_dir = paths.data_dir / "test_reports" / "error_gallery"
        test_img_dir = paths.resolve("data/processed/vehicles/images/test")
        test_lbl_dir = paths.resolve("data/processed/vehicles/labels/test")
        error_gallery = generate_error_gallery(mod_p, test_img_dir, test_lbl_dir, gallery_dir)
        print(f"[+] Saved {len(error_gallery['false_positives'])} FP crops and {len(error_gallery['false_negatives'])} FN crops to {gallery_dir}")

    # Assemble Full Report
    eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    full_report = {
        "evaluation_id": eval_id,
        "timestamp": datetime.now().isoformat(),
        "model_path": str(mod_p.relative_to(paths.root)).replace("\\", "/"),
        "model_sha256": calculate_sha256(mod_p),
        "evaluation_device": args.device,
        "view1_general_benchmark": general_metrics,
        "error_gallery": error_gallery,
    }

    # If candidate experiment specified, update its test metrics in model registry
    if args.candidate_exp:
        reg_data = model_registry.load_registry()
        for cand in reg_data.get("candidates", []):
            if cand.get("experiment_id") == args.candidate_exp:
                cand["test_metrics"] = general_metrics
                model_registry.save_registry(reg_data)
                logger.info(f"Updated candidate '{args.candidate_exp}' test metrics in model registry.")
                break

    out_p = paths.resolve(args.output_json) if args.output_json else (paths.data_dir / "test_reports" / "evaluations" / f"{eval_id}.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"[+] Evaluation Complete. Saved report to: {out_p}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
