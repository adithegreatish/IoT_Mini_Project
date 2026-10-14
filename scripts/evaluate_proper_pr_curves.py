"""
Rigorous Multi-View PR-Curve and Operational Metric Evaluation Engine.
Enforces:
1. Full precision-recall curve integration (101-point COCO interpolation) over conf=[0.001, 1.0].
   No single confidence threshold is ever used to calculate AP or mAP.
2. Operational fixed-threshold metrics (P, R, F1 at conf=0.40) are explicitly separated.
3. Evaluates models on identical frozen test manifests:
   - View 1: General Object-Detection (120 COCO test images)
   - View 2: Parking CCTV (30 PKLot overhead frames)
   - View 3: Difficult Cases (87 hard cases)
   - View 4: Independent Indian CCTV Held-out Test (200 untouched UVH-26 frames)
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple


root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.infrastructure.paths import paths
from app.models.registry import calculate_sha256

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_proper_pr_curves")

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def box_iou(b1: List[float], b2: List[float]) -> float:
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap_from_pr_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes AP via standard 101-point COCO interpolation over recall in [0, 1]."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Compute maximum precision to the right (envelope)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = np.interp(recall_points, mrec, mpre)
    return float(np.mean(interpolated_precisions))


def evaluate_manifest_with_pr_curves(
    model: YOLO,
    manifest_path: Path,
    operational_conf: float = 0.40,
    device: str = "0",
) -> Dict[str, Any]:
    """
    Evaluates model on manifest:
    1. Collects all detections at conf=0.001 across all frames.
    2. Computes true AP@50 and AP@50-95 using full 101-point PR curve.
    3. Separately computes operational P, R, F1 at operational_conf.
    """
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest_data.get("samples", [])

    all_detections = {c: [] for c in range(len(CLASS_NAMES))}
    all_gt = {c: [] for c in range(len(CLASS_NAMES))}
    total_gt_by_class = Counter = {c: 0 for c in range(len(CLASS_NAMES))}

    # Operational metric tallies at operational_conf
    op_metrics = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    latencies = []
    gt_boxes_by_img = defaultdict(list)

    for img_idx, s in enumerate(samples):
        img_p = paths.resolve(s.get("image"))
        if not img_p.exists():
            continue
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]

        lbl_rel = s.get("label")
        gt_boxes = []
        if lbl_rel:
            lbl_p = paths.resolve(lbl_rel)
            if lbl_p.exists():
                for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        if cid < len(CLASS_NAMES):
                            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            x1 = (cx - bw / 2.0) * w
                            y1 = (cy - bh / 2.0) * h
                            x2 = (cx + bw / 2.0) * w
                            y2 = (cy + bh / 2.0) * h
                            box_dict = {"cls": cid, "box": [x1, y1, x2, y2], "img_idx": img_idx, "matched": False}
                            gt_boxes.append(box_dict)
                            gt_boxes_by_img[img_idx].append(box_dict)
                            total_gt_by_class[cid] += 1
                            all_gt[cid].append([x1, y1, x2, y2])



        # Run inference at low conf threshold (0.001) for full PR curve
        t0 = time.time()
        preds = model.predict(im, conf=0.001, imgsz=640, device=device, verbose=False)[0]
        latencies.append((time.time() - t0) * 1000.0)

        # Store predictions
        for b in preds.boxes:
            p_cls = int(b.cls[0].item())
            p_conf = float(b.conf[0].item())
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            if p_cls < len(CLASS_NAMES):
                all_detections[p_cls].append({"conf": p_conf, "box": xyxy, "img_idx": img_idx})

        # Separate: Operational metrics at operational_conf
        op_gt = [dict(g) for g in gt_boxes]
        for b in preds.boxes:
            p_cls = int(b.cls[0].item())
            p_conf = float(b.conf[0].item())
            if p_conf < operational_conf or p_cls >= len(CLASS_NAMES):
                continue
            p_name = CLASS_NAMES[p_cls]
            xyxy = b.xyxy[0].cpu().numpy().tolist()

            best_iou = 0.0
            best_g = None
            for g in op_gt:
                if g["cls"] == p_cls and not g["matched"]:
                    iou = box_iou(xyxy, g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_g = g
            if best_iou >= 0.50 and best_g is not None:
                op_metrics[p_name]["tp"] += 1
                best_g["matched"] = True
            else:
                op_metrics[p_name]["fp"] += 1

        for g in op_gt:
            if not g["matched"]:
                op_metrics[CLASS_NAMES[g["cls"]]]["fn"] += 1

    # Group GT by image index and class for fast matching
    gt_by_img = defaultdict(lambda: defaultdict(list))
    for s_idx, g_list in enumerate(all_gt):
        pass

    # Compute PR-curve AP@50 and AP@50-95 across IoU thresholds [0.50, 0.55, ..., 0.95]
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    per_class_ap = {}

    for c in range(len(CLASS_NAMES)):
        cname = CLASS_NAMES[c]
        dets = all_detections[c]
        dets.sort(key=lambda x: x["conf"], reverse=True)
        num_gt = total_gt_by_class[c]

        if num_gt == 0:
            per_class_ap[cname] = {"ap50": 0.0, "ap50_95": 0.0}
            continue

        ap_per_iou = []
        for iou_th in iou_thresholds:
            matched_gt = set()
            tp = np.zeros(len(dets))
            fp = np.zeros(len(dets))

            for d_idx, det in enumerate(dets):
                d_box = det["box"]
                d_img = det["img_idx"]

                # Find candidate GT in the SAME image only
                best_iou = 0.0
                best_gt_idx = None

                img_gt_boxes = [g for g in gt_boxes_by_img[d_img] if g["cls"] == c]
                for g_idx, g in enumerate(img_gt_boxes):
                    gt_key = f"{d_img}_{c}_{g_idx}"
                    if gt_key in matched_gt:
                        continue
                    iou = box_iou(d_box, g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_key

                if best_iou >= iou_th and best_gt_idx is not None:
                    tp[d_idx] = 1.0
                    matched_gt.add(best_gt_idx)
                else:
                    fp[d_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / max(1, num_gt)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
            ap = compute_ap_from_pr_curve(rec, prec)
            ap_per_iou.append(ap)


        per_class_ap[cname] = {
            "ap50": round(float(ap_per_iou[0]), 4),
            "ap50_95": round(float(np.mean(ap_per_iou)), 4),
        }

    mAP50 = round(float(np.mean([per_class_ap[c]["ap50"] for c in CLASS_NAMES])), 4)
    mAP50_95 = round(float(np.mean([per_class_ap[c]["ap50_95"] for c in CLASS_NAMES])), 4)

    # Operational metrics calculation
    total_tp = sum(op_metrics[c]["tp"] for c in CLASS_NAMES)
    total_fp = sum(op_metrics[c]["fp"] for c in CLASS_NAMES)
    total_fn = sum(op_metrics[c]["fn"] for c in CLASS_NAMES)

    op_p = round(total_tp / max(1, total_tp + total_fp), 4)
    op_r = round(total_tp / max(1, total_tp + total_fn), 4)
    op_f1 = round(2 * op_p * op_r / max(1e-6, op_p + op_r), 4)

    per_class_op = {}
    for c in CLASS_NAMES:
        tp = op_metrics[c]["tp"]
        fp = op_metrics[c]["fp"]
        fn = op_metrics[c]["fn"]
        p = round(tp / max(1, tp + fp), 4)
        r = round(tp / max(1, tp + fn), 4)
        f1 = round(2 * p * r / max(1e-6, p + r), 4)
        per_class_op[c] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    return {
        "view_name": manifest_data.get("view_name", "manifest"),
        "sample_count": len(samples),
        "pr_curve_metrics": {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "per_class": per_class_ap,
        },
        "operational_metrics_at_conf40": {
            "precision": op_p,
            "recall": op_r,
            "f1": op_f1,
            "per_class": per_class_op,
        },
        "latency_mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    }


def evaluate_ultralytics_official(model_pt: Path, data_yaml: Path, split: str = "test", device: str = "0") -> Dict[str, Any]:
    """Runs official Ultralytics validation protocol and extracts metrics."""
    model = YOLO(str(model_pt))
    res = model.val(data=str(data_yaml), split=split, imgsz=640, batch=16, device=device, plots=False, verbose=False)
    metrics = {
        "mAP50": round(float(res.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(res.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(res.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(res.results_dict.get("metrics/recall(B)", 0.0)), 4),
    }
    per_class = {}
    if hasattr(res.box, "r") and len(res.box.r) > 0:
        names = getattr(res, "names", {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"})
        for idx, name in names.items():
            if idx < len(res.box.r):
                per_class[name] = {
                    "precision": round(float(res.box.p[idx]), 4) if idx < len(res.box.p) else 0.0,
                    "recall": round(float(res.box.r[idx]), 4),
                    "ap50": round(float(res.box.ap50[idx]), 4) if hasattr(res.box, "ap50") and idx < len(res.box.ap50) else 0.0,
                    "ap50_95": round(float(res.box.ap[idx]), 4) if hasattr(res.box, "ap") and idx < len(res.box.ap) else 0.0,
                }
    metrics["per_class"] = per_class
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Proper PR Curve and Operational Evaluator")
    parser.add_argument("--model", type=str, default="models/vehicle_detector.pt", help="Path to model (.pt)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0' or 'cpu')")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    model_p = paths.resolve(args.model)
    assert model_p.exists(), f"Model {model_p} not found!"

    logger.info("=" * 70)
    logger.info(f"EVALUATING MODEL WITH PROPER PR-CURVES: {model_p.name}")
    logger.info("=" * 70)

    model = YOLO(str(model_p))

    # 1. View 1: General Test (Official Ultralytics protocol on vehicles_india_cctv.yaml, split=test)
    logger.info("Evaluating View 1 (General COCO Test Benchmark, 120 images)...")
    v1_metrics = evaluate_ultralytics_official(model_p, paths.resolve("data/vehicles_india_cctv.yaml"), split="test", device=args.device)

    # 2. View 4: Independent Indian CCTV Held-out Test (200 untouched images)
    logger.info("Evaluating View 4 (Independent Held-Out Indian CCTV Benchmark, 200 images)...")
    v4_official = evaluate_ultralytics_official(model_p, paths.resolve("data/vehicles_indian_cctv_heldout.yaml"), split="test", device=args.device)
    m4_path = paths.resolve("data/splits/test_indian_cctv_heldout_manifest.json")
    v4_pr_curve = evaluate_manifest_with_pr_curves(model, m4_path, operational_conf=0.40, device=args.device)

    # 3. View 2: Parking Camera (30 images)
    logger.info("Evaluating View 2 (Parking Camera Benchmark, 30 images)...")
    m2_path = paths.resolve("data/splits/test_parking_camera_manifest.json")
    v2_pr_curve = evaluate_manifest_with_pr_curves(model, m2_path, operational_conf=0.40, device=args.device)

    # 4. View 3: Difficult Cases (87 images)
    logger.info("Evaluating View 3 (Difficult Cases Benchmark, 87 images)...")
    m3_path = paths.resolve("data/splits/test_difficult_cases_manifest.json")
    v3_pr_curve = evaluate_manifest_with_pr_curves(model, m3_path, operational_conf=0.40, device=args.device)

    # 5. Historical Validation Set (for leakage comparison: 200 images from previous val split)
    logger.info("Evaluating Historical Validation Set (200 images previously used in training val)...")
    m_hist_path = paths.resolve("data/splits/test_uvh26_indian_traffic_manifest.json")
    v_hist_curve = evaluate_manifest_with_pr_curves(model, m_hist_path, operational_conf=0.40, device=args.device)

    report = {
        "model_name": model_p.name,
        "model_path": str(model_p),
        "model_sha256": calculate_sha256(model_p),
        "evaluation_timestamp": datetime.now().isoformat(),
        "device": args.device,
        "view1_general_coco_test": v1_metrics,
        "view4_independent_indian_cctv_test": {
            "official_ultralytics_val": v4_official,
            "standalone_pr_curve_and_operational": v4_pr_curve,
        },
        "view2_parking_camera": v2_pr_curve,
        "view3_difficult_cases": v3_pr_curve,
        "historical_validation_split": v_hist_curve,
    }

    if args.output_json:
        out_p = Path(args.output_json)
    else:
        out_p = paths.test_reports_dir / f"pr_curve_eval_{model_p.stem}.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Report saved to: {out_p}")


if __name__ == "__main__":
    main()
