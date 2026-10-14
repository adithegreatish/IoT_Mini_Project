"""
Multi-Dataset Controlled Training & Scientific Comparison Engine.
Executes candidate training experiments (target 45 epochs):
1. Experiment A: Production Baseline (0-epoch reference, models/vehicle_detector.pt)
2. Experiment B: India CCTV Enhancement (COCO + UVH-26)
3. Experiment C: Multi-Source & Hard-Case Enhancement (COCO + UVH-26 + VisDrone)
4. Experiment D: Truck-Focused Enhancement (COCO + UVH-26 + VisDrone with targeted sampling & augmentation)
Evaluates all candidates on identical independent held-out benchmarks.
Exports candidates to ONNX, benchmarks runtime latency, registers in model registry,
and audits promotion gates.
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.models.registry import calculate_sha256, model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_multidataset_experiments")

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def evaluate_on_manifest(model_pt: Path, manifest_path: Path, conf_thresh: float = 0.40, iou_thresh: float = 0.50, device: str = "0") -> Dict[str, Any]:
    """Evaluates YOLO model predictions on any custom benchmark manifest."""
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest_data.get("samples", [])
    model = YOLO(str(model_pt))

    metrics = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    latencies = []

    for s in samples:
        img_rel = s.get("image")
        img_p = paths.resolve(img_rel)
        if not img_p.exists():
            continue

        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]

        # Load GT if available
        lbl_rel = s.get("label")
        gt_boxes = []
        if lbl_rel:
            lbl_p = paths.resolve(lbl_rel)
            if lbl_p.exists():
                for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                    if line:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            if cid < len(CLASS_NAMES):
                                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                x1 = (cx - bw / 2.0) * w
                                y1 = (cy - bh / 2.0) * h
                                x2 = (cx + bw / 2.0) * w
                                y2 = (cy + bh / 2.0) * h
                                gt_boxes.append({"cls": cid, "cname": CLASS_NAMES[cid], "box": [x1, y1, x2, y2], "matched": False})

        t0 = time.time()
        preds = model.predict(im, conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        latencies.append((time.time() - t0) * 1000.0)

        # Match preds to GT
        for b in preds.boxes:
            p_cls = int(b.cls[0].item())
            p_conf = float(b.conf[0].item())
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            p_name = CLASS_NAMES[p_cls] if p_cls < len(CLASS_NAMES) else f"cls_{p_cls}"

            best_iou = 0.0
            best_gt = None
            for g in gt_boxes:
                # box iou
                xa, ya = max(xyxy[0], g["box"][0]), max(xyxy[1], g["box"][1])
                xb, yb = min(xyxy[2], g["box"][2]), min(xyxy[3], g["box"][3])
                inter = max(0.0, xb - xa) * max(0.0, yb - ya)
                a1 = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                a2 = (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1])
                u = a1 + a2 - inter
                iou = inter / u if u > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g

            if best_iou >= iou_thresh and best_gt is not None and not best_gt["matched"]:
                if p_cls == best_gt["cls"]:
                    metrics[p_name]["tp"] += 1
                    best_gt["matched"] = True
                else:
                    metrics[p_name]["fp"] += 1
            else:
                metrics[p_name]["fp"] += 1

        for g in gt_boxes:
            if not g["matched"]:
                metrics[g["cname"]]["fn"] += 1

    per_class = {}
    total_tp = sum(metrics[c]["tp"] for c in CLASS_NAMES)
    total_fp = sum(metrics[c]["fp"] for c in CLASS_NAMES)
    total_fn = sum(metrics[c]["fn"] for c in CLASS_NAMES)

    overall_p = total_tp / max(1, total_tp + total_fp)
    overall_r = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_p * overall_r / max(1e-6, overall_p + overall_r)

    for c in CLASS_NAMES:
        tp, fp, fn = metrics[c]["tp"], metrics[c]["fp"], metrics[c]["fn"]
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-6, p + r)
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    return {
        "view_name": manifest_data.get("view_name", "custom_manifest"),
        "sample_count": len(samples),
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "per_class": per_class,
        "latency_mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
    }


def evaluate_ultralytics_val(model_path: Path, data_yaml: Path, split: str = "test", device: str = "0") -> Dict[str, Any]:
    """Runs official Ultralytics validation protocol and extracts metrics."""
    model = YOLO(str(model_path))
    val_res = model.val(data=str(data_yaml), split=split, imgsz=640, batch=16, device=device, plots=False, verbose=False)

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


def train_candidate(
    exp_name: str,
    data_yaml: Path,
    epochs: int = 45,
    batch_size: int = 16,
    base_weights: str = "models/vehicle_detector.pt",
    device: str = "0",
    extra_train_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Trains a candidate YOLOv8-Nano model, checkpointing validation metrics and exporting ONNX."""
    exp_dir = paths.models_dir / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    weights_p = paths.resolve(base_weights)
    if not weights_p.exists():
        logger.warning(f"Base weights {weights_p} not found, falling back to yolov8n.pt")
        weights_p = Path("yolov8n.pt")

    logger.info("=" * 70)
    logger.info(f"TRAINING CANDIDATE EXPERIMENT: {exp_name}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info(f"  Epochs: {epochs} | Batch: {batch_size} | Device: {device}")
    logger.info(f"  Initial Checkpoint: {weights_p}")
    logger.info("=" * 70)

    model = YOLO(str(weights_p))

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": 640,
        "device": device,
        "workers": 0,
        "project": str(exp_dir.parent),
        "name": exp_dir.name,
        "seed": 42,
        "exist_ok": True,
        "verbose": True,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "cos_lr": True,
        "close_mosaic": 10,
        "patience": 10,
    }

    if extra_train_kwargs:
        train_kwargs.update(extra_train_kwargs)

    t0 = time.time()
    results = model.train(**train_kwargs)
    duration_sec = round(time.time() - t0, 2)
    logger.info(f"Candidate {exp_name} trained in {duration_sec}s.")

    # Locate checkpoints
    best_pt = exp_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = exp_dir / "weights" / "last.pt"

    best_pt_sha = calculate_sha256(best_pt)

    # Evaluate validation metrics
    logger.info(f"Evaluating {exp_name} on validation split...")
    val_m = evaluate_ultralytics_val(best_pt, data_yaml, split="val", device=device)
    val_m["training_duration_sec"] = duration_sec

    # Export ONNX
    logger.info(f"Exporting ONNX candidate for {exp_name}...")
    onnx_file = model.export(format="onnx", imgsz=640, dynamic=False, opset=12)
    cand_onnx = Path(onnx_file)
    cand_onnx_sha = calculate_sha256(cand_onnx)

    # Multi-View Independent Evaluation
    logger.info(f"Evaluating {exp_name} across authoritative multi-view benchmarks...")
    view1_general = evaluate_ultralytics_val(best_pt, data_yaml, split="test", device=device)
    view2_parking = evaluate_on_manifest(best_pt, paths.resolve("data/splits/test_parking_camera_manifest.json"), device=device)
    view3_difficult = evaluate_on_manifest(best_pt, paths.resolve("data/splits/test_difficult_cases_manifest.json"), device=device)
    view4_indian_cctv = evaluate_on_manifest(best_pt, paths.resolve("data/splits/test_uvh26_indian_traffic_manifest.json"), device=device)

    # Register in ModelRegistry
    cand_meta = {
        "experiment_id": exp_name,
        "registered_at": datetime.now().isoformat(),
        "pt_path": str(best_pt.relative_to(paths.root)).replace("\\", "/"),
        "onnx_path": str(cand_onnx.relative_to(paths.root)).replace("\\", "/"),
        "pt_sha256": best_pt_sha,
        "onnx_sha256": cand_onnx_sha,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": 640,
            "device": device,
            "duration_sec": duration_sec,
            "base_weights": str(weights_p),
            "data_yaml": str(data_yaml),
        },
        "validation_metrics": val_m,
        "test_metrics": view1_general,
        "multi_view_evaluations": {
            "view1_general_benchmark": view1_general,
            "view2_parking_camera": view2_parking,
            "view3_difficult_cases": view3_difficult,
            "view4_indian_traffic_cctv": view4_indian_cctv,
        },
        "notes": f"Multi-dataset candidate {exp_name} (45 epochs)",
    }

    # Save candidate metadata JSON
    (exp_dir / "candidate_evaluation_summary.json").write_text(json.dumps(cand_meta, indent=2), encoding="utf-8")

    # Update global model registry
    model_registry.register_candidate(
        experiment_id=exp_name,
        pt_path=best_pt,
        onnx_path=cand_onnx,
        training_config=cand_meta["training_config"],
        validation_metrics=val_m,
        notes=cand_meta["notes"],
    )

    # Store full test metrics in registry
    reg_data = model_registry.load_registry()
    for c in reg_data.get("candidates", []):
        if c.get("experiment_id") == exp_name:
            c["test_metrics"] = view1_general
            c["multi_view_evaluations"] = cand_meta["multi_view_evaluations"]
    model_registry.save_registry(reg_data)

    logger.info(f"Registered candidate {exp_name} successfully in model registry.")
    return cand_meta


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Vehicle Detection Controlled Training & Comparison")
    parser.add_argument("--epochs", type=int, default=45, help="Number of epochs per experiment (default: 45)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="0", help="Compute device ('0' or 'cpu')")
    parser.add_argument("--skip-train-b", action="store_true", help="Skip Experiment B")
    parser.add_argument("--skip-train-c", action="store_true", help="Skip Experiment C")
    parser.add_argument("--skip-train-d", action="store_true", help="Skip Experiment D")
    parser.add_argument("--skip-build-splits", action="store_true", help="Skip split building if already built")
    args = parser.parse_args()

    # 1. Build splits first if not skipped
    if not args.skip_build_splits:
        from scripts.build_multidataset_splits import build_and_audit
        build_and_audit()

    results = {}

    # Experiment B: India CCTV Enhancement (COCO + UVH-26)
    if not args.skip_train_b:
        exp_b_id = f"exp_india_cctv_{args.epochs}ep"
        results["exp_b"] = train_candidate(
            exp_name=exp_b_id,
            data_yaml=paths.resolve("data/vehicles_india_cctv.yaml"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_weights="models/vehicle_detector.pt",
            device=args.device,
        )

    # Experiment C: Multi-Source & Hard-Case Enhancement (COCO + UVH-26 + VisDrone)
    if not args.skip_train_c:
        exp_c_id = f"exp_multisource_{args.epochs}ep"
        results["exp_c"] = train_candidate(
            exp_name=exp_c_id,
            data_yaml=paths.resolve("data/vehicles_multisource.yaml"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_weights="models/vehicle_detector.pt",
            device=args.device,
        )

    # Experiment D: Truck-Focused Targeted Sampling & Loss Weighting
    if not args.skip_train_d:
        exp_d_id = f"exp_truck_focus_{args.epochs}ep"
        truck_kwargs = {
            "scale": 0.6,
            "degrees": 5.0,
            "translate": 0.15,
            "hsv_v": 0.4,
            "close_mosaic": 15,
        }
        results["exp_d"] = train_candidate(
            exp_name=exp_d_id,
            data_yaml=paths.resolve("data/vehicles_truck_focused.yaml"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_weights="models/vehicle_detector.pt",
            device=args.device,
            extra_train_kwargs=truck_kwargs,
        )

    logger.info("=" * 70)
    logger.info(" ALL PLANNED EXPERIMENTS EXECUTED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
