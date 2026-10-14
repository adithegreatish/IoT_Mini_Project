"""
Vehicle Detector Training Pipeline & Candidate Experiment Manager.
Orchestrates training with GPU acceleration (RTX 4070), validation checkpointing,
and ONNX candidate export.
STRICT SAFETY INVARIANT: Never overwrites the active production model!
Saves each run to an isolated experiment directory (models/experiments/<exp_id>/)
and registers the model as a CANDIDATE awaiting independent gate evaluation.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional

import yaml

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.infrastructure.platform_info import get_platform_info
from app.models.registry import calculate_sha256, model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_vehicle_detector")

PRESETS = {
    "balanced": {
        "description": "Standard balanced training matching project baseline",
        "epochs": 35,
        "batch_size": 16,
        "imgsz": 640,
        "lr0": 0.002,
        "optimizer": "auto",
        "close_mosaic": 10,
        "augment": True,
    },
    "accuracy_focused": {
        "description": "Maximum precision & mAP with extended learning and cosine schedule",
        "epochs": 45,
        "batch_size": 16,
        "imgsz": 640,
        "lr0": 0.002,
        "cos_lr": True,
        "optimizer": "AdamW",
        "weight_decay": 0.0005,
        "close_mosaic": 15,
        "augment": True,
    },
    "small_object_truck_focus": {
        "description": "Targeted truck recall and small distant vehicle detection",
        "epochs": 35,
        "batch_size": 16,
        "imgsz": 640,
        "scale": 0.6,
        "degrees": 5.0,
        "translate": 0.15,
        "close_mosaic": 10,
        "optimizer": "auto",
        "augment": True,
    },
    "pi_latency_optimized": {
        "description": "Compact 480x480 resolution designed for maximum Raspberry Pi CPU throughput",
        "epochs": 30,
        "batch_size": 16,
        "imgsz": 480,
        "lr0": 0.002,
        "optimizer": "auto",
        "close_mosaic": 8,
        "augment": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train vehicle detector candidate model")
    parser.add_argument("--preset", type=str, default="balanced", choices=list(PRESETS.keys()), help="Training configuration preset")
    parser.add_argument("--experiment-name", type=str, default=None, help="Custom identifier for experiment")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu')")
    parser.add_argument("--data-yaml", type=str, default="data/vehicles.yaml", help="Dataset configuration")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial pretrained weights")
    parser.add_argument("--smoke-test", action="store_true", help="1-epoch sanity test")
    parser.add_argument("--experiment-id", type=str, default=None, help="Explicit experiment identifier")
    parser.add_argument("--notes", type=str, default="", help="Experimental notes")
    args = parser.parse_args()

    # Hardware detection
    import torch
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    device = args.device or ("0" if cuda_avail else "cpu")

    # Resolve settings from preset
    preset_cfg = PRESETS[args.preset].copy()
    epochs = 1 if args.smoke_test else (args.epochs or preset_cfg.get("epochs", 35))
    batch_size = 4 if args.smoke_test else (args.batch_size or preset_cfg.get("batch_size", 16))
    imgsz = args.imgsz or preset_cfg.get("imgsz", 640)

    # Unique experiment directory
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        exp_suffix = f"_{args.experiment_name}" if args.experiment_name else f"_{args.preset}"
        if args.smoke_test:
            exp_suffix += "_smoke"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_{timestamp}{exp_suffix}"

    exp_dir = paths.models_dir / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_yaml_p = paths.resolve(args.data_yaml)
    if not data_yaml_p.exists():
        logger.error(f"Dataset YAML not found at: {data_yaml_p}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" VEHICLE DETECTOR EXPERIMENTAL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Experiment ID:       {experiment_id}")
    print(f"Preset:              {args.preset} ({preset_cfg.get('description')})")
    print(f"Pretrained Weights:  {args.weights}")
    print(f"Dataset YAML:        {data_yaml_p}")
    print(f"Epochs:              {epochs} {'(SMOKE TEST)' if args.smoke_test else ''}")
    print(f"Batch Size:          {batch_size}")
    print(f"Input Size:          {imgsz}x{imgsz}")
    print(f"Compute Device:      {device} ({gpu_name})")
    print(f"Output Directory:    {exp_dir}")
    print("=" * 70 + "\n")

    from ultralytics import YOLO

    model = YOLO(args.weights)

    train_kwargs = {
        "data": str(data_yaml_p),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "workers": 0 if sys.platform == "win32" else 2,
        "project": str(exp_dir.parent),
        "name": exp_dir.name,
        "seed": 42,
        "exist_ok": True,
        "verbose": True,
    }

    if preset_cfg.get("optimizer"):
        train_kwargs["optimizer"] = preset_cfg["optimizer"]
    if preset_cfg.get("close_mosaic"):
        train_kwargs["close_mosaic"] = preset_cfg["close_mosaic"]
    if preset_cfg.get("cos_lr"):
        train_kwargs["cos_lr"] = True
    if preset_cfg.get("scale"):
        train_kwargs["scale"] = preset_cfg["scale"]

    start_time = time.time()
    logger.info("Starting candidate model training...")

    results = model.train(**train_kwargs)
    train_duration = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {train_duration}s.")

    # Locate generated checkpoints
    weights_dir = exp_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    candidate_pt = best_pt if best_pt.exists() else last_pt
    if not candidate_pt.exists():
        logger.error(f"Training failed to produce weight checkpoints in {weights_dir}")
        sys.exit(1)

    pt_sha256 = calculate_sha256(candidate_pt)
    logger.info(f"Candidate checkpoint: {candidate_pt} (SHA-256: {pt_sha256[:16]}...)")

    # Run validation on val split
    logger.info("Evaluating candidate on validation split...")
    val_res = model.val(data=str(data_yaml_p), split="val", imgsz=imgsz, device=device, verbose=False)

    val_metrics = {
        "precision": round(float(val_res.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_res.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        "training_duration_sec": train_duration,
    }

    # Extract per-class
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
    val_metrics["per_class"] = per_class

    print("\n" + "-" * 70)
    print(" CANDIDATE VALIDATION METRICS")
    print("-" * 70)
    print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
    print(f"mAP@50:    {val_metrics['mAP50']:.4f} | mAP@50-95: {val_metrics['mAP50_95']:.4f}")
    for cname, cm in per_class.items():
        print(f"  Class {cname:10s}: AP50={cm['ap50']:.4f}, AP50-95={cm['ap50_95']:.4f}, P={cm['precision']:.4f}, R={cm['recall']:.4f}")
    print("-" * 70 + "\n")

    # Export candidate to ONNX
    logger.info("Exporting candidate ONNX artifact...")
    onnx_file = model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=12)
    candidate_onnx = Path(onnx_file)
    onnx_sha256 = calculate_sha256(candidate_onnx)
    logger.info(f"Exported ONNX: {candidate_onnx} (SHA-256: {onnx_sha256[:16]}...)")

    # Save experiment metadata in experiment directory
    exp_meta = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "preset": args.preset,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "device": device,
            "gpu_name": gpu_name,
            "duration_sec": train_duration,
            "weights": args.weights,
        },
        "weights": {
            "checkpoint_pt": str(candidate_pt.relative_to(paths.root)).replace("\\", "/"),
            "checkpoint_sha256": pt_sha256,
            "onnx": str(candidate_onnx.relative_to(paths.root)).replace("\\", "/"),
            "onnx_sha256": onnx_sha256,
        },
        "validation_metrics": val_metrics,
        "notes": args.notes or preset_cfg.get("description", ""),
    }
    (exp_dir / "experiment_metadata.json").write_text(json.dumps(exp_meta, indent=2), encoding="utf-8")

    # Register candidate in ModelRegistry
    model_registry.register_candidate(
        experiment_id=experiment_id,
        pt_path=candidate_pt,
        onnx_path=candidate_onnx,
        training_config=exp_meta["training_config"],
        validation_metrics=val_metrics,
        notes=args.notes or preset_cfg.get("description", ""),
    )

    print("\n" + "=" * 70)
    print(f"[+] CANDIDATE REGISTRATION COMPLETE: {experiment_id}")
    print(f"[+] Isolated Directory: {exp_dir}")
    print(f"[+] Status: REGISTERED AS CANDIDATE (Awaiting evaluation & promotion gate)")
    print(f"[!] Production models were NOT touched.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
