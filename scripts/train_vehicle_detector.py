"""Orchestrates genuine vehicle detector training, validation, checkpointing, and ONNX export."""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional

import yaml

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.infrastructure.platform_info import get_platform_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_vehicle_detector")


def calculate_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def update_model_manifest(
    model_id: str,
    onnx_path: Path,
    onnx_sha256: str,
    metrics: Dict[str, Any],
) -> None:
    """Updates models/manifest.json with genuine artifact hashes and benchmark metrics."""
    manifest_path = paths.models_manifest_file
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}")
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for m in manifest.get("models", []):
            if m.get("model_id") == model_id:
                m["sha256"] = onnx_sha256
                m["updated_at"] = datetime.now().isoformat()
                m["file_path"] = str(onnx_path.relative_to(paths.root)).replace("\\", "/")
                m.setdefault("metrics", {}).update(metrics)
                logger.info(f"Updated manifest entry for {model_id}: SHA-256={onnx_sha256[:16]}...")
                break
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune YOLO vehicle detector")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu', etc.)")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid 1-epoch smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    paths.ensure_directories()
    cfg_path = paths.resolve(args.config)
    if not cfg_path.exists():
        logger.error(f"Training configuration not found at {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    v_cfg = cfg.get("training", {}).get("vehicle_detector", {})
    architecture = v_cfg.get("architecture", "yolov8n")
    pretrained_weights = v_cfg.get("pretrained_weights", "yolov8n.pt")
    epochs = 1 if args.smoke_test else (args.epochs or v_cfg.get("epochs", 25))
    batch_size = 4 if args.smoke_test else (args.batch_size or v_cfg.get("batch_size", 16))
    imgsz = args.imgsz or v_cfg.get("imgsz", 640)
    workers = 0 if sys.platform == "win32" else v_cfg.get("workers", 2)  # Avoid Windows multiprocessing worker issues
    seed = v_cfg.get("seed", 42)

    save_dir = paths.resolve(v_cfg.get("save_dir", "models/training_runs/vehicle"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Validate dataset config
    dataset_yaml = paths.resolve("data/vehicles.yaml")
    if not dataset_yaml.exists():
        logger.error(f"Vehicle dataset YAML not found at {dataset_yaml}. Run scripts/acquire_datasets.py first!")
        sys.exit(1)

    # Detect hardware
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
    except ImportError:
        logger.error("PyTorch is not installed in current environment.")
        sys.exit(1)

    device = args.device or ("0" if cuda_avail else "cpu")
    if device != "cpu" and not cuda_avail:
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print("\n" + "=" * 65)
    print("VEHICLE DETECTOR TRAINING PIPELINE")
    print("=" * 65)
    print(f"Architecture:       {architecture}")
    print(f"Pretrained Weights: {pretrained_weights}")
    print(f"Dataset Config:     {dataset_yaml}")
    print(f"Epochs:             {epochs} {'(SMOKE TEST)' if args.smoke_test else ''}")
    print(f"Batch Size:         {batch_size}")
    print(f"Image Size:         {imgsz}")
    print(f"Device:             {device} (CUDA: {cuda_avail}, GPU: {gpu_name})")
    print(f"Save Path:          {save_dir}")
    print("=" * 65 + "\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics library not found. Run: pip install ultralytics")
        sys.exit(1)

    # Initialize model
    model = YOLO(pretrained_weights)

    # Execute training
    start_time = time.time()
    logger.info("Starting model training...")

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=str(save_dir.parent),
        name=save_dir.name,
        seed=seed,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
    )

    elapsed_sec = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {elapsed_sec}s.")

    # Locate generated checkpoints
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    if not best_pt.exists():
        logger.warning(f"best.pt not found at {best_pt}. Looking for last.pt...")
        best_pt = last_pt

    if not best_pt.exists():
        logger.error("Training did not produce expected weight checkpoints.")
        sys.exit(1)

    best_sha256 = calculate_sha256(best_pt)
    logger.info(f"Trained checkpoint: {best_pt} (SHA-256: {best_sha256[:16]}...)")

    # Copy to canonical model path
    canonical_pt = paths.models_dir / "vehicle_detector.pt"
    shutil.copy2(best_pt, canonical_pt)

    # Validate trained model on validation set
    logger.info("Validating trained model metrics...")
    val_metrics = model.val(data=str(dataset_yaml), split="val", imgsz=imgsz, device=device)

    mAP50 = float(val_metrics.box.map50) if hasattr(val_metrics.box, "map50") else 0.0
    mAP50_95 = float(val_metrics.box.map) if hasattr(val_metrics.box, "map") else 0.0
    precision = float(val_metrics.box.mp) if hasattr(val_metrics.box, "mp") else 0.0
    recall = float(val_metrics.box.mr) if hasattr(val_metrics.box, "mr") else 0.0

    print("\n" + "=" * 65)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 65)
    print(f"mAP@50:      {mAP50:.4f}")
    print(f"mAP@50-95:   {mAP50_95:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print("=" * 65 + "\n")

    # Export to ONNX format
    logger.info("Exporting trained model to deployment ONNX format...")
    onnx_exported_file = model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=12)
    canonical_onnx = paths.vehicle_model_onnx

    if Path(onnx_exported_file).exists():
        shutil.copy2(onnx_exported_file, canonical_onnx)
        onnx_sha256 = calculate_sha256(canonical_onnx)
        logger.info(f"Canonical ONNX model saved to: {canonical_onnx} (SHA-256: {onnx_sha256[:16]}...)")
    else:
        logger.error(f"ONNX export file {onnx_exported_file} not found.")
        sys.exit(1)

    # Update manifest with real metrics and hash
    metrics_dict = {
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "training_duration_sec": elapsed_sec,
    }
    update_model_manifest("vehicle-yolo-v8n", canonical_onnx, onnx_sha256, metrics_dict)

    # Write training run report
    run_report = {
        "timestamp": datetime.now().isoformat(),
        "model_id": "vehicle-yolo-v8n",
        "architecture": architecture,
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "device": device,
        "duration_sec": elapsed_sec,
        "weights": {
            "checkpoint_pt": paths.to_relative_str(canonical_pt),
            "checkpoint_sha256": best_sha256,
            "onnx": paths.to_relative_str(canonical_onnx),
            "onnx_sha256": onnx_sha256,
        },
        "metrics": metrics_dict,
    }
    run_meta_path = save_dir / "training_run_metadata.json"
    run_meta_path.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    logger.info(f"Training run metadata saved to: {run_meta_path}")

    print("\n" + "=" * 65)
    print("VEHICLE DETECTOR TRAINING & EXPORT COMPLETED SUCCESSFULLY")
    print(f"PyTorch Checkpoint: {canonical_pt}")
    print(f"ONNX Model:         {canonical_onnx}")
    print(f"Manifest Updated:   {paths.models_manifest_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

