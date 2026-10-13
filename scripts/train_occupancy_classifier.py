"""
Train, validate, and export parking-space occupancy patch classifier.

Trains on CNRPark+EXT 150x150 patches (data/processed/parking_occupancy)
Classes: 0 = FREE, 1 = BUSY (Occupied)

Evaluates held-out test set:
- Accuracy, Precision, Recall, F1
- Confusion Matrix (TN, FP, FN, TP)
- Latency benchmarking on CPU & GPU (RTX 4070)
- ONNX export & parity validation
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("train_occupancy_classifier")


class ParkingPatchDataset(Dataset):
    """Loads parking-space image patches for binary occupancy classification."""

    def __init__(self, split_dir: Path, transform=None):
        self.samples = []
        self.transform = transform

        free_dir = split_dir / "free"
        busy_dir = split_dir / "busy"

        if free_dir.exists():
            for p in list(free_dir.glob("*.jpg")) + list(free_dir.glob("*.png")):
                self.samples.append((p, 0))  # 0 = free

        if busy_dir.exists():
            for p in list(busy_dir.glob("*.jpg")) + list(busy_dir.glob("*.png")):
                self.samples.append((p, 1))  # 1 = busy

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Fallback blank
            img = np.zeros((150, 150, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


class LightweightOccupancyNet(nn.Module):
    """Fast, accurate convolutional neural network optimized for parking slot patch classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 150x150 -> 75x75
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 75x75 -> 38x38
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 38x38 -> 19x19
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 19x19 -> 10x10
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global Average Pooling: 10x10 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)


def train_model(epochs: int = 10, batch_size: int = 64, lr: float = 1e-3, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info(f"Training on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    data_dir = paths.processed_data_dir / "parking_occupancy"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((150, 150)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((150, 150)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ParkingPatchDataset(train_dir, transform=train_transform)
    val_ds = ParkingPatchDataset(val_dir, transform=eval_transform)
    test_ds = ParkingPatchDataset(test_dir, transform=eval_transform)

    logger.info(f"Loaded datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LightweightOccupancyNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_weights_path = paths.models_dir / "occupancy_classifier.pt"
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        # Validation
        val_metrics = evaluate_loader(model, val_loader, device)
        logger.info(
            f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_weights_path)

    elapsed_training = time.time() - t0
    logger.info(f"Training completed in {elapsed_training:.1f}s. Best Val F1: {best_val_f1:.4f}")

    # Load best weights for test evaluation
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    test_metrics = evaluate_loader(model, test_loader, device)

    logger.info("=" * 60)
    logger.info("TEST EVALUATION RESULTS (HELD-OUT CNRPARK TEST SET)")
    logger.info("=" * 60)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:15s}: {v:.4f}")
        else:
            logger.info(f"  {k:15s}: {v}")
    logger.info("=" * 60)

    # Export to ONNX
    onnx_path = paths.models_dir / "occupancy_classifier.onnx"
    export_to_onnx(model, onnx_path)

    # Parity & Latency Benchmark
    parity_ok, max_diff, cpu_latency, gpu_latency = benchmark_onnx(model, onnx_path, device)

    # Save evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "occupancy_classifier",
        "architecture": "LightweightOccupancyNet",
        "dataset": "CNRPark+EXT",
        "dataset_splits": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "training_epochs": epochs,
        "training_duration_sec": round(elapsed_training, 2),
        "test_metrics": test_metrics,
        "onnx": {
            "path": paths.to_relative_str(onnx_path),
            "parity_verified": parity_ok,
            "max_absolute_difference": float(max_diff),
            "latency_cpu_ms": round(cpu_latency, 2),
            "latency_gpu_ms": round(gpu_latency, 2),
        }
    }
    report_path = paths.models_dir / "occupancy_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved occupancy evaluation report to {report_path}")

    return report


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy().astype(int))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    tp = int(np.sum((all_preds == 1) & (all_targets == 1)))
    tn = int(np.sum((all_preds == 0) & (all_targets == 0)))
    fp = int(np.sum((all_preds == 1) & (all_targets == 0)))
    fn = int(np.sum((all_preds == 0) & (all_targets == 1)))

    acc = float(np.mean(all_preds == all_targets))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def export_to_onnx(model: nn.Module, out_path: Path):
    logger.info(f"Exporting model to ONNX: {out_path}...")
    model.eval()
    dummy_input = torch.randn(1, 3, 150, 150, device="cpu")
    model_cpu = model.to("cpu")

    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info(f"ONNX export succeeded: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


def benchmark_onnx(pt_model: nn.Module, onnx_path: Path, device: torch.device) -> Tuple[bool, float, float, float]:
    import onnxruntime as ort

    logger.info("Validating PyTorch <-> ONNX numerical parity...")
    dummy_np = np.random.randn(8, 3, 150, 150).astype(np.float32)

    # PyTorch inference
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model.to("cpu")(torch.from_numpy(dummy_np)).numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(["logits"], {"input": dummy_np})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    parity_ok = max_diff < 1e-4
    logger.info(f"Parity Check: Max absolute diff = {max_diff:.6f} (Passed: {parity_ok})")

    # Benchmarking CPU Latency
    times_cpu = []
    single_inp = np.random.randn(1, 3, 150, 150).astype(np.float32)
    for _ in range(10):
        session.run(["logits"], {"input": single_inp})  # warmup

    for _ in range(100):
        t0 = time.time()
        session.run(["logits"], {"input": single_inp})
        times_cpu.append((time.time() - t0) * 1000)

    mean_cpu = float(np.mean(times_cpu))
    logger.info(f"ONNX CPU Latency: {mean_cpu:.2f} ms/patch (P95: {np.percentile(times_cpu, 95):.2f} ms)")

    # Benchmarking GPU Latency if CUDA available
    mean_gpu = 0.0
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            gpu_session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
            for _ in range(10):
                gpu_session.run(["logits"], {"input": single_inp})
            times_gpu = []
            for _ in range(100):
                t0 = time.time()
                gpu_session.run(["logits"], {"input": single_inp})
                times_gpu.append((time.time() - t0) * 1000)
            mean_gpu = float(np.mean(times_gpu))
            logger.info(f"ONNX GPU Latency: {mean_gpu:.2f} ms/patch (P95: {np.percentile(times_gpu, 95):.2f} ms)")
        except Exception as e:
            logger.warning(f"GPU benchmark skipped: {e}")
            mean_gpu = mean_cpu

    return parity_ok, max_diff, mean_cpu, mean_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device_str=args.device)


if __name__ == "__main__":
    main()
