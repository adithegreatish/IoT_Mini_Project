"""
Vehicle Deployment Model Exporter & Parity Verifier.
Exports PyTorch detector checkpoints to optimized ONNX models with:
- Configurable input resolutions (640, 480, 416, 320).
- Strict numerical parity verification against PyTorch output on real test samples.
- File size, SHA-256 checksums, and deployment metadata recording.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.models.registry import calculate_sha256

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_deployment")


def verify_numerical_parity(
    pt_path: Path,
    onnx_path: Path,
    test_img: np.ndarray,
    imgsz: int = 640,
    tolerance: float = 0.05,
) -> Tuple[bool, float, str]:
    """Compares raw PyTorch output against ONNX Runtime output for numerical consistency."""
    import torch
    import onnxruntime as ort
    from ultralytics import YOLO

    # 1. Preprocess test image
    resized = cv2.resize(test_img, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    inp = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

    # 2. Run ONNX Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {inp_name: inp})[0]

    # 3. Run PyTorch model
    pt_model = YOLO(str(pt_path))
    # Ultralytics model predict on raw tensor
    with torch.no_grad():
        pt_tensor = torch.from_numpy(inp)
        pt_raw = pt_model.model(pt_tensor)
        if isinstance(pt_raw, (list, tuple)):
            pt_out = pt_raw[0].cpu().numpy()
        else:
            pt_out = pt_raw.cpu().numpy()

    # Align shapes
    if len(pt_out.shape) == 3 and len(onnx_out.shape) == 3:
        if pt_out.shape != onnx_out.shape:
            if pt_out.shape == (onnx_out.shape[0], onnx_out.shape[2], onnx_out.shape[1]):
                pt_out = pt_out.transpose(0, 2, 1)

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    passed = max_diff < tolerance
    msg = f"Max output difference: {max_diff:.6f} (tolerance: {tolerance})"
    return passed, max_diff, msg


def export_model(
    checkpoint_pt: Path,
    output_onnx: Path,
    imgsz: int = 640,
    opset: int = 12,
    half: bool = False,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    if not checkpoint_pt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_pt}")

    logger.info(f"Loading checkpoint {checkpoint_pt} for ONNX export (imgsz={imgsz}, opset={opset})...")
    model = YOLO(str(checkpoint_pt))

    t0 = time.time()
    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=opset,
        half=half,
    )
    export_duration = round(time.time() - t0, 2)
    exported_p = Path(exported_path_str)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    if exported_p != output_onnx:
        import shutil
        shutil.copy2(exported_p, output_onnx)

    sha256_hash = calculate_sha256(output_onnx)
    size_bytes = output_onnx.stat().st_size

    return {
        "output_path": str(output_onnx.relative_to(paths.root)).replace("\\", "/"),
        "sha256": sha256_hash,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "imgsz": imgsz,
        "opset": opset,
        "export_duration_sec": export_duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to optimized ONNX and verify parity")
    parser.add_argument("--checkpoint", type=str, default="models/vehicle_detector.pt", help="Path to input PyTorch checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to output ONNX file")
    parser.add_argument("--imgsz", type=int, default=640, choices=[320, 416, 480, 640], help="Target inference resolution")
    parser.add_argument("--verify-parity", action="store_true", help="Run numerical parity test against real test image")
    args = parser.parse_args()

    ckpt_p = paths.resolve(args.checkpoint)
    out_p = paths.resolve(args.output) if args.output else (ckpt_p.parent / f"{ckpt_p.stem}_{args.imgsz}.onnx")

    print("\n" + "=" * 70)
    print(" VEHICLE DEPLOYMENT MODEL EXPORT & NUMERICAL PARITY")
    print("=" * 70)
    print(f"Input Checkpoint:  {ckpt_p}")
    print(f"Output ONNX:       {out_p}")
    print(f"Target Resolution: {args.imgsz}x{args.imgsz}")

    info = export_model(ckpt_p, out_p, imgsz=args.imgsz)

    print(f"\n[+] Export Successful ({info['export_duration_sec']}s)")
    print(f"    File:    {info['output_path']}")
    print(f"    Size:    {info['size_mb']} MB ({info['size_bytes']} bytes)")
    print(f"    SHA-256: {info['sha256']}")

    if args.verify_parity:
        print("\n>>> Testing Numerical Inference Parity on Test Sample...")
        test_img_dir = paths.resolve("data/processed/vehicles/images/test")
        sample_img = next(test_img_dir.glob("*.jpg"), None)
        if sample_img:
            img = cv2.imread(str(sample_img))
            passed, max_diff, msg = verify_numerical_parity(ckpt_p, out_p, img, imgsz=args.imgsz)
            print(f"    Sample: {sample_img.name}")
            print(f"    Status: {'PASSED' if passed else 'FAILED'}")
            print(f"    {msg}")
            if not passed:
                logger.error("Numerical parity check failed!")
                sys.exit(1)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
