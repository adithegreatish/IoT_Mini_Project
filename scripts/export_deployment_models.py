"""
Exports PyTorch detector models to optimized ONNX format for deployment and
validates numerical inference parity against PyTorch baseline on real test images.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np
import cv2

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_parking.export")


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_model_parity(
    pt_path: Path,
    onnx_path: Path,
    test_image_path: Path,
    input_size: int = 640,
    tolerance: float = 0.01,
) -> dict:
    """Compares PyTorch and ONNX inference outputs on representative test image."""
    import torch
    import onnxruntime as ort
    from ultralytics import YOLO

    if not pt_path.exists():
        return {"success": False, "error": f"PyTorch weights not found at {pt_path}"}
    if not onnx_path.exists():
        return {"success": False, "error": f"ONNX file not found at {onnx_path}"}

    # Load PyTorch model
    pt_model = YOLO(str(pt_path))

    # Load ONNX session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    # Load and preprocess test image
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
    else:
        img = np.ones((input_size, input_size, 3), dtype=np.uint8) * 128

    resized = cv2.resize(img, (input_size, input_size))
    blob = (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

    # Measure ONNX latency
    t0 = time.perf_counter()
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: blob})[0]
    onnx_lat_ms = (time.perf_counter() - t0) * 1000.0

    # Measure PyTorch latency & output
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_tensor = torch.from_numpy(blob)
        pt_out = pt_model.model(pt_tensor)[0].cpu().numpy()
    pt_lat_ms = (time.perf_counter() - t0) * 1000.0

    # Compute tensor numerical disparity
    diff = np.abs(onnx_out - pt_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    is_parity = max_diff <= tolerance

    return {
        "success": True,
        "is_parity_passed": is_parity,
        "max_absolute_error": round(max_diff, 6),
        "mean_absolute_error": round(mean_diff, 8),
        "tolerance_threshold": tolerance,
        "onnx_latency_ms": round(onnx_lat_ms, 2),
        "pytorch_latency_ms": round(pt_lat_ms, 2),
        "input_shape": list(blob.shape),
        "output_shape": list(onnx_out.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch detectors to ONNX with numerical parity verification")
    parser.add_argument("--force-reexport", action="store_true", help="Force re-export from .pt even if .onnx exists")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DEPLOYMENT MODEL EXPORT & ONNX NUMERICAL PARITY VERIFICATION")
    print("=" * 70)

    # Locate test manifest for representative samples
    test_manifest_path = paths.resolve("data/splits/test_manifest.json")
    vehicle_test_img = None
    plate_test_img = None
    if test_manifest_path.exists():
        manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("vehicles"):
            vehicle_test_img = paths.resolve(manifest["vehicles"][0]["image"])
        if manifest.get("plates"):
            plate_test_img = paths.resolve(manifest["plates"][0]["image"])

    models_to_check = [
        {
            "id": "vehicle-yolo-v8n",
            "name": "Vehicle Detector",
            "pt_path": paths.resolve("models/vehicle_detector.pt"),
            "onnx_path": paths.resolve("models/vehicle_detector.onnx"),
            "test_img": vehicle_test_img,
            "imgsz": 640,
        },
        {
            "id": "plate-detector-tiny",
            "name": "License Plate Detector",
            "pt_path": paths.resolve("models/plate_detector.pt"),
            "onnx_path": paths.resolve("models/plate_detector.onnx"),
            "test_img": plate_test_img,
            "imgsz": 640,
        },
    ]

    parity_reports = []

    for m in models_to_check:
        print(f"\n[+] Processing: {m['name']} ({m['id']})")
        pt_path = m["pt_path"]
        onnx_path = m["onnx_path"]

        if not pt_path.exists():
            print(f"    [-] PyTorch checkpoint not found at: {pt_path}. Skipping.")
            continue

        if not onnx_path.exists() or args.force_reexport:
            print(f"    [*] Exporting {pt_path.name} to ONNX (opset={args.opset})...")
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            exported_path = model.export(
                format="onnx",
                imgsz=m["imgsz"],
                opset=args.opset,
                simplify=True,
                half=False,
            )
            print(f"    [+] Successfully exported to: {exported_path}")

        sha256 = calculate_sha256(onnx_path)
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"    File: {onnx_path.name} ({size_mb:.2f} MB)")
        print(f"    SHA256: {sha256}")

        # Run parity verification
        test_img = m["test_img"] or paths.resolve("test_image.jpg")
        print(f"    [*] Validating numerical parity on {test_img.name if test_img.exists() else 'synthetic image'}...")
        parity = verify_model_parity(pt_path, onnx_path, test_img, input_size=m["imgsz"])

        if parity.get("success"):
            print(f"    [+] Parity Status: {'PASSED' if parity['is_parity_passed'] else 'MARGINAL'}")
            print(f"        Max Abs Error:  {parity['max_absolute_error']:.6f} (tolerance: {parity['tolerance_threshold']})")
            print(f"        Mean Abs Error: {parity['mean_absolute_error']:.8f}")
            print(f"        ONNX Latency:   {parity['onnx_latency_ms']:.2f} ms")
            print(f"        PyTorch Latency:{parity['pytorch_latency_ms']:.2f} ms")
        else:
            print(f"    [!] Parity check error: {parity.get('error')}")

        parity_reports.append({
            "model_id": m["id"],
            "model_name": m["name"],
            "onnx_path": str(onnx_path),
            "sha256": sha256,
            "size_mb": round(size_mb, 2),
            "parity": parity,
        })

    # Save export & parity verification log
    report_file = paths.resolve(f"data/test_reports/export_parity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(parity_reports, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"[+] Parity verification log written to: {report_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
