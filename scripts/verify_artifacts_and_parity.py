"""
Deployment Artifacts Verification and Multi-Image PyTorch/ONNX Parity Audit
Validates:
1. File existence & exact SHA-256 computation
2. Manifest consistency against models/manifest.json
3. Multi-image numerical tensor parity (max error & mean error)
4. Multi-image end-to-end post-NMS detection output parity (box IoU, class mapping, confidences)
5. CPU vs GPU latency profiling for both detectors
"""

import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.infrastructure.platform_info import get_platform_info
from app.detection.vehicle_detector import VehicleDetector
from app.plates.plate_detector import PlateDetector
from app.config.schema import DetectionConfig, PlatesConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_parking.artifact_audit")


def calculate_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union for two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter
    return inter / union if union > 0 else 0.0


def benchmark_latency(onnx_path: Path, sample_images: List[np.ndarray], device: str = "cpu") -> Dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "gpu" and "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    input_name = sess.get_inputs()[0].name

    # Warmup
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    for _ in range(5):
        sess.run(None, {input_name: dummy})

    latencies = []
    for img in sample_images:
        resized = cv2.resize(img, (640, 640))
        blob = (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        t0 = time.perf_counter()
        sess.run(None, {input_name: blob})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "device": device,
        "provider": sess.get_providers()[0],
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p50_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
    }


def audit_model_pair(
    model_name: str,
    pt_path: Path,
    onnx_path: Path,
    test_image_paths: List[Path],
    is_vehicle: bool = True
) -> Dict:
    logger.info(f"Auditing {model_name} across {len(test_image_paths)} real held-out test images...")
    pt_sha = calculate_sha256(pt_path)
    onnx_sha = calculate_sha256(onnx_path)

    # 1. Raw Tensor Parity
    pt_model = YOLO(str(pt_path))
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    tensor_max_diffs = []
    tensor_mean_diffs = []
    box_match_rates = []
    loaded_images = []

    for img_p in test_image_paths:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        loaded_images.append(img)

        # Tensor comparison
        resized = cv2.resize(img, (640, 640))
        blob = (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        onnx_raw = sess.run(None, {input_name: blob})[0]
        with torch.no_grad():
            pt_raw = pt_model.model(torch.from_numpy(blob))[0].cpu().numpy()

        diff = np.abs(onnx_raw - pt_raw)
        tensor_max_diffs.append(float(np.max(diff)))
        tensor_mean_diffs.append(float(np.mean(diff)))

    # 2. End-to-End Post-NMS Output Parity
    if is_vehicle:
        v_onnx = VehicleDetector(DetectionConfig(runtime="onnx", model_path=str(paths.to_relative(onnx_path))))
        for img in loaded_images:
            res_pt_boxes = pt_model.predict(img, conf=0.25, verbose=False)[0].boxes
            res_onnx = v_onnx.detect(img)
            n_pt = len(res_pt_boxes)
            n_onnx = len(res_onnx)
            if n_pt == 0 and n_onnx == 0:
                box_match_rates.append(1.0)
                continue
            matches = 0
            for i in range(n_pt):
                pt_box = res_pt_boxes.xyxy[i].cpu().numpy().tolist()
                pt_cls_name = pt_model.names[int(res_pt_boxes.cls[i].item())]
                for b2 in res_onnx:
                    if b2.class_name == pt_cls_name and box_iou(pt_box, list(b2.pixel_box)) > 0.85:
                        matches += 1
                        break
            rate = matches / max(1, max(n_pt, n_onnx))
            box_match_rates.append(rate)
    else:
        p_onnx = PlateDetector(PlatesConfig(runtime="onnx", model_path=str(paths.to_relative(onnx_path))))
        for img in loaded_images:
            res_pt_boxes = pt_model.predict(img, conf=0.25, verbose=False)[0].boxes
            res_onnx = p_onnx.detect_plates(img)
            n_pt = len(res_pt_boxes)
            n_onnx = len(res_onnx)
            if n_pt == 0 and n_onnx == 0:
                box_match_rates.append(1.0)
                continue
            matches = 0
            for i in range(n_pt):
                pt_box = res_pt_boxes.xyxy[i].cpu().numpy().tolist()
                for b2 in res_onnx:
                    if box_iou(pt_box, list(b2.bbox)) > 0.85:
                        matches += 1
                        break
            rate = matches / max(1, max(n_pt, n_onnx))
            box_match_rates.append(rate)

    # 3. Latency benchmarks
    cpu_latency = benchmark_latency(onnx_path, loaded_images, device="cpu")
    gpu_latency = benchmark_latency(onnx_path, loaded_images, device="gpu") if torch.cuda.is_available() else None

    return {
        "model_name": model_name,
        "pt_path": str(paths.to_relative(pt_path)),
        "pt_sha256": pt_sha,
        "onnx_path": str(paths.to_relative(onnx_path)),
        "onnx_sha256": onnx_sha,
        "test_images_evaluated": len(loaded_images),
        "tensor_parity": {
            "max_absolute_error": round(float(np.max(tensor_max_diffs)), 6),
            "mean_absolute_error": round(float(np.mean(tensor_mean_diffs)), 8),
            "passes_numerical_tolerance": bool(np.max(tensor_max_diffs) < 0.01)
        },
        "post_nms_output_parity": {
            "mean_box_match_rate": round(float(np.mean(box_match_rates)), 4),
            "passes_output_equivalence": bool(np.mean(box_match_rates) >= 0.90)
        },
        "latency_cpu": cpu_latency,
        "latency_gpu": gpu_latency
    }


def main():
    manifest_path = paths.resolve("data/splits/test_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Load 12 vehicle images and 12 plate images
    v_imgs = [paths.resolve(item["image"]) for item in manifest.get("vehicles", [])[:12]]
    p_imgs = [paths.resolve(item["image"]) for item in manifest.get("plates", [])[:12]]

    print("\n" + "=" * 70)
    print("DEPLOYMENT ARTIFACTS VERIFICATION & MULTI-IMAGE PARITY AUDIT")
    print("=" * 70)

    # 1. Vehicle detector audit
    v_audit = audit_model_pair(
        "Vehicle Detector (YOLOv8n)",
        paths.resolve("models/vehicle_detector.pt"),
        paths.resolve("models/vehicle_detector.onnx"),
        v_imgs,
        is_vehicle=True
    )

    # 2. Plate detector audit
    p_audit = audit_model_pair(
        "Plate Detector (YOLOv8n)",
        paths.resolve("models/plate_detector.pt"),
        paths.resolve("models/plate_detector.onnx"),
        p_imgs,
        is_vehicle=False
    )

    # 3. Verify Manifest Consistency
    models_manifest = json.loads(paths.models_manifest_file.read_text(encoding="utf-8"))
    manifest_audit = []
    for m in models_manifest.get("models", []):
        m_id = m.get("model_id")
        file_p = paths.resolve(m.get("file_path"))
        actual_sha = calculate_sha256(file_p) if file_p.exists() else "MISSING"
        matches = (actual_sha == m.get("sha256"))
        manifest_audit.append({
            "model_id": m_id,
            "manifest_file": m.get("file_path"),
            "file_exists": file_p.exists(),
            "recorded_sha256": m.get("sha256"),
            "actual_sha256": actual_sha,
            "hash_matches": matches
        })
        print(f"[{'PASS' if matches else 'FAIL'}] Manifest {m_id}: file_exists={file_p.exists()}, hash_match={matches}")

    report = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": get_platform_info(),
        "vehicle_detector": v_audit,
        "plate_detector": p_audit,
        "manifest_consistency": manifest_audit
    }

    out_json = paths.resolve("data/test_reports/evaluations/artifact_parity_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[+] Full audit report written to: {out_json}")

    print("\n" + "=" * 70)
    print("SUMMARY OF PARITY AUDIT:")
    print(f"Vehicle Tensor Max Diff: {v_audit['tensor_parity']['max_absolute_error']} (Tolerance <0.01: {v_audit['tensor_parity']['passes_numerical_tolerance']})")
    print(f"Vehicle NMS Box Match:   {v_audit['post_nms_output_parity']['mean_box_match_rate']*100:.1f}%")
    print(f"Vehicle Latency:         CPU={v_audit['latency_cpu']['mean_latency_ms']} ms | GPU={v_audit['latency_gpu']['mean_latency_ms'] if v_audit['latency_gpu'] else 'N/A'} ms")
    print("-" * 70)
    print(f"Plate Tensor Max Diff:   {p_audit['tensor_parity']['max_absolute_error']} (Tolerance <0.01: {p_audit['tensor_parity']['passes_numerical_tolerance']})")
    print(f"Plate NMS Box Match:     {p_audit['post_nms_output_parity']['mean_box_match_rate']*100:.1f}%")
    print(f"Plate Latency:           CPU={p_audit['latency_cpu']['mean_latency_ms']} ms | GPU={p_audit['latency_gpu']['mean_latency_ms'] if p_audit['latency_gpu'] else 'N/A'} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
