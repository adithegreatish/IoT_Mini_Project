"""
Comprehensive Inference Latency & Pipeline Benchmarking Tool.
Measures standalone occupancy patch classification, vehicle detection, full end-to-end
camera-to-state pipeline latency, and optional plate OCR.
Reports loading time, warmup, mean/median/p95 latency, FPS, memory, and CPU utilization.
Clearly labels execution environment (Windows vs Raspberry Pi Linux).
Saves benchmark report to data/test_reports/inference_benchmark.json.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.config.loader import load_config
from app.parking.occupancy_classifier import PatchOccupancyClassifier
from app.detection.vehicle_detector import VehicleDetector
from app.parking.calibration_service import CalibrationService
from app.parking.occupancy_service import OccupancyService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def get_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_cpu_temp() -> Optional[float]:
    """Reads Raspberry Pi thermal zone if available."""
    try:
        thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_file.exists():
            return round(float(thermal_file.read_text().strip()) / 1000.0, 1)
    except Exception:
        pass
    return None


def benchmark_patch_classifier(iterations: int = 100) -> Dict[str, Any]:
    logger.info("--- Benchmarking Patch Occupancy Classifier (ONNX CPU) ---")
    model_path = paths.resolve("models/occupancy_classifier.onnx")

    t0 = time.perf_counter()
    classifier = PatchOccupancyClassifier(model_path=model_path, device="cpu", threshold=0.65)
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    dummy_patch = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    dummy_batch50 = [np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8) for _ in range(50)]

    # Warmup
    t_warm0 = time.perf_counter()
    for _ in range(10):
        classifier.predict_patch(dummy_patch)
        classifier.predict_batch(dummy_batch50)
    warmup_time_ms = (time.perf_counter() - t_warm0) * 1000.0

    # Single patch benchmark
    single_latencies = []
    for _ in range(iterations):
        t1 = time.perf_counter()
        classifier.predict_patch(dummy_patch)
        single_latencies.append((time.perf_counter() - t1) * 1000.0)

    # 50-patch batch benchmark (simulating full 50-slot lot)
    batch_latencies = []
    for _ in range(max(20, iterations // 2)):
        t1 = time.perf_counter()
        classifier.predict_batch(dummy_batch50)
        batch_latencies.append((time.perf_counter() - t1) * 1000.0)

    return {
        "model_file": "models/occupancy_classifier.onnx",
        "sha256": get_sha256(model_path),
        "load_time_ms": round(load_time_ms, 2),
        "warmup_time_ms": round(warmup_time_ms, 2),
        "single_patch": {
            "mean_ms": round(float(np.mean(single_latencies)), 3),
            "median_ms": round(float(np.median(single_latencies)), 3),
            "p95_ms": round(float(np.percentile(single_latencies, 95)), 3),
            "fps": round(1000.0 / float(np.mean(single_latencies)), 1)
        },
        "frame_50_slots_batch": {
            "mean_ms": round(float(np.mean(batch_latencies)), 3),
            "median_ms": round(float(np.median(batch_latencies)), 3),
            "p95_ms": round(float(np.percentile(batch_latencies, 95)), 3),
            "fps": round(1000.0 / float(np.mean(batch_latencies)), 1)
        }
    }


def benchmark_vehicle_detector(iterations: int = 50, resolution: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
    logger.info(f"--- Benchmarking Vehicle Detector (ONNX CPU @ {resolution[0]}x{resolution[1]}) ---")
    from app.config.schema import DetectionConfig
    cfg = DetectionConfig(
        model_path="models/vehicle_detector.onnx",
        runtime="onnx",
        input_width=resolution[0],
        input_height=resolution[1]
    )

    t0 = time.perf_counter()
    detector = VehicleDetector(cfg)
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    t_warm0 = time.perf_counter()
    for _ in range(5):
        detector.detect(dummy_frame)
    warmup_time_ms = (time.perf_counter() - t_warm0) * 1000.0

    latencies = []
    for _ in range(iterations):
        t1 = time.perf_counter()
        detector.detect(dummy_frame)
        latencies.append((time.perf_counter() - t1) * 1000.0)

    mean_ms = float(np.mean(latencies))
    return {
        "model_file": "models/vehicle_detector.onnx",
        "sha256": get_sha256(paths.resolve("models/vehicle_detector.onnx")),
        "load_time_ms": round(load_time_ms, 2),
        "warmup_time_ms": round(warmup_time_ms, 2),
        "input_resolution": f"{resolution[0]}x{resolution[1]}",
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "fps": round(1000.0 / mean_ms, 1)
    }


def benchmark_end_to_end_pipeline(
    iterations: int = 50,
    config_file: str = "configs/raspberry_pi.yaml",
    test_image_path: Optional[Path] = None
) -> Dict[str, Any]:
    logger.info(f"--- Benchmarking Full End-to-End Pipeline (Config: {config_file}) ---")
    cfg = load_config(config_file)

    # Prepare representative frame
    if test_image_path and test_image_path.exists():
        frame = cv2.imread(str(test_image_path))
    else:
        # Default representative frame
        candidate = paths.resolve("data/raw/pklot/full_frames/2012-09-11_15_16_58.jpg")
        if candidate.exists():
            frame = cv2.imread(str(candidate))
        else:
            frame = np.random.randint(40, 200, (720, 1280, 3), dtype=np.uint8)

    h, w = frame.shape[:2]

    # Initialize components
    t0 = time.perf_counter()
    calib_svc = CalibrationService(cfg.parking.calibration_file)
    detector = VehicleDetector(cfg.detection)
    occ_svc = OccupancyService(cfg.parking, calib_svc, detector)
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup
    t_warm0 = time.perf_counter()
    for _ in range(5):
        occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    warmup_time_ms = (time.perf_counter() - t_warm0) * 1000.0

    latencies = []
    process = psutil.Process(os.getpid())

    for _ in range(iterations):
        t1 = time.perf_counter()
        snapshot = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
        latencies.append((time.perf_counter() - t1) * 1000.0)

    mean_ms = float(np.mean(latencies))
    fps = round(1000.0 / mean_ms, 2)
    rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    cpu_pct = process.cpu_percent(interval=None)

    # Evaluate against ~5 updates/sec target
    target_fps = 5.0
    target_met = bool(fps >= target_fps)

    return {
        "config_used": config_file,
        "input_frame_shape": f"{w}x{h}",
        "total_calibrated_slots": snapshot["summary"]["total"],
        "load_time_ms": round(load_time_ms, 2),
        "warmup_time_ms": round(warmup_time_ms, 2),
        "latency_stats": {
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(float(np.median(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
        },
        "measured_pipeline_fps": fps,
        "provisional_target_fps": target_fps,
        "provisional_target_met": target_met,
        "resource_utilization": {
            "process_rss_mb": rss_mb,
            "cpu_percent": cpu_pct,
            "cpu_temp_celsius": get_cpu_temp()
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Smart Parking Inference Benchmark")
    parser.add_argument("--iterations", "--runs", type=int, default=50, dest="iterations", help="Benchmark iterations")
    parser.add_argument("--batch", type=int, default=24, help="Batch size for slot patch inference")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing")
    parser.add_argument("--config", type=str, default="configs/raspberry_pi.yaml", help="Config profile")
    parser.add_argument("--image", type=str, default="", help="Optional representative test image path")
    args = parser.parse_args()

    sys_platform = platform.system()
    machine = platform.machine()
    proc = platform.processor()

    # Clear labeling
    is_windows = bool(sys_platform == "Windows")
    env_label = f"Windows Development Workstation ({machine})" if is_windows else f"Raspberry Pi / Linux ({machine})"

    logger.info("=" * 70)
    logger.info(" INFERENCE & PIPELINE PERFORMANCE BENCHMARK")
    logger.info(f" Environment: {env_label}")
    logger.info(f" Python: {platform.python_version()} | ONNX Runtime: {getattr(__import__('onnxruntime'), '__version__', 'N/A')}")
    logger.info("=" * 70)

    patch_res = benchmark_patch_classifier(iterations=max(60, args.iterations))
    det_res = benchmark_vehicle_detector(iterations=max(20, args.iterations // 2), resolution=(480, 480))
    test_img = Path(args.image) if args.image else None
    e2e_res = benchmark_end_to_end_pipeline(iterations=args.iterations, config_file=args.config, test_image_path=test_img)

    report = {
        "timestamp": datetime.now().isoformat(),
        "environment_label": env_label,
        "is_windows_measurement": is_windows,
        "hardware_metadata": {
            "os": sys_platform,
            "machine": machine,
            "processor": proc,
            "cpu_count": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)
        },
        "benchmarks": {
            "patch_occupancy_classifier": patch_res,
            "vehicle_detector": det_res,
            "end_to_end_pipeline": e2e_res
        }
    }

    out_file = paths.resolve("data/test_reports/inference_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved complete benchmark report to: {out_file}")

    # Terminal Summary
    print("\n" + "=" * 70)
    print(f" BENCHMARK SUMMARY [{report['environment_label'].upper()}]")
    if is_windows:
        print(" NOTE: THESE ARE WINDOWS MEASUREMENTS. DO NOT CLAIM AS PI MEASUREMENTS.")
    print("=" * 70)
    print(f" 1. Patch Occupancy Classifier (ONNX CPU):")
    print(f"    - Single Patch Mean:     {patch_res['single_patch']['mean_ms']} ms ({patch_res['single_patch']['fps']} FPS)")
    print(f"    - 50-Slot Lot Batch:     {patch_res['frame_50_slots_batch']['mean_ms']} ms ({patch_res['frame_50_slots_batch']['fps']} FPS)")
    print(f" 2. Vehicle Detector (ONNX YOLOv8n @ 480x480):")
    print(f"    - Mean Latency:          {det_res['mean_ms']} ms ({det_res['fps']} FPS)")
    print(f" 3. Full End-to-End Pipeline (Frame -> Detection -> State Machine):")
    print(f"    - Total Calibrated Bays: {e2e_res['total_calibrated_slots']} bays")
    print(f"    - Mean Latency:          {e2e_res['latency_stats']['mean_ms']} ms (p95: {e2e_res['latency_stats']['p95_ms']} ms)")
    print(f"    - Measured Pipeline FPS: {e2e_res['measured_pipeline_fps']} FPS")
    print(f"    - ~5 FPS Target Status:  {'TARGET MET' if e2e_res['provisional_target_met'] else 'BELOW TARGET'}")
    print(f"    - Process Memory (RSS):  {e2e_res['resource_utilization']['process_rss_mb']} MB")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
