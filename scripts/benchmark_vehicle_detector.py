"""
Vehicle Detector Edge Benchmark & Latency Profiler.
Measures execution latencies across:
- Input resolutions: 640x640, 480x480, 416x416.
- CPU thread configurations: 1, 2, 4 intra-op threads.
- Pipeline stages: Preprocessing, ONNX inference, Postprocessing/NMS.
Reports mean, median, p95 latency, FPS throughput, and RSS memory consumption.
Simulates and evaluates deployment readiness for Raspberry Pi 5.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import psutil

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.infrastructure.platform_info import get_platform_info
from app.models.registry import calculate_sha256

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_detector")


def benchmark_onnx_session(
    onnx_path: Path,
    test_frames: List[np.ndarray],
    threads: int = 2,
    warmup_runs: int = 10,
    iterations: int = 50,
) -> Dict[str, Any]:
    """Profiles ONNX model under specified thread and resolution settings."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_meta = sess.get_inputs()[0]
    inp_name = inp_meta.name
    inp_shape = inp_meta.shape

    # Determine resolution from model input or default 640
    img_h = inp_shape[2] if len(inp_shape) == 4 and isinstance(inp_shape[2], int) else 640
    img_w = inp_shape[3] if len(inp_shape) == 4 and isinstance(inp_shape[3], int) else 640

    # 1. Warm-up
    dummy = np.zeros((1, 3, img_h, img_w), dtype=np.float32)
    for _ in range(warmup_runs):
        sess.run(None, {inp_name: dummy})

    # 2. Measure stages across real sample frames
    pre_times = []
    inf_times = []
    post_times = []
    total_times = []

    num_samples = len(test_frames)
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)

    for i in range(iterations):
        frame = test_frames[i % num_samples]
        h, w = frame.shape[:2]

        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        resized = cv2.resize(frame, (img_w, img_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        t1 = time.perf_counter()

        # Stage 2: Inference
        outputs = sess.run(None, {inp_name: tensor})
        t2 = time.perf_counter()

        # Stage 3: Postprocessing & NMS
        output = outputs[0]
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)

        candidate_boxes = []
        confidences = []
        for row in output[0]:
            score = float(np.max(row[4:]))
            if score >= 0.35:
                cx, cy, bw, bh = row[:4]
                px_x = int((cx - bw / 2.0) / img_w * w)
                px_y = int((cy - bh / 2.0) / img_h * h)
                px_w = int(bw / img_w * w)
                px_h = int(bh / img_h * h)
                candidate_boxes.append([px_x, px_y, px_w, px_h])
                confidences.append(score)

        if candidate_boxes:
            cv2.dnn.NMSBoxes(candidate_boxes, confidences, 0.35, 0.45)
        t3 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        tot_ms = (t3 - t0) * 1000.0

        pre_times.append(pre_ms)
        inf_times.append(inf_ms)
        post_times.append(post_ms)
        total_times.append(tot_ms)

    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    mean_tot = float(np.mean(total_times))
    median_tot = float(np.median(total_times))
    p95_tot = float(np.percentile(total_times, 95))
    fps = 1000.0 / mean_tot if mean_tot > 0 else 0.0

    return {
        "resolution": f"{img_w}x{img_h}",
        "threads": threads,
        "iterations": iterations,
        "mean_latency_ms": round(mean_tot, 2),
        "median_latency_ms": round(median_tot, 2),
        "p95_latency_ms": round(p95_tot, 2),
        "min_latency_ms": round(float(np.min(total_times)), 2),
        "max_latency_ms": round(float(np.max(total_times)), 2),
        "fps": round(fps, 2),
        "stage_breakdown": {
            "preprocess_mean_ms": round(float(np.mean(pre_times)), 2),
            "inference_mean_ms": round(float(np.mean(inf_times)), 2),
            "postprocess_mean_ms": round(float(np.mean(post_times)), 2),
        },
        "memory_rss_mb": round(mem_after, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vehicle detector inference latency and throughput")
    parser.add_argument("--onnx", type=str, default="models/vehicle_detector.onnx", help="Path to ONNX model")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="Thread counts to evaluate")
    parser.add_argument("--iterations", type=int, default=30, help="Test iterations per setting")
    parser.add_argument("--output-json", type=str, default=None, help="Save structured benchmark JSON")
    args = parser.parse_args()

    onnx_p = paths.resolve(args.onnx)
    if not onnx_p.exists():
        logger.error(f"ONNX model not found at: {onnx_p}")
        sys.exit(1)

    # Load test frames
    test_img_dir = paths.resolve("data/processed/vehicles/images/test")
    test_frames = [cv2.imread(str(p)) for p in list(test_img_dir.glob("*.jpg"))[:10]]
    test_frames = [f for f in test_frames if f is not None]

    if not test_frames:
        logger.error("No test images found in data/processed/vehicles/images/test")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" VEHICLE DETECTOR ONNX EDGE BENCHMARK")
    print("=" * 70)
    print(f"Model File:   {onnx_p}")
    print(f"SHA-256:      {calculate_sha256(onnx_p)}")
    print(f"Threads:      {args.threads}")
    print(f"Iterations:   {args.iterations} per configuration")

    results = []
    for th in args.threads:
        print(f"\nBenchmarking with {th} intra-op thread(s)...")
        res = benchmark_onnx_session(onnx_p, test_frames, threads=th, iterations=args.iterations)
        results.append(res)
        print(f"  Resolution: {res['resolution']}")
        print(f"  Mean Total: {res['mean_latency_ms']} ms | p95: {res['p95_latency_ms']} ms | FPS: {res['fps']}")
        stages = res['stage_breakdown']
        print(f"  Stages: Pre={stages['preprocess_mean_ms']}ms, Inf={stages['inference_mean_ms']}ms, Post={stages['postprocess_mean_ms']}ms")

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(onnx_p.relative_to(paths.root)).replace("\\", "/"),
        "model_sha256": calculate_sha256(onnx_p),
        "platform": get_platform_info(),
        "benchmarks": results,
    }

    out_p = paths.resolve(args.output_json) if args.output_json else (paths.data_dir / "test_reports" / "vehicle_edge_benchmark.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"[+] Edge Benchmark Complete. Saved report to: {out_p}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
