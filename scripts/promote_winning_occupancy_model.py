"""
Promote and export winning occupancy classifier (Approach D: Domain-Augmented Combined Model)
to production models/occupancy_classifier.onnx and models/occupancy_classifier.pt.
Validates ONNX parity, measures CPU latency, calculates SHA-256, and updates manifests.
"""

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from scripts.train_and_evaluate_all_approaches import LightweightOccupancyNet

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("promote_model")


def main():
    d_pt_path = paths.models_dir / "occupancy_model_d_domain_augmented.pt"
    if not d_pt_path.exists():
        raise FileNotFoundError(f"Missing {d_pt_path}")

    prod_pt = paths.models_dir / "occupancy_classifier.pt"
    prod_onnx = paths.models_dir / "occupancy_classifier.onnx"

    logger.info(f"Loading Approach D weights from {d_pt_path.name}...")
    model = LightweightOccupancyNet()
    model.load_state_dict(torch.load(d_pt_path, weights_only=True))
    model.eval()

    # Save to production pt
    torch.save(model.state_dict(), prod_pt)
    logger.info(f"Saved production PyTorch weights to {prod_pt}")

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 150, 150)
    torch.onnx.export(
        model,
        dummy_input,
        str(prod_onnx),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info(f"Exported production ONNX model to {prod_onnx}")

    # Numerical Parity Check
    dummy_np = np.random.randn(16, 3, 150, 150).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy_np)).numpy()

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(prod_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    ort_out = session.run(["logits"], {"input": dummy_np})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    mean_diff = float(np.mean(np.abs(pt_out - ort_out)))
    parity_passed = bool(max_diff < 1e-4)

    logger.info(f"ONNX Parity: Max Diff = {max_diff:.6e}, Mean Diff = {mean_diff:.6e} | Passed = {parity_passed}")
    assert parity_passed, "ONNX Parity failed!"

    # Latency benchmarking on CPU
    logger.info("Benchmarking CPU latency...")
    dummy_single = np.random.randn(1, 3, 150, 150).astype(np.float32)
    dummy_batch50 = np.random.randn(50, 3, 150, 150).astype(np.float32)

    for _ in range(10):
        session.run(["logits"], {"input": dummy_single})
        session.run(["logits"], {"input": dummy_batch50})

    single_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        session.run(["logits"], {"input": dummy_single})
        single_times.append((time.perf_counter() - t0) * 1000.0)

    batch50_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(["logits"], {"input": dummy_batch50})
        batch50_times.append((time.perf_counter() - t0) * 1000.0)

    lat_stats = {
        "single_patch_cpu_ms": {
            "mean": round(float(np.mean(single_times)), 3),
            "p50": round(float(np.percentile(single_times, 50)), 3),
            "p95": round(float(np.percentile(single_times, 95)), 3),
            "min": round(float(np.min(single_times)), 3),
            "max": round(float(np.max(single_times)), 3),
        },
        "batch50_cpu_ms": {
            "mean": round(float(np.mean(batch50_times)), 3),
            "p50": round(float(np.percentile(batch50_times, 50)), 3),
            "p95": round(float(np.percentile(batch50_times, 95)), 3),
        }
    }
    logger.info(f"Latency: Single patch mean = {lat_stats['single_patch_cpu_ms']['mean']} ms | 50-slots frame = {lat_stats['batch50_cpu_ms']['mean']} ms")

    # Hashes
    onnx_sha256 = hashlib.sha256(prod_onnx.read_bytes()).hexdigest()
    pt_sha256 = hashlib.sha256(prod_pt.read_bytes()).hexdigest()
    logger.info(f"ONNX SHA-256: {onnx_sha256}")
    logger.info(f"PT   SHA-256: {pt_sha256}")

    # Update multi_model_leave_one_lot_out_evaluation.json
    eval_json = paths.models_dir / "multi_model_leave_one_lot_out_evaluation.json"
    if eval_json.exists():
        data = json.loads(eval_json.read_text(encoding="utf-8"))
        data["winning_model"] = "Approach_D_Domain_Augmented"
        data["selection_rationale"] = (
            "Approach D incorporates domain-appropriate illumination, contrast, and synthetic canopy "
            "shadow augmentation. While matching Approach C on UFPR04 (0.9177 vs 0.9179 macro-F1), it "
            "achieves a decisive +8.92% macro-F1 gain on UFPR05 (0.7895 vs 0.7003) and raises free recall "
            "from 43.70% to 59.10%, producing the highest cross-domain generalization average across all unseen lots."
        )
        data["selected_validation_threshold"] = 0.650
        data["onnx_parity"] = {
            "verified": parity_passed,
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "threshold_used": 0.650
        }
        data["artifacts"] = {
            "onnx_file": "models/occupancy_classifier.onnx",
            "onnx_sha256": onnx_sha256,
            "pt_file": "models/occupancy_classifier.pt",
            "pt_sha256": pt_sha256
        }
        data["latency_benchmarks"] = lat_stats
        eval_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Updated multi_model_leave_one_lot_out_evaluation.json")

    # Update manifest.json
    manifest_p = paths.models_dir / "manifest.json"
    if manifest_p.exists():
        m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
        for m_item in m_data.get("models", []):
            if m_item.get("model_id") == "parking-occupancy-net":
                m_item["sha256"] = onnx_sha256
                m_item["version"] = "2.1.0"
                m_item["updated_at"] = datetime.now().isoformat()
                m_item["winning_approach"] = "Approach_D_Domain_Augmented"
                m_item["metrics"]["selected_validation_threshold"] = 0.650
                m_item["metrics"]["latency_windows_ms"] = lat_stats["single_patch_cpu_ms"]["mean"]
                m_item["metrics"]["latency_batch50_ms"] = lat_stats["batch50_cpu_ms"]["mean"]
                m_item["metrics"]["ufpr04_macro_f1"] = 0.9177
                m_item["metrics"]["ufpr05_macro_f1"] = 0.7895
                m_item["metrics"]["camera09_macro_f1"] = 0.9115
                m_item["metrics"]["historical_benchmark_macro_f1"] = 0.9305
        manifest_p.write_text(json.dumps(m_data, indent=2), encoding="utf-8")
        logger.info("Updated models/manifest.json")


if __name__ == "__main__":
    main()
