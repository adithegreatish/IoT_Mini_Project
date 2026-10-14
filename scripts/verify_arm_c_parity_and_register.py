"""
Verify ONNX parity and CPU latency for Arm C (exp_balanced_multisource_10ep),
emit candidate evaluation summary, and update model registry test_metrics.
"""

import json
import logging
from pathlib import Path
import sys
import time

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.models.registry import calculate_sha256, model_registry
from scripts.audit_evaluation_and_parity import verify_onnx_parity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_arm_c")


def main():
    exp_dir = paths.models_dir / "experiments" / "exp_balanced_multisource_10ep"
    pt_p = exp_dir / "weights" / "best.pt"
    onnx_p = exp_dir / "weights" / "best.onnx"

    assert pt_p.exists(), f"Missing {pt_p}"
    assert onnx_p.exists(), f"Missing {onnx_p}"

    # Sample test images for ONNX parity verification
    test_manifest_p = paths.resolve("data/splits/test_general_manifest.json")
    if not test_manifest_p.exists():
        test_manifest_p = paths.resolve("data/splits/test_indian_cctv_heldout_manifest.json")
    manifest = json.loads(test_manifest_p.read_text(encoding="utf-8"))
    test_imgs = [paths.resolve(s["image"]) for s in manifest["samples"][:15]]

    logger.info("Running ONNX parity verification...")
    parity = verify_onnx_parity(pt_p, onnx_p, test_imgs)
    logger.info(f"Parity results: max_diff={parity['max_abs_diff']:.6f}, pass={parity['parity_pass']}, mean CPU latency={parity['cpu_onnx_latency_mean_ms']}ms")

    # Load PR curve evaluation report for Arm C
    report_p = paths.test_reports_dir / "eval_pr_cand_c.json"
    assert report_p.exists(), f"Missing {report_p}"
    eval_data = json.loads(report_p.read_text(encoding="utf-8"))

    v1 = eval_data["view1_general_coco_test"]
    v4 = eval_data["view4_independent_indian_cctv_test"]["official_ultralytics_val"]
    v4_op = eval_data["view4_independent_indian_cctv_test"]["standalone_pr_curve_and_operational"]["operational_metrics_at_conf40"]

    summary = {
        "experiment_id": "exp_balanced_multisource_10ep",
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pt_path": str(pt_p.relative_to(paths.root)).replace("\\", "/"),
        "onnx_path": str(onnx_p.relative_to(paths.root)).replace("\\", "/"),
        "pt_sha256": calculate_sha256(pt_p),
        "onnx_sha256": calculate_sha256(onnx_p),
        "pt_size_mb": round(pt_p.stat().st_size / (1024 * 1024), 2),
        "onnx_size_mb": round(onnx_p.stat().st_size / (1024 * 1024), 2),
        "training_config": {
            "epochs": 10,
            "batch_size": 16,
            "imgsz": 640,
            "optimizer": "AdamW",
            "cos_lr": True,
            "seed": 42,
            "samples_per_epoch": 3199,
            "sources": {
                "coco_general": 1657,
                "coco_motorcycle_reinforcement": 497,
                "uvh26_indian_cctv_stratified": 700,
                "visdrone_aerial_supplement": 345,
            },
        },
        "test_metrics": {
            "mAP50": v1["mAP50"],
            "mAP50_95": v1["mAP50_95"],
            "precision": v1["precision"],
            "recall": v1["recall"],
            "per_class": v1["per_class"],
        },
        "multi_view_evaluations": {
            "view1_general_coco_test": v1,
            "view4_independent_indian_cctv_heldout": {
                "mAP50": v4["mAP50"],
                "mAP50_95": v4["mAP50_95"],
                "precision": v4["precision"],
                "recall": v4["recall"],
                "per_class": v4["per_class"],
                "operational_metrics_at_conf40": v4_op,
            },
            "view2_parking_camera": eval_data.get("view2_parking_camera", {}),
            "view3_difficult_cases": eval_data.get("view3_difficult_cases", {}),
        },
        "onnx_parity": parity,
        "status": "EXPERIMENTAL_CANDIDATE",
        "notes": "Arm C Balanced Multi-Source: Improves Indian CCTV mAP50 to 0.5631 (+28.2% over baseline), but regresses on General COCO (0.4578 vs 0.5930 baseline) due to VisDrone small-scale anchor interference. Retained as experimental candidate.",
    }

    summary_p = exp_dir / "candidate_evaluation_summary.json"
    summary_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved evaluation summary to {summary_p}")

    # Update model registry
    reg = model_registry.load_registry()
    updated = False
    for c in reg.get("candidates", []):
        if c.get("experiment_id") == "exp_balanced_multisource_10ep":
            c["test_metrics"] = summary["test_metrics"]
            c["multi_view_evaluations"] = summary["multi_view_evaluations"]
            c["notes"] = summary["notes"]
            c["status"] = "EXPERIMENTAL_CANDIDATE"
            updated = True
            break
    if updated:
        model_registry.save_registry(reg)
        logger.info("Updated models/model_registry.json with Arm C test metrics and status.")


if __name__ == "__main__":
    main()
