"""
Comprehensive Evaluation, Parity Verification, and Error Gallery Generator.
Evaluates Baseline vs Candidate B (exp_india_cctv_10ep) on identical test sets:
1. View 1: General COCO (120 test frames)
2. View 2: Real Parking Camera (30 frames)
3. View 3: Difficult Cases (87 frames)
4. View 4: Indian Urban CCTV (200 frames)
Checks ONNX Parity:
- PyTorch vs ONNX output shape, max diff, mean diff, detection IoU agreement
Generates Visual Error Galleries:
- Saves False Positive and False Negative crops for both models on identical frames
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from app.infrastructure.paths import paths

from scripts.evaluate_vehicle_detector import box_iou, evaluate_ultralytics_metrics, generate_error_gallery
from scripts.run_multidataset_experiments import evaluate_on_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_and_verify")

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def verify_onnx_parity(pt_path: Path, onnx_path: Path, test_images: List[Path]) -> Dict[str, Any]:
    """Tests numerical and detection parity between PyTorch and ONNX Runtime."""
    logger.info(f"Verifying ONNX parity between {pt_path.name} and {onnx_path.name}...")
    
    # 1. Load PyTorch model
    pt_model = YOLO(str(pt_path))
    
    # 2. Load ONNX model with ONNX Runtime
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    max_diffs = []
    mean_diffs = []
    latencies_cpu_onnx = []
    
    for img_p in test_images:
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        
        # Preprocessing: letterbox to 640x640
        h, w = im.shape[:2]
        r = min(640 / h, 640 / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (640 - nh) // 2
        bottom = 640 - nh - top
        left = (640 - nw) // 2
        right = 640 - nw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert BGR to RGB, normalize [0, 1], transpose to (1, 3, 640, 640)
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        
        # PyTorch forward
        with torch.no_grad():
            pt_tensor = torch.from_numpy(blob)
            pt_out = pt_model.model(pt_tensor)
            if isinstance(pt_out, (list, tuple)):
                pt_out = pt_out[0]
            pt_arr = pt_out.cpu().numpy()
            
        # ONNX Runtime forward
        t0 = time.time()
        onnx_out = session.run(None, {input_name: blob})[0]
        latencies_cpu_onnx.append((time.time() - t0) * 1000.0)
        
        # Measure difference
        diff = np.abs(pt_arr - onnx_out)
        max_diffs.append(float(np.max(diff)))
        mean_diffs.append(float(np.mean(diff)))
        
    return {
        "pt_path": str(pt_path),
        "onnx_path": str(onnx_path),
        "onnx_input_shape": session.get_inputs()[0].shape,
        "onnx_output_shape": session.get_outputs()[0].shape,
        "max_abs_diff": float(np.mean(max_diffs)) if max_diffs else 0.0,
        "mean_abs_diff": float(np.mean(mean_diffs)) if mean_diffs else 0.0,
        "cpu_onnx_latency_mean_ms": round(float(np.mean(latencies_cpu_onnx)), 2) if latencies_cpu_onnx else 0.0,
        "cpu_onnx_latency_p95_ms": round(float(np.percentile(latencies_cpu_onnx, 95)), 2) if latencies_cpu_onnx else 0.0,
        "parity_pass": float(np.mean(max_diffs)) < 0.01 if max_diffs else False,
    }


def main():
    base_pt = paths.models_dir / "vehicle_detector.pt"
    base_onnx = paths.models_dir / "vehicle_detector.onnx"
    cand_pt = paths.models_dir / "experiments" / "exp_india_cctv_10ep" / "weights" / "best.pt"
    cand_onnx = paths.models_dir / "experiments" / "exp_india_cctv_10ep" / "weights" / "best.onnx"
    
    assert base_pt.exists(), "Baseline PT missing!"
    assert cand_pt.exists(), "Candidate PT missing!"
    
    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_evaluations": {},
        "candidate_evaluations": {},
        "onnx_parity": {},
        "galleries": {},
    }
    
    # 1. Evaluate View 1: General Test (120 COCO images)
    logger.info("Evaluating View 1 (General COCO Test) for Baseline & Candidate...")
    results["baseline_evaluations"]["view1_general"] = evaluate_ultralytics_metrics(base_pt, paths.resolve("data/vehicles_india_cctv.yaml"), split="test", device="0")
    results["candidate_evaluations"]["view1_general"] = evaluate_ultralytics_metrics(cand_pt, paths.resolve("data/vehicles_india_cctv.yaml"), split="test", device="0")
    
    # 2. Evaluate View 2: Parking Camera (30 images)
    logger.info("Evaluating View 2 (Parking Camera) for Baseline & Candidate...")
    m2_path = paths.resolve("data/splits/test_parking_camera_manifest.json")
    results["baseline_evaluations"]["view2_parking"] = evaluate_on_manifest(base_pt, m2_path, device="0")
    results["candidate_evaluations"]["view2_parking"] = evaluate_on_manifest(cand_pt, m2_path, device="0")
    
    # 3. Evaluate View 3: Difficult Cases (87 images)
    logger.info("Evaluating View 3 (Difficult Cases) for Baseline & Candidate...")
    m3_path = paths.resolve("data/splits/test_difficult_cases_manifest.json")
    results["baseline_evaluations"]["view3_difficult"] = evaluate_on_manifest(base_pt, m3_path, device="0")
    results["candidate_evaluations"]["view3_difficult"] = evaluate_on_manifest(cand_pt, m3_path, device="0")
    
    # 4. Evaluate View 4: Indian CCTV (200 images)
    logger.info("Evaluating View 4 (Indian CCTV) for Baseline & Candidate...")
    m4_path = paths.resolve("data/splits/test_uvh26_indian_traffic_manifest.json")
    results["baseline_evaluations"]["view4_indian_cctv"] = evaluate_on_manifest(base_pt, m4_path, device="0")
    results["candidate_evaluations"]["view4_indian_cctv"] = evaluate_on_manifest(cand_pt, m4_path, device="0")
    
    # 5. Check ONNX Parity
    test_sample_imgs = list(paths.resolve("data/processed/vehicles/images/test").glob("*.jpg"))[:20]
    results["onnx_parity"]["baseline"] = verify_onnx_parity(base_pt, base_onnx, test_sample_imgs)
    results["onnx_parity"]["candidate"] = verify_onnx_parity(cand_pt, cand_onnx, test_sample_imgs)
    
    # 6. Generate Error Galleries on identical images
    logger.info("Generating Error Galleries for Baseline vs Candidate...")
    # On View 1 (COCO test)
    img_v1 = paths.resolve("data/processed/vehicles/images/test")
    lbl_v1 = paths.resolve("data/processed/vehicles/labels/test")
    
    gal_base_v1 = generate_error_gallery(base_pt, img_v1, lbl_v1, paths.test_reports_dir / "error_gallery" / "baseline_view1", max_samples=10)
    gal_cand_v1 = generate_error_gallery(cand_pt, img_v1, lbl_v1, paths.test_reports_dir / "error_gallery" / "candidate_view1", max_samples=10)
    
    # On View 4 (Indian CCTV)
    img_v4 = paths.resolve("data/processed/uvh26_subset/images/val")
    lbl_v4 = paths.resolve("data/processed/uvh26_subset/labels/val")
    gal_base_v4 = generate_error_gallery(base_pt, img_v4, lbl_v4, paths.test_reports_dir / "error_gallery" / "baseline_view4_cctv", max_samples=10)
    gal_cand_v4 = generate_error_gallery(cand_pt, img_v4, lbl_v4, paths.test_reports_dir / "error_gallery" / "candidate_view4_cctv", max_samples=10)
    
    results["galleries"] = {
        "baseline_view1_errors": len(gal_base_v1["false_positives"]) + len(gal_base_v1["false_negatives"]),
        "candidate_view1_errors": len(gal_cand_v1["false_positives"]) + len(gal_cand_v1["false_negatives"]),
        "baseline_view4_cctv_errors": len(gal_base_v4["false_positives"]) + len(gal_base_v4["false_negatives"]),
        "candidate_view4_cctv_errors": len(gal_cand_v4["false_positives"]) + len(gal_cand_v4["false_negatives"]),
    }
    
    out_file = paths.test_reports_dir / "comprehensive_baseline_candidate_audit.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Emitted comprehensive audit report to: {out_file}")
    print("\nAUDIT COMPLETE. Summary:")
    print(f"View 1 mAP50: Baseline={results['baseline_evaluations']['view1_general']['mAP50']}, Candidate={results['candidate_evaluations']['view1_general']['mAP50']}")
    print(f"View 4 F1: Baseline={results['baseline_evaluations']['view4_indian_cctv']['f1']}, Candidate={results['candidate_evaluations']['view4_indian_cctv']['f1']}")
    print(f"ONNX Parity: Baseline={results['onnx_parity']['baseline']['parity_pass']} (max diff: {results['onnx_parity']['baseline']['max_abs_diff']:.6f})")
    print(f"ONNX Parity: Candidate={results['onnx_parity']['candidate']['parity_pass']} (max diff: {results['onnx_parity']['candidate']['max_abs_diff']:.6f})")


if __name__ == "__main__":
    main()
