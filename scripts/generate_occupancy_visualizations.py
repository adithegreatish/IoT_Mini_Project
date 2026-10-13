"""
Generate visual proof examples of correct vs incorrect parking occupancy predictions.
Produces visual comparison images across sunny, rainy, and shadowed conditions on PKLot & CNR-EXT.
Saves images to data/test_reports/visualizations/ and outputs a JSON index.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List

import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.parking.occupancy_classifier import PatchOccupancyClassifier

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_visualizations")


def create_annotated_patch(
    img_bgr: np.ndarray,
    pred_occ: bool,
    conf: float,
    gt_label: str,
    meta_text: str
) -> np.ndarray:
    """Renders a diagnostic patch with clear border, prediction, ground truth, and confidence."""
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    if h != 200 or w != 200:
        vis = cv2.resize(vis, (200, 200), interpolation=cv2.INTER_LINEAR)

    gt_is_occ = (gt_label.lower() in ["busy", "occupied", "1"])
    is_correct = (pred_occ == gt_is_occ)

    # Border color: Green if correct, Red/Amber if error
    border_color = (0, 180, 0) if is_correct else (0, 0, 220)
    cv2.rectangle(vis, (0, 0), (199, 199), border_color, 4)

    # Top banner for prediction
    banner_bg = (30, 30, 30)
    cv2.rectangle(vis, (0, 0), (200, 40), banner_bg, -1)

    pred_str = "OCCUPIED" if pred_occ else "VACANT"
    pred_color = (60, 60, 255) if pred_occ else (60, 255, 60)
    cv2.putText(vis, f"PRED: {pred_str}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pred_color, 1, cv2.LINE_AA)
    cv2.putText(vis, f"Conf: {conf:.2f}", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # Bottom banner for Ground Truth & status
    cv2.rectangle(vis, (0, 160), (200, 200), banner_bg, -1)
    status_str = "CORRECT" if is_correct else "MISCLASSIFIED"
    status_color = (0, 220, 0) if is_correct else (0, 80, 255)
    cv2.putText(vis, f"GT: {gt_label.upper()} ({status_str})", (8, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)
    cv2.putText(vis, meta_text[:28], (8, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1, cv2.LINE_AA)

    return vis


def build_comparison_grid(
    patches: List[np.ndarray],
    cols: int = 4,
    pad: int = 10,
    title: str = "Diagnostic Comparison"
) -> np.ndarray:
    """Assembles a clean grid of annotated patches with title header."""
    n = len(patches)
    rows = (n + cols - 1) // cols
    patch_h, patch_w = patches[0].shape[:2]

    header_h = 50
    grid_w = cols * patch_w + (cols + 1) * pad
    grid_h = rows * patch_h + (rows + 1) * pad + header_h

    canvas = np.full((grid_h, grid_w, 3), 20, dtype=np.uint8)

    # Header
    cv2.putText(canvas, title, (pad, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    for idx, p in enumerate(patches):
        r = idx // cols
        c = idx % cols
        y = header_h + pad + r * (patch_h + pad)
        x = pad + c * (patch_w + pad)
        canvas[y:y + patch_h, x:x + patch_w] = p

    return canvas


def main():
    manifest_p = paths.data_dir / "splits" / "leave_one_lot_out_occupancy_manifest.json"
    if not manifest_p.exists():
        logger.error(f"Manifest not found: {manifest_p}")
        return

    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    splits = data["splits"]

    # Load production ONNX classifier
    classifier = PatchOccupancyClassifier(model_path="models/occupancy_classifier.onnx")

    out_dir = paths.resolve("data/test_reports/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect sample sets for visualization:
    # 1. UFPR04 (Held-Out Lot) Sunny Free & Busy
    # 2. UFPR04 Rainy Free & Busy
    # 3. UFPR05 (Held-Out Lot) Cloudy / Shaded Free & Busy
    categories = {
        "UFPR04_Sunny": [r for r in splits["test_leave_one_lot_ufpr04"]["records"] if r.get("weather") == "Sunny"],
        "UFPR04_Rainy": [r for r in splits["test_leave_one_lot_ufpr04"]["records"] if r.get("weather") == "Rainy"],
        "UFPR05_Cloudy_Rainy": [r for r in splits["test_leave_one_lot_ufpr05"]["records"] if r.get("weather") in ["Cloudy", "Rainy"]],
        "CNRPark_Camera09": splits["test_cnr_camera09"]["records"][:30],
    }

    generated_grids = []

    for cat_name, records in categories.items():
        if not records:
            continue

        # Pick diverse subset: 4 free, 4 busy
        free_recs = [r for r in records if r["class_id"] == 0][:4]
        busy_recs = [r for r in records if r["class_id"] == 1][:4]
        selected = free_recs + busy_recs

        annotated_list = []
        for r in selected:
            img_path = paths.resolve(r["path"])
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            pred_occ, conf = classifier.predict_patch(img)
            gt_label = r.get("label", "free" if r["class_id"] == 0 else "busy")
            meta = f"{r.get('parking_lot', '')} | {r.get('weather', '')} | {r.get('source_frame', '')[:10]}"
            ann = create_annotated_patch(img, pred_occ, conf, gt_label, meta)
            annotated_list.append(ann)

        if annotated_list:
            grid = build_comparison_grid(
                annotated_list,
                cols=4,
                title=f"Occupancy Generalization: {cat_name} (Held-Out Evaluation)"
            )
            out_file = out_dir / f"occupancy_vis_{cat_name.lower()}.jpg"
            cv2.imwrite(str(out_file), grid)
            logger.info(f"Generated visualization grid: {out_file.name}")
            generated_grids.append(str(out_file))

    # Also generate a comparison gallery with both correct and any edge cases
    # Find up to 4 correct free, 4 correct busy, and any edge misclassifications
    all_test_recs = (
        splits["test_leave_one_lot_ufpr04"]["records"] +
        splits["test_leave_one_lot_ufpr05"]["records"]
    )
    correct_samples = []
    error_samples = []

    for r in all_test_recs:
        img_path = paths.resolve(r["path"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        pred_occ, conf = classifier.predict_patch(img)
        gt_is_occ = (r["class_id"] == 1)
        meta = f"{r.get('parking_lot', '')} {r.get('weather', '')}"
        ann = create_annotated_patch(img, pred_occ, conf, r.get("label", ""), meta)
        if pred_occ == gt_is_occ:
            if len(correct_samples) < 8:
                correct_samples.append(ann)
        else:
            if len(error_samples) < 4:
                error_samples.append(ann)

    if correct_samples:
        grid_perf = build_comparison_grid(
            correct_samples + error_samples,
            cols=4,
            title="Held-Out PKLot Generalization: Correct Predictions & Borderline Edge Cases"
        )
        perf_file = out_dir / "occupancy_generalization_gallery.jpg"
        cv2.imwrite(str(perf_file), grid_perf)
        logger.info(f"Generated comprehensive gallery: {perf_file.name}")
        generated_grids.append(str(perf_file))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "generated_grids": generated_grids,
        "count": len(generated_grids)
    }
    (out_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Visualization generation complete.")


if __name__ == "__main__":
    main()
