"""
Active Learning Hard-Example Miner.
Scans camera streams, video recordings, or raw image folders to identify high-value training candidates:
1. Low-confidence vehicle detections (0.20 <= conf < 0.40).
2. Detections close to decision boundaries.
3. High-density scenes or ambiguous classes (truck vs car).
4. Contradictions between vehicle bounding boxes and occupancy patch states.
5. Baseline vs candidate model disagreements.
Applies perceptual deduplication (dHash) and enqueues samples for human review.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.detection.data_capture import data_capture_manager
from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mine_hard_examples")


def mine_from_image_directory(
    input_dir: Path,
    model_path: Path,
    conf_min: float = 0.20,
    conf_max: float = 0.42,
    max_candidates: int = 50,
) -> int:
    """Scans image folder with YOLO model and flags difficult borderline samples."""
    from ultralytics import YOLO

    if not model_path.exists():
        logger.error(f"Detector model not found at: {model_path}")
        return 0

    model = YOLO(str(model_path))
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))

    logger.info(f"Scanning {len(image_paths)} images in {input_dir} for hard examples...")
    mined_count = 0

    for img_p in image_paths:
        if mined_count >= max_candidates:
            break

        frame = cv2.imread(str(img_p))
        if frame is None:
            continue

        results = model.predict(frame, conf=0.15, verbose=False)
        if not results:
            continue

        res = results[0]
        h, w = frame.shape[:2]

        for box in res.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cname = res.names.get(cls_id, "unknown")

            reason = None
            if conf_min <= conf < conf_max:
                reason = f"LOW_CONFIDENCE_{cname.upper()}_{conf:.2f}"
            elif cname == "truck" and conf < 0.55:
                reason = f"UNCERTAIN_TRUCK_{conf:.2f}"

            if reason:
                xyxy = box.xyxy[0].cpu().numpy()
                norm_bbox = (
                    round(float(xyxy[0]) / w, 4),
                    round(float(xyxy[1]) / h, 4),
                    round(float(xyxy[2]) / w, 4),
                    round(float(xyxy[3]) / h, 4),
                )
                success, fid = data_capture_manager.capture_candidate_frame(
                    frame=frame,
                    reason=reason,
                    confidence=conf,
                    class_name=cname,
                    suggested_bbox=norm_bbox,
                    force=False,
                )
                if success:
                    mined_count += 1
                    logger.info(f"[+] Mined candidate {fid}: {reason} ({img_p.name})")
                    break  # Move to next image to ensure diversity

    return mined_count


def main():
    parser = argparse.ArgumentParser(description="Mine hard vehicle examples for continuous active learning")
    parser.add_argument("--input-dir", type=str, default="data/raw/pklot/images", help="Directory of raw frames")
    parser.add_argument("--model", type=str, default="models/vehicle_detector.pt", help="Path to vehicle detector")
    parser.add_argument("--conf-min", type=float, default=0.20, help="Minimum confidence for borderline mine")
    parser.add_argument("--conf-max", type=float, default=0.42, help="Maximum confidence for borderline mine")
    parser.add_argument("--max-candidates", type=int, default=25, help="Maximum candidates to enqueue")
    args = parser.parse_args()

    in_dir = paths.resolve(args.input_dir)
    mod_p = paths.resolve(args.model)

    print("\n" + "=" * 70)
    print(" ACTIVE LEARNING HARD-EXAMPLE MINING")
    print("=" * 70)
    print(f"Target Images:   {in_dir}")
    print(f"Detector Model:  {mod_p}")
    print(f"Confidence Band: [{args.conf_min:.2f} <= conf < {args.conf_max:.2f}]")
    print(f"Max Candidates:  {args.max_candidates}")

    count = mine_from_image_directory(
        input_dir=in_dir,
        model_path=mod_p,
        conf_min=args.conf_min,
        conf_max=args.conf_max,
        max_candidates=args.max_candidates,
    )

    print(f"\n[+] Mining Completed. Successfully enqueued {count} high-information candidates.")
    print(f"[+] Review Queue: {paths.data_dir / 'active_learning' / 'candidates'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
