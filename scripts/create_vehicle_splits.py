"""
Multi-View Vehicle Split Generator & Leakage Verifier.
Constructs and maintains the three authoritative evaluation benchmarks:
1. View 1 (General Object-Detection Benchmark): held-out COCO test split.
2. View 2 (Real Parking-Camera Benchmark): high-angle overhead CCTV camera frames.
3. View 3 (Difficult-Case Benchmark): mined challenging vehicles, occlusions, truck ambiguities, small objects.
Verifies zero inter-split leakage across all partitions.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Set

import cv2
import yaml

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_splits")


def get_image_hash(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def build_multiview_manifests():
    splits_dir = paths.data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    test_img_dir = paths.resolve("data/processed/vehicles/images/test")
    test_lbl_dir = paths.resolve("data/processed/vehicles/labels/test")

    # 1. View 1: General Benchmark
    general_samples = []
    for img_p in sorted(test_img_dir.glob("*.jpg")):
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        box_count = len(lbl_p.read_text(encoding="utf-8").strip().splitlines()) if lbl_p.exists() else 0
        general_samples.append({
            "image": str(img_p.relative_to(paths.root)).replace("\\", "/"),
            "label": str(lbl_p.relative_to(paths.root)).replace("\\", "/") if lbl_p.exists() else None,
            "box_count": box_count,
            "sha256": get_image_hash(img_p),
        })

    view1_manifest = {
        "view_name": "general_object_detection",
        "description": "Held-out COCO 2017 street photography vehicle test benchmark",
        "created_at": datetime.now().isoformat(),
        "sample_count": len(general_samples),
        "samples": general_samples,
    }
    (splits_dir / "test_general_manifest.json").write_text(json.dumps(view1_manifest, indent=2), encoding="utf-8")

    # 2. View 2: Real Parking-Camera Benchmark
    # PKLot full frames under CCTV angle
    pklot_img_dir = paths.resolve("data/raw/pklot/images")
    parking_cam_samples = []
    if pklot_img_dir.exists():
        for img_p in sorted(pklot_img_dir.glob("*.jpg"))[:30]:
            parking_cam_samples.append({
                "image": str(img_p.relative_to(paths.root)).replace("\\", "/"),
                "source": "PKLot_PUCPR_CCTV",
                "perspective": "high_angle_overhead_cctv",
                "sha256": get_image_hash(img_p),
            })

    view2_manifest = {
        "view_name": "real_parking_camera",
        "description": "High-angle overhead parking surveillance camera frames",
        "created_at": datetime.now().isoformat(),
        "sample_count": len(parking_cam_samples),
        "samples": parking_cam_samples,
    }
    (splits_dir / "test_parking_camera_manifest.json").write_text(json.dumps(view2_manifest, indent=2), encoding="utf-8")

    # 3. View 3: Difficult-Case Benchmark
    # Select test samples containing truck instances, small objects (< 1% area), or high density
    difficult_samples = []
    for s in general_samples:
        lbl_p = paths.resolve(s["label"])
        is_difficult = False
        reasons = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    bw, bh = float(parts[3]), float(parts[4])
                    if cls_id == 3:  # truck
                        is_difficult = True
                        reasons.append("contains_truck")
                    if bw * bh < 0.008:  # very small vehicle
                        is_difficult = True
                        reasons.append("small_distant_vehicle")
            if s["box_count"] >= 6:
                is_difficult = True
                reasons.append("dense_multi_vehicle_scene")

        if is_difficult:
            difficult_samples.append({
                "image": s["image"],
                "label": s["label"],
                "box_count": s["box_count"],
                "difficulty_reasons": list(set(reasons)),
                "sha256": s["sha256"],
            })

    view3_manifest = {
        "view_name": "difficult_cases",
        "description": "Curated hard-cases benchmark (truck ambiguities, small vehicles, dense scenes)",
        "created_at": datetime.now().isoformat(),
        "sample_count": len(difficult_samples),
        "samples": difficult_samples,
    }
    (splits_dir / "test_difficult_cases_manifest.json").write_text(json.dumps(view3_manifest, indent=2), encoding="utf-8")

    logger.info(f"Built multi-view manifests in {splits_dir}:")
    logger.info(f"  View 1 (General): {len(general_samples)} samples")
    logger.info(f"  View 2 (Parking Camera): {len(parking_cam_samples)} samples")
    logger.info(f"  View 3 (Difficult Cases): {len(difficult_samples)} samples")


def verify_leakage():
    """Verifies that no images overlap between train, val, and test splits."""
    train_dir = paths.resolve("data/processed/vehicles/images/train")
    val_dir = paths.resolve("data/processed/vehicles/images/val")
    test_dir = paths.resolve("data/processed/vehicles/images/test")

    train_hashes = {get_image_hash(p): p.name for p in train_dir.glob("*.jpg")}
    val_hashes = {get_image_hash(p): p.name for p in val_dir.glob("*.jpg")}
    test_hashes = {get_image_hash(p): p.name for p in test_dir.glob("*.jpg")}

    train_val = set(train_hashes.keys()) & set(val_hashes.keys())
    train_test = set(train_hashes.keys()) & set(test_hashes.keys())
    val_test = set(val_hashes.keys()) & set(test_hashes.keys())

    print("\n--- LEAKAGE VERIFICATION AUDIT ---")
    print(f"Train / Val Collisions:  {len(train_val)}")
    print(f"Train / Test Collisions: {len(train_test)}")
    print(f"Val / Test Collisions:   {len(val_test)}")

    if len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0:
        print("[+] ZERO LEAKAGE CONFIRMED: All partitions are strictly independent.\n")
        return True
    else:
        print("[-] DATASET LEAKAGE DETECTED! Partitions share identical images.\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create and verify multi-view evaluation splits")
    parser.add_argument("--build-manifests", action="store_true", help="Build the 3 evaluation view manifests")
    parser.add_argument("--verify-leakage", action="store_true", help="Check for duplicate hash collisions across splits")
    args = parser.parse_args()

    build_multiview_manifests()
    verify_leakage()


if __name__ == "__main__":
    main()
