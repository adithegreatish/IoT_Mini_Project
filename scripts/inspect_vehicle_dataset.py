"""
Detailed Vehicle Dataset Inspection & Statistical Profiler.
Analyzes bounding box distributions, aspect ratios, small vs large object ratios,
image resolutions, and class imbalance metrics for research reporting.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List

import cv2
import numpy as np
import yaml

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def inspect_dataset_statistics(data_yaml_path: Path) -> Dict:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = [cfg["names"][i] for i in sorted(cfg.get("names", {}).keys())] if "names" in cfg else CLASS_NAMES

    report = {"splits": {}, "classes": classes}

    for split in ["train", "val", "test"]:
        rel_img = cfg.get(split)
        if not rel_img:
            continue
        img_dir = paths.resolve(f"data/{rel_img}")
        lbl_dir = img_dir.parent.parent / "labels" / img_dir.name

        box_areas = []
        aspect_ratios = []
        per_class_boxes = {c: [] for c in classes}
        resolutions = []

        for img_p in img_dir.glob("*.jpg"):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    bw, bh = float(parts[3]), float(parts[4])
                    area = bw * bh
                    ar = bw / max(1e-5, bh)
                    box_areas.append(area)
                    aspect_ratios.append(ar)
                    if 0 <= cls_id < len(classes):
                        per_class_boxes[classes[cls_id]].append(area)

        # Compute percentiles
        report["splits"][split] = {
            "total_images": len(list(img_dir.glob("*.jpg"))),
            "total_annotations": len(box_areas),
            "area_distribution": {
                "mean": round(float(np.mean(box_areas)), 4) if box_areas else 0,
                "median": round(float(np.median(box_areas)), 4) if box_areas else 0,
                "p25": round(float(np.percentile(box_areas, 25)), 4) if box_areas else 0,
                "p75": round(float(np.percentile(box_areas, 75)), 4) if box_areas else 0,
                "small_objects_pct": round(float(np.mean([a < 0.01 for a in box_areas]) * 100), 2) if box_areas else 0,
            },
            "aspect_ratio": {
                "mean": round(float(np.mean(aspect_ratios)), 3) if aspect_ratios else 0,
                "median": round(float(np.median(aspect_ratios)), 3) if aspect_ratios else 0,
            },
            "class_instance_counts": {c: len(boxes) for c, boxes in per_class_boxes.items()},
        }

    return report


def main():
    parser = argparse.ArgumentParser(description="Detailed statistical inspection of vehicle dataset")
    parser.add_argument("--data-yaml", type=str, default="data/vehicles.yaml", help="Path to vehicles.yaml")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    yaml_p = paths.resolve(args.data_yaml)
    stats = inspect_dataset_statistics(yaml_p)

    if args.json:
        print(json.dumps(stats, indent=2))
        return

    print("\n" + "=" * 70)
    print(" VEHICLE DATASET STATISTICAL PROFILE")
    print("=" * 70)
    for split, data in stats["splits"].items():
        print(f"\nSplit: [{split.upper()}] ({data['total_images']} images, {data['total_annotations']} annotations)")
        print(f"  Class Breakdown:  {data['class_instance_counts']}")
        area = data["area_distribution"]
        print(f"  Box Area:         Mean={area['mean']:.4f}, Median={area['median']:.4f}, Small (<1%)={area['small_objects_pct']}%")
        ar = data["aspect_ratio"]
        print(f"  Aspect Ratio:     Mean={ar['mean']:.2f}, Median={ar['median']:.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
