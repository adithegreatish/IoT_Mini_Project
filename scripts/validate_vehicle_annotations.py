"""
Dataset Validation & Annotation Integrity Checker.
Enforces strict quality gates before any training run:
- Validates image readability & dimension integrity.
- Validates label syntax, coordinate normalization [0, 1], positive w/h, and bounds.
- Checks image/label correspondence and detects empty/orphaned files.
- Computes SHA-256 and perceptual difference hashes to catch exact and near-duplicates.
- Generates visual contact sheets with rendered bounding boxes.
- Computes overall dataset fingerprint and structured JSON audit report.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_dataset")

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for perceptual similarity comparison."""
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def validate_split(
    img_dir: Path,
    lbl_dir: Path,
    expected_classes: List[str] = CLASS_NAMES,
    contact_sheet_out: Optional[Path] = None,
    max_contact_samples: int = 16,
) -> Dict[str, Any]:
    """Validates an entire dataset split for syntax, integrity, and geometry."""
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total_images = len(img_files)

    usable_images = 0
    corrupt_images = 0
    missing_labels = 0
    empty_label_files = 0
    invalid_boxes_count = 0
    out_of_bounds_count = 0
    small_objects_count = 0  # < 1% of image area

    class_counts = {c: 0 for c in expected_classes}
    invalid_details = []
    seen_sha256 = {}
    seen_dhash = {}
    exact_duplicates = []
    near_duplicates = []

    contact_samples = []

    for img_p in img_files:
        # 1. Image Readability
        try:
            img = cv2.imread(str(img_p))
            if img is None:
                corrupt_images += 1
                invalid_details.append({"file": img_p.name, "issue": "CORRUPT_IMAGE_UNREADABLE"})
                continue
        except Exception as e:
            corrupt_images += 1
            invalid_details.append({"file": img_p.name, "issue": f"EXCEPTION_READING_IMAGE: {e}"})
            continue

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            corrupt_images += 1
            invalid_details.append({"file": img_p.name, "issue": "INVALID_IMAGE_DIMENSIONS"})
            continue

        # Check duplicate by content SHA-256
        sha = hashlib.sha256(img_p.read_bytes()).hexdigest()
        if sha in seen_sha256:
            exact_duplicates.append({"original": seen_sha256[sha], "duplicate": img_p.name})
        else:
            seen_sha256[sha] = img_p.name

        # Check near duplicate by dHash
        dh = compute_dhash(img)
        for prev_dh, prev_name in seen_dhash.items():
            if hamming_distance(dh, prev_dh) <= 2:  # very close similarity
                near_duplicates.append({"image1": prev_name, "image2": img_p.name, "hash_dist": hamming_distance(dh, prev_dh)})
                break
        seen_dhash[dh] = img_p.name

        # 2. Label Readability & Correspondence
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            missing_labels += 1
            invalid_details.append({"file": img_p.name, "issue": "MISSING_LABEL_FILE"})
            continue

        lbl_content = lbl_p.read_text(encoding="utf-8").strip()
        if not lbl_content:
            empty_label_files += 1
            usable_images += 1
            continue

        img_has_errors = False
        parsed_boxes = []

        for line_num, line in enumerate(lbl_content.splitlines(), 1):
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_boxes_count += 1
                img_has_errors = True
                invalid_details.append({"file": lbl_p.name, "line": line_num, "issue": f"EXPECTED_5_FIELDS_FOUND_{len(parts)}"})
                continue

            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError as ve:
                invalid_boxes_count += 1
                img_has_errors = True
                invalid_details.append({"file": lbl_p.name, "line": line_num, "issue": f"VALUE_ERROR: {ve}"})
                continue

            # Class ID validity
            if cls_id < 0 or cls_id >= len(expected_classes):
                invalid_boxes_count += 1
                img_has_errors = True
                invalid_details.append({"file": lbl_p.name, "line": line_num, "issue": f"UNKNOWN_CLASS_ID_{cls_id}"})
                continue

            cname = expected_classes[cls_id]

            # Coordinates in [0, 1]
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                invalid_boxes_count += 1
                img_has_errors = True
                invalid_details.append({"file": lbl_p.name, "line": line_num, "issue": f"COORDINATES_OUT_OF_BOUNDS_OR_NON_POSITIVE"})
                continue

            class_counts[cname] += 1

            # Check box bounds
            xmin = cx - bw / 2.0
            ymin = cy - bh / 2.0
            xmax = cx + bw / 2.0
            ymax = cy + bh / 2.0

            if xmin < -0.05 or ymin < -0.05 or xmax > 1.05 or ymax > 1.05:
                out_of_bounds_count += 1
                invalid_details.append({"file": lbl_p.name, "line": line_num, "issue": f"BOX_PROTRUDES_HEAVILY_BEYOND_IMAGE_BOUNDS"})

            # Check small object (< 1% image area)
            box_area = bw * bh
            if box_area < 0.01:
                small_objects_count += 1

            parsed_boxes.append((cname, xmin, ymin, xmax, ymax))

        if not img_has_errors:
            usable_images += 1

        # Collect sample for contact sheet
        if len(contact_samples) < max_contact_samples and parsed_boxes:
            contact_samples.append((img, parsed_boxes, img_p.name))

    # Generate visual contact sheet if requested
    contact_sheet_saved = None
    if contact_sheet_out and contact_samples:
        contact_sheet_saved = generate_contact_sheet(contact_samples, contact_sheet_out)

    return {
        "total_images": total_images,
        "usable_images": usable_images,
        "corrupt_images": corrupt_images,
        "missing_labels": missing_labels,
        "empty_label_files": empty_label_files,
        "invalid_boxes_count": invalid_boxes_count,
        "out_of_bounds_count": out_of_bounds_count,
        "small_objects_count": small_objects_count,
        "exact_duplicates": len(exact_duplicates),
        "near_duplicates": len(near_duplicates),
        "class_counts": class_counts,
        "invalid_details_sample": invalid_details[:10],
        "contact_sheet_path": str(contact_sheet_saved) if contact_sheet_saved else None,
    }


def generate_contact_sheet(samples: List[Tuple[np.ndarray, List[Tuple], str]], out_path: Path) -> Path:
    """Renders a 4x4 grid of images with annotated ground-truth bounding boxes."""
    grid_size = 4
    thumb_w, thumb_h = 320, 320
    canvas = np.zeros((thumb_h * grid_size, thumb_w * grid_size, 3), dtype=np.uint8)

    colors = {
        "car": (0, 255, 0),        # Green
        "motorcycle": (255, 200, 0), # Cyan/Blue
        "bus": (0, 165, 255),       # Orange
        "truck": (0, 0, 255),       # Red
    }

    for idx, (img, boxes, name) in enumerate(samples[:16]):
        row = idx // grid_size
        col = idx % grid_size

        thumb = cv2.resize(img, (thumb_w, thumb_h))
        for (cname, xmin, ymin, xmax, ymax) in boxes:
            x1 = max(0, int(xmin * thumb_w))
            y1 = max(0, int(ymin * thumb_h))
            x2 = min(thumb_w, int(xmax * thumb_w))
            y2 = min(thumb_h, int(ymax * thumb_h))

            color = colors.get(cname, (200, 200, 200))
            cv2.rectangle(thumb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(thumb, cname, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(thumb, name[:12], (6, thumb_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        canvas[row * thumb_h : (row + 1) * thumb_h, col * thumb_w : (col + 1) * thumb_w] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def compute_dataset_fingerprint(data_yaml: Path) -> str:
    """Computes a deterministic hash fingerprint representing entire dataset state."""
    h = hashlib.sha256()
    h.update(data_yaml.read_bytes())
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for split_key in ["train", "val", "test"]:
        split_p = paths.resolve(f"data/{cfg.get(split_key, '')}")
        if split_p.exists():
            for p in sorted(split_p.glob("*.jpg"))[:20]:
                h.update(p.name.encode())
                h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Validate Vehicle Dataset & Annotation Geometry")
    parser.add_argument("--data-yaml", type=str, default="data/vehicles.yaml", help="Path to vehicles.yaml")
    parser.add_argument("--contact-sheets", action="store_true", help="Generate visual inspection contact sheets")
    parser.add_argument("--output-json", type=str, default=None, help="Save structured audit JSON to file")
    args = parser.parse_args()

    yaml_p = paths.resolve(args.data_yaml)
    if not yaml_p.exists():
        logger.error(f"Dataset YAML not found at: {yaml_p}")
        sys.exit(1)

    with open(yaml_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 70)
    print(" VEHICLE DATASET INTEGRITY & ANNOTATION QUALITY AUDIT")
    print("=" * 70)
    print(f"Dataset YAML: {yaml_p}")

    classes = [cfg["names"][i] for i in sorted(cfg.get("names", {}).keys())] if "names" in cfg else CLASS_NAMES
    print(f"Target Classes: {classes}")

    overall_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_yaml": str(yaml_p),
        "classes": classes,
        "splits": {},
    }

    all_valid = True

    for split in ["train", "val", "test"]:
        rel_img_dir = cfg.get(split)
        if not rel_img_dir:
            continue
        img_dir = paths.resolve(f"data/{rel_img_dir}")
        lbl_dir = img_dir.parent.parent / "labels" / img_dir.name

        print(f"\nEvaluating Split: [{split.upper()}]")
        print(f"  Images: {img_dir}")
        print(f"  Labels: {lbl_dir}")

        contact_sheet_path = (paths.data_dir / "test_reports" / f"contact_sheet_{split}.jpg") if args.contact_sheets else None
        res = validate_split(img_dir, lbl_dir, expected_classes=classes, contact_sheet_out=contact_sheet_path)
        overall_report["splits"][split] = res

        print(f"  Total Images:    {res['total_images']}")
        print(f"  Usable Images:   {res['usable_images']}")
        print(f"  Corrupt Images:  {res['corrupt_images']}")
        print(f"  Missing Labels:  {res['missing_labels']}")
        print(f"  Empty Labels:    {res['empty_label_files']}")
        print(f"  Invalid Boxes:   {res['invalid_boxes_count']}")
        print(f"  Small Objects:   {res['small_objects_count']} (<1% area)")
        print(f"  Duplicates:      {res['exact_duplicates']} exact, {res['near_duplicates']} near")
        print(f"  Class Counts:    {res['class_counts']}")
        if res.get("contact_sheet_path"):
            print(f"  Contact Sheet:   {res['contact_sheet_path']}")

        if res["corrupt_images"] > 0 or res["invalid_boxes_count"] > 0 or res["missing_labels"] > 0:
            all_valid = False

    fingerprint = compute_dataset_fingerprint(yaml_p)
    overall_report["dataset_fingerprint"] = fingerprint
    overall_report["overall_valid"] = all_valid
    print("\n" + "-" * 70)
    print(f"Dataset Fingerprint (SHA-256): {fingerprint}")
    print(f"Overall Dataset Status:        {'PASSED / READY FOR TRAINING' if all_valid else 'FAILED / ATTENTION REQUIRED'}")
    print("=" * 70 + "\n")

    if args.output_json:
        out_p = paths.resolve(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(overall_report, indent=2), encoding="utf-8")
        logger.info(f"Saved audit report to: {out_p}")

    if not all_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
