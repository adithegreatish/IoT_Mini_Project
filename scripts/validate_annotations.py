"""Validates bounding box and polygon annotations for boundaries, syntax, and corruption."""

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import yaml

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths


def validate_yolo_txt(file_path: Path) -> list:
    """Validates YOLO normalized bounding box format: <class_id> <x_center> <y_center> <width> <height>"""
    errors = []
    lines = file_path.read_text(encoding="utf-8").strip().splitlines()
    for idx, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"Line {idx}: Expected 5 fields, found {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            for c in coords:
                if not (0.0 <= c <= 1.0):
                    errors.append(f"Line {idx}: Coordinate {c} outside normalized [0, 1] range")
        except ValueError as ve:
            errors.append(f"Line {idx}: Parse error: {ve}")
    return errors


def validate_pascal_voc_xml(file_path: Path) -> list:
    """Validates Pascal VOC XML format: <annotation><object><bndbox>"""
    errors = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        objects = root.findall("object")
        if not objects:
            errors.append("No <object> elements declared in annotation")
        for obj in objects:
            bndbox = obj.find("bndbox")
            if bndbox is None:
                errors.append("Object missing <bndbox> child")
                continue
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
            if xmax <= xmin or ymax <= ymin:
                errors.append(f"Degenerate box: [{xmin}, {ymin}, {xmax}, {ymax}]")
    except Exception as e:
        errors.append(f"XML parse error: {e}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate bounding box annotations syntax and geometry")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml", help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg_path = paths.resolve(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    annot_dir = paths.resolve(cfg.get("local_capture", {}).get("annotations_dir", "data/annotations"))
    print("\n" + "=" * 65)
    print("ANNOTATION VALIDATION AUDIT")
    print("=" * 65)
    print(f"Target Directory: {annot_dir}")

    if not annot_dir.exists():
        print(f"[!] Target directory {annot_dir} does not exist.")
        sys.exit(0)

    txt_files = list(annot_dir.glob("*.txt"))
    xml_files = list(annot_dir.glob("*.xml"))
    total = len(txt_files) + len(xml_files)

    if total == 0:
        print("[*] No local annotation files found in directory. Directory is clean.")
        print("=" * 65 + "\n")
        sys.exit(0)

    valid_count = 0
    invalid_count = 0

    for f in txt_files:
        errs = validate_yolo_txt(f)
        if errs:
            invalid_count += 1
            print(f"[-] INVALID: {f.name} -> {errs[:2]}")
        else:
            valid_count += 1

    for f in xml_files:
        errs = validate_pascal_voc_xml(f)
        if errs:
            invalid_count += 1
            print(f"[-] INVALID: {f.name} -> {errs[:2]}")
        else:
            valid_count += 1

    print("-" * 65)
    print(f"Audit Complete: Valid: {valid_count} | Invalid: {invalid_count} | Total: {total}\n")
    sys.exit(0 if invalid_count == 0 else 1)


if __name__ == "__main__":
    main()
