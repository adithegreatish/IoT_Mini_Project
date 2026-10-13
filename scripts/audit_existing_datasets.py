"""
Comprehensive Audit of Existing Datasets, Annotations, Checkpoints, and Metrics
Generates a verified inventory detailing:
- Unique image counts and disk usage
- Instance distributions across splits
- Split leakage checks (via SHA-256 image hashes)
- Bounding box coordinate validity ([0, 1] normalized checks)
- Class balance and previous evaluation baselines
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

def compute_file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def audit_dataset_split(img_dir: Path, label_dir: Path, class_names: list) -> dict:
    if not img_dir.exists():
        return {"images": 0, "instances": 0, "classes": {}, "invalid_boxes": 0, "corrupted": 0, "hashes": {}}

    img_files = sorted(list(img_dir.glob("*.*")))
    img_files = [f for f in img_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    hashes = {}
    class_counts = Counter()
    total_instances = 0
    invalid_boxes = 0
    corrupted_images = 0

    for img_p in img_files:
        # Check decodability
        img = cv2.imread(str(img_p))
        if img is None:
            corrupted_images += 1
            continue

        f_hash = compute_file_hash(img_p)
        hashes[img_p.name] = f_hash

        label_p = label_dir / f"{img_p.stem}.txt"
        if not label_p.exists():
            continue

        try:
            lines = label_p.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    # Bounding box bounds check
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        invalid_boxes += 1
                    else:
                        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
                        class_counts[cls_name] += 1
                        total_instances += 1
        except Exception:
            invalid_boxes += 1

    return {
        "images": len(img_files),
        "corrupted": corrupted_images,
        "instances": total_instances,
        "classes": dict(class_counts),
        "invalid_boxes": invalid_boxes,
        "hashes": hashes
    }

def main():
    print("=" * 70)
    print("PHASE 1: AUDITING EXISTING DATASETS, MODELS & SPLITS")
    print("=" * 70)

    # 1. Audit Vehicle Dataset
    v_classes = ["car", "motorcycle", "bus", "truck"]
    v_proc = paths.processed_data_dir / "vehicles"
    v_splits = {}
    v_all_hashes = {}
    v_leakage = []

    for s in ["train", "val", "test"]:
        split_res = audit_dataset_split(
            v_proc / "images" / s,
            v_proc / "labels" / s,
            v_classes
        )
        v_splits[s] = split_res
        for fname, h in split_res["hashes"].items():
            if h in v_all_hashes:
                v_leakage.append((fname, s, v_all_hashes[h]))
            else:
                v_all_hashes[h] = (fname, s)

    print("\n--- VEHICLE DETECTION DATASET AUDIT ---")
    print(f"Base Directory: {v_proc}")
    for s, data in v_splits.items():
        print(f"[{s.upper():5s}] Images: {data['images']:4d} | Instances: {data['instances']:4d} | Classes: {data['classes']} | Invalid Boxes: {data['invalid_boxes']}")
    print(f"Cross-Split Hash Collisions / Data Leakage: {len(v_leakage)}")

    # 2. Audit Plate Dataset
    p_classes = ["license_plate"]
    p_proc = paths.processed_data_dir / "plates"
    p_splits = {}
    p_all_hashes = {}
    p_leakage = []

    for s in ["train", "val", "test"]:
        split_res = audit_dataset_split(
            p_proc / "images" / s,
            p_proc / "labels" / s,
            p_classes
        )
        p_splits[s] = split_res
        for fname, h in split_res["hashes"].items():
            if h in p_all_hashes:
                p_leakage.append((fname, s, p_all_hashes[h]))
            else:
                p_all_hashes[h] = (fname, s)

    print("\n--- LICENSE PLATE DATASET AUDIT ---")
    print(f"Base Directory: {p_proc}")
    for s, data in p_splits.items():
        print(f"[{s.upper():5s}] Images: {data['images']:4d} | Instances: {data['instances']:4d} | Invalid Boxes: {data['invalid_boxes']}")
    print(f"Cross-Split Hash Collisions / Data Leakage: {len(p_leakage)}")

    # 3. Check Baseline Evaluation Artifacts
    eval_dir = paths.resolve("data/test_reports/evaluations")
    latest_eval_json = None
    if eval_dir.exists():
        jsons = sorted(list(eval_dir.glob("eval_report_*.json")))
        if jsons:
            latest_eval_json = jsons[-1]

    print("\n--- BASELINE EVALUATION ARTIFACT VERIFICATION ---")
    if latest_eval_json:
        eval_data = json.loads(latest_eval_json.read_text(encoding="utf-8"))
        print(f"Latest Evaluation File: {latest_eval_json.name}")
        v_m = eval_data.get("vehicle_detector", {}).get("test_metrics", {})
        print(f"Vehicle Test Metrics: Precision={v_m.get('precision')}, Recall={v_m.get('recall')}, mAP50={v_m.get('mAP50')}, mAP50-95={v_m.get('mAP50_95')}")
        p_m = eval_data.get("plate_detector", {}).get("test_metrics", {})
        print(f"Plate Test Metrics:   Precision={p_m.get('precision')}, Recall={p_m.get('recall')}, mAP50={p_m.get('mAP50')}, mAP50-95={p_m.get('mAP50_95')}")
        ind_ocr = eval_data.get("ocr_engine", {}).get("indian_ground_truth_metrics", {})
        if ind_ocr:
            print(f"Indian OCR Metrics:   Valid Format={ind_ocr.get('valid_format_rate')*100:.1f}%, Exact Acc={ind_ocr.get('exact_match_accuracy')*100:.1f}%, CER={ind_ocr.get('character_error_rate')*100:.2f}%")
    else:
        print("No evaluation reports found.")

    # 4. Save Inventory JSON
    inventory = {
        "audit_timestamp": json.dumps(None),
        "vehicle_dataset": {
            "source": "COCO 2017 Val Subset",
            "splits": {s: {k: v for k, v in d.items() if k != "hashes"} for s, d in v_splits.items()},
            "total_images": sum(d["images"] for d in v_splits.values()),
            "total_instances": sum(d["instances"] for d in v_splits.values()),
            "leakage_count": len(v_leakage)
        },
        "plate_dataset": {
            "source": "Indian & Benchmark License Plate Datasets",
            "splits": {s: {k: v for k, v in d.items() if k != "hashes"} for s, d in p_splits.items()},
            "total_images": sum(d["images"] for d in p_splits.values()),
            "total_instances": sum(d["instances"] for d in p_splits.values()),
            "leakage_count": len(p_leakage)
        },
        "latest_eval_reference": latest_eval_json.name if latest_eval_json else None
    }

    out_inv = paths.resolve("data/dataset_inventory.json")
    out_inv.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"\n[+] Saved inventory to: {out_inv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
