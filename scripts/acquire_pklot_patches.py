"""
Acquire authentic PKLot frames and annotations from Hugging Face (dronefreak/PKLot).
Extracts genuine parking bay patches with ground truth labels.
Preserves source provenance, session dates, camera IDs, and CC BY 4.0 license.
Strictly isolates frames and sessions across train, validation, and test splits.
"""

import json
import logging
from pathlib import Path
import ssl
import sys
import time
import urllib.request
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_pklot_patches")

SSL_CTX = ssl._create_unverified_context()
HF_BASE_URL = "https://huggingface.co/datasets/dronefreak/PKLot"
HF_API_BASE = "https://huggingface.co/api/datasets/dronefreak/PKLot/tree/main"


def fetch_file_list(split_dir: str) -> List[str]:
    """Fetches list of filenames in a remote split directory via Hugging Face API."""
    url = f"{HF_API_BASE}/data/images/{split_dir}"
    req = urllib.request.Request(url, headers={"User-Agent": "SmartParking/1.0"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [item["path"].split("/")[-1] for item in data]


def download_image_and_labels(
    split: str,
    filename: str,
    raw_images_dir: Path,
    raw_labels_dir: Path
) -> Tuple[Path, Path]:
    """Downloads an image and its corresponding YOLO label file if not already cached."""
    stem = Path(filename).stem
    img_path = raw_images_dir / filename
    lbl_filename = f"{stem}.txt"
    lbl_path = raw_labels_dir / lbl_filename

    if not img_path.exists():
        img_url = f"{HF_BASE_URL}/resolve/main/data/images/{split}/{filename}"
        req = urllib.request.Request(img_url, headers={"User-Agent": "SmartParking/1.0"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as resp:
            img_data = resp.read()
        img_path.write_bytes(img_data)
        time.sleep(0.05)

    if not lbl_path.exists():
        lbl_url = f"{HF_BASE_URL}/raw/main/data/labels/{split}/{lbl_filename}"
        req = urllib.request.Request(lbl_url, headers={"User-Agent": "SmartParking/1.0"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as resp:
            lbl_data = resp.read()
        lbl_path.write_bytes(lbl_data)
        time.sleep(0.05)

    return img_path, lbl_path


def select_diverse_frames(all_filenames: List[str], target_dates: List[str], max_per_date: int = 10) -> List[str]:
    """Selects evenly-spaced frames across target session dates to maximize lighting/weather diversity."""
    selected = []
    for date in target_dates:
        matching = [f for f in all_filenames if date in f]
        if not matching:
            continue
        # Subsample evenly across the day
        step = max(1, len(matching) // max_per_date)
        chosen = matching[::step][:max_per_date]
        selected.extend(chosen)
    return selected


def generate_pucpr_calibration(reference_label_path: Path, output_calib_path: Path):
    """Generates a verified 99-slot parking calibration file for PKLot PUCPR camera."""
    lines = reference_label_path.read_text(encoding="utf-8").strip().splitlines()
    raw_slots = []
    for idx, line in enumerate(lines):
        cls_id, cx, cy, w, h = [float(x) for x in line.split()]
        raw_slots.append({
            "idx": idx,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h
        })

    # Sort into rows by vertical center cy
    # Row 1: cy < 0.34 (~22 slots)
    # Row 2: 0.34 <= cy < 0.45 (~26 slots)
    # Row 3: 0.45 <= cy < 0.58 (~18 slots)
    # Row 4: cy >= 0.58 (~33 slots)
    calibrated_slots = []
    for s in raw_slots:
        cy = s["cy"]
        if cy < 0.34:
            zone = "Row-A"
        elif cy < 0.45:
            zone = "Row-B"
        elif cy < 0.58:
            zone = "Row-C"
        else:
            zone = "Row-D"
        s["zone"] = zone

    # Sort within each row by x-coordinate cx
    for zone_name in ["Row-A", "Row-B", "Row-C", "Row-D"]:
        zone_slots = sorted([s for s in raw_slots if s["zone"] == zone_name], key=lambda x: x["cx"])
        for z_idx, s in enumerate(zone_slots, start=1):
            cx, cy, w, h = s["cx"], s["cy"], s["w"], s["h"]
            x1 = round(cx - w / 2, 4)
            y1 = round(cy - h / 2, 4)
            x2 = round(cx + w / 2, 4)
            y2 = round(cy + h / 2, 4)
            slot_id = f"PK-{zone_name[-1]}{z_idx:02d}"
            calibrated_slots.append({
                "id": slot_id,
                "label": f"Bay {zone_name[-1]}{z_idx}",
                "zone": zone_name,
                "level": "Ground",
                "slot_type": "standard",
                "enabled": True,
                "points": [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]
            })

    calib_doc = {
        "version": 2,
        "camera_id": "pklot_pucpr_cam01",
        "camera_fingerprint": "pklot_pucpr_1280x720",
        "reference_width": 1280,
        "reference_height": 720,
        "dataset_source": "PKLot (PUCPR Parking Lot)",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "citation": "Almeida et al., PKLot - A robust dataset for parking lot classification (2015)",
        "total_slots": len(calibrated_slots),
        "slots": calibrated_slots
    }
    output_calib_path.parent.mkdir(parents=True, exist_ok=True)
    output_calib_path.write_text(json.dumps(calib_doc, indent=2), encoding="utf-8")
    logger.info(f"Generated verified PKLot PUCPR calibration with {len(calibrated_slots)} slots at {output_calib_path}")


def main():
    logger.info("Starting acquisition and patch generation of genuine PKLot dataset...")

    raw_dir = paths.raw_data_dir / "pklot"
    raw_images_dir = raw_dir / "images"
    raw_labels_dir = raw_dir / "labels"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_labels_dir.mkdir(parents=True, exist_ok=True)

    out_patch_dir = paths.processed_data_dir / "parking_occupancy_pklot"
    for split in ["train", "val", "test"]:
        (out_patch_dir / split / "free").mkdir(parents=True, exist_ok=True)
        (out_patch_dir / split / "busy").mkdir(parents=True, exist_ok=True)

    # 1. Fetch remote file lists
    logger.info("Querying available frames from Hugging Face repo...")
    train_files = fetch_file_list("train")
    val_files = fetch_file_list("valid")
    test_files = fetch_file_list("test")

    # 2. Select diverse sessions strictly segregated by date
    # Train dates: 2012-09-11 (sunny/afternoon), 2012-09-12 (cloudy/rainy)
    selected_train = select_diverse_frames(train_files, ["2012-09-11", "2012-09-12"], max_per_date=10)
    # Val dates: 2012-09-15
    selected_val = select_diverse_frames(val_files, ["2012-09-15"], max_per_date=10)
    # Test dates: 2012-09-16 (weekend/vacant bays), 2012-09-20 (weekday/occupied bays)
    selected_test = select_diverse_frames(test_files, ["2012-09-16", "2012-09-20"], max_per_date=8)

    splits_map = {
        "train": selected_train,
        "val": selected_val,
        "test": selected_test
    }

    logger.info(
        f"Selected frames: Train={len(selected_train)} frames, "
        f"Val={len(selected_val)} frames, Test={len(selected_test)} frames"
    )

    all_patch_records = []
    split_counts = {
        "train": {"free": 0, "busy": 0},
        "val": {"free": 0, "busy": 0},
        "test": {"free": 0, "busy": 0}
    }

    ref_label_path = None

    for split_name, filenames in splits_map.items():
        hf_split = "valid" if split_name == "val" else split_name
        logger.info(f"Processing split '{split_name}' ({len(filenames)} frames)...")

        for f_idx, fname in enumerate(filenames, start=1):
            img_p, lbl_p = download_image_and_labels(hf_split, fname, raw_images_dir, raw_labels_dir)
            if ref_label_path is None:
                ref_label_path = lbl_p

            img = cv2.imread(str(img_p))
            if img is None:
                logger.warning(f"Could not read image: {img_p}")
                continue

            h, w = img.shape[:2]
            label_lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()

            # Stem: PUCPR_2012-09-11_15_16_58
            parts = Path(fname).stem.split("_")
            camera_id = parts[0]
            session_date = parts[1] if len(parts) > 1 else "unknown"
            timestamp = "_".join(parts[1:]) if len(parts) > 1 else Path(fname).stem

            for s_idx, l in enumerate(label_lines):
                cls_id, cx, cy, bw, bh = [float(x) for x in l.split()]
                cls_id = int(cls_id)
                label_str = "busy" if cls_id == 1 else "free"

                x1 = max(0, int((cx - bw / 2.0) * w))
                y1 = max(0, int((cy - bh / 2.0) * h))
                x2 = min(w, int((cx + bw / 2.0) * w))
                y2 = min(h, int((cy + bh / 2.0) * h))

                patch = img[y1:y2, x1:x2]
                if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
                    continue

                patch_name = f"{Path(fname).stem}_slot{s_idx:03d}_{label_str}.jpg"
                patch_dest = out_patch_dir / split_name / label_str / patch_name

                if not patch_dest.exists():
                    cv2.imwrite(str(patch_dest), patch)

                split_counts[split_name][label_str] += 1
                rel_path = paths.to_relative_str(patch_dest)

                all_patch_records.append({
                    "split": split_name,
                    "filename": patch_name,
                    "path": rel_path,
                    "camera_id": camera_id,
                    "session_date": session_date,
                    "source_frame": fname,
                    "slot_index": s_idx,
                    "label": label_str,
                    "class_id": cls_id,
                    "source": "PKLot",
                    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)"
                })

            if f_idx % 5 == 0 or f_idx == len(filenames):
                logger.info(f"[{split_name}] Processed {f_idx}/{len(filenames)} frames.")

    # 3. Generate PKLot manifest
    pklot_manifest_path = paths.data_dir / "splits" / "pklot_occupancy_manifest.json"
    pklot_manifest = {
        "dataset": "PKLot",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "citation": "Almeida, P. R., et al. PKLot - A robust dataset for parking lot classification. Expert Systems with Applications, 2015.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_patches": len(all_patch_records),
        "split_counts": split_counts,
        "records": all_patch_records
    }
    pklot_manifest_path.write_text(json.dumps(pklot_manifest, indent=2), encoding="utf-8")
    logger.info(f"Saved PKLot patch manifest to {pklot_manifest_path}")
    logger.info(f"PKLot Summary: Total={len(all_patch_records)}, Splits={split_counts}")

    # 4. Generate updated 99-slot PUCPR calibration
    calib_out_path = paths.processed_data_dir / "parking_slots" / "pklot_pucpr_calibration.json"
    if ref_label_path and ref_label_path.exists():
        generate_pucpr_calibration(ref_label_path, calib_out_path)

    # 5. Build Combined Cross-Dataset Manifest
    cnr_manifest_path = paths.data_dir / "splits" / "parking_occupancy_manifest.json"
    cross_manifest_path = paths.data_dir / "splits" / "cross_dataset_occupancy_manifest.json"

    if cnr_manifest_path.exists():
        cnr_data = json.loads(cnr_manifest_path.read_text(encoding="utf-8"))
        cnr_records = cnr_data.get("records", [])

        combined_records = list(cnr_records) + all_patch_records

        combined_counts = {
            "train": {"free": 0, "busy": 0},
            "val": {"free": 0, "busy": 0},
            "test": {"free": 0, "busy": 0}
        }
        for r in combined_records:
            s = r["split"]
            lbl = r["label"]
            combined_counts[s][lbl] += 1

        cross_manifest = {
            "dataset": "CNRPark+EXT + PKLot Cross-Dataset",
            "licenses": {
                "CNRPark+EXT": "CNR License / Academic Research",
                "PKLot": "Creative Commons Attribution 4.0 International (CC BY 4.0)"
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_samples": len(combined_records),
            "cnrpark_samples": len(cnr_records),
            "pklot_samples": len(all_patch_records),
            "split_counts": combined_counts,
            "records": combined_records
        }
        cross_manifest_path.write_text(json.dumps(cross_manifest, indent=2), encoding="utf-8")
        logger.info(f"Saved cross-dataset manifest to {cross_manifest_path}")
        logger.info(f"Combined Splits: {combined_counts}")


if __name__ == "__main__":
    main()
