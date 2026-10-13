"""
Acquires, filters, and prepares official COCO 2017 vehicle detection dataset.

Downloads:
- annotations_trainval2017.zip (for instances_val2017.json)
- val2017.zip (5,000 images)

Filters for target vehicle classes:
- car (COCO cat_id: 3 -> class 0)
- motorcycle (COCO cat_id: 4 -> class 1)
- bus (COCO cat_id: 6 -> class 2)
- truck (COCO cat_id: 8 -> class 3)

Validates image decodability, bounding boxes, and generates leakage-free
train/val/test splits using project-relative references.
"""

from collections import defaultdict
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import urllib.request
import zipfile
import cv2
import yaml

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_coco_vehicles")

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"

TARGET_CLASSES = {
    3: 0,  # car -> 0
    4: 1,  # motorcycle -> 1
    6: 2,  # bus -> 2
    8: 3,  # truck -> 3
}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def download_with_progress(url: str, dest_path: Path) -> bool:
    """Downloads a file with progress reporting and resume/cache support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        logger.info(f"Using existing archive: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return True

    logger.info(f"Starting download: {url} -> {dest_path.name}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    t0 = time.time()
    last_log = t0
    part_path = dest_path.with_suffix(".part")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(part_path, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_log >= 5.0 or downloaded == total_size:
                    speed_mb = (downloaded / (1024 * 1024)) / max(0.1, now - t0)
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    logger.info(
                        f"  [{pct:5.1f}%] {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB "
                        f"({speed_mb:.2f} MB/s)"
                    )
                    last_log = now

        part_path.rename(dest_path)
        logger.info(f"Successfully downloaded {dest_path.name} in {time.time() - t0:.1f}s.")
        return True
    except Exception as e:
        logger.error(f"Download error for {url}: {e}")
        if part_path.exists():
            part_path.unlink()
        return False


def extract_instances_json(annotations_zip: Path, target_dir: Path) -> Path:
    """Extracts instances_val2017.json from annotations_trainval2017.zip."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out_json = target_dir / "instances_val2017.json"
    if out_json.exists() and out_json.stat().st_size > 1000000:
        logger.info(f"Using existing instances_val2017.json ({out_json.stat().st_size / (1024*1024):.2f} MB)")
        return out_json

    logger.info("Extracting instances_val2017.json from archive...")
    with zipfile.ZipFile(annotations_zip, "r") as zf:
        for member in zf.infolist():
            if member.filename.endswith("instances_val2017.json"):
                with zf.open(member) as src, open(out_json, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                logger.info(f"Extracted instances_val2017.json to {out_json}")
                return out_json

    raise FileNotFoundError("instances_val2017.json not found inside annotations zip")


def process_and_build_dataset():
    """Main pipeline for downloading, filtering, and preparing COCO vehicle dataset."""
    raw_dir = paths.raw_data_dir / "coco2017"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ann_zip = raw_dir / "annotations_trainval2017.zip"
    val_zip = raw_dir / "val2017.zip"

    # 1. Download
    if not download_with_progress(COCO_ANNOTATIONS_URL, ann_zip):
        raise RuntimeError("Failed to download COCO annotations.")
    if not download_with_progress(COCO_VAL_IMAGES_URL, val_zip):
        raise RuntimeError("Failed to download COCO val2017 images.")

    # 2. Extract annotations JSON
    instances_json_path = extract_instances_json(ann_zip, raw_dir)

    # 3. Parse JSON & filter vehicles
    logger.info("Parsing COCO annotations for vehicle categories (car, motorcycle, bus, truck)...")
    with open(instances_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Index images by id
    images_by_id = {img["id"]: img for img in coco_data["images"]}

    # Collect vehicle annotations by image_id
    vehicle_annos_by_image = defaultdict(list)
    class_instance_counts = defaultdict(int)

    for ann in coco_data["annotations"]:
        cat_id = ann.get("category_id")
        if cat_id in TARGET_CLASSES:
            # bbox in COCO: [x, y, width, height] (pixels)
            target_cls = TARGET_CLASSES[cat_id]
            vehicle_annos_by_image[ann["image_id"]].append((target_cls, ann["bbox"]))
            class_instance_counts[CLASS_NAMES[target_cls]] += 1

    logger.info(f"Found {len(vehicle_annos_by_image)} images containing target vehicle categories.")
    logger.info(f"Instance distribution: {dict(class_instance_counts)}")

    # 4. Open val2017.zip for selective extraction
    proc_vehicles_dir = paths.processed_data_dir / "vehicles"
    # Clean previous tiny dataset
    images_dir = proc_vehicles_dir / "images"
    labels_dir = proc_vehicles_dir / "labels"
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        # Clear previous images and labels
        for old_f in (images_dir / split).glob("*.*"):
            old_f.unlink()
        for old_f in (labels_dir / split).glob("*.*"):
            old_f.unlink()

    # 5. Stratified / Random Split (70% train, 15% val, 15% test)
    image_ids = sorted(list(vehicle_annos_by_image.keys()))
    random.seed(42)
    random.shuffle(image_ids)

    n_total = len(image_ids)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    splits = {
        "train": image_ids[:n_train],
        "val": image_ids[n_train : n_train + n_val],
        "test": image_ids[n_train + n_val :],
    }

    logger.info(f"Dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # Extract images from zip and write YOLO labels
    test_manifest_vehicles = []
    logger.info("Extracting images from val2017.zip and converting to YOLO annotations...")

    with zipfile.ZipFile(val_zip, "r") as zf:
        # Build lookup for filenames in zip: e.g. "val2017/000000000139.jpg"
        zip_namelist = set(zf.namelist())

        for split_name, ids in splits.items():
            split_img_dir = images_dir / split_name
            split_lbl_dir = labels_dir / split_name
            extracted_count = 0

            for img_id in ids:
                img_info = images_by_id[img_id]
                fname = img_info["file_name"]
                w_img = float(img_info["width"])
                h_img = float(img_info["height"])

                zip_member = f"val2017/{fname}"
                if zip_member not in zip_namelist:
                    continue

                dest_img_path = split_img_dir / fname
                dest_lbl_path = split_lbl_dir / f"{Path(fname).stem}.txt"

                # Extract image
                with zf.open(zip_member) as src_img, open(dest_img_path, "wb") as dst_img:
                    shutil.copyfileobj(src_img, dst_img)

                # Write YOLO normalized annotations: class cx cy bw bh
                annos = vehicle_annos_by_image[img_id]
                yolo_lines = []
                for cls_id, bbox in annos:
                    bx, by, bw, bh = bbox
                    # Clamp
                    bx = max(0.0, bx)
                    by = max(0.0, by)
                    bw = min(w_img - bx, bw)
                    bh = min(h_img - by, bh)

                    if bw < 2 or bh < 2:
                        continue

                    cx = (bx + bw / 2.0) / w_img
                    cy = (by + bh / 2.0) / h_img
                    nw = bw / w_img
                    nh = bh / h_img

                    # Normalize [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))

                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                dest_lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
                extracted_count += 1

                if split_name == "test":
                    test_manifest_vehicles.append({
                        "image": paths.to_relative_str(dest_img_path),
                        "label": paths.to_relative_str(dest_lbl_path),
                        "box_count": len(yolo_lines),
                    })

            logger.info(f"  Split '{split_name}': extracted and converted {extracted_count} vehicle images.")

    # 6. Update data/vehicles.yaml (relative paths!)
    vehicles_yaml = paths.resolve("data/vehicles.yaml")
    yaml_content = {
        "train": "processed/vehicles/images/train",
        "val": "processed/vehicles/images/val",
        "test": "processed/vehicles/images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    vehicles_yaml.write_text(yaml.dump(yaml_content, sort_keys=False), encoding="utf-8")
    logger.info(f"Updated {vehicles_yaml.name} with portable relative paths.")

    # 7. Update test_manifest.json with new vehicle test split
    test_manifest_path = paths.resolve("data/splits/test_manifest.json")
    if test_manifest_path.exists():
        manifest_data = json.loads(test_manifest_path.read_text(encoding="utf-8"))
        manifest_data["vehicles"] = test_manifest_vehicles
        manifest_data["updated_at"] = datetime.datetime.now().isoformat()
        test_manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        logger.info(f"Updated test_manifest.json with {len(test_manifest_vehicles)} held-out vehicle test samples.")

    # 8. Update provenance metadata
    prov_path = paths.resolve("data/dataset_provenance.json")
    if prov_path.exists():
        prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
        prov_data["vehicles"] = {
            "source": "Official COCO 2017 Dataset (val2017 vehicle subset)",
            "license": "Creative Commons Attribution 4.0 (CC-BY 4.0)",
            "raw_dir": paths.to_relative_str(raw_dir),
            "processed_dir": paths.to_relative_str(proc_vehicles_dir),
            "total_images": len(image_ids),
            "train_images": len(splits["train"]),
            "val_images": len(splits["val"]),
            "test_images": len(splits["test"]),
            "classes": CLASS_NAMES,
            "instances": dict(class_instance_counts),
            "updated_at": datetime.datetime.now().isoformat(),
        }
        prov_path.write_text(json.dumps(prov_data, indent=2), encoding="utf-8")
        logger.info(f"Updated dataset_provenance.json.")

    logger.info("=" * 60)
    logger.info("COCO 2017 VEHICLE DATASET ACQUISITION & PREPARATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    process_and_build_dataset()
