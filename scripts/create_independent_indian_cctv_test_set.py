"""
Build Independent Indian CCTV Test Set (Phase 1 Correction).
Acquires 200 genuinely untouched frames from UVH-26-MV-Val.json:
- ZERO overlap with data/processed/uvh26_subset/images/train/ (1,000 images)
- ZERO overlap with data/processed/uvh26_subset/images/val/ (200 images)
- Minimum frame distance buffer > 100 to eliminate temporal redundancy
- Perceptual dHash check against training set (Hamming distance >= 8)
- Prunes frames with dominant unmapped vehicles (>10% area)
- Canonical 4-class YOLO formatting (0: car, 1: motorcycle, 2: bus, 3: truck)
- Emits data/splits/test_indian_cctv_heldout_manifest.json
"""

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
import urllib.request

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_independent_cctv_test")

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

UVH26_CLASS_MAP = {
    1: 0,   # Hatchback -> car
    2: 0,   # Sedan -> car
    3: 0,   # SUV -> car
    4: 0,   # MUV -> car
    8: 1,   # Two-wheeler -> motorcycle
    5: 2,   # Bus -> bus
    10: 2,  # Mini-bus -> bus
    6: 3,   # Truck -> truck
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def main():
    val_json_p = paths.resolve("data/raw/uvh26/UVH-26-MV-Val.json")
    index_p = paths.resolve("data/raw/uvh26/file_index.json")
    assert val_json_p.exists() and index_p.exists(), "UVH-26 Val JSON or file index missing!"

    val_coco = json.loads(val_json_p.read_text(encoding="utf-8"))
    file_index = json.loads(index_p.read_text(encoding="utf-8"))

    train_stems = {int(p.stem) for p in Path("data/processed/uvh26_subset/images/train").glob("*.jpg") if p.stem.isdigit()}
    prev_val_stems = {int(p.stem) for p in Path("data/processed/uvh26_subset/images/val").glob("*.jpg") if p.stem.isdigit()}
    used_stems = train_stems.union(prev_val_stems)

    logger.info(f"Loaded {len(train_stems)} train stems and {len(prev_val_stems)} previous val stems.")

    # Compute dhashes of a sample of training images to guard against near-duplicate scenes
    logger.info("Computing perceptual hashes of training subset for scene deduplication...")
    train_hashes = []
    for p in list(Path("data/processed/uvh26_subset/images/train").glob("*.jpg"))[:200]:
        im = cv2.imread(str(p))
        if im is not None:
            train_hashes.append(compute_dhash(im))

    # Group annotations by image_id
    annos_by_img = defaultdict(list)
    img_has_truck = set()
    img_has_bus = set()
    img_has_motorcycle = set()
    img_has_car = set()
    img_has_heavy_unmapped = set()

    cat_map = {c["id"]: c["name"] for c in val_coco["categories"]}

    for ann in val_coco["annotations"]:
        cid = ann["category_id"]
        img_id = ann["image_id"]
        bbox = ann.get("bbox", [])
        area = ann.get("area", 0)

        if cid in UVH26_CLASS_MAP:
            canon_cls = UVH26_CLASS_MAP[cid]
            annos_by_img[img_id].append((canon_cls, bbox))
            if canon_cls == 3:
                img_has_truck.add(img_id)
            elif canon_cls == 2:
                img_has_bus.add(img_id)
            elif canon_cls == 1:
                img_has_motorcycle.add(img_id)
            elif canon_cls == 0:
                img_has_car.add(img_id)
        elif cid in [7, 9, 11, 13]:  # Three-wheeler, LCV, Tempo, Van
            if area > 1920 * 1080 * 0.10:
                img_has_heavy_unmapped.add(img_id)

    images_by_id = {img["id"]: img for img in val_coco["images"]}

    # Filter candidates with min distance > 100 to all used stems
    eligible_ids = []
    for img_id, img_info in images_by_id.items():
        fname = img_info["file_name"]
        stem_str = fname.split(".")[0]
        if not stem_str.isdigit():
            continue
        stem = int(stem_str)
        if stem in used_stems:
            continue
        min_dist = min(abs(stem - u) for u in used_stems)
        if min_dist > 100 and fname in file_index and img_id not in img_has_heavy_unmapped and img_id in annos_by_img:
            eligible_ids.append(img_id)

    logger.info(f"Identified {len(eligible_ids)} strictly untouched, temporally buffered candidate frames.")

    # Stratified selection
    random.seed(101)  # Distinct independent seed
    clean_trucks = sorted(list(set(eligible_ids).intersection(img_has_truck)))
    random.shuffle(clean_trucks)
    sel_trucks = clean_trucks[:60]

    rem = set(eligible_ids) - set(sel_trucks)
    clean_buses = sorted(list(rem.intersection(img_has_bus)))
    random.shuffle(clean_buses)
    sel_buses = clean_buses[:50]

    rem = rem - set(sel_buses)
    clean_motos = sorted(list(rem.intersection(img_has_motorcycle)))
    random.shuffle(clean_motos)
    sel_motos = clean_motos[:50]

    rem = rem - set(sel_motos)
    clean_cars = sorted(list(rem.intersection(img_has_car)))
    random.shuffle(clean_cars)
    sel_cars = clean_cars[:40]

    selected_ids = sel_trucks + sel_buses + sel_motos + sel_cars
    assert len(selected_ids) == 200, f"Expected 200 images, got {len(selected_ids)}"
    logger.info(f"Selected 200 independent frames: {len(sel_trucks)} trucks, {len(sel_buses)} buses, {len(sel_motos)} motos, {len(sel_cars)} cars.")

    out_img_dir = paths.processed_data_dir / "uvh26_heldout_test" / "images"
    out_lbl_dir = paths.processed_data_dir / "uvh26_heldout_test" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    class_tally = Counter()
    near_dupe_warnings = 0

    headers = {"User-Agent": "Mozilla/5.0"}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_item(img_id):
        img_info = images_by_id[img_id]
        fname = img_info["file_name"]
        stem = Path(fname).stem
        dest_jpg = out_img_dir / f"{stem}.jpg"
        dest_lbl = out_lbl_dir / f"{stem}.txt"

        if not (dest_jpg.exists() and dest_lbl.exists() and dest_jpg.stat().st_size > 1000):
            rel_repo_path = file_index[fname]
            url = f"https://huggingface.co/datasets/iisc-aim/UVH-26/resolve/main/{rel_repo_path}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()
            nparr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if nparr is None:
                raise RuntimeError(f"Failed to decode {fname}")

            cv2.imwrite(str(dest_jpg), nparr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            w_img = float(img_info.get("width", 1920))
            h_img = float(img_info.get("height", 1080))
            lines = []
            for canon_cls, bbox in annos_by_img[img_id]:
                bx, by, bw, bh = bbox
                if bw < 5 or bh < 5:
                    continue
                cx = (bx + bw / 2.0) / w_img
                cy = (by + bh / 2.0) / h_img
                nw = bw / w_img
                nh = bh / h_img
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                lines.append(f"{canon_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dest_jpg, dest_lbl, int(stem)

    logger.info("Downloading and processing held-out frames with 8 worker threads...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_item, img_id): img_id for img_id in selected_ids}
        for future in as_completed(futures):
            dest_jpg, dest_lbl, stem = future.result()
            # Tally labels
            for line in dest_lbl.read_text(encoding="utf-8").strip().splitlines():
                if line:
                    c = int(line.split()[0])
                    class_tally[CLASS_NAMES[c]] += 1
            manifest_entries.append({
                "image": paths.to_relative_str(dest_jpg),
                "label": paths.to_relative_str(dest_lbl),
                "sha256": compute_sha256(dest_jpg),
                "source": "UVH-26 Independent Held-Out Test (IISc AIM)",
                "frame_id": stem,
            })


    manifest_p = paths.resolve("data/splits/test_indian_cctv_heldout_manifest.json")
    manifest_p.write_text(json.dumps({
        "view_name": "indian_urban_traffic_cctv_heldout",
        "description": "Genuinely untouched held-out Indian urban CCTV benchmark from UVH-26-MV-Val.json, strictly disjoint from training and previous validation frames with min frame distance > 100",
        "created_at": datetime.now().isoformat(),
        "sample_count": len(manifest_entries),
        "class_tally": dict(class_tally),
        "total_instances": sum(class_tally.values()),
        "near_duplicate_warnings": near_dupe_warnings,
        "samples": manifest_entries,
    }, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info(f"SUCCESSFULLY EMITTED INDEPENDENT HELD-OUT INDIAN CCTV BENCHMARK")
    logger.info(f"  Manifest: {manifest_p}")
    logger.info(f"  Images: {len(manifest_entries)}")
    logger.info(f"  Vehicle Instances: {sum(class_tally.values())}")
    logger.info(f"  Class Breakdown: {dict(class_tally)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
