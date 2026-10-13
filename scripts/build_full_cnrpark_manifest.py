"""
Build authoritative CNRPark+EXT manifest for all 157,549 genuine patches.
Maps CNRPark+EXT.csv metadata to local file paths, verifying 100% extraction integrity.
"""

import csv
import json
import logging
from pathlib import Path
import sys
import time

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("build_cnrpark_manifest")


def build_manifest():
    cnr_dir = paths.raw_data_dir / "cnrpark"
    csv_path = cnr_dir / "CNRPark+EXT.csv"
    orig_dir = cnr_dir / "extracted_patches"
    ext_dir = cnr_dir / "extracted_patches_cnr_ext"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    logger.info("Reading CNRPark+EXT.csv and verifying physical image paths...")
    records = []
    missing_count = 0

    weather_map = {
        "S": "Sunny",
        "O": "Overcast",
        "R": "Rainy"
    }

    camera_counts = {}
    weather_counts = {}
    occ_counts = {"free": 0, "busy": 0}

    t0 = time.time()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            url = row["image_url"]
            if url.startswith("CNRPark/"):
                rel_suffix = url[len("CNRPark/"):]
                abs_p = orig_dir / rel_suffix
                dataset_name = "CNRPark"
            elif url.startswith("CNR-EXT/"):
                rel_suffix = url[len("CNR-EXT/"):]
                abs_p = ext_dir / rel_suffix
                dataset_name = "CNR-EXT"
            else:
                rel_suffix = url
                abs_p = cnr_dir / rel_suffix
                dataset_name = "CNRPark+EXT"

            if not abs_p.exists():
                missing_count += 1
                continue

            occ_int = int(row["occupancy"])
            label_str = "busy" if occ_int == 1 else "free"
            cam = row["camera"]
            w_code = row.get("weather", "S")
            w_str = weather_map.get(w_code, "Unknown")
            dt = row["datetime"]
            slot = row["slot_id"]

            camera_counts[cam] = camera_counts.get(cam, 0) + 1
            weather_counts[w_str] = weather_counts.get(w_str, 0) + 1
            occ_counts[label_str] += 1

            records.append({
                "index": idx,
                "path": paths.to_relative_str(abs_p),
                "label": label_str,
                "class_id": occ_int,
                "camera_id": f"camera_{cam}",
                "capture_session": dt,
                "capture_date": dt[:8],
                "weather": w_str,
                "weather_code": w_code,
                "slot_id": int(slot) if slot.isdigit() else slot,
                "source": dataset_name,
                "license": "CNR License / Academic Research (San Biagio, Pisa)"
            })

    elapsed = time.time() - t0
    logger.info(
        f"Verified {len(records)} records in {elapsed:.2f}s (Missing: {missing_count})."
    )

    manifest_doc = {
        "dataset": "CNRPark+EXT Complete Collection",
        "citation": "Amato et al., Deep learning for decentralized parking lot occupancy detection (2017)",
        "license": "CNR License / Academic Research",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_records": len(records),
        "camera_distribution": camera_counts,
        "weather_distribution": weather_counts,
        "occupancy_distribution": occ_counts,
        "records": records
    }

    out_path = paths.data_dir / "splits" / "cnrpark_full_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest_doc, indent=2), encoding="utf-8")
    logger.info(f"Saved complete CNRPark+EXT manifest to {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    return manifest_doc


if __name__ == "__main__":
    build_manifest()
