"""
Download and unpack standardized PKLot COCO spot annotations for PUCPR, UFPR04, and UFPR05.
Extracts:
- pucpr_spots.json
- ufpr04_spots.json
- ufpr05_spots.json
Audits exact counts of images, slots, polygons, and occupancy distributions across sunny/cloudy/rainy.
"""

import io
import json
import logging
from pathlib import Path
import ssl
import sys
import tarfile
import time
import urllib.request

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_pklot_coco")

SSL_CTX = ssl._create_unverified_context()
BASE_URL = "https://raw.githubusercontent.com/DSBD-Research/MetaPKLot-Dataset/main/annotations/original/spots/PKLot"

FILES = [
    "ufpr04_spots.tar.xz",
    "ufpr05_spots.tar.xz",
    "pucpr_spots.tar.xz"
]


def download_and_unpack():
    out_dir = paths.raw_data_dir / "pklot" / "annotations_metapklot"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_summary = {}

    for fname in FILES:
        stem = fname.replace(".tar.xz", "")
        json_out = out_dir / f"{stem}.json"
        
        if not json_out.exists():
            url = f"{BASE_URL}/{fname}"
            logger.info(f"Downloading {fname} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            t0 = time.time()
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=60) as resp:
                data = resp.read()
            logger.info(f"Downloaded {fname} ({len(data) / (1024*1024):.2f} MB) in {time.time()-t0:.1f}s.")

            logger.info(f"Unpacking {fname}...")
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
                tar.extractall(out_dir)
            logger.info(f"Unpacked to {out_dir}")

        # Audit unpacked JSON
        if json_out.exists():
            logger.info(f"Auditing {json_out.name} ({json_out.stat().st_size / (1024*1024):.2f} MB)...")
            with open(json_out, "r", encoding="utf-8") as f:
                coco = json.load(f)

            images = coco.get("images", [])
            annotations = coco.get("annotations", [])
            climates = coco.get("climates", {})

            # Count occupancy
            occ_counts = {"empty": 0, "occupied": 0}
            climate_counts = {}
            dates = set()

            for img in images:
                c_id = str(img.get("climate", "unknown"))
                climate_counts[c_id] = climate_counts.get(c_id, 0) + 1
                d = img.get("date")
                if d and isinstance(d, list) and len(d) >= 3:
                    dates.add(f"{d[0]:04d}-{d[1]:02d}-{d[2]:02d}")

            for ann in annotations:
                cat = ann.get("category_id")
                if cat == 0:
                    occ_counts["empty"] += 1
                elif cat == 1:
                    occ_counts["occupied"] += 1

            audit_summary[stem] = {
                "total_images": len(images),
                "total_spot_annotations": len(annotations),
                "unique_dates": len(dates),
                "occupancy": occ_counts,
                "climate_image_distribution": climate_counts,
                "json_path": paths.to_relative_str(json_out)
            }
            logger.info(f"{stem} Summary: Images={len(images)}, Annotations={len(annotations)}, Dates={len(dates)}, Occ={occ_counts}")

    report_path = paths.models_dir / "pklot_metapklot_annotations_audit.json"
    report_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    logger.info(f"Saved PKLot COCO annotations audit to {report_path}")
    return audit_summary


if __name__ == "__main__":
    download_and_unpack()
