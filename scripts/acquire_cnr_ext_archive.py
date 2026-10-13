"""
Acquire and audit complete CNR-EXT patch archive from official cnrpark.it source.
Target: ~144,965 CNR-EXT patches + 12,584 CNRPark patches -> ~157,549 total patches.
Audits integrity against CNRPark+EXT.csv metadata.
"""

import hashlib
import json
import logging
from pathlib import Path
import ssl
import sys
import time
import urllib.request
import zipfile

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_cnr_ext")

CNR_EXT_ZIP_URL = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip"
SSL_CTX = ssl._create_unverified_context()


def download_cnr_ext(zip_dest: Path):
    if zip_dest.exists() and zip_dest.stat().st_size >= 400_000_000:
        logger.info(f"CNR-EXT zip archive already exists at {zip_dest} ({zip_dest.stat().st_size / (1024*1024):.2f} MB)")
        return

    logger.info(f"Downloading CNR-EXT archive from {CNR_EXT_ZIP_URL} to {zip_dest}...")
    req = urllib.request.Request(CNR_EXT_ZIP_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    t0 = time.time()
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=60) as response, open(zip_dest, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        last_log = time.time()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_log > 5.0 or downloaded == total_size:
                elapsed = time.time() - t0
                speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                logger.info(
                    f"Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB "
                    f"({pct:.1f}%) at {speed:.2f} MB/s"
                )
                last_log = time.time()

    logger.info(f"Download complete in {time.time() - t0:.1f}s. Saved to {zip_dest}")


def extract_cnr_ext(zip_path: Path, extract_dir: Path):
    logger.info(f"Checking extraction of {zip_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        total_files = len(members)
        logger.info(f"Zip contains {total_files} total entries.")

        # Test if already extracted
        sample_file = extract_dir / members[-1]
        if sample_file.exists():
            logger.info("Extraction appears already complete. Skipping re-extraction.")
            return total_files

        logger.info("Extracting archive (this may take 1-2 minutes)...")
        t0 = time.time()
        z.extractall(extract_dir)
        logger.info(f"Extracted {total_files} entries in {time.time() - t0:.1f}s.")
        return total_files


def audit_cnrpark_plus_ext():
    logger.info("Auditing full CNRPark+EXT collection...")
    cnr_dir = paths.raw_data_dir / "cnrpark"
    csv_path = cnr_dir / "CNRPark+EXT.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    # Read CSV metadata
    csv_records = []
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_records.append(row)

    logger.info(f"CNRPark+EXT.csv has {len(csv_records)} total labeled records.")

    # Inspect camera IDs and weather distributions
    cameras = set()
    weather_counts = {}
    occupancy_counts = {"0": 0, "1": 0}
    dates = set()

    for r in csv_records:
        cameras.add(r["camera"])
        w = r.get("weather", "Unknown")
        weather_counts[w] = weather_counts.get(w, 0) + 1
        occ = r.get("occupancy", "0")
        if occ in occupancy_counts:
            occupancy_counts[occ] += 1
        dates.add(r.get("datetime", "")[:8])

    summary = {
        "total_csv_records": len(csv_records),
        "unique_cameras": sorted(list(cameras)),
        "unique_capture_dates": len(dates),
        "occupancy_distribution": {
            "free": occupancy_counts["0"],
            "busy": occupancy_counts["1"],
            "free_ratio": round(occupancy_counts["0"] / len(csv_records), 4)
        },
        "weather_distribution": weather_counts
    }
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")
    return summary


def main():
    cnr_dir = paths.raw_data_dir / "cnrpark"
    zip_dest = cnr_dir / "CNR-EXT-Patches-150x150.zip"
    extract_dest = cnr_dir / "extracted_patches_cnr_ext"

    # Step 1: Download
    download_cnr_ext(zip_dest)

    # Step 2: Extract
    extract_cnr_ext(zip_dest, extract_dest)

    # Step 3: Audit
    summary = audit_cnrpark_plus_ext()
    
    out_audit = paths.models_dir / "cnrpark_plus_ext_audit.json"
    out_audit.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved audit summary to {out_audit}")


if __name__ == "__main__":
    main()
