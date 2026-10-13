"""Prepares license plate crop dataset and validates resolution constraints."""

import argparse
from pathlib import Path
import sys
import cv2
import yaml

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths


def main():
    parser = argparse.ArgumentParser(description="Prepare license plate crops and validate minimum dimensions")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml", help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg_path = paths.resolve(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_cfg = cfg.get("local_capture", {})
    raw_dir = paths.resolve(local_cfg.get("raw_dir", "data/raw"))
    proc_dir = paths.resolve(local_cfg.get("processed_dir", "data/processed") + "/plates")
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("PLATE CROP DATASET PREPARATION & RESOLUTION AUDIT")
    print("=" * 65)

    images = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
    if not images:
        print("[*] No raw frames found in data/raw to extract plate crops.")
        print("[*] Output directory data/processed/plates is ready.")
        print("=" * 65 + "\n")
        sys.exit(0)

    print(f"[+] Found {len(images)} source images. Checking crops...")
    # Minimum required crop resolution for reliable OCR / HSRP analysis
    min_w = 60
    min_h = 20

    valid_crops = 0
    subscale_crops = 0

    crops = list(proc_dir.glob("*.jpg")) + list(proc_dir.glob("*.png"))
    for c in crops:
        img = cv2.imread(str(c))
        if img is None:
            continue
        h, w = img.shape[:2]
        if w >= min_w and h >= min_h:
            valid_crops += 1
        else:
            subscale_crops += 1

    print(f"[+] Audit of existing crops in {proc_dir}:")
    print(f"    Sufficient resolution (>= {min_w}x{min_h}): {valid_crops}")
    print(f"    Sub-scale / too small (< {min_w}x{min_h}):    {subscale_crops}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
