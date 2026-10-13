"""Prepares parking bay image crops and occupancy dataset formatting."""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Prepare parking patch crops and dataset structure")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml", help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg_path = paths.resolve(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_cfg = cfg.get("local_capture", {})
    raw_dir = paths.resolve(local_cfg.get("raw_dir", "data/raw"))
    proc_dir = paths.resolve(local_cfg.get("processed_dir", "data/processed") + "/parking_slots")
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("PARKING OCCUPANCY DATASET PREPARATION")
    print("=" * 65)

    images = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
    if not images:
        print("[*] No raw frames found in data/raw to crop.")
        print("[*] Output directory data/processed/parking_slots is prepared.")
        print("=" * 65 + "\n")
        sys.exit(0)

    # Process available images using saved calibration if present
    calib_file = paths.resolve("data/calibration.json")
    if not calib_file.exists():
        print("[!] No calibration.json found to map slot polygon crops.")
        sys.exit(1)

    calib = json.loads(calib_file.read_text(encoding="utf-8"))
    slots = calib.get("slots", [])
    print(f"[+] Loaded {len(slots)} bay definitions from calibration.")

    cropped_total = 0
    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        for slot in slots:
            pts = slot.get("points", [])
            if len(pts) < 3:
                continue
            xs = [int(p[0] * w) for p in pts]
            ys = [int(p[1] * h) for p in pts]
            xmin, xmax = max(0, min(xs)), min(w, max(xs))
            ymin, ymax = max(0, min(ys)), min(h, max(ys))
            if xmax > xmin and ymax > ymin:
                crop = img[ymin:ymax, xmin:xmax]
                crop_name = f"{img_p.stem}_{slot['id']}.jpg"
                cv2.imwrite(str(proc_dir / crop_name), crop)
                cropped_total += 1

    print(f"[+] Successfully extracted {cropped_total} slot crops to {proc_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
