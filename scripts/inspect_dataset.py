"""Inspects candidate and local dataset directories, reporting file counts and verification status."""

import argparse
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import yaml
from app.infrastructure.paths import paths


def main():
    parser = argparse.ArgumentParser(description="Inspect candidate datasets and local capture directories")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml", help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg_path = paths.resolve(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Dataset configuration file not found at: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 70)
    print("DATASET INVENTORY & LOCAL CAPTURE AUDIT")
    print("=" * 70)

    # 1. External Candidates
    candidates = cfg.get("datasets", {}).get("candidate_sources", {})
    print(f"\n[+] Registered Candidate Sources: {len(candidates)}")
    print("-" * 70)
    for key, c in candidates.items():
        v_status = "VERIFIED" if c.get("verified", False) else "UNVERIFIED CANDIDATE"
        print(f"  • {c.get('name', key):<30} [{v_status}]")
        print(f"    Source URL: {c.get('url')}")
        print(f"    Purpose:    {c.get('purpose')}")
        print(f"    Format:     {c.get('format')}")

    # 2. Local Directories Inspection
    local_cfg = cfg.get("local_capture", {})
    dirs_to_check = {
        "Raw Frames": paths.resolve(local_cfg.get("raw_dir", "data/raw")),
        "Processed Crops": paths.resolve(local_cfg.get("processed_dir", "data/processed")),
        "Annotations": paths.resolve(local_cfg.get("annotations_dir", "data/annotations")),
        "Split Manifests": paths.resolve(local_cfg.get("splits_dir", "data/splits")),
    }

    print("\n[+] Local Storage Inventory (Truthful Audit)")
    print("-" * 70)
    for label, p in dirs_to_check.items():
        if not p.exists():
            print(f"  • {label:<20} [DIR MISSING]  -> {p}")
            continue

        images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
        annots = list(p.glob("*.xml")) + list(p.glob("*.txt")) + list(p.glob("*.json"))
        print(f"  • {label:<20} Images: {len(images):<5} Annotations/Manifests: {len(annots):<5} ({p})")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
