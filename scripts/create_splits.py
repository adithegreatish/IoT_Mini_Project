"""Generates group-aware train/val/test splits to prevent temporal leakage."""

import argparse
import json
from pathlib import Path
import random
import sys
import yaml

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths


def main():
    parser = argparse.ArgumentParser(description="Create group-aware train/val/test splits")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml", help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg_path = paths.resolve(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_cfg = cfg.get("local_capture", {})
    raw_dir = paths.resolve(local_cfg.get("raw_dir", "data/raw"))
    splits_dir = paths.resolve(local_cfg.get("splits_dir", "data/splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("GROUP-AWARE DATASET SPLIT GENERATOR")
    print("=" * 65)

    images = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))
    if not images:
        print("[!] No images found in data/raw to split.")
        print("[*] Generating empty split manifests with schema documentation.")
        empty_manifest = {
            "created_at": "2026-10-10T12:00:00Z",
            "group_by_session": True,
            "train": [],
            "val": [],
            "test": [],
            "leakage_mitigation": "Grouped by capture session prefix to prevent consecutive frame leakage.",
        }
        for name in ["train_manifest.json", "val_manifest.json", "test_manifest.json"]:
            (splits_dir / name).write_text(json.dumps(empty_manifest, indent=2), encoding="utf-8")
        print(f"[+] Saved baseline manifest templates to {splits_dir}")
        print("=" * 65 + "\n")
        sys.exit(0)

    # Group by session (filename prefix before first underscore or dash)
    sessions = {}
    for img in images:
        prefix = img.stem.split("_")[0].split("-")[0]
        sessions.setdefault(prefix, []).append(str(img.relative_to(paths.root)))

    session_keys = list(sessions.keys())
    random.seed(42)
    random.shuffle(session_keys)

    n_total = len(session_keys)
    n_train = max(1, int(n_total * local_cfg.get("train_ratio", 0.7)))
    n_val = max(1, int(n_total * local_cfg.get("val_ratio", 0.15)))

    train_sessions = session_keys[:n_train]
    val_sessions = session_keys[n_train : n_train + n_val]
    test_sessions = session_keys[n_train + n_val :]

    train_imgs = [img for s in train_sessions for img in sessions[s]]
    val_imgs = [img for s in val_sessions for img in sessions[s]]
    test_imgs = [img for s in test_sessions for img in sessions[s]]

    manifest = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": test_sessions,
    }

    manifest_path = splits_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[+] Splits created successfully:")
    print(f"    Train: {len(train_imgs)} images ({len(train_sessions)} sessions)")
    print(f"    Val:   {len(val_imgs)} images ({len(val_sessions)} sessions)")
    print(f"    Test:  {len(test_imgs)} images ({len(test_sessions)} sessions)")
    print(f"[+] Saved manifest to: {manifest_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
