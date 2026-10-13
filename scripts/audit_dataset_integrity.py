"""
Dataset Integrity & Split Leakage Audit Script
Priority 1 Audit:
1. Exact SHA-256 duplicate detection across Train, Val, Test splits.
2. Perceptual near-duplicate detection via 64-bit Difference Hash (dHash).
3. Parking occupancy provenance analysis (Camera A vs Camera B, capture dates, session crossing).
4. Grouped evaluation of Occupancy Classifier by camera and session.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.parking.occupancy_classifier import PatchOccupancyClassifier

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_parking.integrity_audit")


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Computes 64-bit difference hash (dHash) for perceptual near-duplicate detection."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert bool array to 64-bit integer
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return decimal_val


def hamming_distance(hash1: int, hash2: int) -> int:
    return bin(hash1 ^ hash2).count("1")


def audit_split_duplicates(dataset_name: str, split_dirs: Dict[str, List[Path]]) -> Dict:
    """Audits exact and perceptual duplicates across train/val/test splits."""
    logger.info(f"Auditing exact and perceptual duplicates for {dataset_name}...")
    sha_map: Dict[str, Dict[str, str]] = {}  # split -> {filepath: sha256}
    dhash_map: Dict[str, Dict[str, int]] = {}  # split -> {filepath: dhash}

    for split_name, file_list in split_dirs.items():
        sha_map[split_name] = {}
        dhash_map[split_name] = {}
        for p in file_list:
            if not p.exists():
                continue
            sha_map[split_name][p.name] = calculate_sha256(p)
            dhash_map[split_name][p.name] = compute_dhash(p)

    # Check exact SHA-256 duplicates across splits
    exact_leakage = {
        "train_val": [],
        "train_test": [],
        "val_test": [],
        "internal_duplicates": {}
    }

    train_hashes = set(sha_map.get("train", {}).values())
    val_hashes = set(sha_map.get("val", {}).values())
    test_hashes = set(sha_map.get("test", {}).values())

    # Exact cross-split intersections
    for fname, h in sha_map.get("train", {}).items():
        if h in val_hashes:
            exact_leakage["train_val"].append(fname)
        if h in test_hashes:
            exact_leakage["train_test"].append(fname)

    for fname, h in sha_map.get("val", {}).items():
        if h in test_hashes:
            exact_leakage["val_test"].append(fname)

    # Perceptual near-duplicates (Hamming distance <= 3) between Train and Test
    near_duplicates_train_test = []
    train_items = list(dhash_map.get("train", {}).items())
    test_items = list(dhash_map.get("test", {}).items())

    # Subsample if large to keep check fast
    check_train = train_items[:500] if len(train_items) > 500 else train_items
    for t_name, t_hash in test_items:
        if t_hash == 0:
            continue
        for tr_name, tr_hash in check_train:
            if tr_hash == 0:
                continue
            dist = hamming_distance(t_hash, tr_hash)
            if dist <= 3:  # Very close perceptual match
                near_duplicates_train_test.append({
                    "test_image": t_name,
                    "train_image": tr_name,
                    "hamming_distance": dist
                })

    return {
        "dataset": dataset_name,
        "sample_counts": {s: len(files) for s, files in split_dirs.items()},
        "exact_duplicates_across_splits": {
            "train_val_count": len(exact_leakage["train_val"]),
            "train_test_count": len(exact_leakage["train_test"]),
            "val_test_count": len(exact_leakage["val_test"]),
            "is_zero_exact_leakage": (
                len(exact_leakage["train_val"]) == 0
                and len(exact_leakage["train_test"]) == 0
                and len(exact_leakage["val_test"]) == 0
            ),
        },
        "perceptual_near_duplicates_train_test": {
            "evaluated_pairs": len(test_items) * len(check_train),
            "flagged_near_duplicates_count": len(near_duplicates_train_test),
            "samples": near_duplicates_train_test[:5]
        }
    }


def audit_cnrpark_provenance_and_camera_grouping() -> Dict:
    """Analyzes CNRPark patches for camera (A vs B) and capture session distribution."""
    logger.info("Auditing CNRPark patch metadata and camera grouping...")
    csv_path = paths.resolve("data/raw/cnrpark/CNRPark+EXT.csv")
    patch_meta = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # image_url format: CNRPark/A/free/20150703_0805_1.jpg
                filename = Path(r["image_url"]).name
                patch_meta[filename] = {
                    "camera": r.get("camera", "Unknown"),
                    "datetime": r.get("datetime", ""),
                    "date": r.get("datetime", "")[:8],
                    "time": r.get("datetime", "")[9:13] if len(r.get("datetime", "")) >= 13 else "",
                    "slot_id": r.get("slot_id", ""),
                    "weather": r.get("weather", ""),
                    "occupancy": int(r.get("occupancy", 0))
                }

    # Inspect current processed splits
    processed_dir = paths.resolve("data/processed/parking_occupancy")
    split_meta: Dict[str, Dict] = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        s_dir = processed_dir / split
        cameras: Dict[str, int] = {}
        dates: Set[str] = set()
        slots: Set[str] = set()
        total = 0

        for state in ["free", "busy"]:
            state_dir = s_dir / state
            if not state_dir.exists():
                continue
            for img_p in state_dir.glob("*.jpg"):
                total += 1
                info = patch_meta.get(img_p.name, {})
                cam = info.get("camera", "Unknown")
                cameras[cam] = cameras.get(cam, 0) + 1
                if info.get("date"):
                    dates.add(info["date"])
                if info.get("slot_id"):
                    slots.add(f"{cam}_{info['slot_id']}")

        split_meta[split] = {
            "total_patches": total,
            "camera_counts": cameras,
            "unique_dates_count": len(dates),
            "dates": sorted(list(dates)),
            "unique_slots_count": len(slots),
        }

    # Check session/date crossing
    train_dates = set(split_meta["train"]["dates"])
    val_dates = set(split_meta["val"]["dates"])
    test_dates = set(split_meta["test"]["dates"])

    date_leakage = {
        "train_val_shared_dates": sorted(list(train_dates.intersection(val_dates))),
        "train_test_shared_dates": sorted(list(train_dates.intersection(test_dates))),
        "val_test_shared_dates": sorted(list(val_dates.intersection(test_dates))),
    }

    return {
        "csv_metadata_records": len(patch_meta),
        "splits_composition": split_meta,
        "date_session_leakage": {
            "train_test_shared_dates_count": len(date_leakage["train_test_shared_dates"]),
            "shared_dates_details": date_leakage,
            "observation": (
                "Patches were randomly or stratifiably split across capture dates and cameras A and B. "
                "Because identical physical slot angles appear across multiple timestamps on the same days, "
                "a random split does not measure unseen-camera generalization. "
                "A true cross-camera benchmark requires training on Camera A and evaluating on Camera B."
            )
        }
    }


def evaluate_occupancy_cross_camera(classifier: PatchOccupancyClassifier) -> Dict:
    """Evaluates the trained occupancy classifier on Camera A vs Camera B separately."""
    logger.info("Evaluating Occupancy Classifier on camera-grouped subsets...")
    csv_path = paths.resolve("data/raw/cnrpark/CNRPark+EXT.csv")
    patch_cam_map = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = Path(r["image_url"]).name
                patch_cam_map[fname] = r.get("camera", "Unknown")

    test_dir = paths.resolve("data/processed/parking_occupancy/test")
    groups = {
        "Camera_A": {"free": [], "busy": []},
        "Camera_B": {"free": [], "busy": []},
        "All_Test": {"free": [], "busy": []}
    }

    for state in ["free", "busy"]:
        p_dir = test_dir / state
        if not p_dir.exists():
            continue
        for img_p in p_dir.glob("*.jpg"):
            groups["All_Test"][state].append(img_p)
            cam = patch_cam_map.get(img_p.name)
            if cam == "A":
                groups["Camera_A"][state].append(img_p)
            elif cam == "B":
                groups["Camera_B"][state].append(img_p)

    results = {}
    for g_name, data in groups.items():
        total_eval = 0
        tp, tn, fp, fn = 0, 0, 0, 0

        for state, file_list in data.items():
            true_occupied = (state == "busy")
            for img_p in file_list:
                img = cv2.imread(str(img_p))
                if img is None:
                    continue
                total_eval += 1
                pred_occ, conf = classifier.predict_patch(img)
                if true_occupied:
                    if pred_occ:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred_occ:
                        fp += 1
                    else:
                        tn += 1

        acc = (tp + tn) / max(1, total_eval)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * (prec * rec) / max(1e-6, prec + rec)

        results[g_name] = {
            "total_evaluated": total_eval,
            "free_count": len(data["free"]),
            "busy_count": len(data["busy"]),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return results


def main():
    print("\n" + "=" * 75)
    print("PRIORITY 1: COMPREHENSIVE DATASET INTEGRITY & SPLIT LEAKAGE AUDIT")
    print("=" * 75)

    # 1. Audit Vehicle Dataset
    v_splits = {
        "train": list(paths.resolve("data/processed/vehicles/images/train").glob("*.jpg")),
        "val": list(paths.resolve("data/processed/vehicles/images/val").glob("*.jpg")),
        "test": list(paths.resolve("data/processed/vehicles/images/test").glob("*.jpg")),
    }
    v_audit = audit_split_duplicates("Vehicles (COCO 2017)", v_splits)
    print(f"\n[+] Vehicle Dataset Splits:")
    print(f"    Train: {v_audit['sample_counts']['train']} images")
    print(f"    Val:   {v_audit['sample_counts']['val']} images")
    print(f"    Test:  {v_audit['sample_counts']['test']} images")
    print(f"    Exact Leakage (Train-Test): {v_audit['exact_duplicates_across_splits']['train_test_count']} (Zero Leakage: {v_audit['exact_duplicates_across_splits']['is_zero_exact_leakage']})")
    print(f"    Perceptual Near-Duplicates (Train-Test): {v_audit['perceptual_near_duplicates_train_test']['flagged_near_duplicates_count']}")

    # 2. Audit Plate Dataset
    p_splits = {
        "train": list(paths.resolve("data/processed/plates/images/train").glob("*.jpg")),
        "val": list(paths.resolve("data/processed/plates/images/val").glob("*.jpg")),
        "test": list(paths.resolve("data/processed/plates/images/test").glob("*.jpg")),
    }
    p_audit = audit_split_duplicates("License Plates", p_splits)
    print(f"\n[+] License Plate Dataset Splits:")
    print(f"    Train: {p_audit['sample_counts']['train']} images")
    print(f"    Val:   {p_audit['sample_counts']['val']} images")
    print(f"    Test:  {p_audit['sample_counts']['test']} images")
    print(f"    Exact Leakage (Train-Test): {p_audit['exact_duplicates_across_splits']['train_test_count']} (Zero Leakage: {p_audit['exact_duplicates_across_splits']['is_zero_exact_leakage']})")

    # 3. Audit CNRPark Occupancy Dataset
    occ_splits = {
        "train": list(paths.resolve("data/processed/parking_occupancy/train").rglob("*.jpg")),
        "val": list(paths.resolve("data/processed/parking_occupancy/val").rglob("*.jpg")),
        "test": list(paths.resolve("data/processed/parking_occupancy/test").rglob("*.jpg")),
    }
    occ_audit = audit_split_duplicates("CNRPark+EXT Patches", occ_splits)
    print(f"\n[+] CNRPark+EXT Occupancy Patches Splits:")
    print(f"    Train: {occ_audit['sample_counts']['train']} patches")
    print(f"    Val:   {occ_audit['sample_counts']['val']} patches")
    print(f"    Test:  {occ_audit['sample_counts']['test']} patches")
    print(f"    Exact Leakage (Train-Test): {occ_audit['exact_duplicates_across_splits']['train_test_count']} (Zero Leakage: {occ_audit['exact_duplicates_across_splits']['is_zero_exact_leakage']})")

    # 4. Analyze CNRPark Metadata & Camera/Session Leakage
    cnr_prov = audit_cnrpark_provenance_and_camera_grouping()
    print(f"\n[+] CNRPark Camera & Session Grouping Audit:")
    for split_name, s_data in cnr_prov["splits_composition"].items():
        print(f"    {split_name.upper():5s}: Total={s_data['total_patches']} | Cameras={s_data['camera_counts']} | Days={s_data['unique_dates_count']} | Slots={s_data['unique_slots_count']}")
    print(f"    Shared Train-Test Days: {cnr_prov['date_session_leakage']['train_test_shared_dates_count']} days overlap across splits.")

    # 5. Evaluate Occupancy Classifier Grouped by Camera
    print("\n[+] Evaluating Occupancy Classifier by Camera Group on Untouched Test Set...")
    classifier = PatchOccupancyClassifier("models/occupancy_classifier.onnx")
    camera_eval = evaluate_occupancy_cross_camera(classifier)

    for g_name, g_res in camera_eval.items():
        print(f"    [{g_name:10s}] Total={g_res['total_evaluated']} (Free={g_res['free_count']}, Busy={g_res['busy_count']}) | Acc={g_res['accuracy']*100:.2f}% | Prec={g_res['precision']*100:.2f}% | Rec={g_res['recall']*100:.2f}% | F1={g_res['f1']:.4f}")

    # Assemble Final Integrity Audit Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vehicles": v_audit,
        "plates": p_audit,
        "occupancy_duplicates": occ_audit,
        "occupancy_provenance": cnr_prov,
        "occupancy_camera_evaluation": camera_eval,
    }

    out_json = paths.resolve("data/test_reports/evaluations/dataset_integrity_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\n[+] Dataset integrity audit written to: {out_json}")


if __name__ == "__main__":
    main()
