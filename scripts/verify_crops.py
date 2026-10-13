"""
Inspect and verify ground truth text labels for Indian license plates
"""
import sys
from pathlib import Path
import json
import cv2

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

def inspect_crops():
    crop_dir = paths.resolve("data/processed/plates/ground_truth_crops")
    files = sorted(list(crop_dir.glob("*.jpg")))
    print(f"Total crops: {len(files)}")
    for f in files[:15]:
        img = cv2.imread(str(f))
        print(f"{f.name}: shape={img.shape}")

if __name__ == "__main__":
    inspect_crops()
