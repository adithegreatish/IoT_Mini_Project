"""
Verification and creation of reproducible ground-truth annotations for Indian license plates.
Only plates with clearly legible, unambiguous ground-truth registration characters are included.
Plates that are occluded, corrupted, or unreadable are marked with ground_truth_text=null
to allow honest evaluation of abstention rates.
"""

import json
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

GROUND_TRUTH_DATA = [
    {
        "image": "data/raw/plates/indian/images/image_0001.jpg",
        "ground_truth_plate": "AP 29 AN 0074",
        "state": "AP",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0003.jpg",
        "ground_truth_plate": "JH 05 AW 2117",
        "state": "JH",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0005.jpg",
        "ground_truth_plate": "TN 23 L 4547",
        "state": "TN",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0006.jpg",
        "ground_truth_plate": "TN 18 D 1866",
        "state": "TN",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0010.jpg",
        "ground_truth_plate": "MH 20 BC 5122",
        "state": "MH",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0012.jpg",
        "ground_truth_plate": "KA 08 M 1674",
        "state": "KA",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0013.jpg",
        "ground_truth_plate": "UP 38 P 2859",
        "state": "UP",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0014.jpg",
        "ground_truth_plate": "MP 07 GA 1533",
        "state": "MP",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0018.jpg",
        "ground_truth_plate": "MP 07 HB 8260",
        "state": "MP",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0020.jpg",
        "ground_truth_plate": "AP 28 BW 9994",
        "state": "AP",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0021.jpg",
        "ground_truth_plate": "RJ 22 BR 9769",
        "state": "RJ",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0025.jpg",
        "ground_truth_plate": "BR 33 P 4980",
        "state": "BR",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0028.jpg",
        "ground_truth_plate": "KL 35 H 5834",
        "state": "KL",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0031.jpg",
        "ground_truth_plate": "KL 49 8262",
        "state": "KL",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0032.jpg",
        "ground_truth_plate": "HR 22 AX 7116",
        "state": "HR",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0037.jpg",
        "ground_truth_plate": "TN 52 D 5353",
        "state": "TN",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0038.jpg",
        "ground_truth_plate": "KL 07 BX 7197",
        "state": "KL",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0040.jpg",
        "ground_truth_plate": "KL 10 AG 7249",
        "state": "KL",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0041.jpg",
        "ground_truth_plate": "TN 58 AP 5280",
        "state": "TN",
        "is_two_line": True,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0044.jpg",
        "ground_truth_plate": "MP 15 GA 9462",
        "state": "MP",
        "is_two_line": False,
        "is_legible": True
    },
    {
        "image": "data/raw/plates/indian/images/image_0046.jpg",
        "ground_truth_plate": "MH 14 BA 6092",
        "state": "MH",
        "is_two_line": True,
        "is_legible": True
    },
    # Unreadable / Occluded samples for honest abstention benchmarking
    {
        "image": "data/raw/plates/indian/images/image_0002.jpg",
        "ground_truth_plate": None,
        "state": None,
        "is_two_line": False,
        "is_legible": False,
        "note": "Plate not localized / vehicle side profile"
    },
    {
        "image": "data/raw/plates/indian/images/image_0008.jpg",
        "ground_truth_plate": None,
        "state": None,
        "is_two_line": False,
        "is_legible": False,
        "note": "No plate present on rear tailgate"
    },
    {
        "image": "data/raw/plates/indian/images/image_0015.jpg",
        "ground_truth_plate": None,
        "state": "KL",
        "is_two_line": True,
        "is_legible": False,
        "note": "Severe blur / characters illegible at resolution"
    },
    {
        "image": "data/raw/plates/indian/images/image_0022.jpg",
        "ground_truth_plate": None,
        "state": None,
        "is_two_line": False,
        "is_legible": False,
        "note": "Microscopic plate reflection / illegible"
    }
]

def main():
    out_file = paths.resolve("data/annotations/indian_plates_ground_truth.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(GROUND_TRUTH_DATA, indent=2), encoding="utf-8")
    print(f"Created ground-truth benchmark dataset at {out_file} with {len(GROUND_TRUTH_DATA)} samples.")
    legible = sum(1 for d in GROUND_TRUTH_DATA if d["is_legible"])
    print(f"Legible with verified ground-truth text: {legible}")
    print(f"Illegible/occluded for abstention testing: {len(GROUND_TRUTH_DATA) - legible}")

if __name__ == "__main__":
    main()
