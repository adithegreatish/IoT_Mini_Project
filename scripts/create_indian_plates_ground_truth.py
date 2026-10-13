"""
Inspect and verify ground truth text labels for Indian license plates
"""
import sys
from pathlib import Path
import cv2

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.plates.plate_detector import PlateDetector
from app.config.schema import PlatesConfig
from app.ocr.engine import get_easyocr_reader

def main():
    detector = PlateDetector(PlatesConfig(runtime="onnx"))
    reader = get_easyocr_reader(gpu=True)
    indian_dir = paths.resolve("data/raw/plates/indian/images")
    imgs = sorted(list(indian_dir.glob("*.jpg")))

    out_crops_dir = paths.resolve("data/processed/plates/ground_truth_crops")
    out_crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting {len(imgs)} Indian plate images for GT labeling...")
    data = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        plates = detector.detect_plates(img)
        if not plates:
            print(f"{p.name}: NO PLATE DETECTED")
            continue
        pl = plates[0]
        crop_path = out_crops_dir / f"{p.stem}_crop.jpg"
        cv2.imwrite(str(crop_path), pl.crop)

        tokens = reader.readtext(pl.crop, detail=0)
        print(f"{p.name}: bbox={pl.bbox}, shape={pl.crop.shape}, raw_tokens={tokens}")

if __name__ == "__main__":
    main()
