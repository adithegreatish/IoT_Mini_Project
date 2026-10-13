"""
Detailed inspection of individual plate crop tokens
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

    for img_name in ["image_0001.jpg", "image_0003.jpg", "image_0004.jpg", "image_0014.jpg", "image_0025.jpg", "image_0041.jpg"]:
        img_path = paths.resolve(f"data/raw/plates/indian/images/{img_name}")
        img = cv2.imread(str(img_path))
        plates = detector.detect_plates(img)
        if not plates:
            print(f"[{img_name}] No plate detected")
            continue
        pl = plates[0]
        print(f"\n[{img_name}] bbox={pl.bbox}, crop_shape={pl.crop.shape}")
        tokens = reader.readtext(pl.crop, detail=1)
        for box, txt, conf in tokens:
            ymin = min(p[1] for p in box)
            ymax = max(p[1] for p in box)
            xmin = min(p[0] for p in box)
            xmax = max(p[0] for p in box)
            print(f"  Token: '{txt:15s}' conf={conf:.3f} y=[{ymin:.1f}, {ymax:.1f}] x=[{xmin:.1f}, {xmax:.1f}]")

if __name__ == "__main__":
    main()
