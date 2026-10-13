"""
OCR Diagnostic Script
Investigates raw EasyOCR detections, character bounding boxes, reading order,
spatial layout, character confusion, and Indian registration formatting.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Workspace root
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.config.schema import OCRConfig, PlatesConfig
from app.ocr.engine import OCREngine, get_easyocr_reader
from app.plates.plate_detector import PlateDetector
from app.ocr.normalization import normalize_plate_text

def main():
    detector = PlateDetector(PlatesConfig(runtime="onnx"))
    ocr = OCREngine(OCRConfig(engine="easyocr", confidence_threshold=0.3))
    reader = get_easyocr_reader(gpu=True)

    indian_dir = paths.resolve("data/raw/plates/indian/images")
    imgs = sorted(list(indian_dir.glob("*.jpg")))
    print(f"Total Indian plate images: {len(imgs)}")

    results = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        plates = detector.detect_plates(img)
        if not plates:
            results.append({
                "image": p.name,
                "detected": False,
                "raw": "NO_PLATE",
                "tokens": [],
                "valid": False,
                "error": "No plate detected"
            })
            continue

        pl = plates[0]
        # Raw tokens directly from EasyOCR without filtering
        raw_easyocr = reader.readtext(pl.crop, detail=1, paragraph=False)
        tokens_info = [
            {"text": t[1], "conf": round(float(t[2]), 3), "bbox": [[round(float(c), 1) for c in pt] for pt in t[0]]}
            for t in raw_easyocr
        ]

        res = ocr.recognize(pl.crop)
        results.append({
            "image": p.name,
            "detected": True,
            "plate_bbox": pl.bbox,
            "tokens": tokens_info,
            "engine_raw": res.raw_text,
            "engine_norm": res.normalized_text,
            "engine_valid": res.is_valid_format,
            "engine_err": res.error_reason,
        })

    print("\n" + "="*80)
    print("DETAILED OCR DIAGNOSTIC ON 47 INDIAN VEHICLE PLATES")
    print("="*80)
    valid_count = 0
    detected_count = 0
    for r in results:
        if r["detected"]:
            detected_count += 1
            if r["engine_valid"]:
                valid_count += 1
        tok_strs = [f"'{t['text']}'({t['conf']})" for t in r.get("tokens", [])]
        status = "VALID" if r.get("engine_valid") else "INVALID"
        print(f"[{status:7s}] {r['image']}: raw='{r.get('engine_raw')}' | norm='{r.get('engine_norm')}' | tokens=[{', '.join(tok_strs)}]")

    print(f"\nSummary:")
    print(f"Plates detected: {detected_count}/{len(imgs)} ({detected_count/len(imgs)*100:.1f}%)")
    print(f"Valid formats:   {valid_count}/{detected_count} ({valid_count/detected_count*100:.1f}%)")

if __name__ == "__main__":
    main()
