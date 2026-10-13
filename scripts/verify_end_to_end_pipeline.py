"""
End-to-End Real-Image Verification Script
Processes actual held-out images through:
Vehicle Detection -> Plate Detection -> Crop Preprocessing -> OCR -> Format Validation -> HSRP Visible Cue Analysis.

Verifies:
1. Detected vehicle attributes (bbox, class, confidence)
2. Localized plate attributes (bbox, crop shape, confidence)
3. OCR transcription (raw text, normalized text, valid format flag, unreadable flag)
4. HSRP visible feature observation (blue strip, aspect ratio, rivets, laser PIN)
5. Clear separation between visual cues and authoritative legal verification
6. Strict abstention: never claims successful recognition on unreadable/empty text
"""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List
import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.config.schema import DetectionConfig, PlatesConfig, OCRConfig
from app.detection.vehicle_detector import VehicleDetector
from app.plates.plate_detector import PlateDetector
from app.ocr.engine import OCREngine
from app.plates.hsrp_analyzer import HSRPCueAnalyzer, HSRPOutcome

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_parking.e2e_verify")


def process_image_end_to_end(
    img_path: Path,
    v_detector: VehicleDetector,
    p_detector: PlateDetector,
    ocr_engine: OCREngine,
    hsrp_analyzer: HSRPCueAnalyzer,
) -> Dict:
    img = cv2.imread(str(img_path))
    if img is None:
        return {"image": img_path.name, "status": "ERROR", "error": "Cannot decode image file"}

    h, w = img.shape[:2]
    stage_timing = {}

    # Stage 1: Vehicle Detection
    t0 = time.perf_counter()
    vehicles = v_detector.detect(img)
    stage_timing["vehicle_detection_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # Stage 2: Plate Detection
    t0 = time.perf_counter()
    plates = p_detector.detect_plates(img)
    stage_timing["plate_detection_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    plate_results = []
    for i, pl in enumerate(plates):
        # Stage 3 & 4: OCR Crop Preprocessing & Recognition
        t0 = time.perf_counter()
        ocr_res = ocr_engine.recognize(pl.crop)
        ocr_lat = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 5: HSRP Visible Cue Analysis
        t0 = time.perf_counter()
        hsrp_report = hsrp_analyzer.analyze_crop(pl.crop, is_two_wheeler=False)
        hsrp_lat = round((time.perf_counter() - t0) * 1000.0, 2)

        # Strict safety check: Never report success on unreadable/empty
        if not ocr_res.is_readable:
            pipeline_status = "UNREADABLE_PLATE"
        elif not ocr_res.is_valid_format:
            pipeline_status = "INVALID_REGISTRATION_FORMAT"
        else:
            pipeline_status = "RECOGNIZED_AND_VALIDATED"

        plate_results.append({
            "plate_index": i,
            "plate_bbox": list(pl.bbox),
            "plate_confidence": round(pl.confidence, 3),
            "ocr": {
                "raw_text": ocr_res.raw_text,
                "normalized_text": ocr_res.normalized_text,
                "is_readable": ocr_res.is_readable,
                "is_valid_format": ocr_res.is_valid_format,
                "confidence": ocr_res.confidence,
                "error_reason": ocr_res.error_reason,
                "latency_ms": ocr_lat,
            },
            "hsrp_visual_cues": {
                "outcome": hsrp_report.outcome.value,
                "blue_strip_observed": hsrp_report.blue_ind_strip_detected,
                "observed_aspect_ratio": round(hsrp_report.observed_aspect_ratio, 2),
                "visual_notes": hsrp_report.notes,
                "authoritative_verification": "UNVERIFIED_VISUAL_HEURISTIC_ONLY (Requires official VAHAN integration)"
            },
            "pipeline_status": pipeline_status
        })

    return {
        "image": img_path.name,
        "resolution": f"{w}x{h}",
        "stage_timings_ms": stage_timing,
        "vehicles_detected_count": len(vehicles),
        "vehicles": [
            {
                "class_name": v.class_name,
                "confidence": round(v.confidence, 3),
                "pixel_box": list(v.pixel_box)
            }
            for v in vehicles
        ],
        "plates_detected_count": len(plates),
        "plates": plate_results
    }


def main():
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE VERIFICATION ON REAL HELD-OUT SAMPLES")
    print("=" * 70)

    v_detector = VehicleDetector(DetectionConfig(runtime="onnx", model_path="models/vehicle_detector.onnx"))
    p_detector = PlateDetector(PlatesConfig(runtime="onnx", model_path="models/plate_detector.onnx"))
    ocr_engine = OCREngine(OCRConfig(engine="easyocr"))
    hsrp_analyzer = HSRPCueAnalyzer()

    # Select representative real samples (Indian street/parking lots and benchmark vehicles)
    test_samples = [
        paths.resolve("data/raw/plates/indian/images/image_0001.jpg"),
        paths.resolve("data/raw/plates/indian/images/image_0005.jpg"),
        paths.resolve("data/raw/plates/indian/images/image_0010.jpg"),
        paths.resolve("data/raw/plates/indian/images/image_0020.jpg"),
        paths.resolve("data/raw/plates/indian/images/image_0022.jpg"),  # Unreadable sample
        paths.resolve("data/raw/plates/indian/images/image_0041.jpg"),
    ]

    all_results = []
    for s_path in test_samples:
        if not s_path.exists():
            continue
        res = process_image_end_to_end(s_path, v_detector, p_detector, ocr_engine, hsrp_analyzer)
        all_results.append(res)

        status_str = res["plates"][0]["pipeline_status"] if res["plates"] else "NO_PLATE"
        plate_str = res["plates"][0]["ocr"]["normalized_text"] if res["plates"] else "N/A"
        print(f"[{res['image']:15s}] Vehicles={res['vehicles_detected_count']} | Plates={res['plates_detected_count']} | Status={status_str:25s} | Plate='{plate_str}'")

    out_file = paths.resolve("data/test_reports/evaluations/e2e_pipeline_verification.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[+] E2E pipeline verification successfully saved to {out_file}")


if __name__ == "__main__":
    main()
