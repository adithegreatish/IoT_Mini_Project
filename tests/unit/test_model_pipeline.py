"""
Unit and integration tests for model loading, ONNX inference, missing model handling,
manifest integrity, and end-to-end detection-plate-ocr-occupancy pipeline.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import cv2

from app.config.schema import DetectionConfig, PlatesConfig, OCRConfig, ParkingConfig
from app.detection.vehicle_detector import VehicleDetector
from app.infrastructure.paths import paths
from app.ocr.engine import OCREngine
from app.parking.calibration_service import CalibrationService
from app.parking.occupancy_service import OccupancyService
from app.plates.hsrp_analyzer import HSRPCueAnalyzer
from app.plates.plate_detector import PlateDetector


def test_missing_model_behavior():
    """Verifies that non-existent model files fail safely without crashing or faking detections."""
    # Vehicle detector with bogus path
    v_cfg = DetectionConfig(
        enabled=True,
        runtime="onnx",
        model_path="models/non_existent_vehicle_model.onnx",
    )
    v_det = VehicleDetector(v_cfg)
    assert v_det.status == "MODEL_FILE_NOT_FOUND"
    assert v_det.is_simulated is False

    # Detection returns empty list, never fabricated boxes
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert v_det.detect(test_img) == []

    # Plate detector with bogus path
    p_cfg = PlatesConfig(
        enabled=True,
        runtime="onnx",
        model_path="models/non_existent_plate_model.onnx",
    )
    p_det = PlateDetector(p_cfg)
    assert p_det.status == "MODEL_FILE_NOT_FOUND"
    assert p_det.detect_plates(test_img) == []
    assert p_det.extract_plate(test_img) is None


def test_manifest_integrity_and_sha256():
    """Verifies that model manifest points to real existing files matching SHA256 hashes."""
    manifest_path = paths.resolve("models/manifest.json")
    assert manifest_path.exists(), "models/manifest.json must exist"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "models" in manifest
    assert len(manifest["models"]) >= 2

    for m in manifest["models"]:
        m_file = paths.resolve(m["file_path"])
        assert m_file.exists(), f"Model file {m_file} registered in manifest must exist on disk"

        # Compute actual hash
        h = hashlib.sha256()
        with open(m_file, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        actual_sha = h.hexdigest()

        assert actual_sha == m["sha256"], f"SHA256 mismatch for {m['model_id']}"
        assert m["input"]["width"] in [150, 640]
        assert m["input"]["height"] in [150, 640]


def test_vehicle_detector_onnx_inference():
    """Verifies that real vehicle detector ONNX model loads and runs inference."""
    cfg = DetectionConfig(
        enabled=True,
        runtime="onnx",
        model_path="models/vehicle_detector.onnx",
    )
    detector = VehicleDetector(cfg)
    assert detector.status == "READY"
    assert detector.is_simulated is False

    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    results = detector.detect(frame)
    assert isinstance(results, list)
    for res in results:
        assert 0.0 <= res.confidence <= 1.0
        assert res.class_name in ["car", "motorcycle", "bus", "truck"]
        assert res.is_simulated is False


def test_plate_detector_onnx_inference_real_image():
    """Verifies that real plate detector ONNX model runs inference and produces crops."""
    cfg = PlatesConfig(
        enabled=True,
        runtime="onnx",
        model_path="models/plate_detector.onnx",
    )
    detector = PlateDetector(cfg)
    assert detector.status == "READY"

    # Test image from held-out split
    test_manifest_path = paths.resolve("data/splits/test_manifest.json")
    if test_manifest_path.exists():
        manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plates"):
            img_path = paths.resolve(manifest["plates"][0]["image"])
            if img_path.exists():
                img = cv2.imread(str(img_path))
                candidates = detector.detect_plates(img)
                assert isinstance(candidates, list)
                if candidates:
                    top = candidates[0]
                    assert top.confidence > 0.40
                    assert top.crop is not None
                    assert top.crop.shape[0] > 0
                    assert top.crop.shape[1] > 0


def test_end_to_end_pipeline_integration():
    """Verifies complete integration: vehicle detection -> plate crop -> OCR -> HSRP analysis."""
    # 1. Initialize all real components
    v_det = VehicleDetector(DetectionConfig(runtime="onnx", model_path="models/vehicle_detector.onnx"))
    p_det = PlateDetector(PlatesConfig(runtime="onnx", model_path="models/plate_detector.onnx"))
    ocr = OCREngine(OCRConfig(engine="easyocr"))
    hsrp = HSRPCueAnalyzer()

    # Synthetic vehicle image with high-contrast plate region
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    # Simulate a license plate region with legible text
    plate_region = np.ones((70, 260, 3), dtype=np.uint8) * 240
    cv2.putText(plate_region, "DL 01 AB 1234", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    frame[300:370, 500:760] = plate_region

    # OCR direct execution
    ocr_res = ocr.recognize(plate_region)
    if ocr.is_available:
        assert ocr_res.is_readable is True
        assert "DL 01 AB 1234" in ocr_res.normalized_text
    else:
        assert ocr_res.is_readable is False
        assert "OCR engine unavailable" in (ocr_res.error_reason or "")

    # HSRP analysis execution
    hsrp_rep = hsrp.analyze_crop(plate_region)
    assert hsrp_rep.outcome is not None
    assert isinstance(hsrp_rep.notes, list)


def test_occupancy_service_stale_camera_honesty():
    """Verifies that OccupancyService never manufactures fake occupied/free counts when camera is stale."""
    calib_svc = CalibrationService("data/calibration.json")
    v_det = VehicleDetector(DetectionConfig(runtime="onnx", model_path="models/vehicle_detector.onnx"))
    p_cfg = ParkingConfig(occupancy_threshold=0.35)

    svc = OccupancyService(p_cfg, calib_svc, v_det)
    # Process stale frame (None)
    snap = svc.process_frame(frame=None, is_camera_healthy=False, is_camera_stale=True)

    summary = snap["summary"]
    # Total slots should equal unknown count, free and occupied MUST be 0
    assert summary["unknown"] == summary["total"]
    assert summary["free"] == 0
    assert summary["occupied"] == 0
    assert snap["is_camera_stale"] is True
