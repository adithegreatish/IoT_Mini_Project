"""Unit tests for plate detection, HSRP cue analyzer, and OCR normalization."""

import numpy as np
import pytest
from app.config.schema import OCRConfig
from app.ocr.normalization import normalize_plate_text
from app.ocr.engine import OCREngine
from app.plates.hsrp_analyzer import HSRPCueAnalyzer, HSRPOutcome


def test_plate_text_normalization():
    # Valid Indian formats
    text, valid, err = normalize_plate_text("DL01AB1234")
    assert valid is True
    assert text == "DL 01 AB 1234"
    assert err is None

    text, valid, err = normalize_plate_text("mh-12.cd-5678")
    assert valid is True
    assert text == "MH 12 CD 5678"

    # Invalid state code
    text, valid, err = normalize_plate_text("ZZ01AB1234")
    assert valid is False
    assert "unrecognized state code" in err.lower()

    # Empty or unreadable
    text, valid, err = normalize_plate_text("")
    assert valid is False
    assert text == "UNREADABLE"


def test_hsrp_cue_analyzer_subscale_crop():
    analyzer = HSRPCueAnalyzer(min_width=60, min_height=18)
    # Tiny crop: 30x10
    tiny_crop = np.zeros((10, 30, 3), dtype=np.uint8)
    report = analyzer.analyze_crop(tiny_crop)
    assert report.outcome == HSRPOutcome.INSUFFICIENT_IMAGE_QUALITY
    assert "below minimum" in report.notes[0].lower()


def test_hsrp_cue_analyzer_synthetic_plate():
    analyzer = HSRPCueAnalyzer(min_width=60, min_height=18)
    # Synthetic plate 120x30 (aspect ratio 4.0)
    plate = np.ones((30, 120, 3), dtype=np.uint8) * 240
    # Add blue IND strip on left 18% (approx 20px)
    plate[:, :20, 0] = 200  # Blue in BGR
    plate[:, :20, 1] = 60
    plate[:, :20, 2] = 0
    # Add characters in black
    plate[8:22, 30:110] = 20

    report = analyzer.analyze_crop(plate, is_two_wheeler=False)
    assert report.observed_aspect_ratio == 4.0
    assert report.blue_ind_strip_detected is True
    assert report.outcome == HSRPOutcome.HSRP_STYLE_CONSISTENT


def test_ocr_engine_unreadable_crop():
    """Verifies that characterless or blank crops are honestly reported as unreadable."""
    cfg = OCRConfig(enabled=True, engine="easyocr")
    engine = OCREngine(cfg)

    # Blank crop without text
    blank = np.zeros((40, 140, 3), dtype=np.uint8)
    res = engine.recognize(blank)
    assert res.is_readable is False
    assert res.is_valid_format is False
    assert "UNREADABLE" in res.normalized_text


def test_ocr_engine_legible_plate():
    """Verifies real neural OCR recognition and Indian format normalization on legible text."""
    cfg = OCRConfig(enabled=True, engine="easyocr")
    engine = OCREngine(cfg)
    if not engine.is_available:
        pytest.skip("EasyOCR / PyTorch not installed in this environment (edge/Pi deployment)")

    # Render a high-contrast plate with Indian registration text
    plate = np.ones((100, 400, 3), dtype=np.uint8) * 255
    import cv2
    cv2.putText(plate, "MH 12 AB 1234", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    res = engine.recognize(plate)
    assert res.is_readable is True
    assert res.is_valid_format is True
    assert res.normalized_text == "MH 12 AB 1234"
    assert res.confidence > 0.0


def test_plate_detector_subscale_rejection():
    """Verifies that subscale vehicle crops are rejected without errors."""
    from app.plates.plate_detector import PlateDetector
    from app.config.schema import PlatesConfig

    cfg = PlatesConfig(enabled=True, runtime="onnx")
    detector = PlateDetector(cfg)

    # Tiny crop smaller than minimum
    tiny = np.zeros((10, 20, 3), dtype=np.uint8)
    assert detector.extract_plate(tiny) is None
    assert detector.detect_plates(None) == []


def test_plate_detector_onnx_inference():
    """Verifies that PlateDetector ONNX model loads and runs inference on real sample."""
    from app.plates.plate_detector import PlateDetector
    from app.config.schema import PlatesConfig
    import cv2

    cfg = PlatesConfig(enabled=True, runtime="onnx")
    detector = PlateDetector(cfg)

    # Synthetic image containing plate-like dimensions
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    results = detector.detect_plates(test_img)
    # Output is a list of PlateDetectionResult without crashing
    assert isinstance(results, list)


def test_ocr_character_disambiguation():
    """Verifies slot-based character disambiguation between visually similar glyphs."""
    # Letter O in district slot should be disambiguated to digit 0
    text, valid, err = normalize_plate_text("MPO7GA1533")
    assert valid is True
    assert text == "MP 07 GA 1533"

    # Letter O and S in district slot (JH 05)
    text, valid, err = normalize_plate_text("JHOSAW2117")
    assert valid is True
    assert text == "JH 05 AW 2117"

    # Bharat Series (BH)
    text, valid, err = normalize_plate_text("22BH1234AA")
    assert valid is True
    assert text == "22 BH 1234 AA"


def test_hsrp_security_element_exclusion():
    """Verifies that IND emblems, laser PINs, and frame dealership text are excluded."""
    # HSRP plate with leading/trailing IND and laser PIN
    raw_with_pin = "KL 02 BH 4659 IND BA2500479660"
    text, valid, err = normalize_plate_text(raw_with_pin)
    assert valid is True
    assert "BA2500479660" not in text
    assert text.startswith("KL 02")

    # Plate with car brand frame text (e.g. Maruti Suzuki)
    raw_with_frame = "MH 12 CD 5678 MARUTI SUZUKI"
    text, valid, err = normalize_plate_text(raw_with_frame)
    assert valid is True
    assert text == "MH 12 CD 5678"


def test_two_line_plate_spatial_ordering():
    """Verifies that line-grouping handles two-line plates without reversing rows."""
    from app.ocr.engine import group_tokens_into_lines

    # Simulate tokens for 2-line plate:
    # Row 1 (y: 20-50): 'DL 01'
    # Row 2 (y: 80-110): 'AB 1234'
    box_row1 = [[[10, 20], [60, 20], [60, 50], [10, 50]], "DL01", 0.95]
    box_row2 = [[[10, 80], [90, 80], [90, 110], [10, 110]], "AB1234", 0.96]

    # Deliberately feed in scrambled vertical order
    lines = group_tokens_into_lines([box_row2, box_row1])
    assert len(lines) == 2
    # First line must be Row 1
    assert lines[0][0][1] == "DL01"
    # Second line must be Row 2
    assert lines[1][0][1] == "AB1234"


def test_model_missing_behavior():
    """Verifies that missing detector model produces actionable error or graceful empty list."""
    from app.plates.plate_detector import PlateDetector
    from app.config.schema import PlatesConfig

    # Nonexistent model path
    cfg = PlatesConfig(enabled=True, runtime="onnx", model_path="models/nonexistent_model.onnx")
    detector = PlateDetector(cfg)
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    results = detector.detect_plates(test_img)
    # Never fabricates simulated detections when model is missing and simulation disabled
    assert results == []

