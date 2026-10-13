"""Optical character recognition and normalization package."""

from app.ocr.normalization import normalize_plate_text, INDIAN_STATE_CODES
from app.ocr.engine import OCREngine, OCRResult

__all__ = [
    "normalize_plate_text",
    "INDIAN_STATE_CODES",
    "OCREngine",
    "OCRResult",
]
