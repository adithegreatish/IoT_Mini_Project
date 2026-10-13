"""Real Neural Optical Character Recognition (OCR) engine for vehicle registration plates.

Utilizes PyTorch-backed EasyOCR neural network with bilateral & CLAHE preprocessing,
skew correction, spatial character line ordering, Indian registration syntax validation,
and honest unreadable reporting without fabricated plate numbers.
"""

from dataclasses import dataclass, field
import logging
import math
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.config.schema import OCRConfig
from app.ocr.normalization import normalize_plate_text

logger = logging.getLogger("smart_parking.ocr")

_easyocr_reader = None
_easyocr_lock = threading.Lock()


def get_easyocr_reader(gpu: bool = True):
    """Singleton thread-safe loader for EasyOCR neural network reader."""
    global _easyocr_reader
    with _easyocr_lock:
        if _easyocr_reader is None:
            try:
                import torch
                import easyocr
                use_gpu = gpu and torch.cuda.is_available()
                logger.info(f"Initializing EasyOCR neural reader (GPU accelerated: {use_gpu})...")
                _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                logger.info("EasyOCR reader initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
        return _easyocr_reader


@dataclass
class OCRResult:
    raw_text: str
    normalized_text: str
    confidence: float
    is_valid_format: bool
    is_readable: bool
    error_reason: Optional[str] = None
    detected_tokens: List[Dict[str, Any]] = field(default_factory=list)


def deskew_crop(crop: np.ndarray) -> np.ndarray:
    """Deskews plate crop if a significant tilt angle is detected using edge lines."""
    h, w = crop.shape[:2]
    if h < 30 or w < 60:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(15, w // 4), maxLineGap=10)

    if lines is None or len(lines) == 0:
        return crop

    angles = []
    for line in lines:
        pts = line.flatten()
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = pts[:4]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 10:
            deg = math.degrees(math.atan2(dy, dx))
            # Plate orientation tilt is typically within +/- 30 degrees
            if -30.0 <= deg <= 30.0:
                angles.append(deg)

    if not angles:
        return crop

    median_angle = float(np.median(angles))
    if abs(median_angle) < 1.5:
        return crop

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def group_tokens_into_lines(tokens: List[Any]) -> List[List[Any]]:
    """
    Groups OCR bounding boxes into lines based on vertical overlap,
    ordering lines top-to-bottom and tokens within lines left-to-right.
    """
    if not tokens:
        return []

    token_info = []
    for item in tokens:
        box = item[0]
        text = item[1]
        conf = float(item[2])
        ymin = min(pt[1] for pt in box)
        ymax = max(pt[1] for pt in box)
        xmin = min(pt[0] for pt in box)
        xmax = max(pt[0] for pt in box)
        h = ymax - ymin
        cy = (ymin + ymax) / 2.0
        token_info.append({
            "item": item,
            "ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax,
            "h": h, "cy": cy, "text": text, "conf": conf
        })

    token_info.sort(key=lambda t: t["cy"])
    median_h = max(10.0, float(np.median([t["h"] for t in token_info])))

    lines: List[List[Dict[str, Any]]] = []
    for t in token_info:
        placed = False
        for line in lines:
            line_cy = float(np.mean([item["cy"] for item in line]))
            if abs(t["cy"] - line_cy) < (0.45 * median_h):
                line.append(t)
                placed = True
                break
        if not placed:
            lines.append([t])

    # Sort lines top-to-bottom
    lines.sort(key=lambda l: float(np.mean([item["cy"] for item in l])))

    # Within each line, sort left-to-right
    sorted_lines = []
    for line in lines:
        line.sort(key=lambda item: item["xmin"])
        sorted_lines.append([item["item"] for item in line])

    return sorted_lines


class OCREngine:
    """Real OCR engine extracting text from license plate crops."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.enabled = config.enabled
        self.conf_threshold = config.confidence_threshold
        self.engine_name = getattr(config, "engine", "easyocr")

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses plate crop for optical character recognition:
        - Deskews tilt orientation if horizontal edge lines detected.
        - Standardizes crop height to ~130px preserving aspect ratio.
        - Converts to grayscale.
        - Applies Bilateral Filtering to reduce noise while maintaining character edges.
        - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        h, w = plate_crop.shape[:2]
        # Deskew if tilted
        deskewed = deskew_crop(plate_crop)

        # Scale to standard normalized height for neural character recognition
        target_h = 130
        scale = target_h / float(h)
        target_w = max(int(w * scale), 60)
        resized = cv2.resize(deskewed, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        return enhanced

    def recognize(self, plate_crop: Optional[np.ndarray]) -> OCRResult:
        if not self.enabled:
            return OCRResult("N/A", "N/A", 0.0, False, False, "OCR disabled in configuration")

        if plate_crop is None or plate_crop.size == 0:
            return OCRResult("UNREADABLE", "UNREADABLE", 0.0, False, False, "Null or empty plate image")

        h, w = plate_crop.shape[:2]
        if w < 40 or h < 12:
            return OCRResult("UNREADABLE", "UNREADABLE", 0.0, False, False, f"Crop dimensions {w}x{h} insufficient for OCR")

        # 1. Preprocessing
        enhanced = self.preprocess_plate(plate_crop)

        # 2. Real Neural OCR Recognition
        try:
            reader = get_easyocr_reader(gpu=True)
            ocr_results = reader.readtext(
                enhanced,
                detail=1,
                paragraph=False,
            )
        except Exception as e:
            logger.error(f"EasyOCR inference error: {e}")
            return OCRResult("ERROR", "ERROR", 0.0, False, False, f"OCR engine error: {e}")

        if not ocr_results:
            return OCRResult(
                raw_text="UNREADABLE",
                normalized_text="UNREADABLE",
                confidence=0.0,
                is_valid_format=False,
                is_readable=False,
                error_reason="No legible alphanumeric characters detected",
            )

        # 3. Spatial Line Grouping (handles single-line and two-line plates)
        grouped_lines = group_tokens_into_lines(ocr_results)

        raw_words = []
        token_confs = []
        detected_tokens_info = []

        for line in grouped_lines:
            for bbox, text, conf in line:
                cleaned_word = text.strip().upper()
                if not cleaned_word:
                    continue

                raw_words.append(cleaned_word)
                token_confs.append(float(conf))
                detected_tokens_info.append({
                    "text": cleaned_word,
                    "confidence": round(float(conf), 3),
                    "bbox": [[round(float(c), 1) for c in pt] for pt in bbox]
                })

        raw_text = " ".join(raw_words)
        mean_conf = float(np.mean(token_confs)) if token_confs else 0.0

        # 4. Format Normalization & Validation
        normalized, is_valid, err_reason = normalize_plate_text(raw_text)

        # Report readable based on true neural confidence and non-empty output
        is_readable = mean_conf >= self.conf_threshold and bool(raw_words) and normalized != "UNREADABLE"

        return OCRResult(
            raw_text=raw_text,
            normalized_text=normalized,
            confidence=round(mean_conf, 3),
            is_valid_format=is_valid,
            is_readable=is_readable,
            error_reason=err_reason if not is_valid else None,
            detected_tokens=detected_tokens_info,
        )
