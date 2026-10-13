"""
Prototype and Benchmark for Improved Indian License Plate OCR Pipeline
Tests:
1. Crop padding & aspect-preserving normalization
2. Skew detection and deskewing
3. CLAHE & Bilateral filtering
4. Vertical line grouping and spatial sorting
5. HSRP security element (IND, laser PIN) exclusion
6. Position-aware character disambiguation
7. Comprehensive Indian registration syntax validation
"""

import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.config.schema import OCRConfig, PlatesConfig
from app.plates.plate_detector import PlateDetector
from app.ocr.engine import get_easyocr_reader

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DH", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ",
    "SK", "TN", "TS", "TR", "UA", "UK", "UP", "WB"
}

STANDARD_REGEX = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})$")
BH_REGEX = re.compile(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$")

CHAR_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
DIGIT_TO_CHAR = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

def deskew_crop(crop: np.ndarray) -> np.ndarray:
    """Deskews plate crop if a significant tilt angle is detected using edges."""
    h, w = crop.shape[:2]
    if h < 30 or w < 60:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=w // 4, maxLineGap=10)

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
            # Only consider angles within +/- 30 degrees of horizontal
            if -30.0 <= deg <= 30.0:
                angles.append(deg)

    if not angles:
        return crop

    median_angle = float(np.median(angles))
    if abs(median_angle) < 1.5:
        return crop

    # Rotate around center
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_plate_advanced(crop: np.ndarray) -> np.ndarray:
    """
    Standardizes plate resolution, deskews, and applies CLAHE & bilateral filter.
    Target height is normalized to ~120-140px preserving aspect ratio.
    """
    h, w = crop.shape[:2]
    # Deskew if tilted
    deskewed = deskew_crop(crop)

    # Scale to standard normalized height for OCR
    target_h = 130
    scale = target_h / float(h)
    target_w = max(int(w * scale), 60)
    resized = cv2.resize(deskewed, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Bilateral filter to reduce noise while maintaining character edges
    filtered = cv2.bilateralFilter(gray, 5, 50, 50)
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    return enhanced

def group_tokens_into_lines(tokens: List[Tuple[Any, str, float]]) -> List[List[Tuple[Any, str, float]]]:
    """
    Groups OCR tokens into spatial text lines based on vertical overlap,
    then sorts lines top-to-bottom and tokens within lines left-to-right.
    """
    if not tokens:
        return []

    token_info = []
    for box, text, conf in tokens:
        ymin = min(pt[1] for pt in box)
        ymax = max(pt[1] for pt in box)
        xmin = min(pt[0] for pt in box)
        xmax = max(pt[0] for pt in box)
        h = ymax - ymin
        cy = (ymin + ymax) / 2.0
        token_info.append({
            "box": box, "text": text, "conf": conf,
            "ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax,
            "h": h, "cy": cy
        })

    # Sort tokens by vertical center initially
    token_info.sort(key=lambda t: t["cy"])
    median_h = max(10.0, float(np.median([t["h"] for t in token_info])))

    lines: List[List[Dict[str, Any]]] = []
    for t in token_info:
        placed = False
        for line in lines:
            line_cy = float(np.mean([item["cy"] for item in line]))
            # If vertical center is within half the median height, it's on the same line
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
        sorted_lines.append([(item["box"], item["text"], item["conf"]) for item in line])

    return sorted_lines

def disambiguate_indian_plate(clean_str: str) -> Tuple[str, bool, Optional[str]]:
    """
    Performs position-aware character disambiguation on candidate Indian plate string.
    Only swaps characters when the statutory slot requires digit vs letter.
    """
    if len(clean_str) < 6:
        return clean_str, False, "Too short for Indian registration (<6 chars)"

    # Check BH Series first
    bh_match = BH_REGEX.match(clean_str)
    if bh_match:
        return f"{bh_match.group(1)} BH {bh_match.group(2)} {bh_match.group(3)}", True, None

    # Try standard format directly
    m = STANDARD_REGEX.match(clean_str)
    if m:
        state = m.group(1)
        if state in INDIAN_STATE_CODES:
            series = m.group(3)
            formatted = f"{state} {m.group(2)}" + (f" {series}" if series else "") + f" {m.group(4)}"
            return formatted, True, None

    # Disambiguation attempt
    chars = list(clean_str)
    # Slot 0-1: State code (MUST BE LETTERS)
    for i in (0, 1):
        if i < len(chars) and chars[i] in DIGIT_TO_CHAR:
            chars[i] = DIGIT_TO_CHAR[chars[i]]

    state_candidate = "".join(chars[:2])
    if state_candidate not in INDIAN_STATE_CODES:
        # Check if first 2 chars can form a valid state code
        return clean_str, False, f"Unrecognized state code '{state_candidate}'"

    # Identify remaining segments:
    # Slot 2-3: District code (1 or 2 DIGITS)
    # Followed by 0-3 LETTERS (Series)
    # Followed by 1-4 DIGITS (Number)
    rem = chars[2:]
    if len(rem) < 4:
        return clean_str, False, "Insufficient characters after state code"

    # Fix district code: positions 0 and optionally 1 of rem should be digits
    # Let's test 2 digits vs 1 digit
    for dist_len in (2, 1):
        test_chars = list(rem)
        valid_dist = True
        for i in range(dist_len):
            if test_chars[i] in CHAR_TO_DIGIT:
                test_chars[i] = CHAR_TO_DIGIT[test_chars[i]]
            elif not test_chars[i].isdigit():
                valid_dist = False
                break
        if not valid_dist:
            continue

        dist_str = "".join(test_chars[:dist_len])
        after_dist = test_chars[dist_len:]
        if len(after_dist) < 1:
            continue

        # Last 1-4 characters MUST BE DIGITS (Number)
        # Anything between dist and number is series (LETTERS)
        for num_len in (4, 3, 2, 1):
            if len(after_dist) < num_len:
                continue
            series_chars = list(after_dist[:-num_len])
            num_chars = list(after_dist[-num_len:])

            if len(series_chars) > 3:
                continue

            # Series characters must be letters
            valid_series = True
            for k in range(len(series_chars)):
                if series_chars[k] in DIGIT_TO_CHAR:
                    series_chars[k] = DIGIT_TO_CHAR[series_chars[k]]
                elif not series_chars[k].isalpha():
                    valid_series = False
                    break
            if not valid_series:
                continue

            # Number characters must be digits
            valid_num = True
            for k in range(len(num_chars)):
                if num_chars[k] in CHAR_TO_DIGIT:
                    num_chars[k] = CHAR_TO_DIGIT[num_chars[k]]
                elif not num_chars[k].isdigit():
                    valid_num = False
                    break
            if not valid_num:
                continue

            series_str = "".join(series_chars)
            num_str = "".join(num_chars)
            formatted = f"{state_candidate} {dist_str}" + (f" {series_str}" if series_str else "") + f" {num_str}"
            return formatted, True, None

    return clean_str, False, "Cannot parse valid Indian registration sequence"

def filter_and_normalize_ocr_tokens(lines: List[List[Tuple[Any, str, float]]]) -> Tuple[str, str, float, bool, Optional[str]]:
    """
    Filters HSRP security elements (IND emblem, laser PIN) and formats plate text.
    """
    raw_words = []
    confs = []

    for line in lines:
        for box, text, conf in line:
            clean_word = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            if not clean_word:
                continue

            # Discard standalone IND or INDIA emblems
            if clean_word in ("IND", "INDIA"):
                continue

            # Discard laser-etched permanent identification numbers (typically 10-12 chars starting with letters + digits)
            # if located on bottom-left or separate token
            if len(clean_word) >= 10 and re.match(r"^[A-Z]{2}[0-9]{8,10}$", clean_word):
                continue

            raw_words.append(clean_word)
            confs.append(float(conf))

    if not raw_words:
        return "UNREADABLE", "UNREADABLE", 0.0, False, "No registration characters identified"

    raw_text = " ".join(raw_words)
    mean_conf = float(np.mean(confs)) if confs else 0.0

    # Strip leading/trailing IND if attached to a word
    full_clean = "".join(raw_words)
    if full_clean.startswith("IND") and len(full_clean) > 8:
        full_clean = full_clean[3:]
    if full_clean.endswith("IND") and len(full_clean) > 8:
        full_clean = full_clean[:-3]

    norm_text, is_valid, err = disambiguate_indian_plate(full_clean)
    return raw_text, norm_text, round(mean_conf, 3), is_valid, err

def main():
    detector = PlateDetector(PlatesConfig(runtime="onnx"))
    reader = get_easyocr_reader(gpu=True)

    indian_dir = paths.resolve("data/raw/plates/indian/images")
    imgs = sorted(list(indian_dir.glob("*.jpg")))
    print(f"Testing improved pipeline on {len(imgs)} Indian images...\n")

    valid_count = 0
    detected_count = 0

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        plates = detector.detect_plates(img)
        if not plates:
            print(f"[NO_PLATE] {p.name}")
            continue

        detected_count += 1
        pl = plates[0]

        # Improved preprocessing
        enhanced = preprocess_plate_advanced(pl.crop)

        # EasyOCR with detail=1
        ocr_res = reader.readtext(enhanced, detail=1, paragraph=False)

        # Line grouping
        lines = group_tokens_into_lines(ocr_res)

        # Filter & normalize
        raw, norm, conf, valid, err = filter_and_normalize_ocr_tokens(lines)

        if valid:
            valid_count += 1
            print(f"[VALID  ] {p.name:15s} -> '{norm}' (conf={conf:.2f}, raw='{raw}')")
        else:
            print(f"[INVALID] {p.name:15s} -> raw='{raw}', norm='{norm}', reason='{err}'")

    print("\n" + "="*70)
    print(f"Plates detected: {detected_count}/{len(imgs)} ({detected_count/len(imgs)*100:.1f}%)")
    print(f"Valid Indian formats: {valid_count}/{detected_count} ({valid_count/detected_count*100:.1f}%) [was 5/41 = 12.2%]")
    print("="*70)

if __name__ == "__main__":
    main()
