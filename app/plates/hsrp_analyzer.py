"""Indian HSRP-style visual cue analyzer and explainability engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class HSRPOutcome(str, Enum):
    HSRP_STYLE_CONSISTENT = "HSRP_STYLE_CONSISTENT"
    NON_HSRP_OR_UNCERTAIN = "NON_HSRP_OR_UNCERTAIN"
    INSUFFICIENT_IMAGE_QUALITY = "INSUFFICIENT_IMAGE_QUALITY"
    PLATE_DETECTED = "PLATE_DETECTED"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass
class HSRPAnalysisReport:
    outcome: HSRPOutcome
    confidence: float
    visible_cues: Dict[str, Any]
    observed_aspect_ratio: float
    blue_ind_strip_detected: bool
    crop_width: int
    crop_height: int
    notes: List[str] = field(default_factory=list)
    disclaimer: str = "ADVISORY ONLY: Visual cue analysis does not constitute legal or certified authenticity verification."



class HSRPCueAnalyzer:
    """
    Evaluates visually observable High Security Registration Plate (HSRP) cues:
    - Aspect ratio consistency (~3.5 - 5.0 for cars, ~1.2 - 1.5 for bikes)
    - Blue 'IND' panel on left border
    - High-contrast characters on reflective background
    - Image quality and scale gating
    Does NOT assert cryptographic or legal authenticity.
    """

    def __init__(self, min_width: int = 60, min_height: int = 18):
        self.min_width = min_width
        self.min_height = min_height

    def analyze_crop(self, plate_crop: Optional[np.ndarray], is_two_wheeler: bool = False) -> HSRPAnalysisReport:
        if plate_crop is None or plate_crop.size == 0:
            return HSRPAnalysisReport(
                outcome=HSRPOutcome.NOT_ASSESSED,
                confidence=0.0,
                visible_cues={},
                observed_aspect_ratio=0.0,
                blue_ind_strip_detected=False,
                crop_width=0,
                crop_height=0,
                notes=["Empty or null crop provided."],
            )

        h, w = plate_crop.shape[:2]
        if w < self.min_width or h < self.min_height:
            return HSRPAnalysisReport(
                outcome=HSRPOutcome.INSUFFICIENT_IMAGE_QUALITY,
                confidence=0.0,
                visible_cues={"resolution": f"{w}x{h}"},
                observed_aspect_ratio=round(w / max(1, h), 2),
                blue_ind_strip_detected=False,
                crop_width=w,
                crop_height=h,
                notes=[f"Plate crop resolution {w}x{h} below minimum readability threshold {self.min_width}x{self.min_height}."],
            )

        aspect_ratio = round(w / float(h), 2)
        notes = []
        cues = {}

        # 1. Aspect Ratio Assessment
        expected_aspect = (1.1, 1.8) if is_two_wheeler else (3.2, 5.4)
        aspect_consistent = expected_aspect[0] <= aspect_ratio <= expected_aspect[1]
        cues["aspect_ratio_consistent"] = aspect_consistent
        cues["aspect_ratio"] = aspect_ratio
        if not aspect_consistent:
            notes.append(f"Aspect ratio {aspect_ratio} diverges from typical HSRP range ({expected_aspect[0]}-{expected_aspect[1]}).")

        # 2. Blue 'IND' Panel Detection on Left Margin (first 12% - 20% of plate width)
        left_strip_w = max(4, int(w * 0.18))
        left_strip = plate_crop[:, :left_strip_w]

        hsv_strip = cv2.cvtColor(left_strip, cv2.COLOR_BGR2HSV)
        # Blue in HSV: Hue ~100-135, Saturation > 50, Value > 50
        lower_blue = np.array([95, 50, 50])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv_strip, lower_blue, upper_blue)
        blue_pixel_ratio = float(np.sum(blue_mask > 0) / (left_strip.shape[0] * left_strip.shape[1]))

        has_blue_ind = blue_pixel_ratio >= 0.15
        cues["blue_ind_strip_detected"] = has_blue_ind
        cues["blue_pixel_ratio"] = round(blue_pixel_ratio, 3)

        if has_blue_ind:
            notes.append(f"Observable blue color band detected on left margin ({blue_pixel_ratio * 100:.1f}% pixels).")
        else:
            notes.append("No prominent blue IND margin detected on left edge.")

        # 3. Plate Background / Character Contrast
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        std_contrast = float(np.std(gray))
        mean_lum = float(np.mean(gray))
        cues["contrast_std"] = round(std_contrast, 1)
        cues["mean_luminance"] = round(mean_lum, 1)

        high_contrast = std_contrast > 35.0
        cues["high_contrast"] = high_contrast

        # 4. Synthesize Explainable Outcome
        if has_blue_ind and aspect_consistent and high_contrast:
            outcome = HSRPOutcome.HSRP_STYLE_CONSISTENT
            confidence = 0.85
            notes.append("Visual appearance aligns with Indian HSRP design specifications.")
        elif aspect_consistent and high_contrast:
            outcome = HSRPOutcome.PLATE_DETECTED
            confidence = 0.65
            notes.append("Standard plate geometry detected; IND strip inconclusive.")
        else:
            outcome = HSRPOutcome.NON_HSRP_OR_UNCERTAIN
            confidence = 0.45
            notes.append("Features uncertain or non-standard styling.")

        return HSRPAnalysisReport(
            outcome=outcome,
            confidence=confidence,
            visible_cues=cues,
            observed_aspect_ratio=aspect_ratio,
            blue_ind_strip_detected=has_blue_ind,
            crop_width=w,
            crop_height=h,
            notes=notes,
        )
