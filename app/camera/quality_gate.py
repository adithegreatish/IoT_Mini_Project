"""Image quality gate and diagnostic assessment."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
import cv2
import numpy as np


@dataclass
class QualityAssessment:
    status: Literal["EXCELLENT", "ACCEPTABLE", "POOR", "INVALID"]
    blur_score: float
    brightness: float
    contrast: float
    glare_ratio: float
    width: int
    height: int
    issues: List[str] = field(default_factory=list)


class QualityGate:
    """Evaluates optical image quality for parking monitoring and plate reading."""

    def __init__(
        self,
        min_blur_variance: float = 80.0,
        min_brightness: float = 40.0,
        max_brightness: float = 225.0,
        min_contrast: float = 25.0,
        max_glare_ratio: float = 0.15,
    ):
        self.min_blur_variance = min_blur_variance
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_glare_ratio = max_glare_ratio

    def assess_frame(self, frame: Optional[np.ndarray]) -> QualityAssessment:
        if frame is None or frame.size == 0:
            return QualityAssessment(
                status="INVALID",
                blur_score=0.0,
                brightness=0.0,
                contrast=0.0,
                glare_ratio=0.0,
                width=0,
                height=0,
                issues=["Frame is empty or None"],
            )

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 1. Blur evaluation using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        # 2. Brightness evaluation (mean grayscale intensity)
        brightness = float(np.mean(gray))

        # 3. Contrast evaluation (standard deviation of intensity)
        contrast = float(np.std(gray))

        # 4. Glare evaluation (ratio of pixels saturated > 250)
        glare_pixels = np.sum(gray >= 250)
        glare_ratio = float(glare_pixels / (h * w))

        issues: List[str] = []
        if blur_score < self.min_blur_variance:
            issues.append(f"Image blurred (Laplacian variance {blur_score:.1f} < {self.min_blur_variance:.1f})")
        if brightness < self.min_brightness:
            issues.append(f"Image underexposed/too dark (mean luminance {brightness:.1f} < {self.min_brightness:.1f})")
        elif brightness > self.max_brightness:
            issues.append(f"Image overexposed/too bright (mean luminance {brightness:.1f} > {self.max_brightness:.1f})")
        if contrast < self.min_contrast:
            issues.append(f"Low image contrast (std dev {contrast:.1f} < {self.min_contrast:.1f})")
        if glare_ratio > self.max_glare_ratio:
            issues.append(f"High optical glare detected ({glare_ratio * 100:.1f}% pixels saturated)")

        if not issues:
            status = "EXCELLENT" if blur_score > (self.min_blur_variance * 2) else "ACCEPTABLE"
        elif len(issues) == 1 and blur_score >= (self.min_blur_variance * 0.75):
            status = "ACCEPTABLE"
        else:
            status = "POOR"

        return QualityAssessment(
            status=status,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            glare_ratio=round(glare_ratio, 4),
            width=w,
            height=h,
            issues=issues,
        )
