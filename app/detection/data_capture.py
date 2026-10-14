"""
Local Privacy-Conscious Data Capture & Candidate Mining Engine.
Provides opt-in local frame capture for continuous vehicle detector improvement.
Enforces strict privacy controls:
- Local storage only (never uploads externally).
- Configurable candidate capture interval and rate-limiting.
- Maximum disk storage budget with automatic FIFO cleanup.
- Optional license plate and face blurring on exported review frames.
- Perceptual hash deduplication (dHash) to prevent storing duplicate frames.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.infrastructure.paths import paths

logger = logging.getLogger("smart_parking.data_capture")


@dataclass
class CandidateFrameMetadata:
    frame_id: str
    timestamp: str
    reason: str
    confidence: Optional[float]
    class_name: Optional[str]
    suggested_bbox: Optional[Tuple[float, float, float, float]]
    image_rel_path: str
    review_status: str = "PENDING"  # PENDING, ACCEPTED, REJECTED
    reviewed_at: Optional[str] = None
    corrected_bbox: Optional[Tuple[float, float, float, float]] = None
    corrected_class: Optional[str] = None
    image_width: int = 1280
    image_height: int = 720
    dhash: int = 0


class DataCaptureManager:
    """Manages opt-in local frame recording, storage budgets, and hard-example mining."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_storage_mb: int = 250,
        min_interval_sec: float = 4.0,
        auto_capture_enabled: bool = False,
        anonymize_plates: bool = True,
    ):
        self.storage_dir = storage_dir or (paths.data_dir / "active_learning" / "candidates")
        self.images_dir = self.storage_dir / "images"
        self.metadata_file = self.storage_dir / "candidate_manifest.json"
        self.max_storage_mb = max_storage_mb
        self.min_interval_sec = min_interval_sec
        self.auto_capture_enabled = auto_capture_enabled
        self.anonymize_plates = anonymize_plates

        self._lock = threading.Lock()
        self._last_capture_ts: float = 0.0
        self._seen_dhashes: List[int] = []
        self._candidates: Dict[str, CandidateFrameMetadata] = {}

        self._init_storage()

    def _init_storage(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if self.metadata_file.exists():
            try:
                data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
                for item in data.get("candidates", []):
                    c = CandidateFrameMetadata(**item)
                    self._candidates[c.frame_id] = c
                    if c.dhash:
                        self._seen_dhashes.append(c.dhash)
            except Exception as e:
                logger.warning(f"Could not load candidate manifest: {e}")

    def _compute_dhash(self, image: np.ndarray, hash_size: int = 8) -> int:
        small = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        diff = gray[:, 1:] > gray[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def _hamming_distance(self, h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count("1")

    def is_perceptually_duplicate(self, dhash: int, threshold: int = 5) -> bool:
        """Checks if frame is visually identical to recent captures within threshold hamming distance."""
        for prev in self._seen_dhashes[-30:]:
            if self._hamming_distance(dhash, prev) <= threshold:
                return True
        return False

    def blur_sensitive_regions(
        self,
        frame: np.ndarray,
        plate_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """Applies Gaussian blur to license plate regions before saving for privacy preservation."""
        out = frame.copy()
        h, w = frame.shape[:2]

        if plate_bboxes:
            for (px1, py1, px2, py2) in plate_bboxes:
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(w, px2), min(h, py2)
                if px2 > px1 and py2 > py1:
                    crop = out[py1:py2, px1:px2]
                    blurred = cv2.GaussianBlur(crop, (25, 25), 30)
                    out[py1:py2, px1:px2] = blurred

        return out

    def capture_candidate_frame(
        self,
        frame: np.ndarray,
        reason: str,
        confidence: Optional[float] = None,
        class_name: Optional[str] = None,
        suggested_bbox: Optional[Tuple[float, float, float, float]] = None,
        plate_crops: Optional[List[Tuple[int, int, int, int]]] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Captures a high-value candidate frame into the local active learning queue:
        - Enforces minimum time interval unless force=True.
        - Enforces perceptual deduplication to avoid saving consecutive identical frames.
        - Enforces maximum disk storage budget with automated FIFO pruning.
        - Privacy-safe: optionally blurs plates and stores purely locally.
        """
        now = time.time()
        with self._lock:
            if not force:
                if (now - self._last_capture_ts) < self.min_interval_sec:
                    return False, f"Rate-limited: last capture was {now - self._last_capture_ts:.1f}s ago."

            dh = self._compute_dhash(frame)
            if not force and self.is_perceptually_duplicate(dh):
                return False, "Skipped: frame is visually duplicate to recently captured candidate."

            # Storage management: prune older candidates if budget exceeded
            self._enforce_storage_budget()

            # Generate frame ID and filename
            frame_id = f"cand_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._candidates):04d}"
            img_filename = f"{frame_id}.jpg"
            save_path = self.images_dir / img_filename

            # Anonymize if enabled
            save_frame = frame
            if self.anonymize_plates:
                save_frame = self.blur_sensitive_regions(frame, plate_crops)

            cv2.imwrite(str(save_path), save_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            h, w = frame.shape[:2]

            meta = CandidateFrameMetadata(
                frame_id=frame_id,
                timestamp=datetime.now().isoformat(),
                reason=reason,
                confidence=round(confidence, 3) if confidence is not None else None,
                class_name=class_name,
                suggested_bbox=suggested_bbox,
                image_rel_path=f"data/active_learning/candidates/images/{img_filename}",
                review_status="PENDING",
                image_width=w,
                image_height=h,
                dhash=dh,
            )

            self._candidates[frame_id] = meta
            self._seen_dhashes.append(dh)
            self._last_capture_ts = now
            self._save_manifest()

            logger.info(f"Captured hard-example candidate {frame_id} (Reason: {reason})")
            return True, frame_id

    def _enforce_storage_budget(self) -> None:
        """Removes oldest candidate images when disk usage exceeds max_storage_mb."""
        try:
            total_bytes = sum(f.stat().st_size for f in self.images_dir.glob("*.jpg"))
            max_bytes = self.max_storage_mb * 1024 * 1024
            if total_bytes > max_bytes:
                # Sort by modification time
                files = sorted(self.images_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
                while total_bytes > (max_bytes * 0.85) and files:
                    oldest = files.pop(0)
                    total_bytes -= oldest.stat().st_size
                    fid = oldest.stem
                    oldest.unlink(missing_ok=True)
                    if fid in self._candidates:
                        del self._candidates[fid]
                logger.info(f"Storage budget enforced: trimmed queue to {total_bytes / (1024*1024):.1f} MB.")
        except Exception as e:
            logger.warning(f"Storage enforcement error: {e}")

    def _save_manifest(self) -> None:
        data = {
            "updated_at": datetime.now().isoformat(),
            "total_candidates": len(self._candidates),
            "candidates": [asdict(c) for c in self._candidates.values()],
        }
        self.metadata_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_candidate_queue(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._candidates.values())
            if status:
                items = [c for c in items if c.review_status == status]
            items.sort(key=lambda c: c.timestamp, reverse=True)
            return [asdict(c) for c in items[:limit]]

    def review_candidate(
        self,
        frame_id: str,
        action: str,  # "ACCEPT", "REJECT", "CORRECT"
        corrected_bbox: Optional[Tuple[float, float, float, float]] = None,
        corrected_class: Optional[str] = None,
    ) -> bool:
        """Updates review status of candidate for continuous learning dataset promotion."""
        with self._lock:
            if frame_id not in self._candidates:
                return False
            c = self._candidates[frame_id]
            c.review_status = "ACCEPTED" if action == "ACCEPT" else ("REJECTED" if action == "REJECT" else "CORRECTED")
            c.reviewed_at = datetime.now().isoformat()
            if corrected_bbox:
                c.corrected_bbox = corrected_bbox
            if corrected_class:
                c.corrected_class = corrected_class
            self._save_manifest()
            return True


# Global manager instance
data_capture_manager = DataCaptureManager()
