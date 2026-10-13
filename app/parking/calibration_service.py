"""Calibration management service with revision history and validation."""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.database.connection import DatabaseManager
from app.infrastructure.paths import paths
from app.parking.geometry import validate_slot_polygon, check_slot_overlaps

logger = logging.getLogger("smart_parking.calibration")


class CalibrationService:
    """Manages parking bay calibration files, revisions, and geometry validation."""

    def __init__(self, calib_file_path: str = "data/calibration.json", db_manager: Optional[DatabaseManager] = None):
        self.calib_path = paths.resolve(calib_file_path)
        self.db = db_manager

    def load_calibration(self) -> Dict[str, Any]:
        """Loads and returns current active calibration data."""
        if not self.calib_path.exists():
            return {
                "version": 1,
                "reference_width": 1280,
                "reference_height": 720,
                "camera_fingerprint": "default",
                "created_at": datetime.now().isoformat(),
                "slots": [],
            }
        try:
            return json.loads(self.calib_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read calibration JSON: {e}")
            raise

    def save_calibration(
        self,
        slots: List[Dict[str, Any]],
        ref_width: int = 1280,
        ref_height: int = 720,
        fingerprint: Optional[str] = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validates slots, saves to calibration.json, and records a revision entry in SQLite.
        Returns: (success: bool, validation_warnings: List[str], saved_payload: Dict[str, Any])
        """
        errors = []
        # 1. Validate individual polygons
        seen_ids = set()
        for idx, s in enumerate(slots):
            sid = s.get("id", f"S{idx+1}")
            if sid in seen_ids:
                errors.append(f"Duplicate Slot ID '{sid}' declared.")
            seen_ids.add(sid)

            pts = s.get("points", [])
            valid, err = validate_slot_polygon(pts)
            if not valid:
                errors.append(f"Slot {sid}: {err}")

        if errors:
            raise ValueError(f"Calibration validation failed: {'; '.join(errors)}")

        # 2. Check for overlapping polygons
        overlap_warnings = check_slot_overlaps(slots)

        # 3. Determine revision version
        current_data = self.load_calibration()
        new_version = current_data.get("version", 1) + 1

        payload = {
            "version": new_version,
            "reference_width": ref_width,
            "reference_height": ref_height,
            "camera_fingerprint": fingerprint or "zebronics_uvc",
            "updated_at": datetime.now().isoformat(),
            "slots": slots,
        }

        # Save to file
        self.calib_path.parent.mkdir(parents=True, exist_ok=True)
        self.calib_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Save to database revision history if DB available
        if self.db:
            try:
                with self.db.get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO calibration_revisions (version, reference_width, reference_height, camera_fingerprint, slots_json)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (new_version, ref_width, ref_height, payload["camera_fingerprint"], json.dumps(slots)),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to record calibration revision in DB: {e}")

        logger.info(f"Calibration saved (version {new_version}, {len(slots)} bays). Warnings: {len(overlap_warnings)}")
        return True, overlap_warnings, payload

    def get_revision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves calibration revision history from SQLite."""
        if not self.db:
            return []
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, version, reference_width, reference_height, camera_fingerprint, created_at FROM calibration_revisions ORDER BY id DESC LIMIT ?;",
                    (limit,),
                )
                return [dict(r) for r in cursor.fetchall()]
        except Exception:
            return []
