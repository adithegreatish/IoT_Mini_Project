"""Database repository for parking events, calibrations, and diagnostics."""

import json
import logging
from typing import Any, Dict, List, Optional
from app.database.connection import DatabaseManager

logger = logging.getLogger("smart_parking.repository")


class ParkingRepository:
    """Thread-safe persistence layer for events and calibration revisions."""

    def __init__(self, db_manager: DatabaseManager, allow_raw_retention: bool = False):
        self.db = db_manager
        self.allow_raw_retention = allow_raw_retention

    def mask_plate(self, plate: str) -> str:
        """Masks middle characters of license plate for privacy compliance."""
        clean = plate.replace(" ", "").upper()
        if len(clean) >= 8:
            return f"{clean[:4]}****{clean[-2:]}"
        elif len(clean) >= 4:
            return f"{clean[:2]}**{clean[-2:]}"
        return "****"

    def record_occupancy_event(
        self,
        slot_id: str,
        previous_state: str,
        new_state: str,
        confidence: float,
        track_id: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> int:
        """Records a slot state transition event. Skips identical redundant transitions."""
        if previous_state == new_state:
            return 0

        query = """
        INSERT INTO occupancy_events (slot_id, previous_state, new_state, confidence, track_id, model_version)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (slot_id, previous_state, new_state, float(confidence), track_id, model_version),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def record_plate_event(
        self,
        slot_id: Optional[str],
        raw_plate: str,
        confidence: float,
        crop_quality: str,
        hsrp_outcome: str,
        visible_cues: Optional[Dict[str, Any]] = None,
        ocr_engine: str = "mock",
    ) -> int:
        """Records an OCR plate detection event with privacy masking."""
        masked = self.mask_plate(raw_plate)
        stored_raw = raw_plate if self.allow_raw_retention else None
        cues_json = json.dumps(visible_cues or {})

        query = """
        INSERT INTO plate_events (slot_id, raw_plate, masked_plate, confidence, crop_quality, hsrp_outcome, visible_cues_json, ocr_engine)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (slot_id, stored_raw, masked, float(confidence), crop_quality, hsrp_outcome, cues_json, ocr_engine),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def record_diagnostic_run(
        self,
        run_type: str,
        status: str,
        duration_sec: float,
        total_tests: int,
        passed_tests: int,
        failed_tests: int,
        report_path: Optional[str] = None,
    ) -> int:
        """Logs diagnostic test execution run."""
        query = """
        INSERT INTO diagnostic_runs (run_type, status, duration_sec, total_tests, passed_tests, failed_tests, report_path)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (run_type, status, float(duration_sec), total_tests, passed_tests, failed_tests, report_path),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_recent_occupancy_events(self, limit: int = 50, slot_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries recent occupancy state change events."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if slot_id:
                cursor.execute(
                    "SELECT * FROM occupancy_events WHERE slot_id = ? ORDER BY id DESC LIMIT ?;",
                    (slot_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM occupancy_events ORDER BY id DESC LIMIT ?;",
                    (limit,),
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_recent_plate_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Queries recent license plate detection events."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plate_events ORDER BY id DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                # Parse cues JSON if present
                if item.get("visible_cues_json"):
                    try:
                        item["visible_cues"] = json.loads(item["visible_cues_json"])
                    except Exception:
                        item["visible_cues"] = {}
                results.append(item)
            return results

    def prune_expired_records(self, retention_days: int) -> int:
        """Purges event logs older than retention_days threshold."""
        query_occ = f"DELETE FROM occupancy_events WHERE timestamp < datetime('now', '-{retention_days} days');"
        query_plt = f"DELETE FROM plate_events WHERE timestamp < datetime('now', '-{retention_days} days');"
        query_diag = f"DELETE FROM diagnostic_runs WHERE timestamp < datetime('now', '-{retention_days} days');"

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query_occ)
            c1 = cursor.rowcount
            cursor.execute(query_plt)
            c2 = cursor.rowcount
            cursor.execute(query_diag)
            c3 = cursor.rowcount
            conn.commit()
            total = c1 + c2 + c3
            if total > 0:
                logger.info(f"Pruned {total} expired records older than {retention_days} days.")
            return total
