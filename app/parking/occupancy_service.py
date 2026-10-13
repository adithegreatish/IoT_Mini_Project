"""Parking occupancy service orchestrating detection, association, and state transitions."""

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.config.schema import ParkingConfig
from app.database.repository import ParkingRepository
from app.detection.vehicle_detector import DetectionResult, VehicleDetector
from app.parking.calibration_service import CalibrationService
from app.parking.geometry import calculate_overlap_ratio, denormalize_points
from app.parking.state_machine import SlotState, SlotStateMachine

logger = logging.getLogger("smart_parking.occupancy")


@dataclass
class SlotStatusView:
    id: str
    label: str
    slot_type: str
    state: SlotState
    confidence: float
    reason: str
    points: List[List[float]]
    associated_vehicle: Optional[str] = None


class OccupancyService:
    """Authoritative parking occupancy engine managing bay states and spatial matching."""

    def __init__(
        self,
        config: ParkingConfig,
        calibration_service: CalibrationService,
        detector: Optional[VehicleDetector] = None,
        repository: Optional[ParkingRepository] = None,
        classifier: Optional[Any] = None,
    ):
        self.config = config
        self.calib_service = calibration_service
        self.detector = detector
        self.repo = repository
        self._lock = threading.Lock()

        # Patch classifier support for CPU / Pi edge inference
        self.classifier = classifier
        if self.classifier is None and getattr(config, "inference_mode", "detector") in ("classifier", "hybrid"):
            try:
                from app.parking.occupancy_classifier import PatchOccupancyClassifier
                self.classifier = PatchOccupancyClassifier(
                    model_path=getattr(config, "classifier_model_path", "models/occupancy_classifier.onnx"),
                    threshold=getattr(config, "classifier_threshold", 0.65),
                )
            except Exception as e:
                logger.warning(f"Could not initialize PatchOccupancyClassifier: {e}. Falling back to detector mode.")
                self.classifier = None

        self._state_machines: Dict[str, SlotStateMachine] = {}
        self._slots_def: List[Dict[str, Any]] = []
        self._latest_snapshot: Dict[str, Any] = {}
        self._last_detections: List[DetectionResult] = []

        self.reload_slots()

    def reload_slots(self) -> None:
        """Reloads calibrated slot definitions and initializes state machines."""
        with self._lock:
            calib = self.calib_service.load_calibration()
            self._slots_def = calib.get("slots", [])
            self._state_machines = {
                s["id"]: SlotStateMachine(
                    slot_id=s["id"],
                    smoothing_window=self.config.smoothing_window_frames,
                    hysteresis_confirm=self.config.hysteresis_confirm_frames,
                    unknown_timeout_sec=self.config.unknown_timeout_sec,
                )
                for s in self._slots_def
                if s.get("enabled", True)
            }
            logger.info(f"Loaded {len(self._state_machines)} active parking slots into occupancy engine.")

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        is_camera_healthy: bool = True,
        is_camera_stale: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end occupancy assessment on a single video frame.
        Guarantees that stale or absent camera frames mark bays UNKNOWN, NEVER FREE.
        """
        with self._lock:
            now_iso = datetime.now().isoformat()

            # 1. Camera Disconnect / Stale Protection
            if not is_camera_healthy or is_camera_stale or frame is None:
                slot_views = []
                for s in self._slots_def:
                    sid = s["id"]
                    sm = self._state_machines.get(sid)
                    if sm:
                        _, changed = sm.mark_camera_stale_or_lost("Camera stream offline or stale")
                        if changed and self.repo:
                            self.repo.record_occupancy_event(
                                slot_id=sid,
                                previous_state=SlotState.OCCUPIED.value,
                                new_state=SlotState.UNKNOWN.value,
                                confidence=0.0,
                            )
                        slot_views.append({
                            "id": sid,
                            "label": s.get("label", sid),
                            "slot_type": s.get("slot_type", "car"),
                            "state": SlotState.UNKNOWN.value,
                            "confidence": 0.0,
                            "reason": sm.reason,
                            "points": s.get("points", []),
                        })

                self._latest_snapshot = {
                    "timestamp": now_iso,
                    "summary": {"total": len(slot_views), "free": 0, "occupied": 0, "unknown": len(slot_views)},
                    "slots": slot_views,
                    "is_camera_stale": True,
                }
                return self._latest_snapshot

            # 2. Run Inference according to inference_mode
            mode = getattr(self.config, "inference_mode", "detector")

            detections = []
            if mode in ("detector", "hybrid") and self.detector and getattr(self.detector, "enabled", True):
                detections = self.detector.detect(frame)
            self._last_detections = detections

            slot_views = []
            occupied_count = 0
            free_count = 0
            unknown_count = 0

            # Batch extract patches if in classifier or hybrid mode
            patch_results = {}
            if mode in ("classifier", "hybrid") and self.classifier and frame is not None:
                if np.mean(frame) > 5.0:
                    for s in self._slots_def:
                        pts = s.get("points", [])
                        patch = self.classifier.extract_patch_from_frame(frame, pts)
                        if patch is not None and patch.size > 0:
                            p_occ, p_conf = self.classifier.predict_patch(patch)
                            patch_results[s["id"]] = (p_occ, p_conf)

            # 3. Spatial Slot-to-Vehicle Association & Occupancy Classification
            for s in self._slots_def:
                sid = s["id"]
                sm = self._state_machines.get(sid)
                if not sm:
                    continue

                pts = s.get("points", [])
                best_overlap = 0.0
                associated_class = None

                for det in detections:
                    overlap = calculate_overlap_ratio(pts, det.bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        associated_class = det.class_name

                det_occupied = best_overlap >= self.config.occupancy_threshold
                det_conf = round(best_overlap if det_occupied else (1.0 - best_overlap), 2)

                if mode == "classifier" and sid in patch_results:
                    is_occupied_obs, obs_conf = patch_results[sid]
                elif mode == "hybrid" and sid in patch_results:
                    patch_occ, patch_conf = patch_results[sid]
                    # In hybrid mode: bay is occupied if either patch classifier or vehicle detector detects a car
                    is_occupied_obs = patch_occ or det_occupied
                    obs_conf = max(patch_conf, det_conf) if is_occupied_obs else min(patch_conf, det_conf)
                else:
                    is_occupied_obs = det_occupied
                    obs_conf = det_conf

                prev_state = sm.stable_state
                curr_state, changed = sm.update(is_occupied_obs, obs_conf)

                # Persist state transition to SQLite if state changed
                if changed and self.repo:
                    self.repo.record_occupancy_event(
                        slot_id=sid,
                        previous_state=prev_state.value,
                        new_state=curr_state.value,
                        confidence=obs_conf,
                        track_id=associated_class,
                    )

                if curr_state == SlotState.FREE:
                    free_count += 1
                elif curr_state == SlotState.OCCUPIED:
                    occupied_count += 1
                else:
                    unknown_count += 1

                slot_views.append({
                    "id": sid,
                    "label": s.get("label", sid),
                    "slot_type": s.get("slot_type", "car"),
                    "state": curr_state.value,
                    "confidence": sm.confidence,
                    "reason": sm.reason,
                    "overlap_ratio": round(best_overlap, 3),
                    "associated_vehicle": associated_class if is_occupied_obs else None,
                    "points": pts,
                })

            self._latest_snapshot = {
                "timestamp": now_iso,
                "summary": {
                    "total": len(slot_views),
                    "free": free_count,
                    "occupied": occupied_count,
                    "unknown": unknown_count,
                },
                "slots": slot_views,
                "is_camera_stale": False,
            }
            return self._latest_snapshot

    def get_latest_snapshot(self) -> Dict[str, Any]:
        """Returns the most recent calculated occupancy snapshot."""
        with self._lock:
            return self._latest_snapshot

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Renders parking slot polygons, states, and detection bounding boxes onto frame."""
        with self._lock:
            out = frame.copy()
            h, w = out.shape[:2]

            # 1. Draw Calibrated Slots with Color-Coded State
            for slot in self._latest_snapshot.get("slots", []):
                pts = slot.get("points", [])
                if len(pts) < 3:
                    continue
                pixel_pts = np.array(denormalize_points(pts, w, h), dtype=np.int32)

                state = slot.get("state", "UNKNOWN")
                if state == "FREE":
                    color = (0, 200, 100)  # Green
                elif state == "OCCUPIED":
                    color = (40, 40, 230)  # Red
                else:
                    color = (0, 165, 255)  # Amber

                # Transparent polygon fill
                overlay = out.copy()
                cv2.fillPoly(overlay, [pixel_pts], color)
                cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

                # Polygon boundary line
                cv2.polylines(out, [pixel_pts], isClosed=True, color=color, thickness=2)

                # Text label
                tx, ty = pixel_pts[0]
                label_text = f"{slot['id']}: {state}"
                cv2.putText(out, label_text, (tx + 5, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # 2. Draw Detected Vehicle Bounding Boxes
            for det in self._last_detections:
                bx1, by1, bx2, by2 = det.pixel_box
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
                cv2.putText(
                    out,
                    f"{det.class_name} {det.confidence:.2f}",
                    (bx1 + 4, max(20, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 200, 0),
                    1,
                )

            return out
