"""Parking occupancy service orchestrating detection, association, and state transitions."""

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.config.schema import ParkingConfig
from app.database.repository import ParkingRepository
from app.detection.vehicle_detector import DetectionResult, VehicleDetector
from app.parking.calibration_service import CalibrationService
from app.parking.geometry import calculate_overlap_ratio, calculate_spatial_association_score, denormalize_points
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
        """Reloads calibrated slot definitions, initializes state machines, and seeds initial snapshot."""
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

            # Seed initial snapshot immediately so GET /api/slots reflects new calibration without waiting
            now_iso = datetime.now().isoformat()
            seeded_views = []
            for s in self._slots_def:
                is_enabled = s.get("enabled", True)
                seeded_views.append({
                    "id": s["id"],
                    "label": s.get("label", s["id"]),
                    "zone": s.get("zone", "Zone-A"),
                    "slot_type": s.get("slot_type", "car"),
                    "enabled": is_enabled,
                    "state": SlotState.UNKNOWN.value if is_enabled else "DISABLED",
                    "confidence": 0.0,
                    "reason": "Initialized from calibration" if is_enabled else "Slot disabled in calibration",
                    "points": s.get("points", []),
                })
            self._latest_snapshot = {
                "timestamp": now_iso,
                "summary": {
                    "total": len(seeded_views),
                    "free": 0,
                    "occupied": 0,
                    "unknown": len([v for v in seeded_views if v["state"] == SlotState.UNKNOWN.value]),
                },
                "slots": seeded_views,
                "is_camera_stale": False,
            }

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
            t_frame_start = time.perf_counter()
            now_iso = datetime.now().isoformat()

            # 1. Camera Disconnect / Stale Protection
            if not is_camera_healthy or is_camera_stale or frame is None:
                try:
                    from app.infrastructure.telemetry import telemetry
                    telemetry.record_stale_frame()
                except Exception:
                    pass
                slot_views = []
                for s in self._slots_def:
                    sid = s["id"]
                    is_enabled = s.get("enabled", True)
                    if not is_enabled:
                        slot_views.append({
                            "id": sid,
                            "label": s.get("label", sid),
                            "zone": s.get("zone", "Zone-A"),
                            "slot_type": s.get("slot_type", "car"),
                            "enabled": False,
                            "state": "DISABLED",
                            "confidence": 0.0,
                            "reason": "Slot disabled in calibration",
                            "points": s.get("points", []),
                        })
                        continue

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
                        "zone": s.get("zone", "Zone-A"),
                        "slot_type": s.get("slot_type", "car"),
                        "enabled": is_enabled,
                        "state": SlotState.UNKNOWN.value,
                        "confidence": 0.0,
                        "reason": sm.reason if sm else "Camera stream offline",
                        "points": s.get("points", []),
                    })

                self._latest_snapshot = {
                    "timestamp": now_iso,
                    "summary": {
                        "total": len(slot_views),
                        "free": 0,
                        "occupied": 0,
                        "unknown": len([v for v in slot_views if v["state"] == SlotState.UNKNOWN.value]),
                    },
                    "slots": slot_views,
                    "is_camera_stale": True,
                }
                return self._latest_snapshot

            # 2. Run Inference according to inference_mode
            mode = getattr(self.config, "inference_mode", "detector")

            detections = []
            inference_error = None
            if mode in ("detector", "hybrid") and self.detector and getattr(self.detector, "enabled", True):
                if hasattr(self.detector, "is_healthy") and not self.detector.is_healthy:
                    inference_error = f"Vehicle detector unhealthy ({getattr(self.detector, 'status', 'UNKNOWN')})"
                    logger.warning(f"Inference bypassed: {inference_error}")
                else:
                    try:
                        detections = self.detector.detect(frame)
                    except Exception as exc:
                        inference_error = str(exc)
                        logger.error(f"Inference failure encountered in OccupancyService: {exc}")

            self._last_detections = detections if inference_error is None else []

            # If detection subsystem failed, do NOT mark parking spaces FREE! Mark UNKNOWN.
            if inference_error:
                slot_views = []
                for s in self._slots_def:
                    sid = s["id"]
                    is_enabled = s.get("enabled", True)
                    sm = self._state_machines.get(sid)
                    if sm and is_enabled:
                        sm.mark_camera_stale_or_lost(f"Detector failure: {inference_error}")
                    slot_views.append({
                        "id": sid,
                        "slot_id": sid,
                        "label": s.get("label", sid),
                        "zone": s.get("zone", "Zone-A"),
                        "slot_type": s.get("slot_type", "car"),
                        "enabled": is_enabled,
                        "state": SlotState.UNKNOWN.value if is_enabled else "DISABLED",
                        "confidence": 0.0,
                        "reason": f"Detector error: {inference_error}" if is_enabled else "Slot disabled in calibration",
                        "overlap_ratio": 0.0,
                        "associated_vehicle": None,
                        "points": s.get("points", []),
                    })

                self._latest_snapshot = {
                    "timestamp": now_iso,
                    "summary": {
                        "total": len(slot_views),
                        "free": 0,
                        "occupied": 0,
                        "unknown": len([v for v in slot_views if v["state"] == SlotState.UNKNOWN.value]),
                    },
                    "slots": slot_views,
                    "is_camera_stale": False,
                    "detector_error": inference_error,
                }
                return self._latest_snapshot

            slot_views = []
            occupied_count = 0
            free_count = 0
            unknown_count = 0

            # Batch extract patches if in classifier or hybrid mode
            patch_results = {}
            if mode in ("classifier", "hybrid") and self.classifier and frame is not None:
                t_clf_0 = time.perf_counter()
                if np.mean(frame) > 5.0:
                    for s in self._slots_def:
                        pts = s.get("points", [])
                        patch = self.classifier.extract_patch_from_frame(frame, pts)
                        if patch is not None and patch.size > 0:
                            p_occ, p_conf = self.classifier.predict_patch(patch)
                            patch_results[s["id"]] = (p_occ, p_conf)
                t_clf_1 = time.perf_counter()
                try:
                    from app.infrastructure.telemetry import telemetry
                    telemetry.record_patch_classification((t_clf_1 - t_clf_0) * 1000.0)
                except Exception:
                    pass

            t_assoc_0 = time.perf_counter()
            total_db_ms = 0.0
            # 3. Spatial Slot-to-Vehicle Association & Occupancy Classification
            for s in self._slots_def:
                sid = s["id"]
                is_enabled = s.get("enabled", True)
                if not is_enabled:
                    slot_views.append({
                        "id": sid,
                        "slot_id": sid,
                        "label": s.get("label", sid),
                        "zone": s.get("zone", "Zone-A"),
                        "slot_type": s.get("slot_type", "car"),
                        "enabled": False,
                        "state": "DISABLED",
                        "confidence": 0.0,
                        "reason": "Slot disabled in calibration",
                        "overlap_ratio": 0.0,
                        "associated_vehicle": None,
                        "points": s.get("points", []),
                    })
                    continue

                sm = self._state_machines.get(sid)
                if not sm:
                    continue

                pts = s.get("points", [])
                best_overlap = 0.0
                associated_class = None

                # Log each detection for diagnostics
                for det in detections:
                    logger.info(
                        f"Detection: class={det.class_name}, confidence={det.confidence:.3f}, bbox={det.bbox}, pixel_box={det.pixel_box}"
                    )
                    score, _ = calculate_spatial_association_score(pts, det.bbox, method="hybrid")
                    if score > best_overlap:
                        best_overlap = score
                        associated_class = det.class_name

                # Determine occupancy based on overlap threshold
                det_occupied = best_overlap >= self.config.occupancy_threshold
                det_conf = round(best_overlap if det_occupied else (1.0 - best_overlap), 2)

                # Bottom-center inside slot check
                bottom_center_inside = False
                if detections:
                    best_det = max(detections, key=lambda d: calculate_spatial_association_score(pts, d.bbox, method="hybrid")[0])
                    bc_x = (best_det.bbox[0] + best_det.bbox[2]) / 2.0
                    bc_y = best_det.bbox[3]
                    bottom_center_inside = is_point_in_slot(pts, (bc_x, bc_y))

                # Combine results, enforcing bottom-center inside requirement
                if mode == "classifier" and sid in patch_results:
                    is_occupied_obs, obs_conf = patch_results[sid]
                elif mode == "hybrid" and sid in patch_results:
                    patch_occ, patch_conf = patch_results[sid]
                    is_occupied_obs = (patch_occ or det_occupied) and bottom_center_inside
                    obs_conf = max(patch_conf, det_conf) if is_occupied_obs else min(patch_conf, det_conf)
                else:
                    is_occupied_obs = det_occupied and bottom_center_inside
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
                    "slot_id": sid,
                    "label": s.get("label", sid),
                    "zone": s.get("zone", "Zone-A"),
                    "slot_type": s.get("slot_type", "car"),
                    "enabled": True,
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

            t_frame_end = time.perf_counter()
            try:
                from app.infrastructure.telemetry import telemetry
                telemetry.record_occupancy_update(
                    e2e_ms=(t_frame_end - t_frame_start) * 1000.0,
                    assoc_ms=(t_frame_end - t_assoc_0) * 1000.0,
                    smooth_ms=0.0,
                    db_ms=total_db_ms,
                )
            except Exception:
                pass

            return self._latest_snapshot

    def get_latest_snapshot(self) -> Dict[str, Any]:
        """Returns the most recent calculated occupancy snapshot."""
        with self._lock:
            return self._latest_snapshot

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Renders parking slot polygons, states, and detection bounding boxes onto frame efficiently."""
        with self._lock:
            slots = self._latest_snapshot.get("slots", [])
            detections = self._last_detections

            if not slots and not detections:
                return frame

            out = frame.copy()
            h, w = out.shape[:2]

            # 1. Single-pass transparent polygon fill for all bays combined
            if slots:
                overlay = out.copy()
                has_polys = False
                for slot in slots:
                    pts = slot.get("points", [])
                    if len(pts) < 3:
                        continue
                    has_polys = True
                    pixel_pts = np.array(denormalize_points(pts, w, h), dtype=np.int32)
                    state = slot.get("state", "UNKNOWN")
                    color = (0, 200, 100) if state == "FREE" else ((40, 40, 230) if state == "OCCUPIED" else (0, 165, 255))
                    cv2.fillPoly(overlay, [pixel_pts], color)

                if has_polys:
                    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

                # Draw outlines and labels directly on out without copying
                for slot in slots:
                    pts = slot.get("points", [])
                    if len(pts) < 3:
                        continue
                    pixel_pts = np.array(denormalize_points(pts, w, h), dtype=np.int32)
                    state = slot.get("state", "UNKNOWN")
                    color = (0, 200, 100) if state == "FREE" else ((40, 40, 230) if state == "OCCUPIED" else (0, 165, 255))
                    cv2.polylines(out, [pixel_pts], isClosed=True, color=color, thickness=2)

                    tx, ty = pixel_pts[0]
                    cv2.putText(out, f"{slot['id']}: {state}", (tx + 5, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # 2. Draw Detected Vehicle Bounding Boxes
            for det in detections:
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
