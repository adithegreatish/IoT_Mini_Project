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
from app.parking.geometry import (
    calculate_overlap_ratio,
    calculate_spatial_association_score,
    denormalize_points,
    is_point_in_slot,
)
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
        self._last_diag_log_time: float = 0.0
        self._diag_interval_sec: float = 5.0

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
            if mode in ("detector", "hybrid") and self.detector:
                if not getattr(self.detector, "enabled", True):
                    inference_error = "Vehicle detector is disabled"
                    logger.warning(f"Inference bypassed: {inference_error}")
                elif hasattr(self.detector, "is_healthy") and not self.detector.is_healthy:
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

            # Rate-limited diagnostic logging to prevent terminal/log flooding
            t_now = time.time()
            do_diag_log = (t_now - self._last_diag_log_time >= self._diag_interval_sec) or logger.isEnabledFor(logging.DEBUG)
            if do_diag_log:
                self._last_diag_log_time = t_now
                logger.info(
                    f"[OccupancyDiag] Active bays: {len(self._state_machines)} | Detections: {len(detections)}"
                )
                for det_i, det in enumerate(detections):
                    bc_x = round((det.bbox[0] + det.bbox[2]) / 2.0, 3)
                    bc_y = round(det.bbox[3], 3)
                    logger.info(
                        f"  [Det #{det_i+1}] class={det.class_name} conf={det.confidence:.2f} bbox={det.bbox} bc=({bc_x}, {bc_y})"
                    )

            # Spatial Candidate Matching:
            # Match vehicle detections to bays without assigning one detection to multiple bays.
            candidates = []
            for det_idx, det in enumerate(detections):
                bc_x = (det.bbox[0] + det.bbox[2]) / 2.0
                bc_y = det.bbox[3]
                for s in self._slots_def:
                    if not s.get("enabled", True):
                        continue
                    sid = s["id"]
                    pts = s.get("points", [])
                    if len(pts) < 3:
                        continue
                    score, reason = calculate_spatial_association_score(pts, det.bbox, method="hybrid")
                    bc_inside = is_point_in_slot(pts, (bc_x, bc_y))
                    if score > 0.0 or bc_inside:
                        candidates.append({
                            "det_idx": det_idx,
                            "slot_id": sid,
                            "score": score,
                            "bc_inside": bc_inside,
                            "det": det,
                            "reason": reason,
                        })

            # Sort candidate matches: prioritize bottom-center inside, then highest score
            candidates.sort(key=lambda c: (1 if c["bc_inside"] else 0, c["score"]), reverse=True)

            # Greedy 1-to-1 matching: 1 detection -> at most 1 bay, 1 bay -> at most 1 detection
            matched_slot_detections = {}
            used_det_indices = set()
            for cand in candidates:
                sid = cand["slot_id"]
                d_idx = cand["det_idx"]
                if sid not in matched_slot_detections and d_idx not in used_det_indices:
                    if cand["score"] >= self.config.occupancy_threshold and cand["bc_inside"]:
                        matched_slot_detections[sid] = cand
                        used_det_indices.add(d_idx)

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
                match = matched_slot_detections.get(sid)

                if match:
                    best_overlap = match["score"]
                    matched_vehicle_class = match["det"].class_name
                    matched_conf = match["det"].confidence
                    bottom_center_inside = True
                    det_occupied = True
                    det_conf = round(best_overlap, 2)
                else:
                    best_overlap = 0.0
                    matched_vehicle_class = None
                    matched_conf = 0.0
                    bottom_center_inside = False
                    for det in detections:
                        score, _ = calculate_spatial_association_score(pts, det.bbox, method="hybrid")
                        if score > best_overlap:
                            best_overlap = score
                    det_occupied = False
                    det_conf = round(1.0 - best_overlap, 2)

                # Combine results, supporting detector, hybrid, and patch classifier modes
                if mode == "classifier" and sid in patch_results:
                    is_occupied_obs, obs_conf = patch_results[sid]
                    slot_assoc_vehicle = "vehicle" if is_occupied_obs else None
                elif mode == "hybrid":
                    # Hybrid Mode: Occupancy is strictly grounded in valid vehicle detector observations.
                    # A bay CANNOT be marked occupied without a valid matched vehicle detection.
                    if det_occupied and match:
                        is_occupied_obs = True
                        slot_assoc_vehicle = matched_vehicle_class
                        if sid in patch_results:
                            patch_occ, patch_conf = patch_results[sid]
                            obs_conf = max(det_conf, patch_conf) if patch_occ else round(det_conf * 0.9, 2)
                        else:
                            obs_conf = det_conf
                    else:
                        is_occupied_obs = False
                        slot_assoc_vehicle = None
                        if sid in patch_results:
                            patch_occ, patch_conf = patch_results[sid]
                            obs_conf = patch_conf if not patch_occ else det_conf
                        else:
                            obs_conf = det_conf
                else:
                    # Detector mode (standard)
                    is_occupied_obs = det_occupied
                    obs_conf = det_conf
                    slot_assoc_vehicle = matched_vehicle_class if det_occupied else None

                prev_state = sm.stable_state
                curr_state, changed = sm.update(is_occupied_obs, obs_conf)

                if do_diag_log:
                    if match:
                        logger.info(
                            f"  [Bay {sid}] MATCHED class={matched_vehicle_class} conf={matched_conf:.2f} score={best_overlap:.2f} -> obs={'OCCUPIED' if is_occupied_obs else 'FREE'} stable={curr_state.value}"
                        )
                    else:
                        logger.info(
                            f"  [Bay {sid}] VACANT (best_overlap={best_overlap:.2f}) -> obs={'OCCUPIED' if is_occupied_obs else 'FREE'} stable={curr_state.value}"
                        )

                # Persist state transition to SQLite if state changed
                if changed and self.repo:
                    self.repo.record_occupancy_event(
                        slot_id=sid,
                        previous_state=prev_state.value,
                        new_state=curr_state.value,
                        confidence=obs_conf,
                        track_id=slot_assoc_vehicle,
                    )

                if curr_state == SlotState.FREE:
                    free_count += 1
                elif curr_state == SlotState.OCCUPIED:
                    occupied_count += 1
                else:
                    unknown_count += 1

                # If slot is stable OCCUPIED, associate with the matched vehicle
                effective_assoc_vehicle = None
                if curr_state == SlotState.OCCUPIED:
                    effective_assoc_vehicle = slot_assoc_vehicle or getattr(sm, "_last_vehicle_class", "car")
                    sm._last_vehicle_class = effective_assoc_vehicle
                else:
                    sm._last_vehicle_class = None

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
                    "associated_vehicle": effective_assoc_vehicle,
                    "points": pts,
                })

            active_count = free_count + occupied_count + unknown_count
            self._latest_snapshot = {
                "timestamp": now_iso,
                "summary": {
                    "total": len(slot_views),
                    "active": active_count,
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

    def render_overlay(
        self,
        frame: np.ndarray,
        slots: Optional[List[Dict[str, Any]]] = None,
        detections: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """Renders parking slot polygons, states, and detection bounding boxes onto frame efficiently."""
        with self._lock:
            if slots is None:
                slots = self._latest_snapshot.get("slots", [])
            if detections is None:
                detections = self._last_detections

            if not slots and not detections:
                return frame

            out = frame.copy()
            h, w = out.shape[:2]

            # 1. Restrained transparent polygon fill (alpha = 0.12) to avoid saturating camera footage
            if slots:
                overlay = out.copy()
                has_polys = False
                color_map = {
                    "FREE": (46, 175, 78),          # Refined Emerald Green (BGR)
                    "OCCUPIED": (60, 60, 220),       # Refined Crimson Red (BGR)
                    "RESERVED": (30, 150, 240),      # Refined Amber/Gold (BGR)
                    "UNKNOWN": (140, 140, 140),      # Neutral Slate Gray (BGR)
                    "DISABLED": (80, 80, 80),        # Muted Charcoal (BGR)
                }

                for slot in slots:
                    pts = slot.get("points", [])
                    if len(pts) < 3:
                        continue
                    has_polys = True
                    pixel_pts = np.array(denormalize_points(pts, w, h), dtype=np.int32)
                    state = slot.get("state", "UNKNOWN")
                    color = color_map.get(state, (140, 140, 140))
                    cv2.fillPoly(overlay, [pixel_pts], color)

                if has_polys:
                    cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

                # 2. Draw thin crisp polygon borders and high-contrast label pills
                for slot in slots:
                    pts = slot.get("points", [])
                    if len(pts) < 3:
                        continue
                    pixel_pts = np.array(denormalize_points(pts, w, h), dtype=np.int32)
                    state = slot.get("state", "UNKNOWN")
                    color = color_map.get(state, (140, 140, 140))

                    # Thin, crisp polygon outline
                    cv2.polylines(out, [pixel_pts], isClosed=True, color=color, thickness=2)

                    # Compute anchor point for label pill (top-left bounding area)
                    min_x = min(p[0] for p in pixel_pts)
                    min_y = min(p[1] for p in pixel_pts)
                    tx = max(4, min(w - 100, min_x + 4))
                    ty = max(16, min(h - 6, min_y + 16))

                    label_text = f"{slot['id']} · {state}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.40
                    font_thickness = 1

                    (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                    bg_x1 = max(0, tx - 3)
                    bg_y1 = max(0, ty - txt_h - 4)
                    bg_x2 = min(w - 1, tx + txt_w + 4)
                    bg_y2 = min(h - 1, ty + baseline + 2)

                    # Pill background in dark ink with subtle colored border
                    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (12, 16, 24), -1)
                    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

                    # Pill text
                    cv2.putText(out, label_text, (tx, ty), font, font_scale, (240, 245, 250), font_thickness, cv2.LINE_AA)

            # 3. Draw Detected Vehicle Bounding Boxes and Ground Contact Points
            for det in detections:
                bx1, by1, bx2, by2 = det.pixel_box
                # Crisp thin bounding box (cyan/ice-yellow) distinct from parking bays
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (240, 220, 0), 1, cv2.LINE_AA)

                # Ground contact indicator point (bottom-center)
                bc_x = int((bx1 + bx2) / 2)
                bc_y = int(by2)
                cv2.circle(out, (bc_x, bc_y), 3, (0, 240, 255), -1, cv2.LINE_AA)

                # Small vehicle class and confidence badge
                det_text = f"{det.class_name} {det.confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.36
                font_thickness = 1
                (det_w, det_h), d_base = cv2.getTextSize(det_text, font, font_scale, font_thickness)

                d_lbl_y = max(det_h + 4, by1 - 4)
                cv2.rectangle(out, (bx1, d_lbl_y - det_h - 3), (bx1 + det_w + 4, d_lbl_y + d_base), (10, 14, 22), -1)
                cv2.rectangle(out, (bx1, d_lbl_y - det_h - 3), (bx1 + det_w + 4, d_lbl_y + d_base), (240, 220, 0), 1)
                cv2.putText(out, det_text, (bx1 + 2, d_lbl_y - 1), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            return out
