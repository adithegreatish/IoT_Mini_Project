"""Unit tests for slot state machine and occupancy service."""

from pathlib import Path
import numpy as np
import pytest
from app.config.schema import ParkingConfig, DetectionConfig
from app.detection.vehicle_detector import VehicleDetector
from app.parking.calibration_service import CalibrationService
from app.parking.occupancy_service import OccupancyService
from app.parking.state_machine import SlotState, SlotStateMachine


def test_slot_state_machine_hysteresis_and_transitions():
    sm = SlotStateMachine(slot_id="S01", smoothing_window=5, hysteresis_confirm=3)
    assert sm.stable_state == SlotState.UNKNOWN

    # Observation 1: Occupied -> remains UNKNOWN (pending 1)
    state, changed = sm.update(is_occupied_observation=True, observation_confidence=0.9)
    assert state == SlotState.UNKNOWN
    assert changed is False

    # Observation 2: Occupied -> remains UNKNOWN (pending 2)
    state, changed = sm.update(is_occupied_observation=True, observation_confidence=0.9)
    assert state == SlotState.UNKNOWN
    assert changed is False

    # Observation 3: Occupied -> commits transition to OCCUPIED!
    state, changed = sm.update(is_occupied_observation=True, observation_confidence=0.9)
    assert state == SlotState.OCCUPIED
    assert changed is True

    # Single spurious FREE observation does NOT trigger false transition
    state, changed = sm.update(is_occupied_observation=False, observation_confidence=0.8)
    assert state == SlotState.OCCUPIED
    assert changed is False


def test_slot_state_machine_stale_protection():
    sm = SlotStateMachine(slot_id="S01", smoothing_window=5, hysteresis_confirm=2)
    # Establish occupied state
    sm.update(True, 0.9)
    sm.update(True, 0.9)
    assert sm.stable_state == SlotState.OCCUPIED

    # Disconnect camera -> MUST transition to UNKNOWN, NEVER FREE
    state, changed = sm.mark_camera_stale_or_lost("Camera disconnected")
    assert state == SlotState.UNKNOWN
    assert changed is True
    assert "disconnected" in sm.reason.lower()


def test_occupancy_service_frame_processing(tmp_path: Path):
    calib_file = tmp_path / "calibration.json"
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    d_cfg = DetectionConfig(enabled=True, runtime="mock")

    calib_svc = CalibrationService(str(calib_file))
    sample_slots = [
        {"id": "S01", "label": "Bay 1", "slot_type": "car", "enabled": True, "points": [[0.05, 0.4], [0.25, 0.4], [0.25, 0.9], [0.05, 0.9]]},
        {"id": "S02", "label": "Bay 2", "slot_type": "car", "enabled": True, "points": [[0.30, 0.4], [0.50, 0.4], [0.50, 0.9], [0.30, 0.9]]},
    ]
    calib_svc.save_calibration(sample_slots)

    detector = VehicleDetector(d_cfg)
    occ_svc = OccupancyService(p_cfg, calib_svc, detector)

    # 1. Process empty black frame
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    snapshot = occ_svc.process_frame(black_frame, is_camera_healthy=True, is_camera_stale=False)

    assert "summary" in snapshot
    assert snapshot["summary"]["total"] == 2
    assert snapshot["summary"]["free"] == 2
    assert snapshot["summary"]["occupied"] == 0

    # 2. Process stale camera frame
    stale_snapshot = occ_svc.process_frame(black_frame, is_camera_healthy=False, is_camera_stale=True)
    assert stale_snapshot["summary"]["unknown"] == 2
    assert stale_snapshot["summary"]["free"] == 0  # CRITICAL: unknown must never be counted as free!

    # 3. Test overlay rendering
    overlay = occ_svc.render_overlay(black_frame)
    assert overlay.shape == black_frame.shape
