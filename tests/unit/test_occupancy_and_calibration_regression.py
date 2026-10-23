"""Regression and verification tests for calibration, coordinate mapping, and spatial association."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from app.config.schema import DetectionConfig, ParkingConfig
from app.detection.vehicle_detector import DetectionResult
from app.parking.calibration_service import CalibrationService
from app.parking.geometry import (
    denormalize_points,
    is_point_in_slot,
    normalize_points,
    validate_slot_polygon,
)
from app.parking.occupancy_service import OccupancyService
from app.parking.state_machine import SlotState


def create_sample_multibay_calibration(file_path: Path):
    """Creates a realistic 4-bay parking calibration file."""
    calib_data = {
        "version": 10,
        "reference_width": 1280,
        "reference_height": 720,
        "camera_fingerprint": "zebronics_uvc",
        "slots": [
            {
                "id": "S01",
                "label": "Bay 01",
                "zone": "Zone-A",
                "slot_type": "car",
                "enabled": True,
                "points": [[0.05, 0.40], [0.25, 0.40], [0.25, 0.85], [0.05, 0.85]],
            },
            {
                "id": "S02",
                "label": "Bay 02",
                "zone": "Zone-A",
                "slot_type": "car",
                "enabled": True,
                "points": [[0.27, 0.40], [0.47, 0.40], [0.47, 0.85], [0.27, 0.85]],
            },
            {
                "id": "S03",
                "label": "Bay 03",
                "zone": "Zone-A",
                "slot_type": "car",
                "enabled": True,
                "points": [[0.50, 0.40], [0.70, 0.40], [0.70, 0.85], [0.50, 0.85]],
            },
            {
                "id": "S04",
                "label": "Bay 04 (Disabled)",
                "zone": "Zone-A",
                "slot_type": "car",
                "enabled": False,
                "points": [[0.72, 0.40], [0.92, 0.40], [0.92, 0.85], [0.72, 0.85]],
            },
        ],
    }
    file_path.write_text(json.dumps(calib_data, indent=2), encoding="utf-8")
    return calib_data


def test_loading_multiple_calibrated_bays_schema(tmp_path: Path):
    """Validates loading multiple calibrated bays from the real calibration schema."""
    calib_file = tmp_path / "calibration.json"
    create_sample_multibay_calibration(calib_file)

    calib_svc = CalibrationService(str(calib_file))
    calib = calib_svc.load_calibration()

    assert calib["version"] == 10
    assert len(calib["slots"]) == 4

    # Validate each polygon
    for s in calib["slots"]:
        valid, err = validate_slot_polygon(s["points"])
        assert valid is True, f"Slot {s['id']} polygon invalid: {err}"

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    occ_svc = OccupancyService(p_cfg, calib_svc)

    # 3 enabled bays must have active state machines
    assert len(occ_svc._state_machines) == 3
    assert "S01" in occ_svc._state_machines
    assert "S02" in occ_svc._state_machines
    assert "S03" in occ_svc._state_machines
    assert "S04" not in occ_svc._state_machines  # Disabled bay


def test_preserving_calibration_across_reload(tmp_path: Path):
    """Ensures calibration is preserved, versioned, and reloads without losing slot points."""
    calib_file = tmp_path / "calibration.json"
    create_sample_multibay_calibration(calib_file)

    calib_svc = CalibrationService(str(calib_file))
    initial = calib_svc.load_calibration()
    assert len(initial["slots"]) == 4

    # Update slot label and re-save
    slots_copy = [dict(s) for s in initial["slots"]]
    slots_copy[0]["label"] = "VIP Bay 01"
    success, warnings, saved = calib_svc.save_calibration(slots_copy, ref_width=1280, ref_height=720)
    assert success is True
    assert saved["version"] == initial["version"] + 1

    # Reload from disk
    reloaded = calib_svc.load_calibration()
    assert reloaded["version"] == initial["version"] + 1
    assert reloaded["slots"][0]["label"] == "VIP Bay 01"
    assert reloaded["slots"][0]["points"] == initial["slots"][0]["points"]


def test_coordinate_mapping_resized_preview_to_frame():
    """Verifies normalized coordinates map accurately across different preview and frame sizes."""
    orig_norm_pts = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.5], [0.1, 0.5]]

    # 1. Map to Raspberry Pi native camera resolution 640x480
    pixels_640 = denormalize_points(orig_norm_pts, width=640, height=480)
    assert pixels_640 == [(64, 96), (192, 96), (192, 240), (64, 240)]

    # 2. Map back to normalized coordinates
    norm_again_640 = normalize_points([[p[0], p[1]] for p in pixels_640], width=640, height=480)
    for p_orig, p_new in zip(orig_norm_pts, norm_again_640):
        assert abs(p_orig[0] - p_new[0]) < 1e-3
        assert abs(p_orig[1] - p_new[1]) < 1e-3

    # 3. Map to 1280x720 display canvas
    pixels_1280 = denormalize_points(orig_norm_pts, width=1280, height=720)
    assert pixels_1280 == [(128, 144), (384, 144), (384, 360), (128, 360)]

    norm_again_1280 = normalize_points([[p[0], p[1]] for p in pixels_1280], width=1280, height=720)
    for p_orig, p_new in zip(orig_norm_pts, norm_again_1280):
        assert abs(p_orig[0] - p_new[0]) < 1e-3
        assert abs(p_orig[1] - p_new[1]) < 1e-3


def test_vehicle_bottom_center_inside_bay(tmp_path: Path):
    """Verifies that a vehicle whose bottom-center is inside bay S01 marks S01 OCCUPIED while S02 stays FREE."""
    calib_file = tmp_path / "calibration.json"
    create_sample_multibay_calibration(calib_file)
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    d_cfg = DetectionConfig(enabled=True, runtime="mock")

    calib_svc = CalibrationService(str(calib_file))
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True

    # Vehicle in Bay S01: bbox [0.08, 0.45, 0.22, 0.80] -> bottom-center (0.15, 0.80) inside S01
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.08, 0.45, 0.22, 0.80),
            confidence=0.92,
            class_name="car",
            pixel_box=(51, 216, 140, 384),
        )
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    slot_map = {s["id"]: s["state"] for s in snap["slots"]}

    assert slot_map["S01"] == "OCCUPIED"
    assert slot_map["S02"] == "FREE"
    assert slot_map["S03"] == "FREE"
    assert slot_map["S04"] == "DISABLED"


def test_vehicle_outside_every_calibrated_bay(tmp_path: Path):
    """Verifies that vehicles in non-calibrated areas (e.g. driveway or road) do not mark any bay occupied."""
    calib_file = tmp_path / "calibration.json"
    create_sample_multibay_calibration(calib_file)
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)

    calib_svc = CalibrationService(str(calib_file))
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True

    # Vehicle driving in foreground road: y is 0.90 to 0.98, far below bays (y: 0.40 - 0.85)
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.10, 0.90, 0.40, 0.98),
            confidence=0.88,
            class_name="bus",
            pixel_box=(64, 432, 256, 470),
        ),
        # Another vehicle far to the right outside calibrated zone
        DetectionResult(
            bbox=(0.93, 0.10, 0.99, 0.35),
            confidence=0.85,
            class_name="car",
            pixel_box=(595, 48, 633, 168),
        ),
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    slot_map = {s["id"]: s["state"] for s in snap["slots"]}

    assert slot_map["S01"] == "FREE"
    assert slot_map["S02"] == "FREE"
    assert slot_map["S03"] == "FREE"
    assert snap["summary"]["occupied"] == 0
    assert snap["summary"]["free"] == 3


def test_vehicle_near_shared_polygon_boundary(tmp_path: Path):
    """
    Verifies that a vehicle near a shared boundary is assigned to its primary bay
    and is not double-assigned to both adjacent bays.
    """
    calib_file = tmp_path / "calibration.json"
    # S01: [0.1, 0.2] to [0.3, 0.8], S02: [0.3, 0.2] to [0.5, 0.8] (sharing border x=0.3)
    calib_data = {
        "version": 1,
        "slots": [
            {"id": "S01", "label": "Bay 1", "slot_type": "car", "enabled": True, "points": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.8], [0.1, 0.8]]},
            {"id": "S02", "label": "Bay 2", "slot_type": "car", "enabled": True, "points": [[0.3, 0.2], [0.5, 0.2], [0.5, 0.8], [0.3, 0.8]]},
        ],
    }
    calib_file.write_text(json.dumps(calib_data), encoding="utf-8")

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True

    # Car parked in S01 but near the border: bbox [0.18, 0.30, 0.32, 0.75]
    # bottom-center = (0.25, 0.75), which is cleanly inside S01 (x in [0.1, 0.3])
    # The bounding box extends slightly into S02 (x up to 0.32), but bottom-center is in S01.
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.18, 0.30, 0.32, 0.75),
            confidence=0.91,
            class_name="car",
            pixel_box=(115, 144, 204, 360),
        )
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    slot_map = {s["id"]: s["state"] for s in snap["slots"]}

    assert slot_map["S01"] == "OCCUPIED"
    assert slot_map["S02"] == "FREE"  # Must NOT be double-occupied!
    assert snap["summary"]["occupied"] == 1
    assert snap["summary"]["free"] == 1


def test_camera_loss_and_detector_exceptions(tmp_path: Path):
    """Verifies that camera loss or inference errors mark bays UNKNOWN, NEVER FREE."""
    calib_file = tmp_path / "calibration.json"
    create_sample_multibay_calibration(calib_file)
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True
    mock_det.detect.return_value = []

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Normal frame with no vehicles -> FREE
    snap_normal = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    assert snap_normal["summary"]["free"] == 3

    # 2. Camera Disconnected -> MUST transition to UNKNOWN, free count must be 0
    snap_disconnected = occ_svc.process_frame(None, is_camera_healthy=False, is_camera_stale=False)
    assert snap_disconnected["summary"]["unknown"] == 3
    assert snap_disconnected["summary"]["free"] == 0

    # 3. Detector Exception -> MUST transition to UNKNOWN, free count must be 0
    mock_det.detect.side_effect = RuntimeError("ONNX Runtime engine device fault")
    snap_err = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    assert snap_err["summary"]["unknown"] == 3
    assert snap_err["summary"]["free"] == 0


def test_active_free_occupied_unknown_dashboard_counts(tmp_path: Path):
    """Verifies that active/free/occupied/unknown dashboard counts match exact bay states."""
    calib_file = tmp_path / "calibration.json"
    create_sample_multibay_calibration(calib_file)
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True

    # Vehicle in Bay S01
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.08, 0.45, 0.22, 0.80),
            confidence=0.95,
            class_name="car",
            pixel_box=(51, 216, 140, 384),
        )
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    summary = snap["summary"]

    assert summary["total"] == 4       # 3 active + 1 disabled
    assert summary["active"] == 3      # 3 enabled bays
    assert summary["occupied"] == 1    # S01
    assert summary["free"] == 2        # S02, S03
    assert summary["unknown"] == 0     # Camera is healthy


def test_displayed_counters_agree_with_backend_api_state(tmp_path: Path):
    """Verifies that the merged summary from the API matches the occupancy engine snapshot."""
    calib_file = tmp_path / "calibration.json"
    create_sample_multibay_calibration(calib_file)
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    occ_svc = OccupancyService(p_cfg, calib_svc)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    # Simulate get_parking_slots merge logic
    from app.api import routes
    routes.calibration_service = calib_svc
    routes.occupancy_service = occ_svc
    routes.allocation_service = None

    api_resp = routes.get_parking_slots()
    api_summary = api_resp["summary"]

    assert api_summary["total"] == snap["summary"]["total"]
    assert api_summary["active"] == snap["summary"]["active"]
    assert api_summary["free"] == snap["summary"]["free"]
    assert api_summary["occupied"] == snap["summary"]["occupied"]
