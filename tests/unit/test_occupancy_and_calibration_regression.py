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


def create_ten_bay_calibration(file_path: Path):
    """Creates a canonical 10-bay parking calibration file (T01-T10, 5 cols x 2 rows)."""
    marginX = 5.0
    marginY = 12.0
    gapX = 2.0
    gapY = 4.0
    cols = 5
    rows = 2
    availableW = 100.0 - (2 * marginX) - (gapX * (cols - 1))
    availableH = 75.0 - (gapY * (rows - 1))
    bayW = round((availableW / cols) * 10) / 10
    bayH = round((availableH / rows) * 10) / 10

    slots = []
    count = 1
    for r in range(rows):
        for c in range(cols):
            x = round((marginX + c * (bayW + gapX)) * 10) / 10
            y = round((marginY + r * (bayH + gapY)) * 10) / 10
            nx, ny, nw, nh = x / 100, y / 100, bayW / 100, bayH / 100
            pts = [
                [round(nx, 4), round(ny, 4)],
                [round(nx + nw, 4), round(ny, 4)],
                [round(nx + nw, 4), round(ny + nh, 4)],
                [round(nx, 4), round(ny + nh, 4)],
            ]
            slots.append({
                "id": f"T{count:02d}",
                "label": f"Bay T{count:02d}",
                "zone": "Zone-A",
                "slot_type": "car",
                "enabled": True,
                "points": pts,
            })
            count += 1

    calib_data = {
        "version": 49,
        "reference_width": 1280,
        "reference_height": 720,
        "camera_fingerprint": "zebronics_uvc",
        "slots": slots,
    }
    file_path.write_text(json.dumps(calib_data, indent=2), encoding="utf-8")
    return calib_data


# --- Priority 4 Acceptance Tests: 10 Bays T01-T10 ---

def test_acceptance_1_ten_valid_calibration_polygons_produce_ten_active_bays(tmp_path: Path):
    """1. Proves ten valid calibration polygons produce ten active bays."""
    calib_file = tmp_path / "calibration.json"
    calib_data = create_ten_bay_calibration(calib_file)

    calib_svc = CalibrationService(str(calib_file))
    loaded = calib_svc.load_calibration()
    assert len(loaded["slots"]) == 10

    for s in loaded["slots"]:
        valid, err = validate_slot_polygon(s["points"])
        assert valid is True, f"Bay {s['id']} polygon invalid: {err}"
        assert s["enabled"] is True

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    occ_svc = OccupancyService(p_cfg, calib_svc)

    assert len(occ_svc._state_machines) == 10
    for i in range(1, 11):
        sid = f"T{i:02d}"
        assert sid in occ_svc._state_machines


def test_acceptance_2_valid_vehicle_inside_bay_marks_that_bay_occupied(tmp_path: Path):
    """2. Proves a valid vehicle inside a bay marks that bay occupied and keeps other bays free."""
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    # T02 polygon: x in [0.234, 0.398], y in [0.12, 0.475]
    # Place a vehicle inside T02: bbox [0.25, 0.15, 0.38, 0.45] -> bc = (0.315, 0.45), inside T02
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.25, 0.15, 0.38, 0.45),
            confidence=0.92,
            class_name="car",
            pixel_box=(160, 72, 243, 216),
        )
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    slot_states = {s["id"]: s["state"] for s in snap["slots"]}
    assert slot_states["T02"] == "OCCUPIED"
    assert snap["summary"]["occupied"] == 1
    assert snap["summary"]["free"] == 9

    # Verify associated_vehicle is recorded for T02 and None for vacant bays
    for s in snap["slots"]:
        if s["id"] == "T02":
            assert s["associated_vehicle"] == "car"
            assert s["overlap_ratio"] >= 0.35
        else:
            assert s["state"] == "FREE"
            assert s["associated_vehicle"] is None


def test_acceptance_3_vehicle_outside_all_bays_does_not_occupy_any_bay(tmp_path: Path):
    """3. Proves a vehicle outside all bays does not occupy any bay."""
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    # Vehicle driving in the bottom corridor: y in [0.90, 0.98] (all bays end at y=0.87)
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.40, 0.90, 0.60, 0.98),
            confidence=0.88,
            class_name="truck",
            pixel_box=(256, 432, 384, 470),
        )
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    assert snap["summary"]["occupied"] == 0
    assert snap["summary"]["free"] == 10
    for s in snap["slots"]:
        assert s["state"] == "FREE"
        assert s["associated_vehicle"] is None


def test_acceptance_4_one_vehicle_cannot_occupy_multiple_bays_by_box_overlap(tmp_path: Path):
    """4. Proves one vehicle cannot occupy multiple bays solely because its box overlaps them."""
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    # T01: x in [0.05, 0.214], y in [0.12, 0.475]
    # T02: x in [0.234, 0.398], y in [0.12, 0.475]
    # A wide SUV parked in T01 whose bounding box extends into T02 (x up to 0.26)
    # Bottom-center is (0.13, 0.45) which is cleanly inside T01
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.06, 0.14, 0.26, 0.46),
            confidence=0.94,
            class_name="car",
            pixel_box=(38, 67, 166, 220),
        )
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    slot_states = {s["id"]: s["state"] for s in snap["slots"]}
    assert slot_states["T01"] == "OCCUPIED"
    assert slot_states["T02"] == "FREE"  # T02 must remain FREE!
    assert snap["summary"]["occupied"] == 1
    assert snap["summary"]["free"] == 9


def test_acceptance_5_vehicle_leaving_bay_transitions_state_to_free(tmp_path: Path):
    """5. Proves a vehicle leaving a bay allows the state to transition to free after confirmation period."""
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    # 3 frames hysteresis confirmation
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=3)
    calib_svc = CalibrationService(str(calib_file))

    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Vehicle in T01 for 3 frames -> establishes stable OCCUPIED
    car_det = [
        DetectionResult(
            bbox=(0.06, 0.14, 0.20, 0.46),
            confidence=0.95,
            class_name="car",
            pixel_box=(38, 67, 128, 220),
        )
    ]
    mock_det.detect.return_value = car_det

    for _ in range(3):
        snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    assert snap["slots"][0]["state"] == "OCCUPIED"
    assert snap["summary"]["occupied"] == 1

    # Vehicle leaves: 0 detections
    mock_det.detect.return_value = []

    # Frame 1 after departure: hysteresis pending (remains OCCUPIED)
    snap1 = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    assert snap1["slots"][0]["state"] == "OCCUPIED"

    # Frame 2 after departure: hysteresis pending (remains OCCUPIED)
    snap2 = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    assert snap2["slots"][0]["state"] == "OCCUPIED"

    # Frame 3 after departure: hysteresis confirmed -> transitions to FREE!
    snap3 = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    assert snap3["slots"][0]["state"] == "FREE"
    assert snap3["slots"][0]["associated_vehicle"] is None
    assert snap3["summary"]["occupied"] == 0
    assert snap3["summary"]["free"] == 10


def test_acceptance_6_detector_and_camera_failure_does_not_create_false_free(tmp_path: Path):
    """6. Proves detector/camera failure does not create false free states, marking UNKNOWN."""
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Camera stream disconnects -> All 10 bays must become UNKNOWN, 0 FREE!
    snap_cam_lost = occ_svc.process_frame(None, is_camera_healthy=False, is_camera_stale=False)
    assert snap_cam_lost["summary"]["unknown"] == 10
    assert snap_cam_lost["summary"]["free"] == 0
    assert snap_cam_lost["summary"]["occupied"] == 0

    # 2. Camera frame is stale -> All 10 bays must become UNKNOWN, 0 FREE!
    snap_cam_stale = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=True)
    assert snap_cam_stale["summary"]["unknown"] == 10
    assert snap_cam_stale["summary"]["free"] == 0

    # 3. Detector runtime failure -> All 10 bays must become UNKNOWN, 0 FREE!
    mock_det.detect.side_effect = RuntimeError("V4L2 hardware pipeline timeout")
    snap_det_err = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    assert snap_det_err["summary"]["unknown"] == 10
    assert snap_det_err["summary"]["free"] == 0

    # 4. Detector is unhealthy/unready -> All 10 bays must become UNKNOWN, 0 FREE!
    mock_det.detect.side_effect = None
    mock_det.detect.return_value = []
    mock_det.is_healthy = False
    mock_det.status = "MODEL_INIT_FAILED"
    snap_unhealthy = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    assert snap_unhealthy["summary"]["unknown"] == 10
    assert snap_unhealthy["summary"]["free"] == 0


def test_acceptance_7_dashboard_counters_equal_backend_bay_states(tmp_path: Path):
    """7. Proves dashboard counters equal the totals derived from backend bay states."""
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)
    calib_svc = CalibrationService(str(calib_file))

    # Two cars: in T01 and T05
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True
    mock_det.detect.return_value = [
        DetectionResult(
            bbox=(0.06, 0.14, 0.20, 0.46),
            confidence=0.95,
            class_name="car",
            pixel_box=(38, 67, 128, 220),
        ),
        DetectionResult(
            bbox=(0.80, 0.14, 0.94, 0.46),
            confidence=0.91,
            class_name="car",
            pixel_box=(512, 67, 601, 220),
        ),
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    from app.api import routes
    routes.calibration_service = calib_svc
    routes.occupancy_service = occ_svc
    routes.allocation_service = None

    api_resp = routes.get_parking_slots()
    summary = api_resp["summary"]
    slots = api_resp["slots"]

    # Verify backend API mathematical consistency
    assert summary["total"] == 10
    assert summary["active"] == 10
    assert summary["occupied"] == 2
    assert summary["free"] == 8
    assert summary["unknown"] == 0

    # Simulate dashboard app.js updateOverviewCounters() logic
    active_slots = [s for s in slots if s.get("enabled", True)]
    dash_total = len(active_slots)
    dash_occupied = sum(1 for s in active_slots if s["state"] == "OCCUPIED")
    dash_free = sum(1 for s in active_slots if s["state"] == "FREE")
    dash_unknown = sum(1 for s in active_slots if s["state"] not in ("OCCUPIED", "FREE"))

    assert dash_total == summary["active"] == 10
    assert dash_occupied == summary["occupied"] == 2
    assert dash_free == summary["free"] == 8
    assert dash_unknown == summary["unknown"] == 0


def test_acceptance_hybrid_mode_prevents_false_occupancy_from_patch_classifier(tmp_path: Path):
    """
    Demonstrates the root cause fix:
    In hybrid mode, a patch classifier reporting high probability (due to domain shift)
    must NEVER mark vacant bays as occupied when the detector sees no vehicle.
    """
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    p_cfg = ParkingConfig(
        calibration_file=str(calib_file),
        occupancy_threshold=0.35,
        inference_mode="hybrid",
        classifier_threshold=0.65,
        hysteresis_confirm_frames=1,
    )
    calib_svc = CalibrationService(str(calib_file))

    # Mock detector sees NO vehicle in any bay
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True
    mock_det.detect.return_value = []

    # Mock classifier reports domain-shifted false positives (0.95 conf for all patches)
    mock_clf = MagicMock()
    mock_clf.extract_patch_from_frame.return_value = np.ones((150, 150, 3), dtype=np.uint8) * 128
    mock_clf.predict_patch.return_value = (True, 0.95)

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det, classifier=mock_clf)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    # Vacant bays MUST remain FREE! Cannot be forced OCCUPIED by domain-shifted classifier!
    assert snap["summary"]["occupied"] == 0
    assert snap["summary"]["free"] == 10
    for s in snap["slots"]:
        assert s["state"] == "FREE"
        assert s["associated_vehicle"] is None


def test_acceptance_overlay_rendering_multi_resolution_and_states(tmp_path: Path):
    """
    Validates that render_overlay properly executes across multiple camera frame resolutions
    (640x480, 1280x720, 1920x1080) and renders all semantic bay states (FREE, OCCUPIED,
    RESERVED, UNKNOWN, DISABLED) alongside vehicle detection bounding boxes and ground contact points.
    """
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35)
    calib_svc = CalibrationService(str(calib_file))
    occ_svc = OccupancyService(p_cfg, calib_svc)

    slots_to_render = [
        {"id": "T01", "state": "FREE", "points": [[0.05, 0.12], [0.21, 0.12], [0.21, 0.47], [0.05, 0.47]]},
        {"id": "T02", "state": "OCCUPIED", "points": [[0.23, 0.12], [0.39, 0.12], [0.39, 0.47], [0.23, 0.47]]},
        {"id": "T03", "state": "RESERVED", "points": [[0.41, 0.12], [0.57, 0.12], [0.57, 0.47], [0.41, 0.47]]},
        {"id": "T04", "state": "UNKNOWN", "points": [[0.59, 0.12], [0.75, 0.12], [0.75, 0.47], [0.59, 0.47]]},
        {"id": "T05", "state": "DISABLED", "points": [[0.77, 0.12], [0.93, 0.12], [0.93, 0.47], [0.77, 0.47]]},
    ]

    for (h, w) in [(480, 640), (720, 1280), (1080, 1920)]:
        test_frame = np.zeros((h, w, 3), dtype=np.uint8)
        dets = [
            DetectionResult(
                bbox=(0.25, 0.15, 0.37, 0.45),
                confidence=0.88,
                class_name="car",
                pixel_box=(int(0.25 * w), int(0.15 * h), int(0.37 * w), int(0.45 * h)),
            )
        ]
        rendered = occ_svc.render_overlay(test_frame, slots=slots_to_render, detections=dets)
        assert rendered.shape == (h, w, 3)
        assert np.count_nonzero(rendered) > 0


def test_acceptance_executive_summary_and_reservation_protection(tmp_path: Path):
    """
    Validates that:
    1. Active reservations protect vacant bays from being displayed as FREE.
    2. Occupied bays with active reservations transition to OCCUPIED.
    3. Disabled bays report DISABLED and do not count toward active bays.
    4. 6-card executive summary counts accurately match individual bay states.
    5. Occupancy rate calculation denominator is strictly active bays.
    """
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    calib_data = json.loads(calib_file.read_text(encoding="utf-8"))
    # Disable 1 bay (T10)
    calib_data["slots"][9]["enabled"] = False
    calib_file.write_text(json.dumps(calib_data, indent=2), encoding="utf-8")

    calib_svc = CalibrationService(str(calib_file))
    p_cfg = ParkingConfig(calibration_file=str(calib_file), occupancy_threshold=0.35, hysteresis_confirm_frames=1)

    # Mock detector sees 2 vehicles: one in T01 and one in T02
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True
    mock_det.detect.return_value = [
        DetectionResult(bbox=(0.06, 0.15, 0.20, 0.45), confidence=0.89, class_name="car", pixel_box=(38, 72, 128, 216)),
        DetectionResult(bbox=(0.24, 0.15, 0.38, 0.45), confidence=0.92, class_name="car", pixel_box=(153, 72, 243, 216)),
    ]

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    from app.api import routes
    routes.calibration_service = calib_svc
    routes.occupancy_service = occ_svc

    # Mock allocation service with reservation in T03 (which is physically vacant)
    mock_alloc = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"slot_id": "T03", "reservation_code": "RES-TEST-001", "vehicle_plate": "MH12AB1234", "status": "ACTIVE"}
    ]
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mock_alloc.db.get_connection.return_value = mock_conn
    routes.allocation_service = mock_alloc

    resp = routes.get_parking_slots()
    summary = resp["summary"]
    slots = resp["slots"]

    # Total bays: 10, Active bays: 9 (T10 is disabled)
    assert summary["total"] == 10
    assert summary["active"] == 9
    assert summary["occupied"] == 2  # T01 and T02
    assert summary["reserved"] == 1  # T03 is reserved & physically vacant
    assert summary["free"] == 6      # T04-T09
    # Disabled bay T10 counted in unknown / unavailable
    assert summary["unknown"] == 1

    # Find T03 and verify state is RESERVED, not FREE
    t03 = next(s for s in slots if s["id"] == "T03")
    assert t03["state"] == "RESERVED"
    assert t03["reservation_status"] == "RESERVED"
    assert t03["active_reservation"] == "RES-TEST-001"

    # Find T10 and verify DISABLED state
    t10 = next(s for s in slots if s["id"] == "T10")
    assert t10["state"] == "DISABLED"
    assert t10["enabled"] is False

    # Simulate dashboard app.js updateOverviewCounters() logic
    active_slots = [s for s in slots if s.get("enabled", True)]
    dash_total = len(active_slots)
    dash_occupied = sum(1 for s in active_slots if s["state"] == "OCCUPIED")
    dash_reserved = sum(1 for s in active_slots if s["state"] == "RESERVED")
    dash_free = sum(1 for s in active_slots if s["state"] == "FREE")
    dash_disabled = sum(1 for s in slots if not s.get("enabled", True))
    dash_unknown = sum(1 for s in active_slots if s["state"] not in ("OCCUPIED", "FREE", "RESERVED")) + dash_disabled
    dash_rate = round((dash_occupied / dash_total) * 100, 1)

    assert dash_total == 9
    assert dash_occupied == 2
    assert dash_reserved == 1
    assert dash_free == 6
    assert dash_unknown == 1
    assert dash_rate == 22.2


def test_acceptance_temporal_confirmation_departure(tmp_path: Path):
    """
    Validates that a confirmed OCCUPIED slot requires hysteresis_confirm_frames
    consecutive vacant observations before transitioning back to FREE,
    preventing state flicker during brief occlusion or missed detections.
    """
    calib_file = tmp_path / "calibration.json"
    create_ten_bay_calibration(calib_file)

    p_cfg = ParkingConfig(
        calibration_file=str(calib_file),
        occupancy_threshold=0.35,
        hysteresis_confirm_frames=3,
    )
    calib_svc = CalibrationService(str(calib_file))
    mock_det = MagicMock()
    mock_det.enabled = True
    mock_det.is_healthy = True

    occ_svc = OccupancyService(p_cfg, calib_svc, detector=mock_det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Drive slot T01 to confirmed OCCUPIED (needs 3 consecutive occupied frames)
    car_det = [
        DetectionResult(bbox=(0.06, 0.15, 0.20, 0.45), confidence=0.90, class_name="car", pixel_box=(38, 72, 128, 216))
    ]
    mock_det.detect.return_value = car_det

    for _ in range(3):
        snap = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    t01 = next(s for s in snap["slots"] if s["id"] == "T01")
    assert t01["state"] == "OCCUPIED"

    # 2. Vehicle departs (detector reports empty)
    mock_det.detect.return_value = []

    # Frame 1 of absence: should NOT transition to FREE yet (temporal smoothing / debounce)
    snap1 = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    t01_f1 = next(s for s in snap1["slots"] if s["id"] == "T01")
    assert t01_f1["state"] == "OCCUPIED"

    # Frame 2 of absence: still OCCUPIED
    snap2 = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    t01_f2 = next(s for s in snap2["slots"] if s["id"] == "T01")
    assert t01_f2["state"] == "OCCUPIED"

    # Frame 3 of absence: confirmed departure -> transitions to FREE
    snap3 = occ_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)
    t01_f3 = next(s for s in snap3["slots"] if s["id"] == "T01")
    assert t01_f3["state"] == "FREE"
