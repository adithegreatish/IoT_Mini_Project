"""Unit regression tests for the Excel-style parking space calibration service,
coordinate validation, grid generation, and round-trip persistence.
"""

import json
import pytest
from app.database.connection import DatabaseManager
from app.parking.calibration_service import CalibrationService
from app.parking.geometry import (
    validate_slot_polygon,
    calculate_overlap_ratio,
    check_slot_overlaps,
    normalize_points,
    denormalize_points,
)


def test_validate_slot_polygon_boundaries():
    """Confirms normalized [0, 1] boundaries with floating point epsilon tolerance."""
    # Valid rectangle within [0, 1]
    valid_box = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]
    is_valid, err = validate_slot_polygon(valid_box)
    assert is_valid is True
    assert err is None

    # Slightly exceeds 1.0 due to float rounding (1.00005) - must pass due to epsilon
    slightly_off = [[0.1, 0.1], [1.00005, 0.1], [1.00005, 0.5], [0.1, 0.5]]
    is_valid, err = validate_slot_polygon(slightly_off)
    assert is_valid is True

    # Far outside bounds (1.20) - must fail with explicit error
    invalid_box = [[0.1, 0.1], [1.2, 0.1], [1.2, 0.5], [0.1, 0.5]]
    is_valid, err = validate_slot_polygon(invalid_box)
    assert is_valid is False
    assert "outside normalized" in err


def test_validate_slot_polygon_minimum_area():
    """Allows small bays (e.g. motorcycles) down to 0.0005 normalized area."""
    # 3% width x 3% height = 0.0009 area (motorcycle bay)
    small_bay = [[0.05, 0.05], [0.08, 0.05], [0.08, 0.08], [0.05, 0.08]]
    is_valid, err = validate_slot_polygon(small_bay, min_area=0.0005)
    assert is_valid is True


def test_calibration_service_unique_id_enforcement(tmp_path):
    """Rejects duplicate Slot IDs with clear error message."""
    calib_file = tmp_path / "calibration.json"
    service = CalibrationService(calib_file_path=str(calib_file))

    duplicate_slots = [
        {"id": "S01", "label": "Bay 1", "points": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.3], [0.1, 0.3]]},
        {"id": "S01", "label": "Bay 1 Copy", "points": [[0.3, 0.1], [0.4, 0.1], [0.4, 0.3], [0.3, 0.3]]},
    ]

    with pytest.raises(ValueError, match="Duplicate Slot ID 'S01'"):
        service.save_calibration(duplicate_slots)


def test_calibration_service_persistence_and_revision_history(tmp_path):
    """Saves slots to disk and records revision entry in SQLite."""
    calib_file = tmp_path / "calibration.json"
    db_file = tmp_path / "test_events.sqlite"
    db_mgr = DatabaseManager(str(db_file), cache_size_mb=8, mmap_size_mb=16)
    db_mgr.initialize_schema()

    service = CalibrationService(calib_file_path=str(calib_file), db_manager=db_mgr)

    slots = [
        {
            "id": "A01",
            "label": "Bay A01",
            "zone": "Zone-A",
            "slot_type": "car",
            "enabled": True,
            "points": [[0.05, 0.1], [0.20, 0.1], [0.20, 0.4], [0.05, 0.4]],
        },
        {
            "id": "A02",
            "label": "Bay A02",
            "zone": "Zone-A",
            "slot_type": "ev",
            "enabled": False,
            "points": [[0.25, 0.1], [0.40, 0.1], [0.40, 0.4], [0.25, 0.4]],
        },
    ]

    success, warnings, payload = service.save_calibration(slots, ref_width=1280, ref_height=720)
    assert success is True
    assert len(payload["slots"]) == 2

    # Reload from disk
    loaded = service.load_calibration()
    assert loaded["version"] == 2
    assert len(loaded["slots"]) == 2
    assert loaded["slots"][0]["id"] == "A01"
    assert loaded["slots"][1]["enabled"] is False

    # Check database revision history
    history = service.get_revision_history(limit=5)
    assert len(history) >= 1
    assert history[0]["version"] == 2


def test_check_slot_overlaps_warning():
    """Identifies overlapping parking bays and outputs descriptive warning."""
    overlapping_slots = [
        {"id": "S1", "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]},
        {"id": "S2", "points": [[0.15, 0.1], [0.35, 0.1], [0.35, 0.4], [0.15, 0.4]]},
    ]
    warnings = check_slot_overlaps(overlapping_slots, max_allowed_overlap=0.20)
    assert len(warnings) > 0
    assert "S1 and S2 overlap" in warnings[0]


def test_normalize_and_denormalize_roundtrip():
    """Validates pixel <-> normalized coordinate precision."""
    pixel_pts = [[128, 72], [384, 72], [384, 288], [128, 288]]
    norm = normalize_points(pixel_pts, width=1280, height=720)
    assert norm == [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]

    denorm = denormalize_points(norm, width=1280, height=720)
    assert denorm == [(128, 72), (384, 72), (384, 288), (128, 288)]
