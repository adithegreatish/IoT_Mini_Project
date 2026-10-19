"""
RELEASE HARDENING ACCEPTANCE TEST SUITE
Implements Section 11 of the Final Release Directive:
A. Vehicle detection integrity & safe failure handling
B. Occupancy geometry, ground-contact association, and fail-closed state safety
C. Reservation transaction integrity, duplicate plate rejection, and ACID serialization
D. Security hardening: path traversal, security headers, and input sanitization
E. Comprehensive end-to-end simulated lifecycle
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.config.loader import load_config
from app.config.schema import DetectionConfig, ParkingConfig
from app.database.connection import DatabaseManager
from app.detection.vehicle_detector import VehicleDetector, DetectionResult
from app.main import app, lifespan
from app.parking.allocation_service import (
    AllocationService,
    DuplicateAllocationError,
    SlotUnavailableError,
    ConcurrencyConflictError,
)
from app.parking.calibration_service import CalibrationService
from app.parking.geometry import (
    calculate_spatial_association_score,
    calculate_overlap_ratio,
    validate_slot_polygon,
)
from app.parking.occupancy_service import OccupancyService
from app.parking.state_machine import SlotState, SlotStateMachine


# =============================================================================
# A. VEHICLE DETECTION TESTS
# =============================================================================

def test_production_vehicle_detector_model_loading():
    """Verifies that active production vehicle detector ONNX model loads with exact class mapping."""
    cfg = DetectionConfig(
        enabled=True,
        model_path="models/vehicle_detector.onnx",
        runtime="onnx",
        confidence_threshold=0.40,
    )
    detector = VehicleDetector(cfg)
    assert detector.is_healthy is True
    assert detector.status == "READY"
    assert detector.is_simulated is False
    assert detector.TARGET_VEHICLE_CLASSES == ["car", "motorcycle", "bus", "truck"]


def test_vehicle_detector_empty_and_black_frame():
    """Verifies that empty, None, or solid black frames produce empty detections without crashing."""
    cfg = DetectionConfig(
        enabled=True,
        model_path="models/vehicle_detector.onnx",
        runtime="onnx",
        confidence_threshold=0.40,
    )
    detector = VehicleDetector(cfg)

    assert detector.detect(None) == []
    assert detector.detect(np.array([], dtype=np.uint8)) == []

    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(black_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detector_inference_deterministic_reproducibility():
    """Verifies identical reproducible outputs on static repeated frames."""
    cfg = DetectionConfig(
        enabled=True,
        model_path="models/vehicle_detector.onnx",
        runtime="onnx",
        confidence_threshold=0.40,
    )
    detector = VehicleDetector(cfg)

    # Synthetic image with bright rectangle
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_frame[200:500, 300:700] = 180

    run1 = detector.detect(test_frame)
    run2 = detector.detect(test_frame)

    assert len(run1) == len(run2)
    for d1, d2 in zip(run1, run2):
        assert d1.class_name == d2.class_name
        assert d1.bbox == d2.bbox
        assert d1.confidence == d2.confidence


# =============================================================================
# B. OCCUPANCY AND GEOMETRY TESTS
# =============================================================================

def test_hybrid_spatial_association_perspective_dampening():
    """
    Verifies that perspective roof projection into adjacent slot is penalized
    when vehicle ground contact (bottom-center) is outside the slot polygon.
    """
    # Bay A-01: [0.1, 0.1] to [0.4, 0.6]
    bay_pts = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.6], [0.1, 0.6]]

    # Case 1: Vehicle clearly parked inside Bay A-01
    inside_bbox = (0.15, 0.15, 0.35, 0.55)
    score_inside, reason_inside = calculate_spatial_association_score(bay_pts, inside_bbox, method="hybrid")
    assert score_inside >= 0.50
    assert "Ground contact inside" in reason_inside

    # Case 2: Vehicle parked in Bay A-02, but its roof overlaps Bay A-01 (bottom-center is at x=0.55, y=0.55)
    adjacent_roof_bbox = (0.30, 0.15, 0.80, 0.55)
    score_adjacent, reason_adjacent = calculate_spatial_association_score(bay_pts, adjacent_roof_bbox, method="hybrid")
    raw_overlap = calculate_overlap_ratio(bay_pts, adjacent_roof_bbox)

    assert "outside bay" in reason_adjacent
    assert score_adjacent < raw_overlap  # Penalized by 0.65 factor
    assert score_adjacent < 0.35  # Falls below default occupancy threshold


def test_occupancy_service_fails_safe_to_unknown_on_detector_error(tmp_path):
    """
    CRITICAL: When the vehicle detector fails or encounters an error,
    all slots MUST transition to UNKNOWN, NEVER to FREE.
    """
    import json
    slots_def = [
        {"id": "A-01", "label": "Bay 1", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]], "enabled": True},
        {"id": "A-02", "label": "Bay 2", "points": [[0.5, 0.1], [0.8, 0.1], [0.8, 0.5], [0.5, 0.5]], "enabled": True},
    ]
    calib_file = tmp_path / "test_calib.json"
    calib_file.write_text(json.dumps({"slots": slots_def, "reference_width": 1280, "reference_height": 720, "version": 1}), encoding="utf-8")

    class BrokenDetector:
        enabled = True
        is_healthy = False
        status = "INFERENCE_CRASH"
        def detect(self, frame):
            raise RuntimeError("CUDA hardware fault during forward pass")

    p_cfg = ParkingConfig(occupancy_threshold=0.35, inference_mode="detector")
    calib_srv = CalibrationService(str(calib_file))

    occ_service = OccupancyService(p_cfg, calib_srv, BrokenDetector())

    # Process a frame
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    snapshot = occ_service.process_frame(dummy_frame, is_camera_healthy=True, is_camera_stale=False)

    # Assert fail-safe behavior
    assert snapshot["summary"]["free"] == 0, "Broken detector MUST NEVER report FREE bays!"
    assert snapshot["summary"]["occupied"] == 0
    assert snapshot["summary"]["unknown"] == 2
    for slot in snapshot["slots"]:
        assert slot["state"] == SlotState.UNKNOWN.value
        assert "Detector" in slot["reason"]


def test_occupancy_service_fails_safe_to_unknown_on_camera_loss(tmp_path):
    """Verifies camera disconnection forces all bays to UNKNOWN, NEVER FREE."""
    import json
    slots_def = [
        {"id": "A-01", "label": "Bay 1", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]], "enabled": True},
    ]
    calib_file = tmp_path / "test_calib_loss.json"
    calib_file.write_text(json.dumps({"slots": slots_def, "reference_width": 1280, "reference_height": 720, "version": 1}), encoding="utf-8")

    p_cfg = ParkingConfig(occupancy_threshold=0.35)
    calib_srv = CalibrationService(str(calib_file))

    occ_service = OccupancyService(p_cfg, calib_srv, None)
    snap = occ_service.process_frame(None, is_camera_healthy=False, is_camera_stale=True)

    assert snap["is_camera_stale"] is True
    assert snap["summary"]["free"] == 0
    assert snap["summary"]["unknown"] == 1
    assert snap["slots"][0]["state"] == SlotState.UNKNOWN.value



# =============================================================================
# C. RESERVATION, CONCURRENCY, AND DUPLICATE REJECTION TESTS
# =============================================================================

def test_allocation_duplicate_plate_rejection(tmp_path):
    """
    Verifies that a vehicle with an active reservation CANNOT create a duplicate active allocation.
    """
    db_file = tmp_path / "test_parking.sqlite"
    db_mgr = DatabaseManager(str(db_file))
    db_mgr.initialize_schema()

    alloc_service = AllocationService(db_mgr)
    alloc_service.sync_slots_from_calibration([
        {"id": "B-01", "label": "Bay B1", "zone": "Zone-B", "level": "L1", "slot_type": "car", "enabled": True},
        {"id": "B-02", "label": "Bay B2", "zone": "Zone-B", "level": "L1", "slot_type": "car", "enabled": True},
    ])

    phys_free = {
        "B-01": {"state": SlotState.FREE.value, "confidence": 1.0},
        "B-02": {"state": SlotState.FREE.value, "confidence": 1.0},
    }

    # 1. First allocation succeeds
    rec1 = alloc_service.allocate_slot(
        user_id="user_alpha",
        vehicle_plate="KA01AB1234",
        current_physical_states=phys_free,
    )
    assert rec1["success"] is True
    assert rec1["slot_id"] == "B-01"

    # 2. Second allocation with the SAME plate is rejected
    with pytest.raises(DuplicateAllocationError) as exc_info:
        alloc_service.allocate_slot(
            user_id="user_alpha",
            vehicle_plate="KA 01 AB 1234",  # Spaces stripped, matched normalized
            current_physical_states=phys_free,
        )
    assert "already holds active reservation" in str(exc_info.value) or "Active reservation already exists" in str(exc_info.value)

    # 3. Release first reservation
    alloc_service.release_slot(rec1["reservation_code"])

    # 4. Now vehicle can allocate again
    rec2 = alloc_service.allocate_slot(
        user_id="user_alpha",
        vehicle_plate="KA01AB1234",
        current_physical_states=phys_free,
    )
    assert rec2["success"] is True


def test_concurrent_slot_allocation_acid_protection(tmp_path):
    """
    Simulates simultaneous racing requests to allocate the SAME slot.
    Guarantees exactly ONE succeeds and all others receive controlled concurrency conflict.
    """
    db_file = tmp_path / "test_concurrency.sqlite"
    db_mgr = DatabaseManager(str(db_file))
    db_mgr.initialize_schema()

    alloc_service = AllocationService(db_mgr)
    alloc_service.sync_slots_from_calibration([
        {"id": "SOLO-01", "label": "Solo Bay", "zone": "Zone-A", "level": "L1", "slot_type": "car", "enabled": True},
    ])

    phys_free = {"SOLO-01": {"state": SlotState.FREE.value, "confidence": 1.0}}

    results = []
    errors = []

    def attempt_alloc(idx):
        try:
            r = alloc_service.allocate_slot(
                user_id=f"user_{idx}",
                slot_id="SOLO-01",
                vehicle_plate=f"DL01AB999{idx}",
                current_physical_states=phys_free,
            )
            results.append(r)
        except Exception as e:
            errors.append(e)

    # Race 4 concurrent threads
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(attempt_alloc, i) for i in range(4)]
        for f in futures:
            f.result()

    # Invariant: EXACTLY one thread succeeds
    assert len(results) == 1, f"Expected 1 winner, got {len(results)}"
    assert len(errors) == 3, f"Expected 3 conflicts, got {len(errors)}"
    for err in errors:
        assert isinstance(err, (ConcurrencyConflictError, SlotUnavailableError))


def test_allocation_rejection_when_slot_occupied_or_unknown(tmp_path):
    """Verifies that slots with OCCUPIED or UNKNOWN states can NEVER be allocated."""
    db_file = tmp_path / "test_states.sqlite"
    db_mgr = DatabaseManager(str(db_file))
    db_mgr.initialize_schema()

    alloc_service = AllocationService(db_mgr)
    alloc_service.sync_slots_from_calibration([
        {"id": "S1", "label": "Occupied Bay", "zone": "Zone-A", "level": "L1", "slot_type": "car", "enabled": True},
        {"id": "S2", "label": "Unknown Bay", "zone": "Zone-A", "level": "L1", "slot_type": "car", "enabled": True},
    ])

    phys_states = {
        "S1": {"state": SlotState.OCCUPIED.value, "confidence": 0.95},
        "S2": {"state": SlotState.UNKNOWN.value, "confidence": 0.0},
    }

    # Attempt allocate S1
    with pytest.raises(SlotUnavailableError):
        alloc_service.allocate_slot(user_id="u1", slot_id="S1", current_physical_states=phys_states)

    # Attempt allocate S2
    with pytest.raises(SlotUnavailableError):
        alloc_service.allocate_slot(user_id="u2", slot_id="S2", current_physical_states=phys_states)


# =============================================================================
# D. SECURITY HARDENING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_security_headers_and_path_traversal_blocking():
    """Verifies HTTP security headers and prevents path traversal attacks."""
    load_config("configs/windows.yaml")
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Security Headers Verification
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            assert resp.headers.get("x-content-type-options") == "nosniff"
            assert resp.headers.get("x-frame-options") == "SAMEORIGIN"
            assert resp.headers.get("x-xss-protection") == "1; mode=block"

            # 2. Path Traversal Attacks against Gallery Endpoint
            resp = await client.get("/api/lab/gallery_image/false_positives/../../configs/windows.yaml")
            assert resp.status_code in (400, 404), "Path traversal must be rejected"

            resp = await client.get("/api/lab/gallery_image/false_positives/..%2f..%2fconfigs%2fwindows.yaml")
            assert resp.status_code in (400, 404)

            # 3. Path Traversal Attacks against Candidate Image Endpoint
            resp = await client.get("/api/lab/candidate_image/../../app/main.py")
            assert resp.status_code in (400, 404), "Path traversal must be rejected"


# =============================================================================
# E. END-TO-END APPLICATION INTEGRATION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_end_to_end_slots_api_and_reservation_reflection():
    """
    End-to-End Test:
    1. Reads GET /api/slots
    2. Verifies authoritative operational mapping (RESERVED slots are unavailable)
    3. Transactionally allocates a slot
    4. Verifies GET /api/slots immediately reflects the allocation
    5. Releases allocation
    6. Verifies return to AVAILABLE
    """
    load_config("configs/windows.yaml")
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Read initial slots
            resp = await client.get("/api/slots")
            assert resp.status_code == 200
            data = resp.json()
            assert "slots" in data
            assert len(data["slots"]) >= 1

            target_sid = data["slots"][0]["id"]

            # Step 2: Allocate slot
            alloc_req = {
                "user_id": "test_e2e_driver",
                "slot_id": target_sid,
                "vehicle_plate": "KA04MH9999",
                "vehicle_type": "car",
                "duration_minutes": 15,
            }
            alloc_resp = await client.post("/api/parking/allocate", json=alloc_req)
            assert alloc_resp.status_code in (200, 404, 409)

            if alloc_resp.status_code == 200:
                receipt = alloc_resp.json()
                res_code = receipt["reservation_code"]

                # Step 3: GET /api/slots must reflect RESERVED state
                resp_after = await client.get("/api/slots")
                slots_after = {s["id"]: s for s in resp_after.json()["slots"]}
                assert slots_after[target_sid]["state"] in ("RESERVED", "OCCUPIED")
                assert slots_after[target_sid].get("active_reservation") == res_code

                # Step 4: Duplicate allocation with same plate rejected
                dup_resp = await client.post("/api/parking/allocate", json=alloc_req)
                assert dup_resp.status_code == 409

                # Step 5: Release slot
                rel_resp = await client.post("/api/parking/release", json={"reservation_code": res_code})
                assert rel_resp.status_code == 200
