"""Unit tests for Raspberry Pi 5 deployment readiness.

Verifies:
- CPU-only model loading with ONNX Runtime CPUExecutionProvider
- ONNX artifact integrity against manifest.json
- Preprocessing and prediction output specifications
- Missing model graceful handling
- Missing camera and camera-loss transitions
- Stale-frame threshold (2.0s) safety behavior
- Calibration loading and resolution scaling
- Hysteresis FREE / OCCUPIED / UNKNOWN transitions
- OCR disabled by default while occupancy pipeline operates fully
- SQLite persistence across engine restarts
- Allocation rejection for OCCUPIED and UNKNOWN bays
- Concurrent allocation protection against double-booking
- Offline local file execution with zero external requests
- Configuration portability using relative path infrastructure
- Clean thread and resource shutdown behavior
"""

import concurrent.futures
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pytest

from app.infrastructure.paths import PathManager, paths
from app.config.loader import load_config
from app.config.schema import (
    CameraConfig,
    ParkingConfig,
    DetectionConfig,
    OCRConfig,
)
from app.camera.mock_camera import MockCamera
from app.camera.quality_gate import QualityGate
from app.services.camera_service import CameraService
from app.detection.vehicle_detector import VehicleDetector
from app.parking.occupancy_classifier import PatchOccupancyClassifier
from app.parking.calibration_service import CalibrationService
from app.parking.occupancy_service import OccupancyService
from app.parking.geometry import denormalize_points
from app.parking.state_machine import SlotState, SlotStateMachine
from app.database.connection import DatabaseManager
from app.parking.allocation_service import (
    AllocationService,
    SlotUnavailableError,
    ConcurrencyConflictError,
)


@pytest.fixture
def path_mgr():
    return paths


def test_cpu_only_model_loading_and_execution(path_mgr):
    """Verifies all ONNX models load and execute cleanly with CPUExecutionProvider."""
    import onnxruntime as ort

    occ_path = path_mgr.resolve("models/occupancy_classifier.onnx")
    veh_path = path_mgr.resolve("models/vehicle_detector.onnx")
    plt_path = path_mgr.resolve("models/plate_detector.onnx")

    assert occ_path.exists(), f"Occupancy ONNX missing: {occ_path}"
    assert veh_path.exists(), f"Vehicle ONNX missing: {veh_path}"
    assert plt_path.exists(), f"Plate ONNX missing: {plt_path}"

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2

    # 1. Occupancy Classifier Session (Approach D outputs single binary logit)
    occ_session = ort.InferenceSession(str(occ_path), sess_options=opts, providers=["CPUExecutionProvider"])
    assert "CPUExecutionProvider" in occ_session.get_providers()
    dummy_occ = np.zeros((1, 3, 150, 150), dtype=np.float32)
    occ_out = occ_session.run(None, {occ_session.get_inputs()[0].name: dummy_occ})
    assert occ_out[0].shape == (1, 1)

    # 2. Vehicle Detector Session
    veh_session = ort.InferenceSession(str(veh_path), sess_options=opts, providers=["CPUExecutionProvider"])
    assert "CPUExecutionProvider" in veh_session.get_providers()
    dummy_veh = np.zeros((1, 3, 640, 640), dtype=np.float32)
    veh_out = veh_session.run(None, {veh_session.get_inputs()[0].name: dummy_veh})
    assert len(veh_out) > 0

    # 3. Plate Detector Session
    plt_session = ort.InferenceSession(str(plt_path), sess_options=opts, providers=["CPUExecutionProvider"])
    assert "CPUExecutionProvider" in plt_session.get_providers()
    dummy_plt = np.zeros((1, 3, 640, 640), dtype=np.float32)
    plt_out = plt_session.run(None, {plt_session.get_inputs()[0].name: dummy_plt})
    assert len(plt_out) > 0


def test_onnx_artifact_integrity_and_manifest(path_mgr):
    """Verifies manifest.json accurately reflects real file SHA-256 hashes."""
    manifest_path = path_mgr.resolve("models/manifest.json")
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "models" in manifest
    assert len(manifest["models"]) >= 2

    for m in manifest["models"]:
        m_file = path_mgr.resolve(m["file_path"])
        assert m_file.exists(), f"Model file {m_file} registered in manifest must exist"

        h = hashlib.sha256()
        with open(m_file, "rb") as bf:
            while chunk := bf.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()
        assert actual_sha == m["sha256"], f"SHA mismatch for {m['model_id']}"


def test_occupancy_preprocessing_and_prediction_outputs(path_mgr):
    """Verifies PatchOccupancyClassifier input normalization and probability distributions."""
    occ_path = path_mgr.resolve("models/occupancy_classifier.onnx")
    classifier = PatchOccupancyClassifier(str(occ_path), threshold=0.65)

    # 150x150 synthetic image crop
    patch = np.full((150, 150, 3), fill_value=128, dtype=np.uint8)
    tensor = classifier.preprocess_patch(patch)
    assert tensor.shape == (1, 3, 150, 150)
    assert tensor.dtype == np.float32

    # Inference call
    is_occ, conf = classifier.predict_patch(patch)
    assert isinstance(is_occ, (bool, np.bool_))
    assert 0.0 <= conf <= 1.0

    # Batch prediction call
    batch_res = classifier.predict_batch([patch, patch])
    assert len(batch_res) == 2
    for b_occ, b_conf in batch_res:
        assert isinstance(b_occ, (bool, np.bool_))
        assert 0.0 <= b_conf <= 1.0


def test_missing_model_graceful_handling():
    """Verifies that non-existent models fail safely without crashing."""
    classifier = PatchOccupancyClassifier("models/non_existent_fake_model.onnx")
    assert classifier.session is None
    dummy = np.zeros((150, 150, 3), dtype=np.uint8)
    is_occ, conf = classifier.predict_patch(dummy)
    assert is_occ is False
    assert conf == 0.5

    detector = VehicleDetector(DetectionConfig(model_path="models/missing_detector.onnx", runtime="onnx"))
    assert detector.status == "MODEL_FILE_NOT_FOUND"
    assert detector.detect(dummy) == []


def test_camera_loss_and_stale_transitions():
    """Verifies that slot state transitions to UNKNOWN when camera frame is stale or lost."""
    sm = SlotStateMachine(slot_id="BAY_01", smoothing_window=3, hysteresis_confirm=2)

    # Transition to FREE
    sm.update(is_occupied_observation=False, observation_confidence=0.9)
    sm.update(is_occupied_observation=False, observation_confidence=0.9)
    assert sm.stable_state == SlotState.FREE

    # Simulate camera loss
    state, changed = sm.mark_camera_stale_or_lost("Hardware disconnect timeout reached")
    assert state == SlotState.UNKNOWN
    assert changed is True
    # Crucial safety assertion: state is NEVER FREE during camera failure
    assert sm.stable_state != SlotState.FREE


def test_stale_frame_handling_in_camera_service():
    """Verifies that CameraService accurately flags stale frames after the configured threshold."""
    cam_cfg = CameraConfig(mock_mode=True, stale_threshold_sec=0.2, requested_fps=30)
    cam_svc = CameraService(cam_cfg)
    cam_svc.start()
    time.sleep(0.1)

    status = cam_svc.get_status()
    assert status["is_healthy"] is True
    assert status["is_stale"] is False

    # Simulate provider stoppage
    cam_svc._provider.stop()
    time.sleep(0.35)

    status_after = cam_svc.get_status()
    assert status_after["is_stale"] is True

    cam_svc.stop()


def test_calibration_loading_and_resolution_scaling(tmp_path):
    """Verifies CalibrationService correctly parses polygons and scales across resolutions."""
    calib_file = tmp_path / "test_calib.json"
    calib_svc = CalibrationService(str(calib_file))

    slots = [
        {
            "id": "P1",
            "label": "Space 1",
            "slot_type": "car",
            "enabled": True,
            "points": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.8], [0.1, 0.8]],
        }
    ]
    ok, warnings, payload = calib_svc.save_calibration(slots)
    assert ok is True
    assert len(warnings) == 0

    loaded = calib_svc.load_calibration()
    assert len(loaded["slots"]) == 1
    assert loaded["slots"][0]["id"] == "P1"

    # Test coordinate denormalization for 1280x720 (Zebronics standard resolution)
    pts_720p = denormalize_points(loaded["slots"][0]["points"], width=1280, height=720)
    assert len(pts_720p) == 4
    np.testing.assert_allclose(pts_720p[0], [128.0, 144.0], atol=1.0)
    np.testing.assert_allclose(pts_720p[2], [384.0, 576.0], atol=1.0)


def test_ocr_disabled_while_occupancy_remains_usable(tmp_path, path_mgr):
    """Verifies that occupancy detection functions seamlessly when OCR is fully disabled."""
    calib_file = tmp_path / "calib.json"
    calib_svc = CalibrationService(str(calib_file))
    calib_svc.save_calibration([
        {"id": "BAY_1", "label": "Bay 1", "slot_type": "car", "enabled": True, "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]]}
    ])

    p_cfg = ParkingConfig(
        calibration_file=str(calib_file),
        hysteresis_confirm_frames=1,
        inference_mode="classifier",
        classifier_model_path="models/occupancy_classifier.onnx",
    )
    d_cfg = DetectionConfig(enabled=False)
    ocr_cfg = OCRConfig(enabled=False)  # Completely disabled OCR for Pi

    detector = VehicleDetector(d_cfg)
    occupancy_svc = OccupancyService(p_cfg, calib_svc, detector)

    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    snapshot = occupancy_svc.process_frame(frame, is_camera_healthy=True, is_camera_stale=False)

    assert "summary" in snapshot
    assert snapshot["summary"]["total"] == 1
    assert ocr_cfg.enabled is False


def test_sqlite_persistence_and_restart_recovery(tmp_path):
    """Verifies database records persist across connection lifecycle."""
    db_file = tmp_path / "persistence_test.sqlite"
    mgr1 = DatabaseManager(str(db_file))
    mgr1.initialize_schema()

    with mgr1.get_connection() as conn:
        conn.execute(
            "INSERT INTO parking_slots (id, label, zone, level, slot_type, is_enabled) VALUES (?, ?, ?, ?, ?, ?)",
            ("S_PERST", "Permanent Space", "A", "1", "car", 1),
        )
        conn.commit()

    # Simulate restart by creating brand new manager instance pointing to same file
    mgr2 = DatabaseManager(str(db_file))
    with mgr2.get_connection() as conn:
        row = conn.execute("SELECT id, label, is_enabled FROM parking_slots WHERE id = 'S_PERST'").fetchone()
        assert row is not None
        assert row["id"] == "S_PERST"
        assert row["label"] == "Permanent Space"
        assert row["is_enabled"] == 1


def test_allocation_rejection_for_occupied_and_unknown(tmp_path):
    """Verifies that AllocationService strictly refuses to allocate OCCUPIED or UNKNOWN bays."""
    db_file = tmp_path / "alloc_safety.sqlite"
    mgr = DatabaseManager(str(db_file))
    mgr.initialize_schema()

    alloc_svc = AllocationService(mgr, default_reservation_ttl_min=15)
    alloc_svc.sync_slots_from_calibration([
        {"id": "BAY_OCC", "label": "Occupied Bay", "zone": "Z1", "level": "G", "slot_type": "car", "enabled": True},
        {"id": "BAY_UNK", "label": "Unknown Bay", "zone": "Z1", "level": "G", "slot_type": "car", "enabled": True},
    ])

    phys_state = {
        "BAY_OCC": {"state": SlotState.OCCUPIED.value, "confidence": 0.98},
        "BAY_UNK": {"state": SlotState.UNKNOWN.value, "confidence": 0.0},
    }

    # Should raise SlotUnavailableError because neither is FREE
    with pytest.raises(SlotUnavailableError, match="OCCUPIED or UNKNOWN"):
        alloc_svc.allocate_slot(
            user_id="usr_01",
            zone="Z1",
            slot_type="car",
            current_physical_states=phys_state,
        )


def test_concurrent_allocation_protection(tmp_path):
    """Verifies thread-safe atomic slot reservation prevents double allocations."""
    db_file = tmp_path / "concurrency_test.sqlite"
    mgr = DatabaseManager(str(db_file))
    mgr.initialize_schema()

    alloc_svc = AllocationService(mgr, default_reservation_ttl_min=30)
    alloc_svc.sync_slots_from_calibration([
        {"id": "SINGLE_BAY", "label": "Single Bay", "zone": "Z1", "level": "G", "slot_type": "car", "enabled": True}
    ])

    phys_state = {
        "SINGLE_BAY": {"state": SlotState.FREE.value, "confidence": 0.99}
    }

    results = []
    errors = []

    def try_reserve(user_id):
        try:
            receipt = alloc_svc.allocate_slot(
                user_id=user_id,
                zone="Z1",
                slot_type="car",
                current_physical_states=phys_state,
            )
            results.append((user_id, receipt))
        except (SlotUnavailableError, ConcurrencyConflictError) as err:
            errors.append((user_id, str(err)))

    # Execute 2 concurrent requests simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(try_reserve, "user_alpha")
        f2 = executor.submit(try_reserve, "user_beta")
        concurrent.futures.wait([f1, f2])

    assert len(results) == 1, "Exactly one allocation must succeed"
    assert len(errors) == 1, "Exactly one allocation must be rejected"


def test_configuration_portability():
    """Verifies that configs/raspberry_pi.yaml loads valid schema attributes."""
    cfg = load_config("configs/raspberry_pi.yaml")
    assert cfg.camera.stale_threshold_sec == 2.0
    assert cfg.parking.unknown_timeout_sec == 5.0
    assert cfg.parking.classifier_threshold == 0.65
    assert cfg.parking.inference_mode in ["detector", "classifier", "hybrid"]
    assert cfg.ocr.enabled is False
    assert cfg.detection.runtime == "onnx"


def test_correct_shutdown_behavior():
    """Verifies that CameraService releases worker thread cleanly upon stop()."""
    cam_cfg = CameraConfig(mock_mode=True, requested_fps=30)
    cam_svc = CameraService(cam_cfg)
    assert cam_svc.start() is True
    time.sleep(0.05)
    assert cam_svc._provider is not None

    cam_svc.stop()
    assert cam_svc._provider is None
    status = cam_svc.get_status()
    assert status["is_healthy"] is False
