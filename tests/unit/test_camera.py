"""Unit tests for camera capture workers and quality gate."""

import time
import numpy as np
from app.config.schema import CameraConfig
from app.camera.mock_camera import MockCamera
from app.camera.quality_gate import QualityGate
from app.services.camera_service import CameraService


def test_quality_gate_black_frame():
    gate = QualityGate()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assessment = gate.assess_frame(black_frame)
    assert assessment.status == "POOR"
    assert any("dark" in issue.lower() for issue in assessment.issues)


def test_quality_gate_good_frame():
    gate = QualityGate()
    # Create pattern with sharp contrast
    pattern = np.zeros((480, 640, 3), dtype=np.uint8)
    pattern[:240, :] = 200
    pattern[240:, :] = 80
    assessment = gate.assess_frame(pattern)
    assert assessment.status in ["EXCELLENT", "ACCEPTABLE"]
    assert assessment.blur_score > 0.0


def test_mock_camera_lifecycle_and_frames():
    cfg = CameraConfig(requested_width=640, requested_height=360, requested_fps=15, mock_mode=True)
    cam = MockCamera(cfg)
    assert cam.start() is True
    time.sleep(0.15)

    frame, meta = cam.get_latest_frame()
    assert frame is not None
    assert meta is not None
    assert frame.shape == (360, 640, 3)
    assert meta.is_mock is True
    assert cam.is_healthy() is True

    # Test camera loss scenario
    cam.set_scenario("camera_loss")
    time.sleep(0.1)
    assert cam.is_healthy() is False

    cam.stop()


def test_camera_service_preview_pause():
    cfg = CameraConfig(requested_width=640, requested_height=360, requested_fps=15, mock_mode=True)
    service = CameraService(cfg)
    service.start()
    time.sleep(0.1)

    status = service.get_status()
    assert status["preview_paused"] is False
    assert status["processing_active"] is True

    # Pause preview
    service.pause_preview()
    assert service.get_status()["preview_paused"] is True
    assert service.get_status()["processing_active"] is True  # inference processing unaffected!

    # Resume preview
    service.resume_preview()
    assert service.get_status()["preview_paused"] is False

    service.stop()
