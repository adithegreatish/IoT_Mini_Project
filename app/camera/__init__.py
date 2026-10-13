"""Camera capture, hardware probing, quality assessment, and mocks."""

from app.camera.frame_provider import FrameProvider, FrameMetadata
from app.camera.device_scanner import scan_camera_devices
from app.camera.quality_gate import QualityGate, QualityAssessment
from app.camera.mock_camera import MockCamera
from app.camera.opencv_camera import OpenCVCamera
from app.camera.diagnostic import run_camera_diagnostic

__all__ = [
    "FrameProvider",
    "FrameMetadata",
    "scan_camera_devices",
    "QualityGate",
    "QualityAssessment",
    "MockCamera",
    "OpenCVCamera",
    "run_camera_diagnostic",
]
