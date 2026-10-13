"""Camera self-test and hardware diagnostics utility."""

import time
from pathlib import Path
from typing import Any, Dict
import cv2

from app.camera.device_scanner import scan_camera_devices
from app.camera.quality_gate import QualityGate
from app.camera.mock_camera import MockCamera
from app.config.schema import CameraConfig
from app.infrastructure.paths import paths


def run_camera_diagnostic(camera_config: CameraConfig) -> Dict[str, Any]:
    """Runs a complete diagnostic check of camera hardware and stream quality."""
    devices = scan_camera_devices(max_devices_to_check=3)
    gate = QualityGate()

    snapshot_dir = paths.resolve("data/raw")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "diagnostic_test_frame.jpg"

    selected_index = camera_config.device_index
    physical_available = any(d["device_index"] == selected_index and d["can_read_frame"] for d in devices)

    if physical_available:
        # Test physical capture
        cap = cv2.VideoCapture(selected_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.requested_height)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            cv2.imwrite(str(snapshot_path), frame)
            assessment = gate.assess_frame(frame)
            return {
                "status": "PASS" if assessment.status in ["EXCELLENT", "ACCEPTABLE"] else "WARN",
                "is_physical_camera": True,
                "selected_device_index": selected_index,
                "devices_found": devices,
                "negotiated_resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "quality_assessment": assessment.__dict__,
                "snapshot_saved_at": str(snapshot_path),
            }

    # Physical device not present -> test mock / diagnostic frame
    mock = MockCamera(camera_config)
    mock.start()
    time.sleep(0.1)
    frame, meta = mock.get_latest_frame()
    mock.stop()

    if frame is not None:
        cv2.imwrite(str(snapshot_path), frame)
        assessment = gate.assess_frame(frame)
        return {
            "status": "WARN",  # WARN because physical camera was absent, but fallback operational
            "is_physical_camera": False,
            "selected_device_index": selected_index,
            "devices_found": devices,
            "warning": f"Physical camera index {selected_index} not detected. Synthetic mock feed verified.",
            "negotiated_resolution": f"{frame.shape[1]}x{frame.shape[0]}",
            "quality_assessment": assessment.__dict__,
            "snapshot_saved_at": str(snapshot_path),
        }

    return {
        "status": "FAIL",
        "is_physical_camera": False,
        "selected_device_index": selected_index,
        "devices_found": devices,
        "error": "Failed to acquire frame from both physical and mock providers.",
    }
