"""Camera device enumeration and probing utility."""

import logging
from typing import Any, Dict, List
import cv2

logger = logging.getLogger("smart_parking.camera")


def scan_camera_devices(max_devices_to_check: int = 4) -> List[Dict[str, Any]]:
    """
    Safely probes system camera device indices (e.g. 0..3) to detect available hardware.
    Releases all OpenCV handles immediately.
    """
    found_devices: List[Dict[str, Any]] = []

    for index in range(max_devices_to_check):
        cap = None
        try:
            # Try DirectShow first on Windows if available, else standard fallback
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                # Read properties
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()

                # Grab single test frame to confirm hardware stream readiness
                ret, _ = cap.read()
                found_devices.append({
                    "device_index": index,
                    "is_available": True,
                    "can_read_frame": bool(ret),
                    "default_width": w,
                    "default_height": h,
                    "reported_fps": fps if fps > 0 else 30.0,
                    "backend_name": backend,
                    "display_name": f"Camera Device #{index} ({backend})",
                })
            else:
                # Camera indices are sequential; stop probing further indices
                break
        except Exception as e:
            logger.debug(f"Device probe at index {index} encountered: {e}")
            break
        finally:
            if cap is not None:
                cap.release()

    return found_devices
