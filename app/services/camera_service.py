"""Authoritative camera service owning hardware capture and frame distribution."""

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from app.camera.frame_provider import FrameMetadata, FrameProvider
from app.camera.mock_camera import MockCamera
from app.camera.opencv_camera import OpenCVCamera
from app.config.schema import CameraConfig

logger = logging.getLogger("smart_parking.camera_service")


class CameraService:
    """
    Single authoritative owner of the video capture device.
    Distributes frames to computer vision pipelines and MJPEG streaming endpoints.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._provider: Optional[FrameProvider] = None
        self._lock = threading.Lock()

        # Operational states
        self.preview_paused: bool = False
        self.processing_active: bool = True
        self._last_jpeg: Optional[bytes] = None
        self._last_jpeg_ts: float = 0.0

    def start(self) -> bool:
        with self._lock:
            if self._provider is not None:
                return True

            logger.info("Initializing camera capture worker...")
            if self.config.mock_mode:
                self._provider = MockCamera(self.config)
            else:
                self._provider = OpenCVCamera(self.config)

            return self._provider.start()

    def stop(self) -> None:
        with self._lock:
            if self._provider is not None:
                logger.info("Stopping camera capture worker...")
                self._provider.stop()
                self._provider = None

    def reconnect(self) -> bool:
        with self._lock:
            if self._provider is not None:
                return self._provider.reconnect()
            return self.start()

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Returns the most recent frame for inference or streaming."""
        with self._lock:
            if self._provider is None:
                return None, None
            return self._provider.get_latest_frame()

    def set_scenario(self, scenario: str) -> None:
        """Sets synthetic scenario if running MockCamera."""
        with self._lock:
            if isinstance(self._provider, MockCamera):
                self._provider.set_scenario(scenario)
            elif isinstance(self._provider, OpenCVCamera) and self._provider._fallback_mock:
                self._provider._fallback_mock.set_scenario(scenario)

    def pause_preview(self) -> None:
        """Pauses visual MJPEG stream; does NOT stop inference processing."""
        self.preview_paused = True

    def resume_preview(self) -> None:
        """Resumes visual MJPEG stream."""
        self.preview_paused = False

    def toggle_processing(self) -> bool:
        """Toggles background pipeline processing on/off."""
        self.processing_active = not self.processing_active
        return self.processing_active

    def get_status(self) -> Dict[str, Any]:
        """Returns current camera operational status."""
        frame, meta = self.get_latest_frame()
        is_healthy = self._provider.is_healthy() if self._provider else False

        width = meta.width if meta else self.config.requested_width
        height = meta.height if meta else self.config.requested_height
        observed_fps = meta.observed_fps if meta else 0.0
        is_mock = meta.is_mock if meta else self.config.mock_mode
        is_stale = meta.is_stale if meta else True

        state_label = "STREAMING" if (is_healthy and not is_stale) else ("STALE" if is_stale else "DISCONNECTED")

        return {
            "state": state_label,
            "is_healthy": is_healthy,
            "is_stale": is_stale,
            "is_mock": is_mock,
            "device_index": self.config.device_index,
            "backend": self.config.backend,
            "requested_resolution": f"{self.config.requested_width}x{self.config.requested_height}",
            "negotiated_resolution": f"{width}x{height}",
            "observed_fps": observed_fps,
            "preview_paused": self.preview_paused,
            "processing_active": self.processing_active,
            "last_frame_timestamp": meta.timestamp if meta else None,
        }

    def generate_mjpeg_frame(self, overlay_renderer=None) -> Optional[bytes]:
        """Encodes latest frame to JPEG bytes with optional overlay and caching."""
        if self.preview_paused:
            if self._last_jpeg:
                return self._last_jpeg

        frame, meta = self.get_latest_frame()
        if frame is None:
            # Render a clean "NO SIGNAL / DISCONNECTED" banner
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "CAMERA SIGNAL LOST / RECONNECTING...",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
            _, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 65])
            return buf.tobytes()

        # Cache check: if the frame timestamp is identical, reuse the cached JPEG
        frame_ts = meta.timestamp if meta else 0.0
        if self._last_jpeg is not None and frame_ts > 0 and frame_ts == self._last_jpeg_ts:
            return self._last_jpeg

        # Apply overlay if provided
        if overlay_renderer is not None and callable(overlay_renderer):
            frame = overlay_renderer(frame)

        # Quality 55 provides ultra-fast 30+ FPS encoding with ~70% lower bandwidth on ARM
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ret:
            self._last_jpeg = buf.tobytes()
            self._last_jpeg_ts = frame_ts
            return self._last_jpeg
        return None
