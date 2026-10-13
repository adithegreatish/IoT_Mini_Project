"""Physical OpenCV camera capture worker with buffer draining and auto-reconnect."""

import logging
import platform
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from app.camera.frame_provider import FrameProvider, FrameMetadata
from app.camera.mock_camera import MockCamera
from app.config.schema import CameraConfig

logger = logging.getLogger("smart_parking.camera")


class OpenCVCamera(FrameProvider):
    """
    Dedicated camera service managing OpenCV VideoCapture.
    Owns device connection, measures actual negotiated resolution and FPS,
    prevents buffer lag by draining frames, and handles reconnection.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.device_index = config.device_index
        self.requested_width = config.requested_width
        self.requested_height = config.requested_height
        self.requested_fps = config.requested_fps

        self.actual_width = 0
        self.actual_height = 0
        self.observed_fps = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None

        self._current_frame: Optional[np.ndarray] = None
        self._metadata: Optional[FrameMetadata] = None
        self._frame_count = 0
        self._last_frame_time = 0.0

        self._fallback_mock: Optional[MockCamera] = None
        self._is_using_mock = False

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="OpenCVCameraWorker")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()
        if self._fallback_mock:
            self._fallback_mock.stop()
            self._fallback_mock = None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        if self._is_using_mock and self._fallback_mock:
            return self._fallback_mock.get_latest_frame()

        with self._lock:
            if self._current_frame is None:
                return None, None
            # Check staleness
            now = time.time()
            is_stale = (now - self._last_frame_time) > self.config.stale_threshold_sec
            meta = self._metadata
            if meta and is_stale:
                meta.is_stale = True
            return self._current_frame, meta

    def is_healthy(self) -> bool:
        if self._is_using_mock and self._fallback_mock:
            return self._fallback_mock.is_healthy()
        with self._lock:
            if not self._running or self._current_frame is None:
                return False
            return (time.time() - self._last_frame_time) <= self.config.stale_threshold_sec

    def reconnect(self) -> bool:
        logger.info("Manual camera reconnect requested.")
        self._release_capture()
        return True

    def _open_capture(self) -> bool:
        """Attempts opening the physical device."""
        backend = cv2.CAP_ANY
        if self.config.backend == "DSHOW" and platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
            backend = cv2.CAP_DSHOW
        elif self.config.backend == "V4L2" and platform.system() == "Linux" and hasattr(cv2, "CAP_V4L2"):
            backend = cv2.CAP_V4L2

        try:
            logger.info(f"Opening camera index {self.device_index} with backend {self.config.backend}...")
            cap = cv2.VideoCapture(self.device_index, backend)
            if not cap.isOpened():
                cap.release()
                return False

            # 1. Set requested resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

            # 2. Enforce MJPG FourCC AFTER setting resolution so V4L2 driver does not fallback to uncompressed YUYV
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

            # 3. Request high FPS and minimal buffer queue
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                return False

            h, w = test_frame.shape[:2]
            self.actual_width = w
            self.actual_height = h
            self._cap = cap
            logger.info(f"Camera opened successfully. Negotiated: {w}x{h} @ {self.requested_fps} FPS (backend: {self.config.backend})")
            return True
        except Exception as e:
            logger.warning(f"Error opening camera {self.device_index}: {e}")
            return False

    def _release_capture(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def _worker_loop(self) -> None:
        fps_history = []
        last_calc_time = time.time()

        while self._running:
            # 1. Ensure camera is open
            if self._cap is None or not self._cap.isOpened():
                success = self._open_capture()
                if not success:
                    if self.config.mock_mode or self._is_using_mock:
                        if not self._is_using_mock:
                            logger.info("Physical camera unavailable; falling back to MockCamera.")
                            self._fallback_mock = MockCamera(self.config)
                            self._fallback_mock.start()
                            self._is_using_mock = True
                        time.sleep(1.0)
                        continue
                    else:
                        logger.warning(
                            f"Physical camera {self.device_index} unavailable. Retrying in {self.config.reconnect_interval_sec}s..."
                        )
                        time.sleep(self.config.reconnect_interval_sec)
                        continue
                else:
                    if self._is_using_mock and self._fallback_mock:
                        self._fallback_mock.stop()
                        self._fallback_mock = None
                        self._is_using_mock = False

            # 2. Read frame (blocks for next hardware frame interrupt in V4L2)
            ret, frame = self._cap.read()
            now = time.time()

            if not ret or frame is None:
                logger.warning("Failed to grab frame from physical camera.")
                self._release_capture()
                time.sleep(self.config.reconnect_interval_sec)
                continue

            self._frame_count += 1
            self._last_frame_time = now

            # Calculate FPS moving average
            fps_history.append(now)
            if len(fps_history) > 30:
                fps_history.pop(0)
            if len(fps_history) > 1:
                duration = fps_history[-1] - fps_history[0]
                if duration > 0:
                    self.observed_fps = round((len(fps_history) - 1) / duration, 1)

            with self._lock:
                self._current_frame = frame
                self._metadata = FrameMetadata(
                    timestamp=now,
                    frame_index=self._frame_count,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    observed_fps=self.observed_fps,
                    is_mock=False,
                    is_stale=False,
                )
