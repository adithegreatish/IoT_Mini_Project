"""Synthetic camera frame generator for offline development, tests, and demo mode."""

import math
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from app.camera.frame_provider import FrameProvider, FrameMetadata
from app.config.schema import CameraConfig


class MockCamera(FrameProvider):
    """Generates synthetic parking lot camera feeds with customizable scenarios."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.width = config.requested_width
        self.height = config.requested_height
        self.target_fps = max(1, config.requested_fps)
        self.scenario = config.synthetic_scenario

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._metadata: Optional[FrameMetadata] = None
        self._frame_count = 0
        self._start_time = 0.0

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="MockCameraWorker")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._current_frame = None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        with self._lock:
            if self._current_frame is None:
                return None, None
            return self._current_frame.copy(), self._metadata

    def is_healthy(self) -> bool:
        if self.scenario == "camera_loss":
            return False
        return self._running and self._current_frame is not None

    def reconnect(self) -> bool:
        self.stop()
        return self.start()

    def set_scenario(self, scenario: str) -> None:
        with self._lock:
            self.scenario = scenario

    def _worker_loop(self) -> None:
        interval = 1.0 / self.target_fps
        while self._running:
            loop_start = time.time()

            if self.scenario == "camera_loss":
                # Simulate loss of signal
                with self._lock:
                    self._current_frame = None
                    self._metadata = FrameMetadata(
                        timestamp=loop_start,
                        frame_index=self._frame_count,
                        width=self.width,
                        height=self.height,
                        observed_fps=0.0,
                        is_mock=True,
                        is_stale=True,
                    )
                time.sleep(interval)
                continue

            frame = self._render_synthetic_scene(self._frame_count)
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._start_time
            observed_fps = self._frame_count / max(0.001, elapsed)

            with self._lock:
                self._current_frame = frame
                self._metadata = FrameMetadata(
                    timestamp=now,
                    frame_index=self._frame_count,
                    width=self.width,
                    height=self.height,
                    observed_fps=round(observed_fps, 1),
                    is_mock=True,
                    is_stale=False,
                )

            delay = interval - (time.time() - loop_start)
            if delay > 0:
                time.sleep(delay)

    def _render_synthetic_scene(self, step: int) -> np.ndarray:
        w, h = self.width, self.height
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Dark asphalt background with gradient
        for y in range(h):
            grad = int(35 + (y / h) * 20)
            img[y, :] = [grad, grad, grad]

        # 2. Draw parking slot guide markings (white and yellow lines)
        slot_xs = [int(w * 0.05), int(w * 0.28), int(w * 0.51), int(w * 0.74), int(w * 0.95)]
        y_top = int(h * 0.40)
        y_bottom = int(h * 0.92)

        for sx in slot_xs:
            cv2.line(img, (sx, y_top), (sx, y_bottom), (240, 240, 240), 4)

        # Baseline bay dividers
        cv2.line(img, (slot_xs[0], y_top), (slot_xs[-1], y_top), (220, 220, 220), 4)
        cv2.line(img, (slot_xs[0], y_bottom), (slot_xs[-1], y_bottom), (220, 220, 220), 4)

        # Bay numbers text
        cv2.putText(img, "S01", (slot_xs[0] + 40, y_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(img, "S02", (slot_xs[1] + 40, y_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(img, "S03", (slot_xs[2] + 40, y_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(img, "S04", (slot_xs[3] + 40, y_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # 3. Determine vehicle presence based on scenario
        t = time.time()
        s01_occ = False
        s02_occ = False
        s03_occ = False
        s04_occ = False

        if self.scenario == "all_occupied":
            s01_occ = s02_occ = s03_occ = s04_occ = True
        elif self.scenario == "all_free":
            s01_occ = s02_occ = s03_occ = s04_occ = False
        elif self.scenario == "mixed":
            s01_occ = True
            s03_occ = True
        elif self.scenario == "straddling":
            # Vehicle straddles between S01 and S02
            self._draw_synthetic_vehicle(img, (slot_xs[0] + slot_xs[1]) // 2 + 30, y_top + 40, is_car=True, color=(160, 40, 40), plate_text="DL01AA9999")
        else:  # "normal" periodic cycling
            cycle = (t % 40)
            s01_occ = True
            s02_occ = cycle > 15
            s03_occ = cycle < 28
            s04_occ = math.sin(t * 0.4) > 0.0

        if s01_occ and self.scenario != "straddling":
            self._draw_synthetic_vehicle(img, slot_xs[0] + 35, y_top + 30, is_car=True, color=(180, 60, 40), plate_text="DL 01 AB 1234")

        if s02_occ:
            self._draw_synthetic_vehicle(img, slot_xs[1] + 35, y_top + 30, is_car=True, color=(40, 140, 200), plate_text="MH 12 CD 5678")

        if s03_occ:
            self._draw_synthetic_vehicle(img, slot_xs[2] + 35, y_top + 30, is_car=True, color=(60, 180, 60), plate_text="KA 03 EF 9012")

        if s04_occ:
            self._draw_synthetic_vehicle(img, slot_xs[3] + 45, y_top + 50, is_car=False, color=(200, 180, 40), plate_text="HR 26 GH 3456")

        # 4. Simulation Banner and metadata
        cv2.rectangle(img, (0, 0), (w, 36), (20, 24, 33), -1)
        cv2.putText(
            img,
            f"SIMULATION MODE - SCENARIO: {self.scenario.upper()} - FPS: {self.target_fps}",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 240, 255),
            2,
        )
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, time_str, (w - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return img

    def _draw_synthetic_vehicle(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        is_car: bool = True,
        color: Tuple[int, int, int] = (150, 50, 50),
        plate_text: str = "DL01AB1234",
    ) -> None:
        """Draws a vehicle polygon with windshield, headlights, and HSRP-style license plate."""
        if is_car:
            vw = int(self.width * 0.17)
            vh = int(self.height * 0.44)

            # Vehicle chassis
            cv2.rectangle(img, (x, y), (x + vw, y + vh), color, -1)
            cv2.rectangle(img, (x, y), (x + vw, y + vh), (30, 30, 30), 2)

            # Windshield
            win_y = y + int(vh * 0.15)
            win_h = int(vh * 0.30)
            cv2.rectangle(img, (x + 10, win_y), (x + vw - 10, win_y + win_h), (40, 50, 60), -1)

            # Headlights
            cv2.circle(img, (x + 15, y + vh - 15), 10, (220, 240, 255), -1)
            cv2.circle(img, (x + vw - 15, y + vh - 15), 10, (220, 240, 255), -1)

            # License Plate with Blue IND Region
            pw = int(vw * 0.58)
            ph = int(vh * 0.14)
            px = x + (vw - pw) // 2
            py = y + vh - ph - 8

            # White plate background
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 0, 0), 2)

            # Blue IND strip on left edge
            ind_w = max(6, int(pw * 0.16))
            cv2.rectangle(img, (px, py), (px + ind_w, py + ph), (180, 70, 0), -1)  # BGR blue
            cv2.putText(img, "IND", (px + 1, py + ph - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

            # Plate characters
            cv2.putText(img, plate_text.replace(" ", ""), (px + ind_w + 3, py + ph - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

        else:
            # Motorcycle / scooter
            vw = int(self.width * 0.09)
            vh = int(self.height * 0.38)

            cv2.rectangle(img, (x, y), (x + vw, y + vh), color, -1)
            cv2.circle(img, (x + vw // 2, y + 20), 12, (20, 20, 20), -1)
            # Rear plate
            pw = int(vw * 0.75)
            ph = int(vh * 0.16)
            px = x + (vw - pw) // 2
            py = y + vh - ph - 6
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 0, 0), 1)
            cv2.putText(img, plate_text[:6], (px + 2, py + ph - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
