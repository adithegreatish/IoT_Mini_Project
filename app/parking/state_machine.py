"""Slot state machine with hysteresis, temporal smoothing, and stale protection."""

from collections import deque
from enum import Enum
import time
from typing import Dict, List, Optional, Tuple


class SlotState(str, Enum):
    FREE = "FREE"
    OCCUPIED = "OCCUPIED"
    UNKNOWN = "UNKNOWN"


class SlotStateMachine:
    """
    Manages state transitions for a single parking slot.
    Applies temporal window smoothing and hysteresis confirmation to prevent flicker.
    Protects against false FREE transitions during camera signal loss.
    """

    def __init__(
        self,
        slot_id: str,
        smoothing_window: int = 5,
        hysteresis_confirm: int = 3,
        unknown_timeout_sec: float = 5.0,
    ):
        self.slot_id = slot_id
        self.smoothing_window = smoothing_window
        self.hysteresis_confirm = hysteresis_confirm
        self.unknown_timeout_sec = unknown_timeout_sec

        self.stable_state: SlotState = SlotState.UNKNOWN
        self.confidence: float = 0.5
        self.reason: str = "Awaiting initial observation"

        self._recent_observations: deque = deque(maxlen=smoothing_window)
        self._pending_state: Optional[SlotState] = None
        self._pending_count: int = 0
        self._last_verified_time: float = time.time()

    def update(self, is_occupied_observation: bool, observation_confidence: float) -> Tuple[SlotState, bool]:
        """
        Ingests a frame-level occupancy observation.
        Returns: (current_stable_state, state_changed_flag)
        """
        now = time.time()
        self._last_verified_time = now

        obs_state = SlotState.OCCUPIED if is_occupied_observation else SlotState.FREE
        self._recent_observations.append((obs_state, observation_confidence))

        # Check for state transition candidate
        if obs_state != self.stable_state:
            if obs_state == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state = obs_state
                self._pending_count = 1

            # Hysteresis gate: must be consistent across hysteresis_confirm observations
            if self._pending_count >= self.hysteresis_confirm:
                prev_state = self.stable_state
                self.stable_state = obs_state
                self.confidence = observation_confidence
                self.reason = f"Consistent evidence over {self._pending_count} frames"
                self._pending_state = None
                self._pending_count = 0
                return self.stable_state, True
        else:
            self._pending_state = None
            self._pending_count = 0
            self.confidence = max(0.6, observation_confidence)
            self.reason = "Consistent with stable state"

        return self.stable_state, False

    def mark_camera_stale_or_lost(self, reason: str = "Camera disconnected or stale") -> Tuple[SlotState, bool]:
        """
        Invoked when camera frames are stale or absent.
        Guarantees that lost cameras NEVER report slots as FREE!
        """
        changed = False
        if self.stable_state != SlotState.UNKNOWN:
            self.stable_state = SlotState.UNKNOWN
            changed = True
        self.confidence = 0.0
        self.reason = reason
        self._pending_state = None
        self._pending_count = 0
        return self.stable_state, changed
