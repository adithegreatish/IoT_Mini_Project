"""Abstract frame provider interface for video and camera capture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class FrameMetadata:
    timestamp: float
    frame_index: int
    width: int
    height: int
    observed_fps: float
    is_mock: bool = False
    is_stale: bool = False


class FrameProvider(ABC):
    """Abstract interface defining camera capture workers."""

    @abstractmethod
    def start(self) -> bool:
        """Starts the capture worker thread."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops capture and releases hardware resources."""
        pass

    @abstractmethod
    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Returns the most recent available frame and its metadata."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Returns whether the capture stream is currently active and fresh."""
        pass

    @abstractmethod
    def reconnect(self) -> bool:
        """Triggers reconnection attempt."""
        pass
