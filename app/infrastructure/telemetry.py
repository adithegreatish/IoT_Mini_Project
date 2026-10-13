"""System telemetry, latency profiling, and performance metrics collector.

Measures the 7 core rates and fine-grained latency per pipeline stage without
synchronous I/O or logging bottlenecks:
1. Camera capture FPS
2. Live video delivery FPS
3. Vehicle detector inference FPS
4. Occupancy classifier update rate
5. Effective occupancy state update rate
6. End-to-end frame processing latency (mean, p50, p95)
7. Dropped / stale frame count
"""

import collections
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import psutil

from app.infrastructure.paths import paths


class StageTimer:
    """Context manager or manual stop-timer for lightweight stage latency recording."""

    __slots__ = ("_telemetry", "_stage_name", "_start_time")

    def __init__(self, telemetry: "TelemetryCollector", stage_name: str):
        self._telemetry = telemetry
        self._stage_name = stage_name
        self._start_time = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._telemetry.record_stage_latency(self._stage_name, duration_ms)


class TelemetryCollector:
    """Thread-safe, low-overhead rolling metrics and latency profiler."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._lock = threading.Lock()

        # Timestamps for rate calculation
        self._capture_timestamps = collections.deque(maxlen=window_size)
        self._delivery_timestamps = collections.deque(maxlen=window_size)
        self._detector_timestamps = collections.deque(maxlen=window_size)
        self._classifier_timestamps = collections.deque(maxlen=window_size)
        self._occupancy_timestamps = collections.deque(maxlen=window_size)

        # Stage latency deques (in milliseconds)
        self._stage_latencies: Dict[str, collections.deque] = {
            "camera_capture": collections.deque(maxlen=window_size),
            "frame_conversion": collections.deque(maxlen=window_size),
            "resize_normalization": collections.deque(maxlen=window_size),
            "onnx_inference": collections.deque(maxlen=window_size),
            "postprocessing": collections.deque(maxlen=window_size),
            "geometry_association": collections.deque(maxlen=window_size),
            "patch_classification": collections.deque(maxlen=window_size),
            "temporal_smoothing": collections.deque(maxlen=window_size),
            "jpeg_encoding": collections.deque(maxlen=window_size),
            "streaming_delivery": collections.deque(maxlen=window_size),
            "db_writes": collections.deque(maxlen=window_size),
            "e2e_pipeline": collections.deque(maxlen=window_size),
        }

        # Counters
        self._dropped_frames = 0
        self._stale_frames = 0
        self._total_frames_captured = 0
        self._start_time = time.time()

    def record_camera_capture(self, latency_ms: float = 0.0) -> None:
        now = time.time()
        with self._lock:
            self._capture_timestamps.append(now)
            self._total_frames_captured += 1
            if latency_ms > 0:
                self._stage_latencies["camera_capture"].append(latency_ms)

    def record_video_delivery(self, latency_ms: float = 0.0) -> None:
        now = time.time()
        with self._lock:
            self._delivery_timestamps.append(now)
            if latency_ms > 0:
                self._stage_latencies["streaming_delivery"].append(latency_ms)

    def record_detector_inference(
        self,
        onnx_ms: float = 0.0,
        preprocess_ms: float = 0.0,
        postprocess_ms: float = 0.0,
    ) -> None:
        now = time.time()
        with self._lock:
            self._detector_timestamps.append(now)
            if onnx_ms > 0:
                self._stage_latencies["onnx_inference"].append(onnx_ms)
            if preprocess_ms > 0:
                self._stage_latencies["resize_normalization"].append(preprocess_ms)
            if postprocess_ms > 0:
                self._stage_latencies["postprocessing"].append(postprocess_ms)

    def record_patch_classification(self, latency_ms: float = 0.0) -> None:
        now = time.time()
        with self._lock:
            self._classifier_timestamps.append(now)
            if latency_ms > 0:
                self._stage_latencies["patch_classification"].append(latency_ms)

    def record_occupancy_update(
        self,
        e2e_ms: float = 0.0,
        assoc_ms: float = 0.0,
        smooth_ms: float = 0.0,
        db_ms: float = 0.0,
    ) -> None:
        now = time.time()
        with self._lock:
            self._occupancy_timestamps.append(now)
            if e2e_ms > 0:
                self._stage_latencies["e2e_pipeline"].append(e2e_ms)
            if assoc_ms > 0:
                self._stage_latencies["geometry_association"].append(assoc_ms)
            if smooth_ms > 0:
                self._stage_latencies["temporal_smoothing"].append(smooth_ms)
            if db_ms > 0:
                self._stage_latencies["db_writes"].append(db_ms)

    def record_stage_latency(self, stage: str, duration_ms: float) -> None:
        with self._lock:
            if stage not in self._stage_latencies:
                self._stage_latencies[stage] = collections.deque(maxlen=self.window_size)
            self._stage_latencies[stage].append(duration_ms)

    def record_dropped_frame(self) -> None:
        with self._lock:
            self._dropped_frames += 1

    def record_stale_frame(self) -> None:
        with self._lock:
            self._stale_frames += 1

    def time_stage(self, stage_name: str) -> StageTimer:
        return StageTimer(self, stage_name)

    @staticmethod
    def _calculate_rate(timestamps: collections.deque) -> float:
        """Computes instantaneous operations per second across deque window."""
        if len(timestamps) < 2:
            return 0.0
        span = timestamps[-1] - timestamps[0]
        if span <= 0:
            return 0.0
        return round((len(timestamps) - 1) / span, 2)

    @staticmethod
    def _calculate_stats(samples: collections.deque) -> Dict[str, float]:
        """Calculates count, mean, p50, and p95 latency in ms."""
        if not samples:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        vals = sorted(samples)
        n = len(vals)
        mean_v = sum(vals) / n
        p50_v = vals[n // 2]
        p95_idx = min(int(n * 0.95), n - 1)
        p95_v = vals[p95_idx]
        return {
            "count": n,
            "mean_ms": round(mean_v, 2),
            "p50_ms": round(p50_v, 2),
            "p95_ms": round(p95_v, 2),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Extracts complete telemetry report covering all 7 mandatory rates and stages."""
        with self._lock:
            # 1. Core rates
            capture_fps = self._calculate_rate(self._capture_timestamps)
            delivery_fps = self._calculate_rate(self._delivery_timestamps)
            detector_fps = self._calculate_rate(self._detector_timestamps)
            classifier_fps = self._calculate_rate(self._classifier_timestamps)
            occupancy_update_rate = self._calculate_rate(self._occupancy_timestamps)

            # 2. End-to-end latency stats
            e2e_stats = self._calculate_stats(self._stage_latencies["e2e_pipeline"])

            # 3. Stage latencies breakdown
            stages_breakdown = {
                k: self._calculate_stats(v) for k, v in self._stage_latencies.items()
            }

            # 4. Process Resource snapshot
            process_mem_mb = 0.0
            cpu_pct = 0.0
            threads = 0
            try:
                proc = psutil.Process()
                process_mem_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
                cpu_pct = round(proc.cpu_percent(interval=None), 1)
                threads = proc.num_threads()
            except Exception:
                pass

            uptime_sec = round(time.time() - self._start_time, 1)

            return {
                "uptime_seconds": uptime_sec,
                # Seven Core Measurement Rates
                "core_metrics": {
                    "1_camera_capture_fps": capture_fps,
                    "2_live_video_delivery_fps": delivery_fps,
                    "3_detector_inference_fps": detector_fps,
                    "4_classifier_update_rate": classifier_fps,
                    "5_effective_occupancy_update_rate": occupancy_update_rate,
                    "6_e2e_latency_ms": e2e_stats,
                    "7_dropped_frames_count": self._dropped_frames,
                    "7_stale_frames_count": self._stale_frames,
                },
                # Detailed Latency Breakdown per Stage
                "stage_latencies_ms": stages_breakdown,
                # Process Memory and CPU Footprint
                "system_footprint": {
                    "process_rss_mb": process_mem_mb,
                    "process_cpu_pct": cpu_pct,
                    "active_threads": threads,
                    "total_frames_captured": self._total_frames_captured,
                },
            }

    def export_benchmark_report(self, filepath: Optional[str] = None) -> str:
        """Exports JSON benchmark output file and returns absolute path."""
        target = paths.resolve(filepath or "data/test_reports/telemetry_benchmark.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.get_metrics()
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(target)


# Global singleton telemetry collector
telemetry = TelemetryCollector(window_size=120)
