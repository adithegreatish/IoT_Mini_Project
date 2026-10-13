"""Unit tests verifying bounded memory configurations and the 7 telemetry rates."""

import json
import time
import pytest
from app.config.schema import CameraConfig
from app.database.connection import DatabaseManager
from app.infrastructure.telemetry import TelemetryCollector
from app.services.camera_service import CameraService


def test_sqlite_memory_pragmas(tmp_path):
    """Confirms SQLite cache_size is 8 MiB (-8192 KiB) and mmap_size is 16 MiB."""
    db_file = tmp_path / "test_memory.sqlite"
    mgr = DatabaseManager(str(db_file), cache_size_mb=8, mmap_size_mb=16)

    conn = mgr.get_connection()
    cursor = conn.cursor()

    # Query effective cache_size from SQLite
    cursor.execute("PRAGMA cache_size;")
    cache_size = cursor.fetchone()[0]
    # Negative number indicates size in KiB (-8192 = 8 MiB)
    assert cache_size == -8192

    # Query effective mmap_size from SQLite
    cursor.execute("PRAGMA mmap_size;")
    mmap_size = cursor.fetchone()[0]
    assert mmap_size == 16 * 1024 * 1024

    # Confirm WAL mode is active
    cursor.execute("PRAGMA journal_mode;")
    journal = cursor.fetchone()[0].upper()
    assert journal == "WAL"

    conn.close()


def test_camera_service_bounded_ring_buffer():
    """Confirms camera service uses small bounded buffer (e.g. 4 frames max)."""
    cfg = CameraConfig(mock_mode=True)
    svc = CameraService(cfg, max_buffer_mb=1024, ring_buffer_max_frames=4)

    assert svc.ring_buffer_max_frames == 4
    assert svc._ring_buffer.maxlen == 4

    # Status must report max_buffer_mb as 1024
    status = svc.get_status()
    assert status["max_buffer_mb"] == 1024
    assert status["ring_buffer_frames"] == 0


def test_telemetry_collector_seven_core_rates(tmp_path):
    """Confirms TelemetryCollector tracks all 7 mandatory rates and exports report."""
    collector = TelemetryCollector(window_size=50)

    # Simulate 5 camera captures
    for _ in range(5):
        collector.record_camera_capture(latency_ms=2.5)
        time.sleep(0.005)

    # Simulate 5 deliveries
    for _ in range(5):
        collector.record_video_delivery(latency_ms=1.2)
        time.sleep(0.005)

    # Simulate detector runs
    for _ in range(3):
        collector.record_detector_inference(onnx_ms=25.0, preprocess_ms=1.5, postprocess_ms=2.0)
        time.sleep(0.005)

    # Simulate classifier runs
    for _ in range(3):
        collector.record_patch_classification(latency_ms=4.0)

    # Simulate occupancy update with e2e latency
    for _ in range(3):
        collector.record_occupancy_update(e2e_ms=32.0, assoc_ms=2.1, smooth_ms=0.5, db_ms=1.0)

    collector.record_dropped_frame()
    collector.record_stale_frame()

    metrics = collector.get_metrics()
    core = metrics["core_metrics"]

    # All 7 core metrics must be present and non-ambiguous
    assert "1_camera_capture_fps" in core
    assert "2_live_video_delivery_fps" in core
    assert "3_detector_inference_fps" in core
    assert "4_classifier_update_rate" in core
    assert "5_effective_occupancy_update_rate" in core
    assert "6_e2e_latency_ms" in core
    assert "7_dropped_frames_count" in core
    assert "7_stale_frames_count" in core

    assert core["7_dropped_frames_count"] == 1
    assert core["7_stale_frames_count"] == 1
    assert core["6_e2e_latency_ms"]["count"] == 3
    assert core["6_e2e_latency_ms"]["p50_ms"] == 32.0

    # Stage latencies breakdown
    stages = metrics["stage_latencies_ms"]
    assert "onnx_inference" in stages
    assert "streaming_delivery" in stages
    assert stages["onnx_inference"]["p50_ms"] == 25.0

    # Export benchmark report
    out_file = tmp_path / "benchmark.json"
    exported_path = collector.export_benchmark_report(str(out_file))
    assert out_file.exists()
    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["core_metrics"]["7_dropped_frames_count"] == 1
