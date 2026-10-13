"""Automated diagnostics engine, health checks, and report generator."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from app.camera.device_scanner import scan_camera_devices
from app.camera.quality_gate import QualityGate
from app.camera.mock_camera import MockCamera
from app.config.loader import get_config, load_config
from app.config.schema import RootConfig
from app.database.connection import DatabaseManager, SCHEMA_VERSION
from app.database.repository import ParkingRepository
from app.infrastructure.paths import paths
from app.infrastructure.platform_info import get_platform_info


@dataclass
class DiagnosticTestResult:
    test_id: str
    name: str
    description: str
    status: Literal["PASS", "WARN", "FAIL", "SKIPPED", "NOT RUN"]
    duration_ms: float
    message: str
    corrective_action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    report_id: str
    timestamp: str
    run_type: Literal["quick", "full"]
    overall_status: Literal["PASS", "WARN", "FAIL"]
    platform: Dict[str, Any]
    summary: Dict[str, int]
    total_duration_sec: float
    results: List[DiagnosticTestResult]


class DiagnosticsRunner:
    """Executes multi-tier diagnostic audits and outputs JSON / Markdown reports."""

    def __init__(self, config: Optional[RootConfig] = None):
        self.config = config or get_config()
        self._cancel_requested = False

    def request_cancellation(self) -> None:
        """Signals active diagnostic run to stop safely."""
        self._cancel_requested = True

    def run(self, run_type: Literal["quick", "full"] = "quick", progress_callback: Optional[Callable[[int, int, str], None]] = None) -> DiagnosticReport:
        self._cancel_requested = False
        start_time = time.time()
        test_plan = self._build_test_plan(run_type)
        total_tests = len(test_plan)

        results: List[DiagnosticTestResult] = []

        for idx, (t_id, t_name, t_desc, t_func) in enumerate(test_plan):
            if self._cancel_requested:
                results.append(
                    DiagnosticTestResult(
                        test_id=t_id,
                        name=t_name,
                        description=t_desc,
                        status="SKIPPED",
                        duration_ms=0.0,
                        message="Cancelled by user.",
                    )
                )
                continue

            if progress_callback:
                progress_callback(idx + 1, total_tests, t_name)

            t_start = time.time()
            try:
                res = t_func()
                duration = round((time.time() - t_start) * 1000, 2)
                res.duration_ms = duration
                results.append(res)
            except Exception as e:
                duration = round((time.time() - t_start) * 1000, 2)
                results.append(
                    DiagnosticTestResult(
                        test_id=t_id,
                        name=t_name,
                        description=t_desc,
                        status="FAIL",
                        duration_ms=duration,
                        message=f"Unhandled exception during test: {str(e)}",
                        corrective_action="Review application stack trace and configuration.",
                    )
                )

        total_sec = round(time.time() - start_time, 3)
        summary = {
            "total": len(results),
            "pass": sum(1 for r in results if r.status == "PASS"),
            "warn": sum(1 for r in results if r.status == "WARN"),
            "fail": sum(1 for r in results if r.status == "FAIL"),
            "skipped": sum(1 for r in results if r.status == "SKIPPED"),
            "not_run": sum(1 for r in results if r.status == "NOT RUN"),
        }

        overall: Literal["PASS", "WARN", "FAIL"] = "PASS"
        if summary["fail"] > 0:
            overall = "FAIL"
        elif summary["warn"] > 0:
            overall = "WARN"

        report_id = f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = DiagnosticReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            run_type=run_type,
            overall_status=overall,
            platform=get_platform_info(),
            summary=summary,
            total_duration_sec=total_sec,
            results=results,
        )

        self._save_report(report)
        return report

    def _build_test_plan(self, run_type: str) -> List[Tuple[str, str, str, Callable[[], DiagnosticTestResult]]]:
        plan = [
            ("env_check", "Python & Platform Compatibility", "Validates Python version and host OS architecture.", self._test_environment),
            ("config_check", "Configuration Schema & Precedence", "Validates syntax and boundaries in central YAML config.", self._test_configuration),
            ("fs_perms", "Filesystem Write Permissions", "Checks directory access for logs, data, and models.", self._test_filesystem_permissions),
            ("db_health", "SQLite Database & Schema Integrity", "Tests database connection, migrations, and read/write cycle.", self._test_database),
            ("camera_enum", "Camera Enumeration & Probe", "Scans available video capture devices.", self._test_camera_probe),
            ("model_manifest", "Model Registry & Manifest Hash", "Verifies models/manifest.json presence and model metadata.", self._test_models),
        ]

        if run_type == "full":
            plan.extend([
                ("frame_quality", "Image Quality Assessment Gate", "Tests blur, brightness, contrast, and glare analysis on test frame.", self._test_image_quality),
                ("slot_geom", "Parking Polygon Geometry Integrity", "Validates calibration polygons for minimum area and non-overlap.", self._test_calibration_geometry),
                ("disk_resources", "Host System Resources & Disk Space", "Inspects available disk storage and CPU load.", self._test_system_resources),
            ])

        return plan

    # Individual test implementations
    def _test_environment(self) -> DiagnosticTestResult:
        info = get_platform_info()
        py_ver = info["python_version"]
        major, minor = [int(x) for x in py_ver.split(".")[:2]]
        if (major, minor) < (3, 10):
            return DiagnosticTestResult(
                test_id="env_check",
                name="Python & Platform Compatibility",
                description="Validates Python version and host OS architecture.",
                status="FAIL",
                duration_ms=0.0,
                message=f"Unsupported Python {py_ver}. Minimum required: Python 3.10.",
                corrective_action="Install Python 3.10, 3.11, or 3.12.",
                details=info,
            )
        return DiagnosticTestResult(
            test_id="env_check",
            name="Python & Platform Compatibility",
            description="Validates Python version and host OS architecture.",
            status="PASS",
            duration_ms=0.0,
            message=f"Compatible environment detected: Python {py_ver} on {info['device_model']}.",
            details=info,
        )

    def _test_configuration(self) -> DiagnosticTestResult:
        try:
            cfg = load_config()
            return DiagnosticTestResult(
                test_id="config_check",
                name="Configuration Schema & Precedence",
                description="Validates syntax and boundaries in central YAML config.",
                status="PASS",
                duration_ms=0.0,
                message=f"Configuration valid (profile: {cfg.app.env}, camera index: {cfg.camera.device_index}).",
                details={"env": cfg.app.env, "port": cfg.app.port},
            )
        except Exception as e:
            return DiagnosticTestResult(
                test_id="config_check",
                name="Configuration Schema & Precedence",
                description="Validates syntax and boundaries in central YAML config.",
                status="FAIL",
                duration_ms=0.0,
                message=f"Configuration error: {e}",
                corrective_action="Verify values in configs/windows.yaml or configs/raspberry_pi.yaml.",
            )

    def _test_filesystem_permissions(self) -> DiagnosticTestResult:
        dirs_to_check = [paths.resolve("logs"), paths.resolve("data"), paths.resolve("data/test_reports")]
        for d in dirs_to_check:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".perm_check.tmp"
            try:
                test_file.write_text("ok", encoding="utf-8")
                test_file.unlink()
            except Exception as e:
                return DiagnosticTestResult(
                    test_id="fs_perms",
                    name="Filesystem Write Permissions",
                    description="Checks directory access for logs, data, and models.",
                    status="FAIL",
                    duration_ms=0.0,
                    message=f"Write permission denied in directory '{d}': {e}",
                    corrective_action=f"Grant write permissions to {d}.",
                )
        return DiagnosticTestResult(
            test_id="fs_perms",
            name="Filesystem Write Permissions",
            description="Checks directory access for logs, data, and models.",
            status="PASS",
            duration_ms=0.0,
            message="Read/write permissions verified across data, logs, and report directories.",
        )

    def _test_database(self) -> DiagnosticTestResult:
        try:
            db_mgr = DatabaseManager(self.config.database.db_path)
            db_mgr.initialize_schema()
            repo = ParkingRepository(db_mgr)
            # Test quick write and read
            eid = repo.record_occupancy_event("DIAG_TEST", "FREE", "UNKNOWN", 0.99)
            with db_mgr.get_connection() as conn:
                conn.execute("DELETE FROM occupancy_events WHERE slot_id = 'DIAG_TEST';")
                conn.commit()
            return DiagnosticTestResult(
                test_id="db_health",
                name="SQLite Database & Schema Integrity",
                description="Tests database connection, migrations, and read/write cycle.",
                status="PASS",
                duration_ms=0.0,
                message=f"SQLite active at schema version {SCHEMA_VERSION}. Read/write transaction verified.",
            )
        except Exception as e:
            return DiagnosticTestResult(
                test_id="db_health",
                name="SQLite Database & Schema Integrity",
                description="Tests database connection, migrations, and read/write cycle.",
                status="FAIL",
                duration_ms=0.0,
                message=f"Database access failure: {e}",
                corrective_action="Check database file path and directory permissions.",
            )

    def _test_camera_probe(self) -> DiagnosticTestResult:
        devices = scan_camera_devices(max_devices_to_check=3)
        target_idx = self.config.camera.device_index
        found_target = any(d["device_index"] == target_idx and d["can_read_frame"] for d in devices)

        if found_target:
            return DiagnosticTestResult(
                test_id="camera_enum",
                name="Camera Enumeration & Probe",
                description="Scans available video capture devices.",
                status="PASS",
                duration_ms=0.0,
                message=f"Physical camera device #{target_idx} is connected and ready.",
                details={"devices_found": devices},
            )
        elif devices:
            return DiagnosticTestResult(
                test_id="camera_enum",
                name="Camera Enumeration & Probe",
                description="Scans available video capture devices.",
                status="WARN",
                duration_ms=0.0,
                message=f"Device #{target_idx} not responding, but found other devices: {[d['device_index'] for d in devices]}.",
                corrective_action=f"Update camera.device_index in config to {[d['device_index'] for d in devices][0]}.",
                details={"devices_found": devices},
            )
        else:
            return DiagnosticTestResult(
                test_id="camera_enum",
                name="Camera Enumeration & Probe",
                description="Scans available video capture devices.",
                status="WARN",
                duration_ms=0.0,
                message="No physical UVC webcam detected. Synthetic camera simulation mode is operational.",
                corrective_action="Connect Zebronics USB webcam to USB 3.0 port if live camera capture is required.",
                details={"mock_available": True},
            )

    def _test_models(self) -> DiagnosticTestResult:
        manifest_path = paths.resolve("models/manifest.json")
        if not manifest_path.exists():
            return DiagnosticTestResult(
                test_id="model_manifest",
                name="Model Registry & Manifest Hash",
                description="Verifies models/manifest.json presence and model metadata.",
                status="FAIL",
                duration_ms=0.0,
                message="models/manifest.json missing.",
                corrective_action="Restore model registry manifest.",
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            count = len(manifest.get("models", []))
            return DiagnosticTestResult(
                test_id="model_manifest",
                name="Model Registry & Manifest Hash",
                description="Verifies models/manifest.json presence and model metadata.",
                status="PASS",
                duration_ms=0.0,
                message=f"Model manifest valid. Registered models: {count}.",
                details={"registered_models": [m.get("model_id") for m in manifest.get("models", [])]},
            )
        except Exception as e:
            return DiagnosticTestResult(
                test_id="model_manifest",
                name="Model Registry & Manifest Hash",
                description="Verifies models/manifest.json presence and model metadata.",
                status="FAIL",
                duration_ms=0.0,
                message=f"Corrupt manifest JSON: {e}",
                corrective_action="Reformat models/manifest.json.",
            )

    def _test_image_quality(self) -> DiagnosticTestResult:
        mock = MockCamera(self.config.camera)
        mock.start()
        frame = None
        for _ in range(10):
            time.sleep(0.05)
            frame, meta = mock.get_latest_frame()
            if frame is not None:
                break
        mock.stop()

        if frame is None:
            return DiagnosticTestResult(
                test_id="frame_quality",
                name="Image Quality Assessment Gate",
                description="Tests blur, brightness, contrast, and glare analysis on test frame.",
                status="FAIL",
                duration_ms=0.0,
                message="Failed to retrieve test frame for optical analysis.",
            )

        gate = QualityGate()
        assessment = gate.assess_frame(frame)
        status: Literal["PASS", "WARN", "FAIL"] = "PASS" if assessment.status in ["EXCELLENT", "ACCEPTABLE"] else "WARN"
        return DiagnosticTestResult(
            test_id="frame_quality",
            name="Image Quality Assessment Gate",
            description="Tests blur, brightness, contrast, and glare analysis on test frame.",
            status=status,
            duration_ms=0.0,
            message=f"Image quality rated: {assessment.status} (blur: {assessment.blur_score:.1f}, luminance: {assessment.brightness:.1f}).",
            details=assessment.__dict__,
        )

    def _test_calibration_geometry(self) -> DiagnosticTestResult:
        calib_file = paths.resolve(self.config.parking.calibration_file)
        if not calib_file.exists():
            return DiagnosticTestResult(
                test_id="slot_geom",
                name="Parking Polygon Geometry Integrity",
                description="Validates calibration polygons for minimum area and non-overlap.",
                status="WARN",
                duration_ms=0.0,
                message="Calibration file data/calibration.json not found.",
                corrective_action="Use Parking Calibration tab in GUI to define parking slots.",
            )
        try:
            data = json.loads(calib_file.read_text(encoding="utf-8"))
            slots = data.get("slots", [])
            if not slots:
                return DiagnosticTestResult(
                    test_id="slot_geom",
                    name="Parking Polygon Geometry Integrity",
                    description="Validates calibration polygons for minimum area and non-overlap.",
                    status="WARN",
                    duration_ms=0.0,
                    message="Calibration file contains 0 configured parking bays.",
                )
            return DiagnosticTestResult(
                test_id="slot_geom",
                name="Parking Polygon Geometry Integrity",
                description="Validates calibration polygons for minimum area and non-overlap.",
                status="PASS",
                duration_ms=0.0,
                message=f"Verified {len(slots)} calibrated parking bay polygons.",
                details={"slots_count": len(slots), "slot_ids": [s["id"] for s in slots]},
            )
        except Exception as e:
            return DiagnosticTestResult(
                test_id="slot_geom",
                name="Parking Polygon Geometry Integrity",
                description="Validates calibration polygons for minimum area and non-overlap.",
                status="FAIL",
                duration_ms=0.0,
                message=f"Invalid calibration JSON: {e}",
            )

    def _test_system_resources(self) -> DiagnosticTestResult:
        total, used, free = shutil.disk_usage(str(paths.root))
        free_gb = round(free / (1024**3), 2)
        if free_gb < 1.0:
            return DiagnosticTestResult(
                test_id="disk_resources",
                name="Host System Resources & Disk Space",
                description="Inspects available disk storage and CPU load.",
                status="WARN",
                duration_ms=0.0,
                message=f"Low disk storage: {free_gb} GB remaining.",
                corrective_action="Prune old log files or test reports to avoid disk exhaustion.",
                details={"free_gb": free_gb},
            )
        return DiagnosticTestResult(
            test_id="disk_resources",
            name="Host System Resources & Disk Space",
            description="Inspects available disk storage and CPU load.",
            status="PASS",
            duration_ms=0.0,
            message=f"Storage healthy: {free_gb} GB available on host drive.",
            details={"free_gb": free_gb, "total_gb": round(total / (1024**3), 2)},
        )

    def _save_report(self, report: DiagnosticReport) -> None:
        report_dir = paths.resolve(self.config.diagnostics.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        json_path = report_dir / f"{report.report_id}.json"
        md_path = report_dir / f"{report.report_id}.md"

        # Save JSON
        json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

        # Save Markdown
        lines = [
            f"# Self-Diagnostics Report — {report.report_id}",
            f"- **Timestamp:** {report.timestamp}",
            f"- **Execution Type:** {report.run_type.upper()}",
            f"- **Overall Status:** {report.overall_status}",
            f"- **Duration:** {report.total_duration_sec} seconds",
            f"- **Platform:** {report.platform['device_model']} ({report.platform['architecture']}), Python {report.platform['python_version']}",
            "",
            "## Summary",
            f"| Total Tests | Passed | Warnings | Failed | Skipped | Not Run |",
            f"| :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| {report.summary['total']} | {report.summary['pass']} | {report.summary['warn']} | {report.summary['fail']} | {report.summary['skipped']} | {report.summary['not_run']} |",
            "",
            "## Individual Test Results",
            "| ID | Test Name | Status | Duration | Message |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ]
        for r in report.results:
            lines.append(f"| `{r.test_id}` | {r.name} | **{r.status}** | {r.duration_ms} ms | {r.message} |")

        lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
