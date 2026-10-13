"""
Preflight Self-Diagnostics Script for Raspberry Pi 5 Deployment.
Validates CPU architecture, Python version, dependencies, model checksums,
configuration files, filesystem permissions, database initialization, and camera devices.
Outputs formatted terminal results and JSON report to data/test_reports/preflight_report.json.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("preflight")


class PreflightChecker:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def record(self, category: str, name: str, status: str, details: str, meta: Dict = None):
        """status must be PASS, WARN, FAIL, or NOT_RUN."""
        self.results.append({
            "category": category,
            "name": name,
            "status": status,
            "details": details,
            "metadata": meta or {}
        })

    def run_all_checks(self) -> Dict[str, Any]:
        logger.info("Executing Raspberry Pi Preflight Diagnostics Suite...")

        # 1. Platform & CPU Architecture
        arch = platform.machine().lower()
        system = platform.system()
        if arch in ["aarch64", "arm64"] and system == "Linux":
            self.record("Platform", "Architecture", "PASS", f"ARM64 Linux confirmed ({arch})", {"arch": arch, "system": system})
        else:
            self.record(
                "Platform", "Architecture", "WARN",
                f"Currently running on {system} ({arch}). Physical Pi 5 requires aarch64 Linux.",
                {"arch": arch, "system": system, "target": "Linux aarch64"}
            )

        # 2. Python Version
        v = sys.version_info
        py_str = f"{v.major}.{v.minor}.{v.micro}"
        if v.major == 3 and v.minor >= 10:
            self.record("Platform", "Python Version", "PASS", f"Python {py_str} meets requirements (>=3.10)", {"version": py_str})
        else:
            self.record("Platform", "Python Version", "FAIL", f"Python {py_str} is unsupported. Python 3.10+ required.", {"version": py_str})

        # 3. Memory & Storage
        try:
            import psutil
            total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
            avail_ram_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)
            free_disk_gb = round(shutil.disk_usage(str(paths.root)).free / (1024 ** 3), 2)

            if total_ram_gb >= 1.8:
                self.record("Hardware", "RAM Capacity", "PASS", f"{total_ram_gb} GB Total RAM ({avail_ram_gb} GB free)", {"total_gb": total_ram_gb, "free_gb": avail_ram_gb})
            else:
                self.record("Hardware", "RAM Capacity", "WARN", f"{total_ram_gb} GB RAM detected (Pi 5 8GB recommended)", {"total_gb": total_ram_gb})

            if free_disk_gb >= 1.5:
                self.record("Hardware", "Disk Space", "PASS", f"{free_disk_gb} GB free storage available", {"free_gb": free_disk_gb})
            else:
                self.record("Hardware", "Disk Space", "FAIL", f"Low storage: only {free_disk_gb} GB free. At least 1.5 GB required.", {"free_gb": free_disk_gb})
        except Exception as e:
            self.record("Hardware", "Resources", "WARN", f"Could not query psutil: {e}")

        # 4. Core Runtime Dependencies
        deps = [
            ("onnxruntime", "ONNX Runtime CPU Inference Engine"),
            ("cv2", "OpenCV Image Processing"),
            ("numpy", "NumPy Numerical Computing"),
            ("fastapi", "FastAPI Asynchronous Web Framework"),
            ("uvicorn", "Uvicorn ASGI Server"),
            ("pydantic", "Pydantic Configuration Validation"),
        ]
        for mod_name, label in deps:
            try:
                mod = __import__(mod_name)
                ver = getattr(mod, "__version__", "installed")
                self.record("Dependencies", label, "PASS", f"Module '{mod_name}' available (v{ver})", {"module": mod_name, "version": str(ver)})
            except ImportError as e:
                self.record("Dependencies", label, "FAIL", f"Module '{mod_name}' is missing: {e}", {"module": mod_name})

        # 5. Model Artifacts & SHA-256 Checksums
        expected_models = [
            ("occupancy_classifier.onnx", "e42e18e40b9337fde295c2e269845bfb9cb3df54d341971a03c6ae8b5d80105f", True, "Winning Approach D Model"),
            ("vehicle_detector.onnx", "8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc", True, "Vehicle Detector Model"),
            ("plate_detector.onnx", "efddfbc0d555a374389cb7847db06b776a22fed2b706eac56ce51596bae5cea4", False, "Plate Localizer Model (Optional)"),
            ("occupancy_classifier_baseline_cnrpark.onnx", "835781c13fb70a5e80505ab29c324a37aceac5803ec218a5ed249ef789e3e8bc", False, "Baseline Rollback ONNX (ARM64 CPU)"),
            ("occupancy_classifier_baseline_cnrpark.pt", None, False, "Baseline Rollback Checkpoint"),
        ]

        for filename, expected_sha, required, desc in expected_models:
            m_path = paths.models_dir / filename
            if not m_path.exists():
                if required:
                    self.record("Models", filename, "FAIL", f"Required model file missing: {m_path}", {"file": filename})
                else:
                    self.record("Models", filename, "WARN", f"Optional model file not found: {m_path}", {"file": filename})
                continue

            # Compute sha256
            hasher = hashlib.sha256()
            with open(m_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest()

            if expected_sha and actual_sha != expected_sha:
                self.record("Models", filename, "FAIL", f"Checksum mismatch! Expected {expected_sha[:16]}..., got {actual_sha[:16]}...", {"expected": expected_sha, "actual": actual_sha})
            else:
                size_mb = round(m_path.stat().st_size / (1024 * 1024), 2)
                self.record("Models", filename, "PASS", f"{desc} verified ({size_mb} MB, SHA: {actual_sha[:16]}...)", {"file": filename, "sha256": actual_sha, "size_mb": size_mb})

        # 6. Configuration Validity
        config_files = ["configs/raspberry_pi.yaml", "configs/windows.yaml"]
        for c_file in config_files:
            cp = paths.resolve(c_file)
            if cp.exists():
                try:
                    from app.config.loader import load_config
                    cfg = load_config(str(cp))
                    self.record("Configuration", Path(c_file).name, "PASS", f"Valid YAML schema (env={cfg.app.env}, stale={cfg.camera.stale_threshold_sec}s)", {"file": c_file})
                except Exception as e:
                    self.record("Configuration", Path(c_file).name, "FAIL", f"Config parse error: {e}", {"file": c_file})
            else:
                self.record("Configuration", Path(c_file).name, "FAIL", f"Missing config file: {cp}", {"file": c_file})

        # 7. Filesystem Writable Permissions
        writable_dirs = ["data", "logs", "data/test_reports"]
        for d in writable_dirs:
            dp = paths.resolve(d)
            dp.mkdir(parents=True, exist_ok=True)
            test_file = dp / ".write_test_tmp"
            try:
                test_file.write_text("ok", encoding="utf-8")
                test_file.unlink()
                self.record("Filesystem", f"Write: {d}/", "PASS", f"Directory is writable ({dp})", {"dir": d})
            except Exception as e:
                self.record("Filesystem", f"Write: {d}/", "FAIL", f"Directory not writable: {e}", {"dir": d})

        # 8. Database Initialization & Schema Readiness
        try:
            from app.database.connection import DatabaseManager
            db_path = paths.resolve("data/parking_events.sqlite")
            db = DatabaseManager(str(db_path))
            db.initialize_schema()
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row["name"] for row in cursor.fetchall()]
            required_tables = {"parking_slots", "occupancy_events", "slot_reservations"}
            if required_tables.issubset(set(tables)):
                self.record("Database", "SQLite Schema", "PASS", f"All schema tables present: {', '.join(required_tables)}", {"tables": tables})
            else:
                self.record("Database", "SQLite Schema", "FAIL", f"Missing tables: {required_tables - set(tables)}", {"tables": tables})
        except Exception as e:
            self.record("Database", "SQLite Schema", "FAIL", f"Database error: {e}")

        # 9. Camera Hardware Discoverability
        if system == "Linux":
            video_nodes = list(Path("/dev").glob("video*"))
            if video_nodes:
                names = [v.name for v in video_nodes]
                self.record("Camera", "V4L2 Video Nodes", "PASS", f"Detected video nodes: {', '.join(names)}", {"nodes": names})
            else:
                self.record("Camera", "V4L2 Video Nodes", "WARN", "No /dev/video* devices found. Connect USB webcam.", {"nodes": []})
        else:
            self.record("Camera", "Hardware Video Device", "WARN", "Running on Windows. Connect USB webcam before running on physical Pi.", {"system": system})

        # 10. Live CPU Model Preflight Execution
        try:
            from app.parking.occupancy_classifier import PatchOccupancyClassifier
            clf = PatchOccupancyClassifier(device="cpu", threshold=0.65)
            import numpy as np
            test_patch = np.full((150, 150, 3), 120, dtype=np.uint8)
            is_occ, conf = clf.predict_patch(test_patch)
            self.record("Model Preflight", "ONNX CPU Inference", "PASS", f"Patch inference verified (is_occupied={is_occ}, conf={conf:.2f})", {"threshold": clf.threshold})
        except Exception as e:
            self.record("Model Preflight", "ONNX CPU Inference", "FAIL", f"Model test failed: {e}")

        # Summary counts
        pass_count = sum(1 for r in self.results if r["status"] == "PASS")
        warn_count = sum(1 for r in self.results if r["status"] == "WARN")
        fail_count = sum(1 for r in self.results if r["status"] == "FAIL")

        overall_status = "PASS" if fail_count == 0 else "FAIL"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "target_platform": "Raspberry Pi 5 (8 GB, aarch64)",
            "execution_platform": f"{platform.system()} ({platform.machine()})",
            "overall_status": overall_status,
            "counts": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
            "checks": self.results
        }

        # Save to JSON
        out_dir = paths.resolve("data/test_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_file = out_dir / "preflight_report.json"
        report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary


def print_terminal_report(summary: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(" RASPBERRY PI 5 PREFLIGHT SELF-DIAGNOSTIC REPORT")
    print(f" Execution Platform: {summary['execution_platform']} | Target: {summary['target_platform']}")
    print("=" * 70)

    for item in summary["checks"]:
        st = item["status"]
        if st == "PASS":
            badge = "\033[32m[PASS]\033[0m"
        elif st == "WARN":
            badge = "\033[33m[WARN]\033[0m"
        elif st == "FAIL":
            badge = "\033[31m[FAIL]\033[0m"
        else:
            badge = "\033[37m[SKIP]\033[0m"

        print(f" {badge:16} {item['category']:15} | {item['name']:25} : {item['details']}")

    counts = summary["counts"]
    print("=" * 70)
    print(f" SUMMARY: {counts['pass']} Passed | {counts['warn']} Warnings | {counts['fail']} Failed")
    if counts["fail"] == 0:
        print("\033[32m >>> OVERALL PREFLIGHT STATUS: PASSED (System Ready for Deployment) <<<\033[0m")
    else:
        print("\033[31m >>> OVERALL PREFLIGHT STATUS: FAILED (Resolve failures above before starting) <<<\033[0m")
    print("=" * 70 + "\n")


def main():
    checker = PreflightChecker()
    summary = checker.run_all_checks()
    print_terminal_report(summary)
    sys.exit(0 if summary["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
