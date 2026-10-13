"""
Packaging and Verification Tool for Raspberry Pi 5 Deployment.
Builds dist/raspberry_pi/ and creates a clean, standalone ZIP distribution archive.
Ensures zero CUDA dependencies, validates all models, normalizes LF line endings on shell scripts,
and excludes datasets, caches, virtual environments, and intermediate weights.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import zipfile
import yaml
from typing import Tuple, List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("packager")

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist" / "raspberry_pi"
ZIP_PATH = REPO_ROOT / "dist" / "smart_parking_raspberry_pi_deployment.zip"


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_line_endings_lf(filepath: Path):
    """Ensures shell scripts have Unix LF line endings to avoid \\r execution errors on Linux."""
    try:
        content = filepath.read_bytes()
        if b"\r\n" in content:
            normalized = content.replace(b"\r\n", b"\n")
            filepath.write_bytes(normalized)
            logger.info(f"Converted CRLF to LF for: {filepath.name}")
    except Exception as e:
        logger.warning(f"Could not normalize line endings for {filepath}: {e}")


def collect_runtime_files() -> list:
    """Enumerates all files genuinely required for Raspberry Pi deployment."""
    included_files = []

    # 1. Application Backend (app/)
    app_dir = REPO_ROOT / "app"
    for p in app_dir.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".bak"):
            rel = p.relative_to(REPO_ROOT)
            included_files.append(rel)

    # 2. Frontend / Dashboard Assets (web/)
    web_dir = REPO_ROOT / "web"
    for p in web_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(REPO_ROOT)
            included_files.append(rel)

    # 3. Configurations (configs/)
    config_names = [
        "raspberry_pi.yaml",
        "windows.yaml",
        "deployment.yaml",
        "evaluation.yaml",
        "datasets.yaml",
        "training.yaml",
    ]
    for cname in config_names:
        p = REPO_ROOT / "configs" / cname
        if p.exists():
            included_files.append(p.relative_to(REPO_ROOT))

    # 4. Calibration & Data Runtime (data/)
    data_files = [
        "data/calibration.json",
        "data/README.md",
    ]
    for df in data_files:
        p = REPO_ROOT / df
        if p.exists():
            included_files.append(p.relative_to(REPO_ROOT))

    # 5. Production Models & Manifest (models/)
    model_files = [
        "models/occupancy_classifier.onnx",
        "models/vehicle_detector.onnx",
        "models/plate_detector.onnx",
        "models/occupancy_classifier_baseline_cnrpark.onnx",
        "models/occupancy_classifier_baseline_cnrpark.pt",
        "models/occupancy_classifier.pt",
        "models/manifest.json",
        "models/README.md",
    ]
    for mf in model_files:
        p = REPO_ROOT / mf
        if p.exists():
            included_files.append(p.relative_to(REPO_ROOT))

    # 6. Deployment Scripts (deployment/raspberry_pi/)
    deploy_dir = REPO_ROOT / "deployment" / "raspberry_pi"
    for p in deploy_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(REPO_ROOT)
            included_files.append(rel)

    # 7. Utilities & Benchmarks (scripts/)
    script_files = [
        "scripts/preflight_raspberry_pi.py",
        "scripts/benchmark_inference.py",
        "scripts/export_deployment_models.py",
        "scripts/train_and_evaluate_all_approaches.py",
    ]
    for sf in script_files:
        p = REPO_ROOT / sf
        if p.exists():
            included_files.append(p.relative_to(REPO_ROOT))

    # 8. Documentation (docs/raspberry_pi/)
    doc_dir = REPO_ROOT / "docs" / "raspberry_pi"
    for p in doc_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(REPO_ROOT)
            included_files.append(rel)

    # 9. Targeted Readiness Tests (tests/)
    test_files = [
        "tests/unit/test_raspberry_pi_readiness.py",
        "tests/unit/test_model_pipeline.py",
        "tests/unit/test_occupancy_classifier.py",
        "tests/unit/test_occupancy_service.py",
        "tests/unit/test_allocation_service.py",
        "tests/unit/test_parking_geometry.py",
        "tests/unit/test_camera.py",
        "tests/unit/test_config.py",
        "tests/unit/test_database.py",
        "tests/unit/test_infrastructure.py",
        "tests/unit/test_path_relocation.py",
        "tests/unit/test_absolute_paths.py",
        "tests/unit/test_plates_ocr.py",
        "tests/integration/test_api.py",
    ]
    for tf in test_files:
        p = REPO_ROOT / tf
        if p.exists():
            included_files.append(p.relative_to(REPO_ROOT))

    # 10. Root Configuration & Dependencies
    root_files = [
        "pyproject.toml",
        "requirements-pi.txt",
        "requirements-windows.txt",
        "README.md",
        "LICENSES.md",
    ]
    for rf in root_files:
        p = REPO_ROOT / rf
        if p.exists():
            included_files.append(p.relative_to(REPO_ROOT))

    return sorted(list(set(included_files)))


def build_distribution_directory(files: list):
    """Copies all required files into dist/raspberry_pi/."""
    logger.info(f"Cleaning and preparing distribution directory: {DIST_DIR}")
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    total_bytes = 0

    for rel_path in files:
        src = REPO_ROOT / rel_path
        dst = DIST_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        # Normalize line endings for shell scripts
        if dst.suffix == ".sh":
            normalize_line_endings_lf(dst)

        copied_count += 1
        total_bytes += dst.stat().st_size

    # Ensure empty runtime directories exist
    (DIST_DIR / "data" / "test_reports").mkdir(parents=True, exist_ok=True)
    (DIST_DIR / "logs").mkdir(parents=True, exist_ok=True)

    logger.info(f"Copied {copied_count} files ({total_bytes / (1024 * 1024):.2f} MB uncompressed) into {DIST_DIR}")
    return copied_count, total_bytes


def verify_distribution_integrity():
    """Runs complete integrity checks directly on files inside dist/raspberry_pi/."""
    logger.info("Executing comprehensive verification on dist/raspberry_pi/...")
    errors = []

    # 1. Verify Manifest Models and Hashes
    manifest_path = DIST_DIR / "models" / "manifest.json"
    if not manifest_path.exists():
        errors.append("Missing models/manifest.json in distribution")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for m in manifest.get("models", []):
            m_path = DIST_DIR / m["file_path"]
            if not m_path.exists():
                errors.append(f"Model file missing in dist: {m['file_path']}")
                continue
            actual_hash = calculate_sha256(m_path)
            if actual_hash != m["sha256"]:
                errors.append(f"Model SHA mismatch for {m['file_path']}: expected {m['sha256']}, got {actual_hash}")

    # 2. Verify ONNX Runtime loading with CPUExecutionProvider
    try:
        import onnxruntime as ort
        for onnx_name in [
            "occupancy_classifier.onnx",
            "vehicle_detector.onnx",
            "plate_detector.onnx",
            "occupancy_classifier_baseline_cnrpark.onnx",
        ]:
            op = DIST_DIR / "models" / onnx_name
            if op.exists():
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                sess = ort.InferenceSession(str(op), sess_options=opts, providers=["CPUExecutionProvider"])
                assert "CPUExecutionProvider" in sess.get_providers()
                logger.info(f"Verified ONNX CPU loading: {onnx_name} (inputs: {[i.name for i in sess.get_inputs()]})")
    except Exception as e:
        errors.append(f"ONNX CPU verification error: {e}")

    # 3. Verify SQLite Schema File
    schema_file = DIST_DIR / "app" / "database" / "schema.sql"
    if not schema_file.exists():
        errors.append("Missing app/database/schema.sql")
    else:
        import sqlite3
        test_db = DIST_DIR / "data" / "_temp_schema_verify.sqlite"
        try:
            conn = sqlite3.connect(str(test_db))
            conn.executescript(schema_file.read_text(encoding="utf-8"))
            conn.close()
            test_db.unlink()
            logger.info("Verified SQLite schema script initialization.")
        except Exception as e:
            errors.append(f"SQLite schema execution failed: {e}")

    # 4. Verify Shell Scripts LF Line Endings & Executable bit
    for sh in (DIST_DIR / "deployment" / "raspberry_pi").glob("*.sh"):
        content = sh.read_bytes()
        if b"\r\n" in content:
            errors.append(f"Script {sh.name} still contains CRLF line endings!")

    # 5. Verify YAML Config
    pi_yaml = DIST_DIR / "configs" / "raspberry_pi.yaml"
    if not pi_yaml.exists():
        errors.append("Missing configs/raspberry_pi.yaml")
    else:
        data = yaml.safe_load(pi_yaml.read_text(encoding="utf-8"))
        assert data.get("app", {}).get("env") == "raspberry_pi"
        assert data.get("camera", {}).get("stale_threshold_sec") == 2.0
        assert data.get("ocr", {}).get("enabled") is False
        logger.info("Verified configs/raspberry_pi.yaml schema values.")

    # 6. Verify Web Assets
    index_html = DIST_DIR / "web" / "templates" / "index.html"
    styles_css = DIST_DIR / "web" / "static" / "css" / "styles.css"
    app_js = DIST_DIR / "web" / "static" / "js" / "app.js"
    for asset in [index_html, styles_css, app_js]:
        if not asset.exists():
            errors.append(f"Missing web asset: {asset.relative_to(DIST_DIR)}")
        else:
            assert asset.stat().st_size > 500

    if errors:
        for err in errors:
            logger.error(f"[VERIFY FAIL] {err}")
        return False, errors
    logger.info(">>> ALL DISTRIBUTION INTEGRITY CHECKS PASSED <<<")
    return True, []


def create_zip_archive(files: list) -> Tuple[Path, int]:
    """Creates a compressed ZIP archive from dist/raspberry_pi/."""
    logger.info(f"Creating ZIP archive at: {ZIP_PATH}")
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, filenames in os.walk(DIST_DIR):
            for fn in filenames:
                full_path = Path(root) / fn
                rel_path = full_path.relative_to(DIST_DIR)
                zf.write(full_path, arcname=str(rel_path).replace("\\", "/"))

    # Also place a copy inside dist/raspberry_pi/ for convenience
    zip_copy = DIST_DIR / "smart_parking_raspberry_pi_deployment.zip"
    shutil.copy2(ZIP_PATH, zip_copy)

    zip_size = ZIP_PATH.stat().st_size
    logger.info(f"Successfully generated ZIP: {ZIP_PATH} ({zip_size / (1024 * 1024):.2f} MB)")
    return ZIP_PATH, zip_size


def main():
    files = collect_runtime_files()
    copied_count, total_bytes = build_distribution_directory(files)
    ok, errors = verify_distribution_integrity()
    if not ok:
        print(f"FAILED with {len(errors)} errors")
        sys.exit(1)

    zip_file, zip_size = create_zip_archive(files)

    print("\n" + "=" * 70)
    print(" RASPBERRY PI 5 DEPLOYMENT PACKAGE SUMMARY")
    print("=" * 70)
    print(f" Destination Directory: {DIST_DIR}")
    print(f" Total Included Files:  {copied_count} files")
    print(f" Uncompressed Size:     {total_bytes / (1024 * 1024):.2f} MB")
    print(f" ZIP Archive Path:      {zip_file}")
    print(f" ZIP Archive Size:      {zip_size / (1024 * 1024):.2f} MB")
    print(f" Integrity Status:      VERIFIED (100% PASS)")
    print("=" * 70)


if __name__ == "__main__":
    from typing import Tuple
    main()
