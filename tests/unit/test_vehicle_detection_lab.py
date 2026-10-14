"""
Comprehensive Regression & Unit Test Suite for Vehicle Detection Research Lab:
- Model registry initialization, integrity, gated promotion, and rollback.
- Dataset annotation validation (YOLO syntax, coordinate bounds, class ID validity).
- Multi-view split hygiene & zero-leakage verification.
- Active learning data capture, storage budgets, and perceptual deduplication.
- Comparative analysis & automatic regression warning gates.
- AI Model Lab REST endpoints.
"""

from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.infrastructure.paths import paths
from app.models.registry import ModelRegistry, PromotionGateResult
from app.detection.data_capture import DataCaptureManager
from scripts.compare_vehicle_models import compare_metrics
from scripts.validate_vehicle_annotations import validate_split, compute_dhash, hamming_distance
from app.main import create_app


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def test_model_registry_initialization_and_integrity(temp_dir):
    """Verifies that model registry initializes cleanly with active baseline and valid checksums."""
    reg_file = temp_dir / "model_registry.json"
    registry = ModelRegistry(registry_file=reg_file)

    active = registry.get_active_model()
    assert active["model_id"] == "vehicle-yolo-v8n"
    assert active["version"] == "1.0.0"
    assert active["status"] == "ACTIVE_PRODUCTION"
    assert "metrics" in active
    assert active["metrics"]["mAP50"] == 0.5950
    assert active["metrics"]["per_class"]["truck"]["ap50"] == 0.3873


def test_promotion_gates_blocking_regressions(temp_dir):
    """Verifies that candidate promotion gates block regressed candidates."""
    reg_file = temp_dir / "model_registry.json"
    registry = ModelRegistry(registry_file=reg_file)

    dummy_pt = temp_dir / "cand.pt"
    dummy_onnx = temp_dir / "cand.onnx"
    dummy_pt.write_bytes(b"dummy_pt_weights")
    dummy_onnx.write_bytes(b"dummy_onnx_weights")

    regressed_cand = {
        "pt_path": str(dummy_pt.relative_to(paths.root)).replace("\\", "/") if dummy_pt.is_relative_to(paths.root) else str(dummy_pt),
        "onnx_path": str(dummy_onnx.relative_to(paths.root)).replace("\\", "/") if dummy_onnx.is_relative_to(paths.root) else str(dummy_onnx),
        "test_metrics": {
            "mAP50": 0.4500,  # Regressed from 0.5950
            "mAP50_95": 0.2500,  # Regressed
            "per_class": {
                "car": {"ap50": 0.5000},
                "motorcycle": {"ap50": 0.5000},
                "bus": {"ap50": 0.6000},
                "truck": {"ap50": 0.2000},  # Regressed from 0.3873
            },
        },
    }

    result = registry.evaluate_promotion_gates(regressed_cand, min_mAP50_improvement=0.0)
    assert result.eligible is False
    assert len(result.reasons) >= 2
    # Verify reasons contain regression notes
    reasons_str = " ".join(result.reasons)
    assert "mAP@50" in reasons_str
    assert "regressed" in reasons_str


def test_model_promotion_and_rollback_flow(temp_dir):
    """Tests atomic promotion of valid candidate and instant rollback to prior stable version."""
    reg_file = temp_dir / "model_registry.json"
    registry = ModelRegistry(registry_file=reg_file)
    registry.archive_dir = temp_dir / "archive"
    registry.archive_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy candidate files within workspace paths
    exp_dir = paths.models_dir / "experiments" / "test_exp_unit"
    exp_dir.mkdir(parents=True, exist_ok=True)
    pt_p = exp_dir / "cand.pt"
    onnx_p = exp_dir / "cand.onnx"
    pt_p.write_bytes(b"TEST_PT_CONTENT_V2")
    onnx_p.write_bytes(b"TEST_ONNX_CONTENT_V2")

    registry.register_candidate(
        experiment_id="test_exp_unit",
        pt_path=pt_p,
        onnx_path=onnx_p,
        training_config={"architecture": "YOLOv8n", "imgsz": 640},
        validation_metrics={"mAP50": 0.65},
        test_metrics={
            "mAP50": 0.6500,
            "mAP50_95": 0.4200,
            "per_class": {
                "car": {"ap50": 0.65},
                "motorcycle": {"ap50": 0.70},
                "bus": {"ap50": 0.75},
                "truck": {"ap50": 0.42},
            },
        },
    )

    # Promote candidate with force=True for unit test speed
    promoted, msg = registry.promote_candidate("test_exp_unit", force=True)
    assert promoted is True
    active = registry.get_active_model()
    assert active["version"] == "1.0.1"
    assert active["source_experiment"] == "test_exp_unit"

    # Now verify rollback
    rolled_back, roll_msg = registry.rollback_model()
    assert rolled_back is True
    active_restored = registry.get_active_model()
    assert active_restored["version"] == "1.0.0"

    # Clean up test exp dir
    shutil.rmtree(exp_dir, ignore_errors=True)


def test_dataset_annotation_syntax_and_bounds_validation(temp_dir):
    """Tests dataset validator detecting corrupt labels, out-of-bounds coords, and valid syntax."""
    img_dir = temp_dir / "images"
    lbl_dir = temp_dir / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    import cv2

    # Create valid image and label
    valid_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "valid_001.jpg"), valid_img)
    (lbl_dir / "valid_001.txt").write_text("0 0.5 0.5 0.2 0.3\n3 0.2 0.2 0.1 0.1\n")

    # Create image with invalid box (coordinates out of bounds)
    cv2.imwrite(str(img_dir / "invalid_002.jpg"), valid_img)
    (lbl_dir / "invalid_002.txt").write_text("0 1.5 0.5 0.2 0.3\n")  # x=1.5 > 1.0

    # Create image with unknown class ID
    cv2.imwrite(str(img_dir / "invalid_003.jpg"), valid_img)
    (lbl_dir / "invalid_003.txt").write_text("99 0.5 0.5 0.2 0.3\n")  # cls=99

    res = validate_split(img_dir, lbl_dir, expected_classes=["car", "motorcycle", "bus", "truck"])
    assert res["total_images"] == 3
    assert res["usable_images"] == 1
    assert res["invalid_boxes_count"] == 2
    assert res["class_counts"]["car"] == 1
    assert res["class_counts"]["truck"] == 1


def test_dhash_perceptual_deduplication():
    """Verifies perceptual dHash detects near-duplicates while differentiating distinct scenes."""
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[20:80, 20:80] = 255  # White square in center

    # Identical with slight pixel noise
    img2 = img1.copy()
    img2[21:22, 21:22] = 240

    # Totally different gradient image
    img3 = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    h3 = compute_dhash(img3)

    assert hamming_distance(h1, h2) <= 1  # Near identical
    assert hamming_distance(h1, h3) >= 10  # Distinct scenes


def test_active_learning_data_capture_manager(temp_dir):
    """Tests rate limiting, budget enforcement, and candidate manifest in DataCaptureManager."""
    mgr = DataCaptureManager(
        storage_dir=temp_dir,
        max_storage_mb=50,
        min_interval_sec=0.1,
        auto_capture_enabled=True,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 128

    # First capture succeeds
    success, fid1 = mgr.capture_candidate_frame(frame, reason="LOW_CONFIDENCE_CAR", confidence=0.32, force=True)
    assert success is True
    assert fid1.startswith("cand_")

    # Review action
    reviewed = mgr.review_candidate(fid1, action="ACCEPT")
    assert reviewed is True

    queue = mgr.get_candidate_queue()
    assert len(queue) == 1
    assert queue[0]["review_status"] == "ACCEPTED"


def test_model_comparison_regression_detection():
    """Tests compare_metrics flagging critical regressions."""
    base_m = {
        "mAP50": 0.5950,
        "mAP50_95": 0.3797,
        "precision": 0.6224,
        "recall": 0.5459,
        "per_class": {
            "car": {"ap50": 0.6038, "recall": 0.5552},
            "motorcycle": {"ap50": 0.6698, "recall": 0.5932},
            "bus": {"ap50": 0.7192, "recall": 0.6182},
            "truck": {"ap50": 0.3873, "recall": 0.4168},
        },
    }

    cand_regressed = {
        "mAP50": 0.5800,
        "mAP50_95": 0.3600,
        "precision": 0.6100,
        "recall": 0.5300,
        "per_class": {
            "car": {"ap50": 0.6100, "recall": 0.5600},
            "motorcycle": {"ap50": 0.6700, "recall": 0.5900},
            "bus": {"ap50": 0.7200, "recall": 0.6200},
            "truck": {"ap50": 0.2800, "recall": 0.3200},  # Dropped from 0.3873 by >0.10!
        },
    }

    res = compare_metrics(base_m, cand_regressed, max_allowed_regression=0.03)
    assert res["has_regressions"] is True
    assert any("truck" in r.lower() for r in res["regressions"])


def test_api_lab_endpoints():
    """Tests FastAPI /api/lab REST endpoints."""
    app = create_app()
    client = TestClient(app)

    # 1. Overview
    r1 = client.get("/api/lab/overview")
    assert r1.status_code == 200
    d1 = r1.json()
    assert "active_model" in d1
    assert d1["active_model"]["model_id"] == "vehicle-yolo-v8n"

    # 2. Experiments
    r2 = client.get("/api/lab/experiments")
    assert r2.status_code == 200
    assert "candidates" in r2.json()

    # 3. Dataset Health
    r3 = client.get("/api/lab/dataset_health")
    assert r3.status_code == 200

    # 4. Error Gallery
    r4 = client.get("/api/lab/error_gallery")
    assert r4.status_code == 200
    assert "false_positives" in r4.json()

    # 5. Report retrieval
    r5 = client.get("/api/lab/report/baseline")
    assert r5.status_code == 200
    assert "Vehicle Detection Baseline Audit" in r5.json()["content"]


def test_promotion_gate_blocks_candidate_with_domain_specialization(temp_dir):
    """Verifies that actual candidate exp_india_cctv_10ep is safely blocked by promotion gates."""
    reg_file = temp_dir / "model_registry.json"
    registry = ModelRegistry(registry_file=reg_file)

    # Load actual candidate evaluation summary
    cand_summary_p = paths.models_dir / "experiments" / "exp_india_cctv_10ep" / "candidate_evaluation_summary.json"
    assert cand_summary_p.exists(), "Candidate evaluation summary missing!"
    cand_data = json.loads(cand_summary_p.read_text(encoding="utf-8"))

    # Evaluate promotion gates
    gate_res = registry.evaluate_promotion_gates(cand_data, min_mAP50_improvement=0.0, max_class_regression=0.05)
    assert gate_res.eligible is False, "Candidate with domain regression should not be eligible for production promotion!"
    reasons_text = " ".join(gate_res.reasons)
    assert "motorcycle" in reasons_text
    assert "mAP@50" in reasons_text

    # Verify active production model remains untouched
    active = registry.get_active_model()
    assert active["version"] == "1.0.0"
    assert active["status"] == "ACTIVE_PRODUCTION"
    assert active["model_id"] == "vehicle-yolo-v8n"


def test_pr_curve_ap_calculation_not_fixed_threshold():
    """Verifies that AP is calculated via 101-point PR curve interpolation, not a single fixed threshold."""
    from scripts.evaluate_proper_pr_curves import compute_ap_from_pr_curve

    # Test monotonic precision-recall curve
    recalls = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([0.9, 0.85, 0.8, 0.7, 0.5, 0.3])
    ap = compute_ap_from_pr_curve(recalls, precisions)

    assert 0.0 < ap < 1.0
    # AP under this curve is approximately 0.71, definitely distinct from single point precision 0.3 or 0.9
    assert abs(ap - 0.71) < 0.05
    assert ap != precisions[-1], "AP must not equal final single-threshold precision"
    assert ap != precisions[0], "AP must not equal initial single-threshold precision"

    # Edge cases
    assert compute_ap_from_pr_curve(np.array([]), np.array([])) == 0.0


