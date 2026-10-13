"""Unit tests for ONNX PatchOccupancyClassifier."""

from pathlib import Path
import numpy as np
import pytest

from app.infrastructure.paths import paths
from app.parking.occupancy_classifier import PatchOccupancyClassifier


@pytest.fixture
def classifier():
    model_path = paths.resolve("models/occupancy_classifier.onnx")
    if not model_path.exists():
        pytest.skip("models/occupancy_classifier.onnx not yet trained")
    return PatchOccupancyClassifier(model_path=model_path, device="cpu")


def test_classifier_loads_session(classifier):
    assert classifier.session is not None
    assert classifier.input_name == "input"


def test_preprocess_patch_shape_and_range(classifier):
    dummy_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tensor = classifier.preprocess_patch(dummy_bgr)
    assert tensor.shape == (1, 3, 150, 150)
    assert tensor.dtype == np.float32


def test_single_patch_prediction(classifier):
    dummy_bgr = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    is_occ, conf = classifier.predict_patch(dummy_bgr)
    assert isinstance(is_occ, bool)
    assert 0.0 <= conf <= 1.0


def test_batch_prediction(classifier):
    patches = [np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8) for _ in range(5)]
    results = classifier.predict_batch(patches)
    assert len(results) == 5
    for is_occ, conf in results:
        assert isinstance(is_occ, bool)
        assert 0.0 <= conf <= 1.0


def test_extract_patch_from_frame(classifier):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Define normalized polygon points [0.1, 0.1] to [0.3, 0.4]
    slot_points = [
        [0.1, 0.1],
        [0.3, 0.1],
        [0.3, 0.4],
        [0.1, 0.4],
    ]
    crop = classifier.extract_patch_from_frame(frame, slot_points)
    assert crop is not None
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
