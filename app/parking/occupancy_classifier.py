"""
Parking-space patch occupancy classification inference engine.

Loads ONNX model (models/occupancy_classifier.onnx) with CPU/CUDA execution providers.
Classifies cropped parking slot image patches into FREE (0) or OCCUPIED (1) states.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from app.infrastructure.paths import paths

logger = logging.getLogger("smart_parking.occupancy_classifier")


class PatchOccupancyClassifier:
    """Production ONNX inference wrapper for parking slot occupancy classification."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        threshold: float = 0.65
    ):
        resolved_path = paths.resolve(model_path or "models/occupancy_classifier.onnx")
        self.model_path = resolved_path
        self.threshold = threshold

        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        if resolved_path.exists():
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(resolved_path), sess_options=opts, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Loaded Occupancy Classifier from {resolved_path.name} on {providers[0]}")
        else:
            self.session = None
            logger.warning(f"Occupancy Classifier model not found at {resolved_path}. Inference disabled.")

        # Normalization constants (ImageNet mean & std)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))

    def preprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        """Preprocesses a single image patch (BGR) to (1, 3, 150, 150) normalized tensor."""
        if patch.shape[0] != 150 or patch.shape[1] != 150:
            patch = cv2.resize(patch, (150, 150), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std
        # HWC to CHW
        chw = np.transpose(norm, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def predict_patch(self, patch: np.ndarray) -> Tuple[bool, float]:
        """
        Classifies a single parking slot patch.
        Returns: (is_occupied: bool, confidence: float in [0.0, 1.0])
        """
        if self.session is None or patch is None or patch.size == 0:
            return False, 0.5

        tensor = self.preprocess_patch(patch)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        logit = float(logits[0][0])
        prob = 1.0 / (1.0 + np.exp(-logit))  # Sigmoid

        is_occupied = bool(prob >= self.threshold)
        conf = float(prob if is_occupied else (1.0 - prob))
        return is_occupied, round(conf, 4)

    def predict_batch(self, patches: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Classifies a batch of slot patches efficiently."""
        if self.session is None or not patches:
            return [(False, 0.5) for _ in patches]

        tensors = [self.preprocess_patch(p) for p in patches]
        batch_tensor = np.concatenate(tensors, axis=0)

        logits = self.session.run(None, {self.input_name: batch_tensor})[0]
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))

        results = []
        for prob in probs:
            is_occ = bool(prob >= self.threshold)
            conf = float(prob if is_occ else (1.0 - prob))
            results.append((is_occ, round(conf, 4)))
        return results

    def extract_patch_from_frame(self, frame: np.ndarray, points: List[List[float]]) -> Optional[np.ndarray]:
        """
        Extracts an aligned or axis-aligned bounding patch from a full camera frame
        using normalized polygon vertices [0.0, 1.0].
        """
        if frame is None or len(points) < 3:
            return None

        h, w = frame.shape[:2]
        pts_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in points], dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts_px)

        # Boundary checks
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        return frame[y1:y2, x1:x2].copy()
