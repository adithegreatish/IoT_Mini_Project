"""Vehicle detection module with ONNX Runtime backend, OpenCV NMS, and strict model validation."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.config.schema import DetectionConfig
from app.infrastructure.paths import paths

logger = logging.getLogger("smart_parking.detection")


@dataclass
class DetectionResult:
    bbox: Tuple[float, float, float, float]  # Normalized [xmin, ymin, xmax, ymax] in [0, 1]
    confidence: float
    class_name: str
    pixel_box: Tuple[int, int, int, int]    # [xmin, ymin, xmax, ymax] in pixels
    is_simulated: bool = False


class VehicleDetector:
    """Detects vehicles using ONNX Runtime model with strict verification and NMS."""

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
    ]
    TARGET_VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.enabled = config.enabled
        self.conf_threshold = config.confidence_threshold
        self.input_width = config.input_width
        self.input_height = config.input_height

        self._session = None
        self._is_simulated = False
        self._model_status = "INITIALIZING"
        self._last_error: Optional[str] = None
        self._init_session()

    @property
    def status(self) -> str:
        return self._model_status

    @property
    def is_healthy(self) -> bool:
        """Returns True if the detector is operational and ready to process real or simulated frames."""
        if not self.enabled:
            return False
        if self._is_simulated:
            return True
        return self._session is not None and self._model_status == "READY"

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def is_simulated(self) -> bool:
        return self._is_simulated

    def _init_session(self) -> None:
        if not self.enabled:
            logger.info("Vehicle detector disabled in configuration.")
            self._model_status = "DISABLED"
            return

        model_path = paths.resolve(self.config.model_path)

        if self.config.runtime == "onnx":
            if not model_path.exists():
                logger.error(f"Configured ONNX model file not found at: {model_path}")
                self._model_status = "MODEL_FILE_NOT_FOUND"
                return

            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.enable_mem_pattern = True
                self._session = ort.InferenceSession(
                    str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
                # Auto-align dimensions with compiled model if static dimensions are defined
                inp_shape = self._session.get_inputs()[0].shape
                if len(inp_shape) == 4 and isinstance(inp_shape[2], int) and isinstance(inp_shape[3], int):
                    self.input_height = inp_shape[2]
                    self.input_width = inp_shape[3]

                logger.info(f"Loaded ONNX vehicle detector successfully from {model_path} ({self.input_width}x{self.input_height})")
                self._model_status = "READY"
                self._is_simulated = False
                return
            except Exception as e:
                logger.error(f"Failed to load ONNX model from {model_path}: {e}")
                self._model_status = f"LOAD_ERROR: {e}"
                return

        elif self.config.runtime == "mock":
            logger.info("Vehicle detector explicitly configured in development mock mode.")
            self._model_status = "MOCK_SIMULATED"
            self._is_simulated = True
        else:
            logger.error(f"Unsupported detection runtime: {self.config.runtime}")
            self._model_status = "UNSUPPORTED_RUNTIME"

    def detect(self, frame: Optional[np.ndarray]) -> List[DetectionResult]:
        """Detects vehicles in frame and returns bounding boxes."""
        if not self.enabled or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        if self._session is not None and not self._is_simulated:
            try:
                return self._run_onnx_inference(frame, w, h)
            except Exception as e:
                self._model_status = f"INFERENCE_ERROR: {e}"
                self._last_error = str(e)
                logger.error(f"Inference exception encountered in VehicleDetector: {e}", exc_info=True)
                raise RuntimeError(f"ONNX inference failure: {e}") from e
        elif self._is_simulated:
            return self._run_heuristic_mock_detection(frame, w, h)
        else:
            logger.debug(f"Vehicle detector unavailable (status: {self._model_status}). Returning empty detections.")
            return []


    def _run_onnx_inference(self, frame: np.ndarray, w: int, h: int) -> List[DetectionResult]:
        """Runs genuine ONNX model inference with OpenCV Non-Maximum Suppression (NMS) and telemetry."""
        import time
        from app.infrastructure.telemetry import telemetry

        # 1. Preprocessing stage
        t0 = time.perf_counter()
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        t1 = time.perf_counter()

        # 2. ONNX Model Inference stage
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})
        t2 = time.perf_counter()

        # 3. Postprocessing & NMS stage
        output = outputs[0]
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)

        candidate_boxes = []
        confidences = []
        class_ids = []

        num_classes = output.shape[2] - 4

        for row in output[0]:
            cx, cy, bw, bh = row[:4]
            scores = row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])

            if score >= self.conf_threshold:
                if num_classes == 4:
                    cname = self.TARGET_VEHICLE_CLASSES[class_id] if class_id < len(self.TARGET_VEHICLE_CLASSES) else "car"
                else:
                    cname = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else "car"

                if cname in self.TARGET_VEHICLE_CLASSES:
                    px_cx = (cx / self.input_width) * w
                    px_cy = (cy / self.input_height) * h
                    px_w = (bw / self.input_width) * w
                    px_h = (bh / self.input_height) * h

                    px_x = max(0, int(px_cx - px_w / 2.0))
                    px_y = max(0, int(px_cy - px_h / 2.0))

                    candidate_boxes.append([px_x, px_y, int(px_w), int(px_h)])
                    confidences.append(score)
                    class_ids.append(cname)

        if not candidate_boxes:
            t3 = time.perf_counter()
            telemetry.record_detector_inference(
                onnx_ms=(t2 - t1) * 1000.0,
                preprocess_ms=(t1 - t0) * 1000.0,
                postprocess_ms=(t3 - t2) * 1000.0,
            )
            return []

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            bboxes=candidate_boxes,
            scores=confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=0.45,
        )

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = candidate_boxes[idx]
                score = confidences[idx]
                cname = class_ids[idx]

                xmin = round(max(0.0, bx / float(w)), 4)
                ymin = round(max(0.0, by / float(h)), 4)
                xmax = round(min(1.0, (bx + bw) / float(w)), 4)
                ymax = round(min(1.0, (by + bh) / float(h)), 4)

                results.append(DetectionResult(
                    bbox=(xmin, ymin, xmax, ymax),
                    confidence=round(score, 3),
                    class_name=cname,
                    pixel_box=(bx, by, bx + bw, by + bh),
                    is_simulated=False,
                ))

        t3 = time.perf_counter()
        telemetry.record_detector_inference(
            onnx_ms=(t2 - t1) * 1000.0,
            preprocess_ms=(t1 - t0) * 1000.0,
            postprocess_ms=(t3 - t2) * 1000.0,
        )
        return results

    def _run_heuristic_mock_detection(self, frame: np.ndarray, w: int, h: int) -> List[DetectionResult]:
        """
        Explicit development simulation mode:
        Only invoked when config.runtime == 'mock' is explicitly requested.
        Visibly flags detections as simulated.
        """
        results: List[DetectionResult] = []

        roi_y1 = int(h * 0.40)
        roi_y2 = int(h * 0.92)
        roi = frame[roi_y1:roi_y2, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        vehicle_mask = (sat > 50) | (val > 190)
        mask_uint8 = vehicle_mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_veh_area = (w * h) * 0.015

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_veh_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                full_y = by + roi_y1
                xmin = round(bx / w, 3)
                ymin = round(full_y / h, 3)
                xmax = round((bx + bw) / w, 3)
                ymax = round((full_y + bh) / h, 3)

                aspect = bw / max(1, bh)
                cname = "motorcycle" if aspect < 0.6 else "car"

                results.append(DetectionResult(
                    bbox=(xmin, ymin, xmax, ymax),
                    confidence=0.88,
                    class_name=cname,
                    pixel_box=(bx, full_y, bx + bw, full_y + bh),
                    is_simulated=True,
                ))

        return results

