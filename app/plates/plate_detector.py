"""License plate detector extracting high-resolution plate crops using trained ONNX model."""

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.config.schema import PlatesConfig
from app.infrastructure.paths import paths

logger = logging.getLogger("smart_parking.plates")


@dataclass
class PlateDetectionResult:
    crop: np.ndarray
    bbox: Tuple[int, int, int, int]  # [xmin, ymin, xmax, ymax] within vehicle crop / image
    confidence: float
    is_simulated: bool = False


class PlateDetector:
    """Localizes license plate regions using trained YOLOv8 ONNX model with NMS."""

    def __init__(self, config: PlatesConfig):
        self.config = config
        self.enabled = config.enabled
        self.conf_threshold = config.confidence_threshold
        self.min_w = config.min_crop_width
        self.min_h = config.min_crop_height
        self._session = None
        self._status = "INITIALIZING"
        self._init_session()

    @property
    def status(self) -> str:
        return self._status

    def _init_session(self) -> None:
        if not self.enabled:
            logger.info("Plate detector disabled in configuration.")
            self._status = "DISABLED"
            return

        model_path = paths.resolve(self.config.model_path)

        if self.config.runtime == "onnx":
            if not model_path.exists():
                logger.warning(f"Plate detector ONNX model file not found at: {model_path}. Falling back to heuristic.")
                self._status = "MODEL_FILE_NOT_FOUND"
                return

            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self._session = ort.InferenceSession(
                    str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
                logger.info(f"Loaded ONNX plate detector successfully from {model_path}")
                self._status = "READY"
            except Exception as e:
                logger.error(f"Failed to load ONNX plate detector from {model_path}: {e}")
                self._status = f"LOAD_ERROR: {e}"
        elif self.config.runtime == "heuristic":
            self._status = "HEURISTIC_MODE"
        else:
            self._status = f"RUNTIME_{self.config.runtime.upper()}"

    def detect_plates(self, image: Optional[np.ndarray]) -> List[PlateDetectionResult]:
        """Detects all candidate license plates in an image or vehicle crop."""
        if not self.enabled or image is None or image.size == 0:
            return []

        if self._session is not None and self._status == "READY":
            return self._run_onnx_inference(image)

        # Fallback to heuristic edge detector if model unavailable or in heuristic mode
        heuristic_res = self._run_heuristic_extraction(image)
        return [heuristic_res] if heuristic_res is not None else []

    def extract_plate(self, vehicle_crop: Optional[np.ndarray]) -> Optional[PlateDetectionResult]:
        """
        Extracts top candidate license plate crop from vehicle image.
        Maintains full compatibility with existing vehicle detection pipeline.
        """
        candidates = self.detect_plates(vehicle_crop)
        if not candidates:
            return None
        # Return candidate with highest detection confidence
        return max(candidates, key=lambda c: c.confidence)

    def _run_onnx_inference(self, image: np.ndarray) -> List[PlateDetectionResult]:
        """Runs genuine ONNX inference and applies OpenCV Non-Maximum Suppression."""
        orig_h, orig_w = image.shape[:2]
        if orig_h < 15 or orig_w < 15:
            return []

        # YOLOv8 input preprocessing (640x640 letterbox/resize RGB)
        resized = cv2.resize(image, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})

        output = outputs[0]
        # YOLOv8 output shape: [1, 5, 8400] -> transpose to [1, 8400, 5]
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)

        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        boxes_for_nms = []
        confidences_for_nms = []

        for row in output[0]:
            cx, cy, bw, bh, score = row[:5]
            score = float(score)
            if score >= self.conf_threshold:
                pixel_cx = cx * scale_x
                pixel_cy = cy * scale_y
                pixel_w = bw * scale_x
                pixel_h = bh * scale_y

                px = int(pixel_cx - pixel_w / 2.0)
                py = int(pixel_cy - pixel_h / 2.0)
                pw = int(pixel_w)
                ph = int(pixel_h)

                boxes_for_nms.append([px, py, pw, ph])
                confidences_for_nms.append(score)

        if not boxes_for_nms:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, confidences_for_nms, score_threshold=self.conf_threshold, nms_threshold=0.45
        )

        results = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                x, y, w, h = boxes_for_nms[idx]
                conf = confidences_for_nms[idx]

                # Clamp to image bounds with 3-4% margin to prevent character edge clipping
                pad_x = int(0.03 * w)
                pad_y = int(0.04 * h)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(orig_w, x + w + pad_x)
                y2 = min(orig_h, y + h + pad_y)

                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w >= self.min_w and crop_h >= self.min_h:
                    plate_crop = image[y1:y2, x1:x2].copy()
                    results.append(
                        PlateDetectionResult(
                            crop=plate_crop,
                            bbox=(x1, y1, x2, y2),
                            confidence=round(conf, 3),
                            is_simulated=False,
                        )
                    )

        return results

    def _run_heuristic_extraction(self, vehicle_crop: np.ndarray) -> Optional[PlateDetectionResult]:
        """Heuristic Sobel-edge based plate candidate extraction when ONNX model is absent."""
        vh, vw = vehicle_crop.shape[:2]
        if vw < 60 or vh < 60:
            return None

        # Look for plate in lower 40% of vehicle chassis
        search_y1 = int(vh * 0.60)
        roi = vehicle_crop[search_y1:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_crop = None
        best_area = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(max(1, h))
            if 2.0 <= aspect <= 6.0 and w >= self.min_w and h >= self.min_h:
                area = w * h
                if area > best_area:
                    best_area = area
                    full_y = y + search_y1
                    plate_img = vehicle_crop[full_y : full_y + h, x : x + w].copy()
                    # Calculate actual edge density as honest heuristic confidence metric
                    edge_density = float(np.count_nonzero(thresh[y : y + h, x : x + w])) / max(1, area)
                    heur_conf = min(0.75, max(0.30, round(edge_density * 2.0, 2)))
                    best_crop = PlateDetectionResult(
                        crop=plate_img,
                        bbox=(x, full_y, x + w, full_y + h),
                        confidence=heur_conf,
                        is_simulated=True,
                    )

        return best_crop
