"""Configuration schemas for the Smart Parking & HSRP system."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "Smart Parking & HSRP System"
    version: str = "1.0.0"
    env: Literal["windows", "raspberry_pi", "linux", "test"] = "windows"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True


class CameraConfig(BaseModel):
    device_index: int = 0
    backend: Literal["AUTO", "DSHOW", "V4L2", "MSMF"] = "AUTO"
    requested_width: int = 1280
    requested_height: int = 720
    requested_fps: int = 30
    buffer_size: int = 1
    reconnect_interval_sec: float = 2.0
    reconnect_max_attempts: int = 10
    stale_threshold_sec: float = 2.0
    mock_mode: bool = False
    synthetic_scenario: Literal["normal", "all_free", "all_occupied", "mixed", "camera_loss", "straddling"] = "normal"


class ParkingConfig(BaseModel):
    calibration_file: str = "data/calibration.json"
    occupancy_threshold: float = Field(default=0.35, ge=0.05, le=1.0)
    classifier_threshold: float = Field(default=0.65, ge=0.05, le=1.0)
    inference_mode: Literal["detector", "classifier", "hybrid"] = "detector"
    classifier_model_path: str = "models/occupancy_classifier.onnx"
    smoothing_window_frames: int = Field(default=5, ge=1, le=60)
    hysteresis_confirm_frames: int = Field(default=3, ge=1, le=30)
    unknown_timeout_sec: float = Field(default=5.0, ge=1.0)


class DetectionConfig(BaseModel):
    enabled: bool = True
    model_id: str = "vehicle-yolo-v8n"
    model_path: str = "models/vehicle_detector.onnx"
    runtime: Literal["onnx", "opencv", "mock"] = "onnx"
    confidence_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    input_width: int = 640
    input_height: int = 640
    classes: List[str] = Field(default_factory=lambda: ["car", "motorcycle", "bus", "truck"])
    inference_interval_ms: int = Field(default=66, ge=10, le=1000)


class PlatesConfig(BaseModel):
    enabled: bool = True
    model_id: str = "plate-detector-tiny"
    model_path: str = "models/plate_detector.onnx"
    runtime: Literal["onnx", "opencv", "heuristic", "mock"] = "onnx"
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_crop_width: int = Field(default=60, ge=10)
    min_crop_height: int = Field(default=20, ge=5)
    hsrp_cue_analysis_enabled: bool = True


class OCRConfig(BaseModel):
    enabled: bool = True
    engine: Literal["easyocr", "tesseract", "paddle", "heuristic", "mock"] = "easyocr"
    confidence_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    registration_pattern: str = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$"
    allow_uncertain: bool = True


class DatabaseConfig(BaseModel):
    db_path: str = "data/parking_events.sqlite"
    retention_days: int = Field(default=30, ge=1)
    vacuum_on_startup: bool = False


class DiagnosticsConfig(BaseModel):
    report_dir: str = "data/test_reports"
    quick_timeout_sec: float = 10.0
    full_timeout_sec: float = 60.0


class PrivacyConfig(BaseModel):
    mask_plates_by_default: bool = True
    redact_log_plates: bool = True
    allow_raw_retention: bool = False


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file_path: str = "logs/app.log"
    max_bytes: int = 10485760  # 10 MB
    backup_count: int = 5


class RootConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    parking: ParkingConfig = Field(default_factory=ParkingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    plates: PlatesConfig = Field(default_factory=PlatesConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
