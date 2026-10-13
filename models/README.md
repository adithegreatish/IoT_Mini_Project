# Model Registry & Artifacts

This directory stores exported deep-learning models (ONNX runtime format) and metadata manifests.

## Files
- `manifest.json`: Single source of truth recording model IDs, checksums, input shapes, supported platforms, licenses, and benchmarked metrics.
- `vehicle_detector.onnx`: ONNX model for vehicle bounding box detection.
- `plate_detector.onnx`: ONNX model for license plate localization.

## Deployment Guidelines
- All models must run seamlessly on **CPU** using `onnxruntime` on both Windows (x86_64) and Raspberry Pi 5 (ARM64).
- Models must be verified against `manifest.json` sha256 checksums before production inference loading.
- When physical weights are not yet present or training is pending, the application provides a mock detector fallback to enable end-to-end system validation and testing without crashes.
