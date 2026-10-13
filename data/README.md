# Data Directory Structure

This directory houses local runtime databases, calibrations, datasets, and diagnostic reports:

- `calibration.json`: Saved parking bay polygon definitions with normalized coordinates.
- `parking_events.sqlite`: Local SQLite database containing state change events and camera diagnostics.
- `raw/`: Unprocessed footage and webcam snapshots (strictly kept local; never committed).
- `processed/`: Formatted image crops and normalized datasets.
- `annotations/`: YOLO/Pascal VOC annotation XMLs/TXTs.
- `splits/`: Group-aware train/val/test split manifests.
- `demo/`: Permitted fixture images/videos for offline demonstrations.
- `test_reports/`: Timestamped JSON and HTML self-test reports.
