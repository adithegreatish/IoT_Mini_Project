# Troubleshooting & Operational Guide

## 1. Camera Capture Issues
- **Camera Not Found:** Ensure the webcam is connected and recognized in Windows Device Manager (under Cameras/Imaging Devices) or via `lsusb` / `v4l2-ctl --list-devices` on Linux. Check if another application (e.g., Zoom, Skype, Chrome) is already locking the camera device.
- **Low Frame Rate / Latency:** USB 2.0 hubs can saturate bandwidth. Connect directly to a native USB port on the motherboard / Raspberry Pi blue USB 3.0 port. Lower requested resolution from 1080p to 720p.

## 2. Occupancy False Positives & Flickering
- **Tree Shadows / Changing Sunlight:** Increase `occupancy_threshold` in `configs/windows.yaml` from 0.35 to 0.45.
- **Flickering Transitions:** Increase `smoothing_window_frames` (default 5) and `hysteresis_confirm_frames` (default 3) to demand longer temporal consistency before triggering a state change.

## 3. High CPU Load on Raspberry Pi 5
- Ensure `onnxruntime` is utilizing 4 threads (`num_threads: 4`).
- Decrease detector input resolution from `640x640` to `480x480` in `configs/raspberry_pi.yaml`.
- Increase `inference_interval_ms` to 120ms (~8 FPS), which is plenty for parking lot turnover without overheating the CPU.
