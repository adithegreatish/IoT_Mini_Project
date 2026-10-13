# Raspberry Pi 5 Deployment Troubleshooting Guide

This guide addresses common real-world failure modes encountered when deploying on physical Raspberry Pi 5 hardware.

---

## 1. Camera & Video Capture Issues

### Issue 1.1: `Permission denied: '/dev/video0'`
- **Symptom:** The camera worker fails to open device 0 or logs `Permission denied`.
- **Cause:** Current Linux user lacks membership in the `video` group.
- **Fix:**
  ```bash
  sudo usermod -a -G video $(whoami)
  # Log out and log back in, or run:
  newgrp video
  ```

### Issue 1.2: `Device or resource busy`
- **Symptom:** `cv2.VideoCapture(0)` fails with error code -16.
- **Cause:** Another process (e.g., Motion, another OpenCV worker, or WebRTC) is holding an exclusive V4L2 lock on the webcam.
- **Fix:**
  ```bash
  # Identify the process holding /dev/video0
  sudo fuser -v /dev/video0
  # Terminate the locking process (replace <PID> with actual process ID)
  sudo kill -9 <PID>
  ```

### Issue 1.3: Camera Disconnects and System Reports Stale/Unknown
- **Symptom:** All bays suddenly change to `UNKNOWN` in yellow on the dashboard.
- **Cause:** Expected fail-safe behavior. The USB webcam disconnected, lost power, or stopped sending frames for $> 2.0$ seconds.
- **Fix:**
  1. Check physical USB connection. Ensure webcam is connected to a native USB 3.0 port (blue port).
  2. Check kernel logs:
     ```bash
     dmesg -T | grep -i usb | tail -n 20
     ```
  3. The `OpenCVCamera` service will automatically attempt reconnection every 2.0 seconds. Once frames resume, the state machine will restore stable observations.

### Issue 1.4: Low Video Frame Rate or Negotiated Resolution Mismatch
- **Symptom:** The webcam defaults to $640 \times 480$ @ 5 FPS instead of $1280 \times 720$.
- **Cause:** The USB webcam is negotiating uncompressed YUYV format, which saturates USB 2.0 bandwidth.
- **Fix:**
  Check supported formats using `v4l-utils`:
  ```bash
  v4l2-ctl --list-formats-ext
  ```
  If MJPEG is available, configure `configs/raspberry_pi.yaml`:
  ```yaml
  camera:
    requested_width: 1280
    requested_height: 720
    requested_fps: 15
  ```

---

## 2. System Performance & Thermal Envelope

### Issue 2.1: Pi 5 CPU Throttling Under Continuous Load
- **Symptom:** CPU clock drops from 2.4 GHz to 1.5 GHz; logs report high latency.
- **Cause:** Broadcom BCM2712 SoC reaching $80^\circ\text{C}$ without active cooling.
- **Fix:**
  1. Verify temperature:
     ```bash
     vcgencmd measure_temp
     vcgencmd get_throttled
     ```
  2. Ensure the official **Raspberry Pi 5 Active Cooler** is installed and spinning.
  3. In `configs/raspberry_pi.yaml`, optimize the inference cadence:
     ```yaml
     detection:
       inference_interval_ms: 150  # ~6.6 FPS to maintain cool thermals (<65°C)
     ```

### Issue 2.2: Memory Pressure / Out of Memory (OOM)
- **Symptom:** Process killed by Linux kernel OOM killer (`dmesg` reports `Out of memory: Kill process`).
- **Cause:** Heavy dependencies (like PyTorch / EasyOCR) loaded into RAM.
- **Fix:**
  Ensure `ocr.enabled: false` in `configs/raspberry_pi.yaml`. The core parking occupancy engine with ONNX Runtime uses only **~160 MB RSS**, leaving $> 7.5\text{ GB}$ free on an 8 GB Pi 5.

---

## 3. Dependency & Shared Library Issues

### Issue 3.1: `ImportError: libGL.so.1: cannot open shared object file`
- **Symptom:** OpenCV fails upon import in Python.
- **Cause:** Missing OpenGL runtime library on minimal/headless Linux installations.
- **Fix:**
  ```bash
  sudo apt update && sudo apt install -y libgl1 libglib2.0-0
  ```

### Issue 3.2: `No module named onnxruntime`
- **Symptom:** Python raises `ModuleNotFoundError` when loading models.
- **Cause:** Running outside the dedicated virtual environment `.venv-pi`.
- **Fix:**
  Always activate the Linux environment before running commands:
  ```bash
  source .venv-pi/bin/activate
  ```

---

## 4. systemd Service Troubleshooting

### Issue 4.1: Service Fails on Boot (`Exit Code 1`)
- **Fix:**
  Check detailed service logs:
  ```bash
  sudo journalctl -u smart-parking.service -n 50 --no-pager
  ```
  Common causes:
  1. `WorkingDirectory` path does not match the actual user directory. Ensure `smart-parking.service` has the correct path (e.g., `/home/pi/smart-parking`).
  2. Database file permissions:
     ```bash
     sudo chown -R $(whoami):$(whoami) data/ logs/
     ```

---

## 5. Network & API Access

### Issue 5.1: Dashboard Inaccessible from Other Machines on the Local Network
- **Symptom:** Works on `http://127.0.0.1:8000`, but `http://192.168.1.x:8000` is refused.
- **Cause:** `configs/raspberry_pi.yaml` binds to `127.0.0.1` (localhost only).
- **Fix:**
  Update `configs/raspberry_pi.yaml`:
  ```yaml
  app:
    host: "0.0.0.0"  # Binds to all local network interfaces
    port: 8000
  ```
  Restart service:
  ```bash
  sudo systemctl restart smart-parking
  ```
