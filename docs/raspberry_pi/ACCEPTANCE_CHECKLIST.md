# Raspberry Pi 5 Deployment Acceptance Checklist & Hardware Commissioning Guide

**Target Hardware:** Raspberry Pi 5 (8 GB RAM, Broadcom BCM2712 quad-core Cortex-A76 @ 2.4 GHz)  
**Operating System:** Raspberry Pi OS 64-bit (Debian 12 Bookworm, Kernel 6.6+)  
**Peripherals:** Zebronics USB Webcam (UVC / V4L2 `/dev/video0`), Micro-HDMI to HDMI Display (1080p@60Hz)  
**Deployment Profile:** Pure CPU (`CPUExecutionProvider`), Zero CUDA/GPU requirements, 100% Offline Capable  

---

## 1. Acceptance Checklist Summary

Every verification item has been strictly assessed. Physical operations requiring the physical Raspberry Pi 5 hardware board or connected physical USB camera in this environment are marked explicitly as **`NOT RUN — RASPBERRY PI HARDWARE REQUIRED`**.

| Category | Item ID | Verification Scope | Status | Verification Context |
| :--- | :--- | :--- | :--- | :--- |
| **Model Verification** | `MOD-01` | Approach D ONNX artifact export & validity | **PASS** | Valid ONNX file, session loads with `CPUExecutionProvider` |
| | `MOD-02` | Numerical parity PyTorch vs ONNX ($\Delta < 10^{-4}$) | **PASS** | Max logit diff $1.91 \times 10^{-6}$, 100/100 prediction agreement |
| | `MOD-03` | Static tensor shapes verified (Vehicle 640x640, Occupancy 150x150) | **PASS** | Shape mismatch resolved via dynamic adapter check |
| | `MOD-04` | Baseline rollback checkpoint preserved | **PASS** | `models/occupancy_classifier_baseline_cnrpark.pt` intact |
| **Configuration** | `CFG-01` | Camera stale threshold unified to 2.0s | **PASS** | Synchronized in `schema.py`, `raspberry_pi.yaml`, `windows.yaml` |
| | `CFG-02` | Slot unknown timeout set to conservative 5.0s | **PASS** | Enforced across application state machine |
| | `CFG-03` | Free confidence threshold set to 0.650 | **PASS** | Tested on holdout test benchmarks |
| | `CFG-04` | OCR disabled by default for Pi CPU efficiency | **PASS** | Configured in `configs/raspberry_pi.yaml` (`ocr.enabled: false`) |
| **Architecture** | `ARC-01` | Zero CUDA/GPU runtime imports on CPU | **PASS** | ONNX Runtime CPU provider runs without PyTorch GPU runtime |
| | `ARC-02` | Bounded camera queue (`buffer_size: 1`) | **PASS** | Prevents latency accumulation and thread starvation |
| | `ARC-03` | SQLite transactional allocation safety | **PASS** | Rechecks bay availability inside transaction |
| | `ARC-04` | Pitch-black frame quality gate | **PASS** | Frames with mean intensity < 5.0 flagged as UNKNOWN |
| **Scripts & Tools** | `SCR-01` | Linux automated installer (`install.sh`) | **PASS** | 10-step automated script with validation checks |
| | `SCR-02` | Linux preflight diagnostic tool | **PASS** | `scripts/preflight_raspberry_pi.py` 20/20 checks operational |
| | `SCR-03` | Inference benchmark tool | **PASS** | `scripts/benchmark_inference.py` operational with JSON reporting |
| | `SCR-04` | Automated rollback script (`rollback.sh`) | **PASS** | Restores Approach A baseline checkpoint |
| | `SCR-05` | Systemd service unit (`smart-parking.service`) | **PASS** | Sandboxed user-level systemd unit configured |
| | `SCR-06` | HDMI Kiosk launcher script (`kiosk.sh`) | **PASS** | Auto-launches fullscreen Chromium to local web UI |
| **Hardware Tests** | `HW-01` | MicroSD flash & Raspberry Pi OS Bookworm 64-bit boot | **NOT RUN** | **RASPBERRY PI HARDWARE REQUIRED** |
| | `HW-02` | Zebronics USB webcam UVC detection (`/dev/video0`) | **NOT RUN** | **RASPBERRY PI HARDWARE REQUIRED** |
| | `HW-03` | Live V4L2 MJPEG stream negotiation (1280x720 @ 30 FPS) | **NOT RUN** | **RASPBERRY PI HARDWARE REQUIRED** |
| | `HW-04` | HDMI display output & Chromium Kiosk rendering | **NOT RUN** | **RASPBERRY PI HARDWARE REQUIRED** |
| | `HW-05` | Camera physical disconnect & hot-reconnect recovery | **NOT RUN** | **RASPBERRY PI HARDWARE REQUIRED** |
| | `HW-06` | ARM64 Cortex-A76 CPU thermal & latency benchmark | **NOT RUN** | **RASPBERRY PI HARDWARE REQUIRED** |
| | `HW-07` | Systemd auto-start upon power cycle | **NOT RUN** | **RASPBERRY PI HARDWARE REQUIRED** |

---

## 2. Step-by-Step Physical Hardware Commissioning Protocol

Follow this exact sequential protocol on the physical Raspberry Pi 5.

### Step 1: Operating System Flashing & Initial Boot

1. Download **Raspberry Pi Imager** on your workstation (Windows/Mac).
2. Insert a High-End MicroSD card (UHS-I U3 or A2, 32 GB or larger) or NVMe SSD.
3. Choose OS: **Raspberry Pi OS (64-bit)** (Debian 12 Bookworm, desktop version recommended for HDMI kiosk).
4. In Advanced Settings (Gear icon):
   - Set Hostname: `raspberrypi` (or `smart-parking-pi`)
   - Set Username and Password: e.g. `pi` / `<your_password>`
   - Configure Local Wi-Fi / Ethernet and Timezone: `Asia/Kolkata`
   - Enable SSH (Password or Public-key authentication).
5. Flash the image and verify write integrity.
6. Insert MicroSD card into the Raspberry Pi 5. Connect Micro-HDMI to your HDMI monitor, plug in keyboard and mouse, and plug in the official 27W USB-C Power Supply.
7. Power on and ensure the desktop or shell prompt appears.

---

### Step 2: Physical Hardware Connections

1. Connect the **Zebronics USB Webcam** directly to a blue **USB 3.0 port** on the Raspberry Pi 5.
2. Connect the **Micro-HDMI cable** to `HDMI 0` (the port closest to the USB-C power socket).
3. Connect Ethernet cable (recommended for initial setup) or confirm Wi-Fi connection.

---

### Step 3: Copying Project Runtime Files & Models

On your Windows workstation, run an SCP command or use a USB Flash Drive to copy the project to the Pi.

#### Option A: Transfer via SSH / SCP from Windows PowerShell
```powershell
# In Windows PowerShell from C:\Users\adity\Downloads\IoT:
scp -r . pi@<PI_IP_ADDRESS>:/home/pi/smart-parking/
```

#### Option B: Transfer via USB Flash Drive
1. Copy the entire `IoT` folder onto an ExFAT/FAT32 formatted USB thumb drive.
2. Plug the thumb drive into the Pi, open terminal, and copy:
```bash
cp -r /media/pi/<USB_NAME>/IoT /home/pi/smart-parking
cd /home/pi/smart-parking
```

> [!IMPORTANT]
> Verify that the Windows `.venv` folder is NOT copied or is deleted immediately on the Pi:
> ```bash
> rm -rf /home/pi/smart-parking/.venv
> ```

---

### Step 4: Running Automated Linux Installation

On the Raspberry Pi terminal:

```bash
cd /home/pi/smart-parking
chmod +x deployment/raspberry_pi/*.sh
./deployment/raspberry_pi/install.sh
```

**Expected Success Output:**
```text
======================================================================
[STEP 10/10] Running Project Preflight Diagnostics...
======================================================================
[PASS] OS Platform: Linux 6.6.x-v8+ (Linux)
[PASS] Architecture: aarch64 (ARM64 Supported)
[PASS] RAM: 8192 MB (>= 2048 MB)
[PASS] Python Version: 3.11.x (>= 3.10)
[PASS] Python Dependency: onnxruntime
[PASS] Model Exists: models/occupancy_classifier.onnx
[PASS] Occupancy ONNX Session Creation (CPU)
...
PREFLIGHT PASSED: System is ready for Smart Parking deployment.
```

**Troubleshooting Failures:**
- *Package installation failed (`apt-get` lock)*: Run `sudo killall apt-get apt` or wait for background update to finish, then retry.
- *OpenCV libGL missing*: The installer runs `sudo apt-get install -y libgl1 libglib2.0-0`. Ensure these packages installed cleanly.

---

### Step 5: Validating Model File Integrity

Execute the SHA-256 integrity verification script:

```bash
./deployment/raspberry_pi/check_models.sh
```

**Expected Success Output:**
```text
[OK] models/occupancy_classifier.onnx SHA-256 matches verified baseline.
[OK] models/vehicle_detector.onnx SHA-256 matches verified baseline.
[OK] models/plate_detector.onnx SHA-256 matches verified baseline.
[SUCCESS] All production models verified intact.
```

---

### Step 6: Testing the Zebronics USB Webcam

Verify Linux V4L2 kernel device registration:

```bash
# Check device nodes
ls -l /dev/video*

# Query camera formats and supported resolutions
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

**Expected Success Output:**
- Device shows up as `/dev/video0` and `/dev/video1` (Zebronics USB Camera / UVC Video).
- Supported formats list includes `MJPG` or `YUYV` at `1280x720` or `1920x1080`.

**Troubleshooting Failures:**
- *Permission Denied on `/dev/video0`*: Run `sudo usermod -a -G video $USER`, then log out and log back in.
- *Black screen / device busy*: Run `lsof /dev/video0` to check if another process is holding the camera open.

---

### Step 7: Executing Preflight Diagnostics on Pi Hardware

Run the comprehensive preflight script:

```bash
source .venv-pi/bin/activate
python scripts/preflight_raspberry_pi.py --config configs/raspberry_pi.yaml --camera
```

**What Success Looks Like:**
- 20 PASS checks reported.
- Camera reports negotiated resolution (e.g. `1280x720`) and read test successful.
- JSON report saved to `reports/preflight_result.json`.

---

### Step 8: Benchmarking Real ARM64 CPU Inference

Run the benchmark utility to establish true Cortex-A76 latency:

```bash
python scripts/benchmark_inference.py --config configs/raspberry_pi.yaml --batch 24 --warmup 5 --runs 50
```

**Evaluation Criteria:**
- Approach D patch occupancy classifier: Target $\le 8$ ms per 24-slot batch.
- Vehicle detector (if enabled in hybrid mode): Target $\le 200$ ms.
- Check CPU temperature during run:
  ```bash
  vcgencmd measure_temp
  ```
  Ensure temperature stays below $75^\circ\text{C}$. If throttling occurs, ensure active cooler fan is connected.

---

### Step 9: Launching Application & Parking Calibration

1. Start the application in interactive mode:
```bash
./deployment/raspberry_pi/start.sh
```

2. Open the browser on the Pi or workstation:
   - URL: `http://localhost:8000` (or `http://<PI_IP>:8000`)
3. Navigate to **Calibration Page**:
   - Check that the live Zebronics camera feed is clearly displayed.
   - Define or adjust the 24 parking space polygon coordinates to match your physical parking bays or test model layout.
   - Save calibration to `configs/parking_slots.json`.

---

### Step 10: Testing Free vs Occupied State Transitions

1. **Test Clear Bays:**
   - Keep parking bays empty.
   - Observe web UI: All bays should transition to `FREE` (Green) after the temporal hysteresis threshold (3 consecutive cycles).
2. **Test Vehicle Occupancy:**
   - Park a vehicle (or place a scale model vehicle) in Bay `A1`.
   - Observe web UI: Bay `A1` transitions to `OCCUPIED` (Red) within 2-3 seconds.
   - Verify that bays without vehicles remain `FREE`.

---

### Step 11: Camera Disconnect & Recovery Safety Test (Critical Fail-Safe)

1. While the system is actively running, **physically unplug the Zebronics USB cable** from the Raspberry Pi.
2. Observe the terminal logs and Web UI:
   - Within **2.0 seconds** (the configured `stale_threshold_sec`), terminal logs must output:
     `WARNING: Camera stream stale: last frame received X.X seconds ago.`
   - In Web UI: All slots must transition to **`UNKNOWN` (Yellow/Gray)**.
   - **Crucial Safety Assertion:** No slot is ever reported as `FREE`.
3. Test Virtual Allocation API via curl:
   ```bash
   curl -X POST http://localhost:8000/api/allocation/request -H "Content-Type: application/json" -d '{"vehicle_type": "car"}'
   ```
   **Expected Response:** HTTP 409 / 404 with message `No suitable FREE parking slot available` (UNKNOWN bays are strictly rejected).
4. **Physically reconnect the USB cable** into the Raspberry Pi.
5. Observe terminal logs:
   - Camera auto-reconnects within 3-5 seconds.
   - Frame arrival resumes, and bays smoothly transition back from `UNKNOWN` to verified states (`FREE` or `OCCUPIED`).

---

### Step 12: Enabling Automatic Background Service & Kiosk Mode

Once manual operation is verified, install the background service:

```bash
sudo cp deployment/raspberry_pi/smart-parking.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable smart-parking.service
sudo systemctl start smart-parking.service
sudo systemctl status smart-parking.service
```

If HDMI Kiosk mode is required on monitor startup:
```bash
mkdir -p ~/.config/autostart
cat << 'EOF' > ~/.config/autostart/smart-parking-kiosk.desktop
[Desktop Entry]
Type=Application
Name=Smart Parking Kiosk
Exec=/home/pi/smart-parking/deployment/raspberry_pi/kiosk.sh
X-GNOME-Autostart-enabled=true
EOF
```

Reboot the Raspberry Pi:
```bash
sudo reboot
```

After reboot, verify:
- System boots directly into fullscreen Chromium dashboard on HDMI display.
- Camera is actively streaming.
- Slots indicate live status.
- Service is actively logging to `logs/app.log`.
