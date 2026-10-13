# Raspberry Pi 5 Deployment Guide — Smart Parking & HSRP

This guide walks you through setting up and running the **Smart Parking & HSRP Management System** on a physical **Raspberry Pi 5 (8 GB RAM)** running 64-bit Raspberry Pi OS (Debian 12 Bookworm).

---

## 1. Hardware Architecture & Requirements

- **Compute Unit:** Raspberry Pi 5 (8 GB RAM, Quad-Core Cortex-A76 @ 2.4 GHz)
- **Power Supply:** Official 27W USB-C Power Supply (5V / 5A)
- **Operating System:** Raspberry Pi OS 64-bit (with Desktop or Lite)
- **Primary Sensor:** Zebronics USB Webcam (UVC-compliant, standard USB 2.0/3.0 port)
- **Display (Optional):** Micro-HDMI to Standard HDMI connected to an external monitor
- **Network:** Wi-Fi or Ethernet (required only during initial package installation)

```
 +-------------------------------------------------------------------------+
 |                            RASPBERRY PI 5                               |
 |                                                                         |
 |  [Zebronics USB Webcam]                                                 |
 |            |                                                            |
 |            v                                                            |
 |   /dev/video0 (V4L2)                                                    |
 |            |                                                            |
 |            +---> [CameraService] (Single-owner, drained buffer=1)       |
 |                        |                                                |
 |                        v                                                |
 |             [OccupancyService]                                          |
 |                        |                                                |
 |         +--------------+--------------+                                 |
 |         |                             |                                 |
 |         v                             v                                 |
 |  [PatchOccupancyClassifier]   [VehicleDetector]                         |
 |    (ONNX Runtime CPU:            (ONNX Runtime CPU:                     |
 |     50 bays in ~18-50ms)          YOLOv8n @ 640x640)                    |
 |         |                             |                                 |
 |         +--------------+--------------+                                 |
 |                        |                                                |
 |                        v                                                |
 |              [SlotStateMachine] (Hysteresis & Stale=2.0s protection)    |
 |                        |                                                |
 |                        v                                                |
 |              [AllocationService] (ACID SQLite Concurrency Protection)   |
 |                        |                                                |
 |                        v                                                |
 |              FastAPI REST API & Technical Web Dashboard (Port 8000)     |
 |                        |                                                |
 |                        +---> [HDMI Monitor: Chromium Kiosk Mode]        |
 +-------------------------------------------------------------------------+
```

---

## 2. Quick-Start Setup (On Physical Raspberry Pi)

### Step 1: Copy Project onto the Raspberry Pi
On your Windows machine, open PowerShell and copy the repository to the Pi:
```powershell
# Replace 'pi' and 'raspberrypi.local' with your Pi username and hostname/IP
scp -r C:\Users\adity\Downloads\IoT pi@raspberrypi.local:~/smart-parking
```

### Step 2: Run the Automated Installer
SSH into your Raspberry Pi or open a terminal:
```bash
ssh pi@raspberrypi.local
cd ~/smart-parking

# Make scripts executable and run installer
chmod +x deployment/raspberry_pi/*.sh
bash deployment/raspberry_pi/install.sh
```
The installer automatically:
1. Validates 64-bit ARM architecture (`aarch64`).
2. Checks available RAM and storage.
3. Installs system libraries (`libgl1`, `libglib2.0-0`, `v4l-utils`).
4. Creates a fresh Linux virtual environment (`.venv-pi`).
5. Installs ARM64 CPU dependencies (`requirements-pi.txt`).
6. Verifies model files and their SHA-256 checksums (`check_models.sh`).
7. Initializes the SQLite database.
8. Runs the preflight self-diagnostic suite.

---

## 3. Running the Application

### Option A: Manual Foreground Launch (Testing & Calibration)
```bash
cd ~/smart-parking
bash deployment/raspberry_pi/start.sh
```
Open a browser on your monitor or network: `http://localhost:8000` (or `http://<pi-ip-address>:8000`).

### Option B: Automatic Startup via systemd Service
To keep the application running automatically in the background upon booting:
```bash
# 1. Substitute current username and path into service unit
sudo cp deployment/raspberry_pi/smart-parking.service /etc/systemd/system/smart-parking.service
sudo sed -i "s/%i/$(whoami)/g" /etc/systemd/system/smart-parking.service

# 2. Reload systemd daemon, enable, and start service
sudo systemctl daemon-reload
sudo systemctl enable smart-parking
sudo systemctl start smart-parking

# 3. Check live status and logs
sudo systemctl status smart-parking
sudo journalctl -u smart-parking -f
```

### Option C: Fullscreen HDMI Kiosk Mode
If an HDMI monitor is attached to the Pi for live operator monitoring:
```bash
cd ~/smart-parking
bash deployment/raspberry_pi/kiosk.sh
```

---

## 4. Operational Safety Guarantees

1. **Stale Camera Protection:**
   The frame staleness threshold is set to a conservative **2.0 seconds** in `configs/raspberry_pi.yaml`. If the USB webcam freezes or is unplugged, all bays immediately transition to `UNKNOWN`.
2. **Double-Allocation Prevention:**
   The virtual allocation engine uses SQLite `BEGIN IMMEDIATE;` transaction locks. Even under concurrent allocation requests, double allocation is physically impossible.
3. **Rejection of Unknown Bays:**
   Slots in state `UNKNOWN` or `OCCUPIED` are strictly rejected by the allocation API with a `409 Conflict`.
4. **Offline Capability:**
   Once installed, the system operates completely offline without internet connectivity.

---

## 5. Model Rollback Procedure

If you ever encounter an atypical parking lot geometry that requires rolling back to the original CNRPark baseline classifier:
```bash
cd ~/smart-parking
bash deployment/raspberry_pi/rollback.sh
```
The rollback script automatically preserves backups and restores the baseline weights.
