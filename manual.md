# Raspberry Pi 5 (8 GB) Comprehensive Setup & Operations Manual

This manual provides complete, step-by-step instructions for deploying and running the **Smart Parking Management & Indian HSRP Recognition System** on a brand-new **Raspberry Pi 5**. Every command is documented from initial OS configuration through background system service installation and kiosk display mode.

---

## Table of Contents
1. [Hardware & Prerequisites](#1-hardware--prerequisites)
2. [Initial Operating System Preparation](#2-initial-operating-system-preparation)
3. [Transferring the Project to the New Pi](#3-transferring-the-project-to-the-new-pi)
4. [Method A: Fast-Track Automated Setup (Recommended)](#4-method-a-fast-track-automated-setup-recommended)
5. [Method B: Manual Step-by-Step Installation](#5-method-b-manual-step-by-step-installation)
6. [Hardware & Webcam Configuration](#6-hardware--webcam-configuration)
7. [Starting the Application](#7-starting-the-application)
8. [Verifying System Operation & In-App Diagnostics](#8-verifying-system-operation--in-app-diagnostics)
9. [Configuring Automatic Boot Service (systemd)](#9-configuring-automatic-boot-service-systemd)
10. [Configuring HDMI Fullscreen Kiosk Display](#10-configuring-hdmi-fullscreen-kiosk-display)
11. [Troubleshooting & Common Issues](#11-troubleshooting--common-issues)

---

## 1. Hardware & Prerequisites

Ensure the following equipment is ready before beginning:
- **Board:** Raspberry Pi 5 (4 GB or 8 GB RAM recommended).
- **Power Supply:** Official Raspberry Pi 27W USB-C Power Supply (prevents CPU throttling under neural network inference).
- **Storage:** MicroSD Card (32 GB+ Class 10 / A2) or NVMe SSD with **Raspberry Pi OS 64-bit (Debian Bookworm)**.
- **Camera:** Standard UVC USB Webcam (e.g., Zebronics, Logitech C270/C920) plugged into a blue USB 3.0 port.
- **Network:** Active Wi-Fi or Ethernet connection.
- **Display (Optional):** Micro-HDMI to HDMI cable connected to a monitor for desktop/kiosk usage.

---

## 2. Initial Operating System Preparation

On your new Raspberry Pi 5, open a terminal (or connect via SSH):

```bash
# 1. Verify 64-bit ARM architecture (must return aarch64 or arm64)
uname -m

# 2. Update package repositories and upgrade existing system packages
sudo apt update && sudo apt upgrade -y

# 3. Install core system packages, build libraries, and video utilities
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    git-lfs \
    v4l-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    chromium-browser

# 4. Add your current user to the 'video' group for UVC webcam access
sudo usermod -a -G video $USER

# 5. Apply the video group permissions to your active shell
newgrp video
```

---

## 3. Transferring the Project to the New Pi

Choose **one** of the two transfer methods below:

### Option 3.1: Clone via Git (If Hosted on Git Repository)
```bash
cd ~
git clone https://github.com/adithegreatish/IoT_Mini_Project.git smart-parking
cd smart-parking

# Pull binary ONNX model weights via Git LFS
git lfs install
git lfs pull
```

### Option 3.2: Transfer from Development PC via SCP / SFTP / USB
On your development machine (Windows PowerShell):
```powershell
# Copy the pre-packaged deployment archive to the Pi
scp c:\Users\adity\Downloads\IoT\dist\smart_parking_raspberry_pi_deployment.zip pi@<RASPBERRY_PI_IP>:~/
```

On your Raspberry Pi terminal:
```bash
cd ~
unzip smart_parking_raspberry_pi_deployment.zip -d smart-parking
cd smart-parking
```

---

## 4. Method A: Fast-Track Automated Setup (Recommended)

The project includes an all-in-one automated deployment script that verifies system specs, initializes the virtual environment, downloads dependencies, checks model SHA-256 hashes, and sets up the SQLite database.

```bash
# Navigate to the project root
cd ~/smart-parking

# Ensure shell scripts have execution permissions and UNIX line endings
chmod +x deployment/raspberry_pi/*.sh
sed -i 's/\r$//' deployment/raspberry_pi/*.sh

# Run the automated installer
bash deployment/raspberry_pi/install.sh

# Launch the system
bash deployment/raspberry_pi/start.sh
```

If you prefer to configure everything manually step-by-step, proceed to [Method B](#5-method-b-manual-step-by-step-installation).

---

## 5. Method B: Manual Step-by-Step Installation

### Step 5.1: Create Native ARM64 Python Virtual Environment
> [!IMPORTANT]
> Never copy the `.venv` folder from a Windows computer. Always build a clean native virtual environment on ARM64 Linux.

```bash
cd ~/smart-parking

# Create virtual environment named .venv-pi
python3 -m venv .venv-pi

# Activate virtual environment
source .venv-pi/bin/activate

# Upgrade pip, wheel, and setuptools
pip install --upgrade pip setuptools wheel
```

### Step 5.2: Install Python Dependencies
```bash
# Install optimized headless CPU packages for Raspberry Pi 5
pip install -r requirements-pi.txt
```

Verify installed core packages:
```bash
python -c "import cv2, onnxruntime, fastapi, uvicorn, shapely; print('All core libraries imported successfully!')"
```

### Step 5.3: Verify Model Weights Integrity
Ensure model weights are valid ONNX binaries (not placeholder text):
```bash
bash deployment/raspberry_pi/check_models.sh
```
Expected output:
- `Parking Occupancy Net (Winning Approach D)`: `[PASS]`
- `Vehicle Detector (YOLOv8n)`: `[PASS]`

### Step 5.4: Initialize SQLite Database
Initialize the database tables and enable WAL (Write-Ahead Logging) mode:
```bash
python -c "
from app.infrastructure.paths import paths
from app.database.connection import DatabaseManager
db_path = paths.resolve('data/parking_events.sqlite')
db = DatabaseManager(str(db_path))
db.initialize_schema()
print(f'Database initialized at: {db_path}')
"
```

### Step 5.5: Run Preflight Self-Diagnostics
Run the self-diagnostic suite to confirm hardware, CPU, memory, and model readiness:
```bash
python scripts/preflight_raspberry_pi.py
```
Check generated report summary in terminal and in `data/test_reports/preflight_report.json`.

---

## 6. Hardware & Webcam Configuration

### Step 6.1: Identify Video Device Index
Plug your USB webcam into a blue USB 3.0 port and identify its node:
```bash
v4l2-ctl --list-devices
```
You will see output similar to:
```text
Zebronics USB Camera (usb-xhci-hcd.0-1):
    /dev/video0
    /dev/video1
```
Note the first video node (usually `/dev/video0`).

### Step 6.2: Adjust Configuration (`configs/raspberry_pi.yaml`)
Open the Pi configuration file in nano:
```bash
nano configs/raspberry_pi.yaml
```

Check the following fields:
1. **Network Binding:**
   - To access the dashboard only on the Pi monitor: `host: "127.0.0.1"`
   - To access the dashboard from other laptops/smartphones on your Wi-Fi: `host: "0.0.0.0"`
2. **Camera Index:**
   - If your webcam was detected on `/dev/video0`: `device_index: 0`
   - If your webcam was detected on `/dev/video1`: `device_index: 1`
   - If bench-testing without a physical camera plugged in: `mock_mode: true`

Save and exit in nano by pressing `Ctrl + O`, `Enter`, then `Ctrl + X`.

---

## 7. Starting the Application

### Option 7.1: Using Startup Script
```bash
bash deployment/raspberry_pi/start.sh
```

### Option 7.2: Using Direct Command
```bash
cd ~/smart-parking
source .venv-pi/bin/activate
export SMART_PARKING_ROOT="$(pwd)"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python app/main.py --config configs/raspberry_pi.yaml
```

### Step 7.3: Access the Web Dashboard
Open your web browser and navigate to:
- **From Raspberry Pi browser:** `http://localhost:8000`
- **From another computer on same LAN:** `http://<RASPBERRY_PI_IP>:8000` *(Run `hostname -I` on the Pi to find its IP address)*.

To stop the application at any time, press **`Ctrl + C`** in the terminal.

---

## 8. Verifying System Operation & In-App Diagnostics

With the application running, you can test the system components:

### 1. API Health Check
Open a second terminal on the Pi and test:
```bash
curl -s http://localhost:8000/api/health | python3 -m json.tool
```
Expected response:
```json
{
    "status": "healthy",
    "timestamp": "...",
    "database": "connected",
    "camera": "streaming"
}
```

### 2. Live Stream Endpoint
Test the MJPEG video stream:
```bash
curl -I http://localhost:8000/api/stream/video_feed
# Expect: HTTP/1.1 200 OK, Content-Type: multipart/x-mixed-replace; boundary=frame
```

### 3. Run Automated System Diagnostic Tool
In your activated virtual environment:
```bash
python -m app.diagnostics --quick
```

### 4. Run Acceptance Test Suite
```bash
pytest tests/ -v
```

---

## 9. Configuring Automatic Boot Service (systemd)

To ensure the parking system launches automatically on power-up and restarts automatically if terminated, install the systemd service.

### Step 9.1: Configure the Service File
Open the service file template:
```bash
nano deployment/raspberry_pi/smart-parking.service
```
Replace `%i` with your actual Linux username (e.g. `pi`):
```ini
[Unit]
Description=Smart Parking & HSRP Management Service
After=network.target multi-user.target
Wants=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-parking
Environment="SMART_PARKING_ROOT=/home/pi/smart-parking"
Environment="PATH=/home/pi/smart-parking/.venv-pi/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/pi/smart-parking/.venv-pi/bin/python app/main.py --config configs/raspberry_pi.yaml
Restart=always
RestartSec=5s

# Security sandboxing
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=/home/pi/smart-parking/data /home/pi/smart-parking/logs

StandardOutput=journal
StandardError=journal
SyslogIdentifier=smart-parking

[Install]
WantedBy=multi-user.target
```

### Step 9.2: Install and Enable the Service
```bash
# 1. Copy service file to systemd directory
sudo cp deployment/raspberry_pi/smart-parking.service /etc/systemd/system/smart-parking.service

# 2. Reload systemd daemon
sudo systemctl daemon-reload

# 3. Enable service to start on every boot
sudo systemctl enable smart-parking.service

# 4. Start the service immediately
sudo systemctl start smart-parking.service

# 5. Check operational status
sudo systemctl status smart-parking.service
```

### Step 9.3: Manage the Service
```bash
# View live application logs
journalctl -u smart-parking.service -f

# Restart service
sudo systemctl restart smart-parking.service

# Stop service
sudo systemctl stop smart-parking.service
```

---

## 10. Configuring HDMI Fullscreen Kiosk Display

If your Raspberry Pi 5 is connected to an HDMI monitor for live demonstration:

### Manual Kiosk Launch
```bash
bash deployment/raspberry_pi/kiosk.sh
```

### Autostart Kiosk on Desktop Boot
Depending on your window manager (Wayland / Wayfire vs X11):

**For Raspberry Pi OS Bookworm (Wayfire):**
Edit `~/.config/wayfire.ini`:
```ini
[autostart]
parking_kiosk = bash /home/pi/smart-parking/deployment/raspberry_pi/kiosk.sh
```

**For X11 / Openbox Desktop:**
Edit `~/.config/openbox/autostart` or `/etc/xdg/lxsession/LXDE-pi/autostart` and add:
```bash
@bash /home/pi/smart-parking/deployment/raspberry_pi/kiosk.sh
```

---

## 11. Troubleshooting & Common Issues

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `bash: \r: command not found` | Scripts transferred with Windows CRLF line endings. | Run `sed -i 's/\r$//' deployment/raspberry_pi/*.sh` to convert to UNIX line endings. |
| `Permission denied: '/dev/video0'` | User not in `video` Linux group. | Run `sudo usermod -a -G video $USER`, then log out and log back in, or run `newgrp video`. |
| `Checksum MISMATCH / File is Git LFS pointer` | Git LFS weights were not downloaded during clone. | Run `sudo apt install -y git-lfs && git lfs install && git lfs pull`. |
| `Address already in use (Port 8000)` | Previous instance or systemd service is already running. | Stop existing instance: `sudo systemctl stop smart-parking.service` or `sudo fuser -k 8000/tcp`. |
| `Camera frame read timeout / Stale camera` | Webcam disconnected or wrong device index. | Run `v4l2-ctl --list-devices` and update `device_index` in `configs/raspberry_pi.yaml`. Or set `mock_mode: true` for bench testing. |
| `Low voltage warning / CPU throttling` | Non-standard 5V phone charger used. | Connect official Raspberry Pi 27W (5.1V / 5.0A) power adapter to maintain maximum inference clock speed. |

---

## Quick Reference Summary Sheet

```bash
# === Quick Command Cheatsheet ===

# Navigate to project
cd ~/smart-parking

# Activate environment
source .venv-pi/bin/activate

# Start application directly
python app/main.py --config configs/raspberry_pi.yaml

# Manage background service
sudo systemctl start smart-parking.service
sudo systemctl stop smart-parking.service
sudo systemctl status smart-parking.service
journalctl -u smart-parking.service -f

# Check camera
v4l2-ctl --list-devices

# Health check
curl -s http://localhost:8000/api/health
```
