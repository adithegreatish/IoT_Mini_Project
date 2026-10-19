# PRODUCTION DEPLOYMENT & OPERATION GUIDE

**Target System:** Smart Parking + Indian HSRP Management System  
**Platforms Supported:** Windows 10/11 (Development/Demo) & Raspberry Pi 4 / Debian Linux (Edge Deployment)  
**Document Revision:** Final Release Submission  

---

## 1. System Requirements & Dependencies

### Hardware Requirements
- **Development / Demonstration Host:**
  - CPU: 4+ physical cores (Intel i5/i7 or AMD Ryzen).
  - RAM: 8 GB minimum.
  - Video Input: Standard UVC USB Webcam (e.g., Zebronics, Logitech) or DirectShow camera.
- **Edge Deployment Host (Raspberry Pi 4):**
  - Board: Raspberry Pi 4 Model B (2 GB, 4 GB, or 8 GB RAM).
  - Storage: MicroSD Card (16 GB+ Class 10 / A2) or USB 3.0 SSD.
  - Camera: USB UVC Webcam or Raspberry Pi Camera Module (CSI).
  - Power: Official 5.1V 3.0A USB-C Power Supply.

### Software Requirements
- Python 3.10, 3.11, or 3.12 (64-bit).
- Git.
- Operating System: Windows 11 or Raspberry Pi OS 64-bit (Debian Bookworm/Bullseye).

---

## 2. Installation & Setup

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone repository
git clone https://github.com/adithegreatish/IoT_Mini_Project.git
cd IoT_Mini_Project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / Raspberry Pi:
source .venv/bin/activate
```

### Step 2: Install Core Python Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Local Environment
```bash
# Copy template configuration
cp .env.example .env
```

---

## 3. Windows Execution & Demonstration

### Step 1: Verify Configuration
Inspect `configs/windows.yaml` to ensure:
```yaml
app:
  host: "127.0.0.1"
  port: 8000
camera:
  device_index: 0
  backend: "DSHOW"
  mock_mode: true  # Automatically falls back to synthetic video if camera 0 is unplugged
detection:
  model_path: "models/vehicle_detector.onnx"
  runtime: "onnx"
```

### Step 2: Start the Application
```bash
python -m app.main --config configs/windows.yaml
```

The application will start:
- Uvicorn server on `http://127.0.0.1:8000`.
- SQLite database initialized in WAL mode at `data/parking_events.sqlite`.
- Background inference thread launched at ~15 FPS.
- Live MJPEG stream served on `/api/stream/video_feed`.

### Step 3: Access the Dashboard
Open your web browser and navigate to:
```
http://localhost:8000
```

### Step 4: Graceful Shutdown
Press `Ctrl+C` in the terminal. The lifespan manager safely flushes SQLite WAL buffers, stops the camera grabber, joins the background inference worker, and exits cleanly.

---

## 4. Raspberry Pi 4 Deployment

### Step 1: Install System Packages
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1-mesa-glx v4l-utils
```

### Step 2: Grant Camera Permissions
```bash
sudo usermod -a -G video $USER
```

### Step 3: Identify Camera Device Index
```bash
v4l2-ctl --list-devices
```
Note the `/dev/videoX` index (typically `/dev/video0`) and configure `device_index` in `configs/rpi4.yaml`.

### Step 4: Run Application
```bash
python3 -m app.main --config configs/rpi4.yaml
```

### Step 5: (Optional) Systemd Daemon Service Setup
To automatically run the application on boot:
```ini
# /etc/systemd/system/smart-parking.service
[Unit]
Description=Smart Parking and HSRP Management Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/IoT_Mini_Project
ExecStart=/home/pi/IoT_Mini_Project/.venv/bin/python3 -m app.main --config configs/rpi4.yaml
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable smart-parking.service
sudo systemctl start smart-parking.service
```

---

## 5. System Health Verification & Smoke Tests

Run the following curl commands to verify service readiness:

1. **Service Health Check:**
   ```bash
   curl -s http://127.0.0.1:8000/api/health
   # Expected response: {"uptime_sec": ..., "platform": "Windows/Linux", "status": "healthy"}
   ```

2. **Camera Hardware Status:**
   ```bash
   curl -s http://127.0.0.1:8000/api/camera/status
   # Confirms negotiated resolution, actual FPS, and connection health.
   ```

3. **Active Parking Slots:**
   ```bash
   curl -s http://127.0.0.1:8000/api/slots
   # Returns calibrated bays merged with real-time occupancy and reservation status.
   ```

4. **Run Full Test Suite:**
   ```bash
   pytest tests/ -v
   # All 114 tests must pass.
   ```

---

## 6. Backup & Recovery Procedures

### Database Backup
The SQLite database stores historical occupancy events, active reservations, and audit logs:
```bash
# Safely snapshot SQLite in WAL mode
sqlite3 data/parking_events.sqlite ".backup data/backup_parking_events.sqlite"
```

### Calibration Backup
Parking bay coordinates are stored in human-readable JSON:
```bash
cp data/calibration.json data/calibration_backup.json
```

### Restoring from Backup
```bash
cp data/backup_parking_events.sqlite data/parking_events.sqlite
cp data/calibration_backup.json data/calibration.json
```

---

## 7. Scripted 5-Minute Demonstration Walkthrough

When presenting to evaluators, execute this sequence:

1. **Dashboard Overview (Tab 1: Overview):**
   - Point out live video stream with real-time detection bounding boxes.
   - Show status counters: Total Calibrated Bays, Free Bays, Occupied Bays, Unknown Bays.
2. **Virtual Allocation Demonstration (Tab 2: Virtual Allocation):**
   - In the Quick Allotment panel, select a Free bay (e.g., Bay A-01).
   - Enter Vehicle Plate `MH12AB1234` and Driver ID `prof_evaluator`.
   - Click **Allot Bay**. Point out that the bay instantly turns amber/cyan (`RESERVED`) with an active reservation code.
   - Explain ACID concurrency: try submitting the same form again. Show that duplicate allocation is rejected with an explanatory conflict message.
3. **Parking Bay Calibration CAD Editor (Tab 3: Parking Calibration):**
   - Switch to the Calibration tab. Show the blueprint CAD grid and the interactive polygon canvas.
   - Click on an existing bay to show the coordinate row highlight in the Excel-style spreadsheet.
   - Demonstrate drag-to-resize or vertex editing.
4. **License Plate & HSRP Analysis (Tab 4: Events):**
   - Show recent vehicle arrival events.
   - Display the advisory HSRP visual analysis report and note the privacy masking (`MH12****34`).
5. **System Diagnostics (Tab 7: Diagnostics):**
   - Click **Run Self-Diagnostics**. Show real-time telemetry: camera FPS, ONNX inference latency (~15 ms), memory footprint (<500 MB), and SQLite journal mode (`WAL`).
