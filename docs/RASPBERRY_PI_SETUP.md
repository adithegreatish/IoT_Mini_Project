# Raspberry Pi 5 (8 GB) Deployment Guide

This guide describes deploying the application onto a target **Raspberry Pi 5 with 8 GB RAM** running standard 64-bit Raspberry Pi OS (Bookworm).

---

## 1. Hardware Checklist
- Raspberry Pi 5 (8 GB RAM recommended)
- Compatible 27W USB-C official power supply
- Standard 32 GB or 64 GB high-endurance A2 microSD card
- Zebronics USB webcam plugged into a blue USB 3.0 port
- Micro-HDMI to HDMI cable connected to a 1080p local monitor

---

## 2. Operating System & System Prerequisites
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git v4l-utils libgl1 libglib2.0-0 chromium-browser
```

---

## 3. Python Virtual Environment Setup
> **CRITICAL RULE:** Do NOT copy the `.venv` directory from Windows to the Raspberry Pi. Create a fresh virtual environment on ARM64:

```bash
cd /home/pi
git clone <repository_url> smart-parking
cd smart-parking

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt
```

---

## 4. Hardware Verification
Verify that the Zebronics USB webcam is enumerated via Video4Linux2:
```bash
v4l2-ctl --list-devices
```
Inspect supported capture formats:
```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

---

## 5. Startup & In-App Diagnostics
Run quick diagnostics:
```bash
python -m app.diagnostics --quick
```
Launch the system with the Pi configuration profile:
```bash
python -m app.main --config configs/raspberry_pi.yaml
```

---

## 6. Autostart Systemd Service
Create `/etc/systemd/system/smart-parking.service`:
```ini
[Unit]
Description=Smart Parking & HSRP System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-parking
ExecStart=/home/pi/smart-parking/.venv/bin/python -m app.main --config configs/raspberry_pi.yaml
Restart=on-failure
RestartSec=5s
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable smart-parking.service
sudo systemctl start smart-parking.service
sudo systemctl status smart-parking.service
```

---

## 7. Chromium Kiosk Mode Setup (HDMI Display)
To launch the dashboard automatically fullscreen on boot:
Add to `~/.config/wayfire.ini` (or Openbox autostart depending on desktop environment):
```ini
[autostart]
kiosk = chromium-browser --noerrdialogs --disable-infobars --kiosk http://127.0.0.1:8000
```
