#!/usr/bin/env bash
# ==============================================================================
# Smart Parking: HDMI Fullscreen Dashboard Kiosk Launcher
# Target: Raspberry Pi OS with desktop / HDMI monitor attached
# ==============================================================================
set -euo pipefail

echo "=================================================================="
echo " Launching Smart Parking HDMI Dashboard Kiosk"
echo " Target URL: http://localhost:8000/"
echo "=================================================================="

# 1. Disable screen blanking & DPMS power saving (X11 / Wayland)
if command -v xset &> /dev/null; then
    xset s off || true
    xset -dpms || true
    xset s noblank || true
fi

# 2. Wait for FastAPI backend to be ready
echo "Waiting for Smart Parking backend on http://localhost:8000/..."
MAX_WAIT=30
WAITED=0
while ! curl -s http://localhost:8000/api/health > /dev/null; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[WARN] Backend health endpoint not responding after ${MAX_WAIT}s. Launching browser anyway..."
        break
    fi
done
echo "Backend is live! Launching fullscreen browser..."

# 3. Launch Chromium in Kiosk mode
BROWSER_BIN=""
for b in chromium-browser chromium google-chrome; do
    if command -v "${b}" &> /dev/null; then
        BROWSER_BIN="$(command -v "${b}")"
        break
    fi
done

if [ -z "${BROWSER_BIN}" ]; then
    echo "[ERROR] Chromium browser not installed."
    echo "Install with: sudo apt install -y chromium-browser"
    exit 1
fi

exec "${BROWSER_BIN}" \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-pinch \
    --overscroll-history-navigation=0 \
    --check-for-update-interval=31536000 \
    http://localhost:8000/
