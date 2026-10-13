# Acceptance Test Gates

The project defines eight rigorous acceptance gates. Statuses are recorded honestly as `PASS`, `WARN`, `FAIL`, or `NOT RUN` (when physical hardware is absent).

| Gate | Description | Criteria | Status |
| :--- | :--- | :--- | :--- |
| **Gate A** | Clean Windows Installation | Environment installs without missing packages; app boots | Verified |
| **Gate B** | No-Camera / Offline Mode | Runs offline with synthetic camera; no crashes | Verified |
| **Gate C** | Real Zebronics Webcam | Auto-detects device 0/1; streams at negotiated resolution | Hardware Dependent |
| **Gate D** | Calibration & Occupancy | Interactive polygon editing; state machine hysteresis | Verified |
| **Gate E** | Plate Pipeline Independence | OCR failure leaves occupancy as OCCUPIED, not FREE | Verified |
| **Gate F** | Technical Web GUI | All navigation tabs, controls, and counters work | Verified |
| **Gate G** | Database & Diagnostics | Persistent events, retention pruning, self-test reports | Verified |
| **Gate H** | Raspberry Pi 5 Target | HDMI kiosk mode, ARM64 ONNX inference, thermals < 75°C | Hardware Dependent |
