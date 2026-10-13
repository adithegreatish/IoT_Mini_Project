# Security Architecture & Hardening

## 1. Network Boundary
- The application binds to `127.0.0.1` (localhost) by default to prevent unauthorized network access.
- Exposing the dashboard across a Local Area Network (LAN) requires explicitly altering `host: "0.0.0.0"` in configuration and implementing appropriate firewall restrictions.

## 2. API & Input Validation
- All REST endpoints strictly validate payloads against Pydantic schemas.
- Path traversal protections ensure file downloads (diagnostic reports, calibration exports) remain confined to authorized directory paths (`data/`, `data/test_reports/`).
- SQL injection prevention is guaranteed through SQLite parameterized queries in the repository layer.

## 3. Physical Hardware Security
- On deployment on the Raspberry Pi 5, the HDMI display runs in kiosk mode with context menus and keyboard shortcut inspection disabled to prevent tampering in unattended environments.
