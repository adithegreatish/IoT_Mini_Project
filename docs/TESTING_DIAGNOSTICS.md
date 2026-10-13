# Testing & Automated Diagnostics

The system provides three complementary verification tiers to ensure stability on developer machines, continuous integration, and target hardware:

---

## 1. Quick Startup Health Check
A non-destructive sanity check run automatically during system initialization or on-demand:
```bash
python -m app.diagnostics --quick
```
Verifies:
- Python version and platform architecture.
- Central YAML configuration schema validity.
- Read/write permissions for databases, logs, and report directories.
- SQLite connection and schema migration readiness.
- Camera device enumeration (safe probe).
- Model files existence and manifest integrity.

---

## 2. Full In-App Self-Diagnostics & Automated Reports
A comprehensive end-to-end audit:
```bash
python -m app.diagnostics --full
```
Can also be triggered directly from the **System Diagnostics** tab in the web GUI.
Outputs:
- Live test execution progress.
- Clean cancellation support without dangling threads.
- Timestamped JSON report: `data/test_reports/diagnostics_<timestamp>.json`
- Human-readable Markdown summary: `data/test_reports/diagnostics_<timestamp>.md`

---

## 3. Developer Test Suite (pytest)
Runs isolated unit, integration, and failure-injection tests without requiring physical hardware:
```bash
pytest -v
```
Key tests covered:
- Geometry validation, polygon area, and point-in-polygon checks.
- Occupancy state machine hysteresis (verifying stale cameras never report false FREE states).
- Camera capture worker, mock generator, and auto-reconnect backoff.
- Database repository transactions, event pruning, and plate masking.
- API endpoints and streaming response contracts.
