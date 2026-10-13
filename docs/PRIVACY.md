# Privacy & Data Retention Policy

Vehicle registration numbers and timestamps represent sensitive operational data. This system implements privacy-by-design principles:

---

## 1. Core Privacy Controls
1. **Default Plate Masking:** Registration plate strings displayed in the user interface, tables, and inspection screens are masked by default (e.g., `DL 01 ** 1234`). Unmasking requires explicit UI toggle action.
2. **Log Redaction:** Application log files (`logs/app.log`) automatically redact license plate strings to prevent accidental leakage into system monitors or bug reports.
3. **Local Storage Exclusively:** All inferences, event databases, and cropped images remain strictly on local storage (Windows local drive or Raspberry Pi SD card). No telemetry or cloud analytics are transmitted.
4. **No Facial Recognition:** Human detection, facial tracking, and person profiling are deliberately omitted from the architecture.
5. **Configurable Retention:** Automated SQLite retention pruning purges state events and image snapshots older than the configured threshold (default 30 days on Windows, 14 days on Pi).
