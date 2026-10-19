# SECURITY AUDIT & VULNERABILITY MITIGATION REPORT

**Target System:** Smart Parking & Indian HSRP Management System  
**Audit Scope:** FastAPI Backend, SQLite Storage, Image Serving Endpoints, Camera Handlers, Model Inference, Configuration  
**Audit Date:** 2026-10-14  
**Audit Status:** **HARDENED & VERIFIED**  

---

## 1. Access Control & Network Attack Surface

### 1.1 Endpoint Surface & State Mutation Review
Every state-changing endpoint was audited to verify input validation, transaction boundaries, and error containment:

| Endpoint | Method | Scope | Protection Mechanism |
| :--- | :---: | :--- | :--- |
| `/api/slots` | POST | Calibrated bay geometry update | Pydantic schema validation (`SlotsUpdateRequest`), coordinate bounds checking `[0.0, 1.0]`, self-intersection check. |
| `/api/parking/allocate` | POST | Virtual bay allocation | Pydantic validation, `BEGIN IMMEDIATE` lock, duplicate plate verification, refusal of OCCUPIED/UNKNOWN slots. |
| `/api/parking/release` | POST | Reservation release | Parameterized SQL lookup, code format check. |
| `/api/parking/cancel` | POST | Reservation cancellation | Parameterized SQL lookup, reason recording in audit trail. |
| `/api/camera/pause_preview` | POST | Stream preview pause | Local loopback control, idempotency check. |
| `/api/camera/resume_preview` | POST | Stream preview resume | Local loopback control, idempotency check. |
| `/api/camera/set_scenario` | POST | Synthetic video simulation | Whitelisted scenario enum validation (`normal`, `all_free`, `all_occupied`, `mixed`, `camera_loss`, `straddling`). |

### 1.2 Cross-Origin Resource Sharing (CORS) Hardening
- **Previous State:** Unconfigured / unrestricted default.
- **Hardened State:** FastAPI `CORSMiddleware` configured strictly to authorized local development origins:
  - `http://localhost:8000`
  - `http://127.0.0.1:8000`
  - `http://localhost:3000`
  - `http://127.0.0.1:3000`
- **Enforcement:** Disallows wildcard `*` origins on credentialed transactions.

### 1.3 HTTP Security Headers
All HTTP responses now include defense-in-depth security headers via middleware in `app/main.py`:
- `X-Content-Type-Options: nosniff` — Prevents MIME-type sniffing.
- `X-Frame-Options: SAMEORIGIN` — Mitigates clickjacking.
- `X-XSS-Protection: 1; mode=block` — Enables legacy browser XSS filters.
- `Referrer-Policy: strict-origin-when-cross-origin` — Protects internal URI leaks.

---

## 2. Injection & Path Traversal Protections

### 2.1 SQL Injection Prevention
- **Audit Finding:** All database access is routed through `DatabaseManager` (`app/database/connection.py`) and `ParkingRepository` (`app/database/repository.py`).
- **Verification:** 100% of SQL statements across the codebase use parameterized query placeholders (`?`). String interpolation, concatenation, and formatting (`f"SELECT ... {var}"`) are strictly prohibited and zero instances exist in application code.
- **Rollback Safety:** All allocation transactions execute inside `try ... conn.commit() except ... conn.rollback()`.

### 2.2 Path Traversal in Image Endpoints (Defect 5 - Fixed)
- **Vulnerability:** Endpoints `/lab/gallery_image/{category}/{filename}` and `/lab/candidate_image/{filename}` previously concatenated raw string parameters into filesystem paths.
- **Remediation Implemented:**
  1. Base directory is resolved to absolute canonical path (`base_dir.resolve()`).
  2. Input filename is stripped of path components using `Path(filename).name`.
  3. Strict extension whitelist enforcement (`.jpg`, `.jpeg`, `.png`).
  4. Path containment verification using `img_p.is_relative_to(base_dir)`.
  5. Any traversal attempt (`../`, `..\`) triggers an immediate HTTP 400 or 404 response.
- **Verification:** Verified by automated test `test_security_headers_and_path_traversal_blocking` in `tests/integration/test_release_hardening.py`.

### 2.3 Model Loading Protection
- The system rejects loading arbitrary user-supplied model weights over HTTP. Model loading is strictly restricted to paths defined in local YAML configuration files and verified against `models/manifest.json` SHA-256 hashes.

---

## 3. Secrets and Privacy Hardening

### 3.1 Hard-Coded Credentials Audit
- **Grep Audit:** Searched all tracked repository files for API keys, passwords, private camera RTSP credentials, and access tokens. Zero secrets found.
- **Environment Template:** Created safe `.env.example` documenting all configuration keys without exposing real secret values.
- **Version Control Exclusions:** Confirmed `.gitignore` excludes `.env`, `*.key`, `*.pem`, `*.pfx`, `secrets.yaml`, and local SQLite database files (`*.sqlite`).

### 3.2 Vehicle & Registration Plate Privacy
- **Masking at Rest & Display:** `ParkingRepository` automatically masks license plate characters (e.g., `KA01AB1234` -> `KA01****34`) before saving to event tables unless `allow_raw_retention: true` is explicitly configured.
- **Debug Logs:** The logging subsystem applies `RedactionFilter` to redact license plates and sensitive tokens from console and log outputs.
- **Raw Crops:** Unnecessary raw plate crops are purged from memory after OCR inference.

### 3.3 HSRP Legal Advisory Notice
- Under Directive Section 7, all HSRP analysis visual cues are explicitly labelled as **advisory visual heuristics**.
- `HSRPAnalysisReport` now attaches the mandatory disclaimer:
  > `"ADVISORY ONLY: Visual cue analysis does not constitute legal or certified authenticity verification."`
- The system never claims certified or legal plate authentication.

---

## 4. Operational Security for Demonstration

1. **Localhost Binding:** `configs/windows.yaml` binds to `127.0.0.1:8000`. It is NOT exposed to public networks.
2. **Demonstration Network:** If demoing across a local WiFi router to an external tablet/browser:
   - Temporarily bind `host: "0.0.0.0"` in configuration.
   - Restrict port 8000 via Windows Firewall to the presenter\'s subnet.
   - Do NOT expose to the public internet or port-forward on routers.
