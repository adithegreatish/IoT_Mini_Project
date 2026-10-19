# FINAL RELEASE & VERIFICATION AUDIT REPORT

**System:** Smart Parking + Indian HSRP Management System  
**Submission Target:** Final College Project Examination  
**Release Date:** 2026-10-14  
**Authoritative Verdict:** **READY FOR DEMONSTRATION**  

---

## 1. Executive Summary & Release Scope

In accordance with the Final Release Directive, this repository has completed the transition from experimental research to verified, hardened production release. All machine-learning experimentation has been frozen. Zero new datasets were downloaded, no unverified model promotions were performed, and no destructive rewrites were executed.

The existing codebase was thoroughly audited, critical defects in slot state reporting, allocation integrity, and security endpoints were diagnosed and repaired, and a release-blocking automated test suite of 114 tests was executed with a 100% pass rate.

---

## 2. Active Model Verification & Cryptographic Provenance

In strict compliance with Directive Section 1.1, the production detector baseline has remained frozen and active. Its artifact files and cryptographic SHA-256 signatures have been verified directly on the filesystem:

| Component | Model Identifier | Architecture | Filesystem Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Active Production Detector** | `vehicle-yolo-v8n` (v1.0.0) | YOLOv8-Nano (ONNX) | `models/vehicle_detector.onnx` | `8EA5414F0CF2BC4F81861FC19DB26ECE955CADC736DEC8FA56EBE01A049244BC` | **ACTIVE** |
| **Production PyTorch Checkpoint** | `vehicle-yolo-v8n` (v1.0.0) | YOLOv8-Nano (PyTorch) | `models/vehicle_detector.pt` | `0F8C08DC8F4AFF7F0B52582F38677C016B8CAEDF73AB4B39C182046A086E6089` | **VERIFIED** |
| **Plate Localization Model** | `plate-detector-tiny` (v1.0.0) | Tiny ONNX (640x640) | `models/plate_detector.onnx` | Verified in manifest | **ACTIVE** |
| **Occupancy Patch Classifier** | `patch-classifier-v1` (v1.0.0) | MobileNet/CNN (150x150) | `models/occupancy_classifier.onnx` | Verified in manifest | **ACTIVE** |
| **Research Candidate B** | `exp_india_cctv_10ep` | YOLOv8-Nano | `models/training_runs/exp_india_cctv_10ep/weights/best.onnx` | Cataloged in registry | ARCHIVED |
| **Research Candidate C** | `exp_cctv_balanced_10ep` | YOLOv8-Nano | `models/training_runs/exp_cctv_balanced_10ep/weights/best.onnx` | Cataloged in registry | ARCHIVED |

**Decision Justification:**  
While research Candidates B and C showed specialized domain adaptation on Indian CCTV frames, Candidate B suffered from historical validation leakage and Candidate C required further edge-case testing under varying lighting before meeting production promotion gates. In strict adherence to Directive Section 1.1, the baseline production model (`vehicle-yolo-v8n`) is retained as the authoritative deployment model. Candidates B and C remain cataloged in `models/model_registry.json` and `models/training_runs/` for academic traceability.

---

## 3. Discovered Defects & Implemented Root-Cause Fixes

During the comprehensive audit, 8 defects were identified and systematically resolved:

### Defect 1: Bug in `GET /api/slots` State Mapping (CRITICAL)
- **Root Cause:** In `app/api/routes.py`, `snap_map` was keyed by `s["slot_id"]`. However, `OccupancyService` output dictionaries used key `"id"`. Because of this mismatch, `snap_map` was always empty. The fallback logic defaulted missing live observations to `state = "FREE"`, causing occupied and unknown bays to falsely render as FREE in the dashboard.
- **Correction:** 
  1. Updated key extraction to `(s.get("id") or s.get("slot_id"))`.
  2. Integrated authoritative reservation state lookup from `AllocationService`.
  3. Changed fallback logic for unobserved bays to `UNKNOWN`, never `FREE`.
  4. Updated summary recalculation to derive true live counts for Free, Occupied, Reserved, and Unknown.

### Defect 2: Duplicate Active Allocations & Missing DB Constraints (CRITICAL)
- **Root Cause:** In `app/parking/allocation_service.py`, `allocate_slot()` failed to verify whether a vehicle already held an active reservation. Repeated requests allowed a single vehicle to occupy multiple parking bays simultaneously.
- **Correction:**
  1. Added normalization (stripping spaces, hyphens, and converting to uppercase).
  2. Added pre-check querying `slot_reservations` for active allocations matching the plate.
  3. Added partial unique indexes in SQLite schema (`uq_active_slot_reservation` and `uq_active_plate_reservation` WHERE `status = 'ACTIVE'`) to enforce database-level ACID serialization against concurrent race conditions.
  4. Added `DuplicateAllocationError` exception mapping to HTTP 409 Conflict.

### Defect 3: Safe Model Failure Handling & Fail-Closed State Transition (CRITICAL)
- **Root Cause:** If `VehicleDetector` experienced model loading errors or inference exceptions, `detect()` returned an empty list. `OccupancyService` treated this as "zero vehicles detected", falsely transitioning all bays to `FREE`.
- **Correction:**
  1. Added `is_healthy` property and exception tracking to `VehicleDetector`.
  2. Updated `OccupancyService.process_frame()` to catch detector exceptions and inspect `is_healthy`.
  3. When an inference fault occurs, all active bays transition to `SlotState.UNKNOWN` with reason `Detector failure: <msg>`, and capacity is reported as `free: 0, occupied: 0, unknown: total`. Bays are NEVER marked free on detector failures.

### Defect 4: Perspective Roof Distortion & Adjacent Bay Stealing (CORRECTNESS)
- **Root Cause:** `OccupancyService` used raw 2D bounding-box polygon intersection (`calculate_overlap_ratio`). For elevated CCTV cameras, tall vehicles (SUVs, trucks) project roof bounding boxes into adjacent parking bays while their tires remain in their own bay, causing adjacent empty bays to be classified as occupied.
- **Correction:**
  1. Replaced raw overlap with `calculate_spatial_association_score(pts, det.bbox, method="hybrid")`.
  2. The hybrid spatial association checks bottom-center ground contact containment. If ground contact is inside the bay, association is confirmed. If ground contact is outside the bay, overlap is dampened by 0.65x factor, keeping adjacent slots below the occupancy threshold.

### Defect 5: Path Traversal Vulnerability in Image Serving Endpoints (SECURITY)
- **Root Cause:** Endpoints `/lab/gallery_image/{category}/{filename}` and `/lab/candidate_image/{filename}` concatenated user-controlled parameters into filesystem paths.
- **Correction:**
  1. Applied `Path(filename).name` sanitization.
  2. Restricted extensions strictly to `.jpg`, `.jpeg`, `.png`.
  3. Enforced boundary check using `img_p.is_relative_to(base_dir)`. Rejects traversal attempts with HTTP 400/404.

### Defect 6: Missing CORS and Security Headers (SECURITY)
- **Root Cause:** Application lacked CORS origins restrictions and standard HTTP security headers.
- **Correction:**
  1. Mounted FastAPI `CORSMiddleware` restricted to local development origins (`http://localhost:8000`, `http://127.0.0.1:8000`, `http://localhost:3000`).
  2. Added HTTP middleware attaching `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `X-XSS-Protection: 1; mode=block`, and `Referrer-Policy: strict-origin-when-cross-origin`.

### Defect 7: Safe Environment Template (DEPLOYMENT)
- **Root Cause:** No `.env.example` existed to guide reproducible deployment without checking in secrets.
- **Correction:** Created `.env.example` documenting all configuration keys with safe placeholder values.

### Defect 8: HSRP Advisory Disclosure (LEGAL / HONESTY)
- **Root Cause:** Visual feature analysis did not explicitly declare its advisory limitations on every report.
- **Correction:** Added mandatory disclaimer field to `HSRPAnalysisReport`:
  `"ADVISORY ONLY: Visual cue analysis does not constitute legal or certified authenticity verification."`

---

## 4. Test Suite Execution & Verification Summary

The test suite was executed across all unit, integration, and newly implemented acceptance test files:

```bash
.venv\Scripts\python -m pytest tests/ -v
```

### Result:
```
======================= 114 passed, 1 warning in 13.24s =======================
```
- **Total Tests Executed:** 114
- **Passed:** 114 (100%)
- **Failed:** 0
- **Regression Defects:** None.

---

## 5. Hardware Demarcation

- **Desktop Validation (Fully Verified):** Executed on Windows 11 host with DirectShow webcam capture, synthetic UVC mock fallback, and ONNX Runtime CPU execution provider.
- **Raspberry Pi Validation (Simulated & Portable):** Validated via automated unit tests verifying headless execution, bounded ring buffer memory (<128 MB), thread quotas (intra=2, inter=1), and relative path relocation. Physical Raspberry Pi 4 board execution requires physical camera connection as documented in `docs/KNOWN_LIMITATIONS.md`.

---

## 6. Release Deliverables Inventory

1. [docs/FINAL_RELEASE_REPORT.md](file:///c:/Users/adity/Downloads/IoT/docs/FINAL_RELEASE_REPORT.md) — Authoritative release audit and defect resolution history.
2. [docs/DEPLOYMENT_GUIDE.md](file:///c:/Users/adity/Downloads/IoT/docs/DEPLOYMENT_GUIDE.md) — Reproducible setup, startup, shutdown, backup, and demonstration guide.
3. [docs/SECURITY_AUDIT.md](file:///c:/Users/adity/Downloads/IoT/docs/SECURITY_AUDIT.md) — Security boundary audit, injection defenses, and CORS report.
4. [docs/ACCEPTANCE_TEST_RESULTS.md](file:///c:/Users/adity/Downloads/IoT/docs/ACCEPTANCE_TEST_RESULTS.md) — Verified acceptance test matrix with exact pass/fail records.
5. [docs/KNOWN_LIMITATIONS.md](file:///c:/Users/adity/Downloads/IoT/docs/KNOWN_LIMITATIONS.md) — Truthful disclosure of AI, optical, and hardware boundaries.
6. [.env.example](file:///c:/Users/adity/Downloads/IoT/.env.example) — Safe configuration template.
7. [tests/integration/test_release_hardening.py](file:///c:/Users/adity/Downloads/IoT/tests/integration/test_release_hardening.py) — 11 automated release-blocking acceptance tests.
8. [README.md](file:///c:/Users/adity/Downloads/IoT/README.md) — Updated master guide.

---

## 7. Official Go/No-Go Release Decision

### Final Verdict: **READY FOR DEMONSTRATION**

**Rationale:**
1. **Core Workflows Verified:** Vehicle detection, bay geometry association, ground-contact perspective dampening, ACID reservation allocation, release, and live dashboard reflection have all passed automated verification without failure.
2. **Defects Eliminated:** The critical slot state mapping bug in `GET /api/slots`, duplicate vehicle allocation loopholes, detector crash fail-free risks, and path traversal vulnerabilities have been completely resolved and covered by regression tests.
3. **Traceability Preserved:** Production baseline ONNX and PyTorch hashes are verified and unmodified. Research candidates B and C are cataloged and preserved.
4. **Transparent Disclosures:** Known limitations regarding physical Raspberry Pi connectivity, extreme camera perspective angles, and the advisory nature of HSRP visual cues are fully documented without fabrication.

The application is stable, secure, and ready for evaluation and demonstration.
