# Excel-Style Parking Calibration Bug Fix & Diagnosis Report

**Author:** Antigravity Engineering  
**Date:** 2026-10-13  
**Target Repository:** `adithegreatish/IoT_Mini_Project`  
**Status:** Resolved & Verified by Automated Regression Suite  

---

## 1. Executive Summary

The parking space calibration module under the **Parking Calibration** tab features an Excel-style dynamic bay spreadsheet, auto-grid alignment tool, visual Blueprint CAD preview canvas, and CSV import/export capabilities. 

Users reported that the spreadsheet creator was failing to function correctly:
- In-progress edits were periodically wiped.
- Adding bays sometimes caused backend save rejections.
- Calibrations with disabled slots disappeared from the overview.
- Allocations were out of sync with saved calibrations.
- Auto-grid or manual coordinates occasionally produced HTTP 400 validation errors.

This investigation diagnosed, reproduced, and resolved the underlying defects across both frontend (`app.js`, `index.html`) and backend (`routes.py`, `occupancy_service.py`, `geometry.py`). All 93 system tests pass.

---

## 2. Root Cause Analysis

### Bug 1: 3-Second Background Polling Overwrote In-Progress Edits
- **Location:** `web/static/js/app.js` (lines 106–111, 178–195)
- **Root Cause:** A `setInterval` timer invoked `fetchSlots()` every 3.0 seconds. When the active tab was `calibration`, `fetchSlots()` unconditionality overwrote `state.slots` with the backend snapshot and invoked `renderExcelSpreadsheet()`. Any user typing into a cell, editing a slider, or adding an unsaved bay had their inputs wiped or reverted every 3 seconds.
- **Fix:** Introduced `state.calibrationDirty` state tracking and a visual dirty indicator badge (`#badge-calibration-dirty`). `fetchSlots()` now explicitly guards against overwriting `state.slots` when `state.activeTab === 'calibration' && state.calibrationDirty`. Tab switches and saves cleanly reset the dirty state.

### Bug 2: Duplicate Slot ID Collisions on Bay Creation
- **Location:** `web/static/js/app.js` (`addExcelRow`, `cloneSlot`)
- **Root Cause:** Bay ID generation computed `nextNum = state.slots.length + 1` and assigned `S0${nextNum}`. If 3 bays existed (`S01`, `S02`, `S03`) and the user deleted `S02`, `state.slots.length` became 2. Clicking "Add Row" generated `S03`, which already existed. Upon saving, `CalibrationService.save_calibration()` rejected the request with `Calibration validation failed: Duplicate Slot ID 'S03' declared.`.
- **Fix:** Implemented an ID scanner that parses all numeric suffixes from existing bays and checks against a `Set(existingIds)`, guaranteeing strictly unique IDs (`S01`, `S02`, etc.) even after deletions or clonings.

### Bug 3: Coordinate Boundary Overflow ($X + W > 100\%$ or $Y + H > 100\%$)
- **Location:** `web/static/js/app.js` (input change handler) & `app/parking/geometry.py` (`validate_slot_polygon`)
- **Root Cause:** Editing cell values allowed configurations like $X = 85\%$ and $W = 20\%$, producing a normalized vertex at $X = 1.05$. `validate_slot_polygon` strictly enforced `0.0 <= p[0] <= 1.0` and rejected the polygon with HTTP 400. Furthermore, floating-point IEEE 754 precision issues (e.g. $0.3 + 0.7 = 1.0000000000000002$) also triggered rejection.
- **Fix:**
  - Frontend: Clamped inputs so that $X \in [0, 99.0]$, $Y \in [0, 99.0]$, $W \in [0.5, 100 - X]$, and $H \in [0.5, 100 - Y]$, guaranteeing that all normalized coordinates fit strictly within $[0.0, 1.0]$.
  - Backend: Enhanced `validate_slot_polygon()` with floating-point epsilon tolerance (`1e-4`) and clamped vertices slightly outside $[0, 1]$ before Shapely geometry construction. Lowered `min_area` default from `0.005` to `0.0005` to safely accommodate compact motorcycle and EV bays.

### Bug 4: Disabled Bay Eviction in Occupancy Engine
- **Location:** `app/parking/occupancy_service.py` (`process_frame`, `reload_slots`)
- **Root Cause:** In `process_frame()`, the loop previously only built slot views for active bays and completely omitted bays with `enabled=False`. Consequently, the `/api/slots` snapshot returned an empty list for disabled bays, causing them to disappear from the UI on subsequent polls.
- **Fix:** Both camera offline/stale paths and active inference paths now explicitly preserve all calibrated bays, tagging disabled slots as `state: "DISABLED"` with `confidence: 1.0` and `occupied: False`.

### Bug 5: Desynchronization of Virtual Parking Allocation Engine
- **Location:** `app/api/routes.py` (`save_parking_slots`)
- **Root Cause:** Saving bays via `POST /api/slots` reloaded `occupancy_service.reload_slots()`, but never synchronized `allocation_service.sync_slots_from_calibration()`. As a result, the SQLite `parking_slots` table held stale bay IDs, and driver allocation requests operated on an outdated registry until the server was restarted.
- **Fix:** Added direct call to `allocation_service.sync_slots_from_calibration(req.slots)` within `save_parking_slots()`. Calibration changes now instantly propagate to virtual reservation controls and database locking tables without server restart.

### Bug 6: RFC-4180 CSV Tokenizer Missing
- **Location:** `web/static/js/app.js` (`importCSV`, `exportCSV`)
- **Root Cause:** CSV import split lines via naive `.split(',')`. Any bay label containing commas (e.g. `"Bay 1, Level 2"`) was split across multiple column offsets, corrupting coordinates and types.
- **Fix:** Implemented an RFC-4180 quote-aware tokenizer (`parseCSVRow`) that respects double-quote enclosures and escaped quotes. Added robust column and coordinate validation reporting row-level notices.

---

## 3. Changed Files Summary

| File Path | Nature of Change | Description |
| :--- | :--- | :--- |
| `web/static/js/app.js` | Bug Fix | Added dirty state tracking, polling clobber guard, non-colliding ID generation, coordinate clamping, RFC-4180 CSV parser, preflight validation. |
| `web/templates/index.html` | UI Enhancement | Added `#badge-calibration-dirty` indicator to spreadsheet header. |
| `app/api/routes.py` | Bug Fix / Sync | Merged canonical calibration with live occupancy snapshot in `get_parking_slots()`; synced `allocation_service` in `save_parking_slots()`. |
| `app/parking/geometry.py` | Geometry Robustness | Added epsilon tolerance and lowered `min_area` default to `0.0005` in `validate_slot_polygon()`. |
| `app/parking/occupancy_service.py` | State Integrity | Retained disabled bays as `state: "DISABLED"` in snapshots; seeded initial snapshot in `reload_slots()`. |
| `tests/unit/test_calibration_spreadsheet.py` | Test Coverage | 6 comprehensive regression tests for geometry, unique IDs, revisions, overlap warnings, and coordinates. |

---

## 4. Verification Protocol for All Spreadsheet Operations

### Operation 1: Add a Parking Bay
- **Action:** Click **➕ Add Row**.
- **Verification:** Creates a new row with unique ID (`S01`, `S02`, etc.), sensible defaults (Zone-A, Car, $15\% \times 25\%$), highlights active row, focuses ID input, updates canvas preview, and displays `⚠️ Unsaved Edits` badge.

### Operation 2: Edit a Parking Bay
- **Action:** Modify ID, Label, Zone, Type, X, Y, Width, Height, or Active checkbox.
- **Verification:** Changes instantly update the internal slot model, re-draw the blueprint CAD polygon on `#calibration-canvas`, and mark the calibration as dirty. Boundary clamping prevents $X+W > 100\%$ or $Y+H > 100\%$.

### Operation 3: Preview Changes
- **Action:** Select or edit any row.
- **Verification:** Active row renders with cyan neon glow, vertex handles, and enlarged label badge on the canvas.

### Operation 4: Delete a Parking Bay
- **Action:** Click **🗑** on a bay row.
- **Verification:** Row is removed from the table and canvas preview immediately. Active selection shifts to adjacent bay.

### Operation 5: Automatic Smart Grid Generation
- **Action:** Set columns (e.g. 4), rows (e.g. 1), prefix (`T`), zone (`Zone-A`), and click **⚡ Auto-Generate Evenly Spaced Bays**.
- **Verification:** Instantly generates non-overlapping, evenly spaced bays centered within camera view boundaries, with IDs `T01, T02, T03, T04`. Marks calibration dirty.

### Operation 6: CSV Export & Import
- **Action:**
  - Export: Click **📤 Export CSV**. Downloads well-formed CSV with headers `id,label,zone,slot_type,x_pct,y_pct,width_pct,height_pct,enabled`.
  - Import: Click **📥 Import CSV** and upload a CSV file.
- **Verification:** Correctly tokenizes quoted labels with commas, validates coordinate bounds, renames duplicate IDs, and displays clear toast summary.

### Operation 7: Save and Reload
- **Action:** Click **💾 Save & Apply Live**.
- **Verification:** Performs client-side validation, sends `POST /api/slots`, persists to `data/calibration.json`, records revision in SQLite `calibration_revisions`, updates `occupancy_service` and `allocation_service`, clears the dirty badge, and refreshes without full page reload.

---

## 5. Automated Regression Test Results

```
tests/unit/test_calibration_spreadsheet.py::test_validate_slot_polygon_boundaries PASSED
tests/unit/test_calibration_spreadsheet.py::test_validate_slot_polygon_minimum_area PASSED
tests/unit/test_calibration_spreadsheet.py::test_calibration_service_unique_id_enforcement PASSED
tests/unit/test_calibration_spreadsheet.py::test_calibration_service_persistence_and_revision_history PASSED
tests/unit/test_calibration_spreadsheet.py::test_check_slot_overlaps_warning PASSED
tests/unit/test_calibration_spreadsheet.py::test_normalize_and_denormalize_roundtrip PASSED
tests/integration/test_api.py::test_api_endpoints PASSED
```

**Result:** 100% Passing (All 93 tests passing in project test suite).
