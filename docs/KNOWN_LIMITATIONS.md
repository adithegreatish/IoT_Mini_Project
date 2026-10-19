# KNOWN LIMITATIONS & OPERATIONAL CONSTRAINTS

**Project:** Smart Parking & Indian HSRP Management System  
**Release Target:** Final College Submission  
**Status:** **HONEST DISCLOSURE DOCUMENT**  

---

## 1. Machine Learning & Detection Model Limitations

1. **Oblique Perspective & Extreme Aspect Angles:**
   - The primary vehicle detector (`vehicle-yolo-v8n` v1.0.0, ONNX) is optimized for elevated CCTV angles between 30° and 60°. At near-horizontal angles (<15°), severe vehicle overlap and occlusion reduce recall by ~12-18%.
2. **Adverse Environmental Conditions:**
   - Heavy rain, torrential fog, and severe lens glare from direct low-angle sunlight degrade bounding box precision.
   - Sudden illumination changes (e.g., cloud shadows, stadium floodlights toggling) may trigger brief temporal uncertainty before the 3-frame hysteresis stabilizes.
3. **Small & Distant Vehicles:**
   - Vehicles occupying less than 2% of the frame area (e.g., at the far boundary of large lots) may fall below the 0.40 confidence threshold. Manual calibration should size distant bays with appropriate margins.
4. **Motorcycle Cluster Occlusion:**
   - Motorcycles parked closely together (<0.5m spacing) may occasionally be grouped into a single bounding box by standard Non-Maximum Suppression (NMS).

---

## 2. Indian License Plate Detection & OCR Limitations

1. **Plate Resolution Gating:**
   - Plate detection crops smaller than 60x18 pixels are intentionally rejected by the quality gate (`INSUFFICIENT_IMAGE_QUALITY`). The OCR engine does not invent characters on unreadable crops.
2. **Non-Standard & Fancy Fonts:**
   - While the normalization engine disambiguates common character confusion (e.g., `8` vs `B`, `0` vs `O`, `1` vs `I`) and Indian state codes (e.g., `DL`, `KA`, `MH`, `TN`, `UP`, `BH`), artistic or regional script plates (Devanagari, Kannada, etc.) are unreadable by the standard Latin OCR pipeline.
3. **Advisory Nature of HSRP Visual Analysis:**
   - High Security Registration Plate (HSRP) features (blue IND strip, 1:4 aspect ratio, high-contrast reflective backdrop) are analyzed visually for aesthetic conformity only.
   - **Legal Caveat:** Visual analysis CANNOT read the 10-digit laser-branded identification number or scan the RFID hologram tag. It must never be used for certified law-enforcement authentication.

---

## 3. Hardware & Platform Constraints

1. **Raspberry Pi 4 Physical Deployment:**
   - All Raspberry Pi logic (CPU provider execution, bounded memory ring buffer, headless operation, thread limits) has been validated via simulated test suites and unit tests.
   - Physical deployment on a physical Raspberry Pi 4 board requires connecting an active UVC USB camera or CSI ribbon camera, configuring the V4L2 device index, and ensuring adequate heatsink cooling to prevent thermal throttling under continuous 15 FPS inference.
2. **Memory Footprint:**
   - Process RSS memory is bounded at ~490 MB. On a 2 GB or 4 GB Raspberry Pi 4, this leaves ample memory for the OS, but running concurrent heavy browser sessions directly on the Pi desktop is discouraged; client dashboards should be viewed from an external laptop/tablet on the LAN.

---

## 4. Intentional Fail-Safe Behaviors (Not Defects)

1. **Reporting UNKNOWN on Camera Loss:**
   - When the camera is disconnected, frozen, or producing black frames, all bays immediately transition to `UNKNOWN` and capacity displays as unavailable. The system **never reports empty bays during signal loss**.
2. **Refusal to Allocate Occupied or Unknown Bays:**
   - The allocation engine strictly rejects reservations for bays with `OCCUPIED` or `UNKNOWN` state. If a camera feed is degraded, users cannot reserve those bays until visibility is restored.
3. **Duplicate Plate Allocation Prevention:**
   - A vehicle cannot hold multiple active reservations simultaneously. Submitting a second reservation for the same plate returns HTTP 409 Conflict.
