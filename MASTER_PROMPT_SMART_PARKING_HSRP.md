# MASTER BUILD SPECIFICATION
## Smart Parking Management & Indian HSRP-Style Registration-Plate Recognition
### Windows Development · Raspberry Pi 5 (8 GB) Deployment · Professional GUI · Automated Self-Testing

> **Purpose:** This is the single source of truth for building the complete application. Follow it as an implementation contract, not as a request for a plan or mock-up. Build the real repository, test it honestly, document it, and prepare it for deployment.

---

# 0. Mission, Constraints, and Rules of Engagement

Act as a cross-functional senior engineering team consisting of a computer-vision engineer, machine-learning engineer, Python architect, backend engineer, frontend/UI engineer, QA engineer, dataset engineer, privacy/security reviewer, and Raspberry Pi deployment specialist.

Design, implement, validate, and document a complete, maintainable academic prototype called **Smart Parking Management System with HSRP-Style Plate Detection and Vehicle Identification**.

The system must be developed primarily on Windows and run locally on a Raspberry Pi 5 with 8 GB RAM. It uses a **normal Zebronics USB webcam** and the **HDMI monitor/main display** connected to the Raspberry Pi. **Do not use or assume a Raspberry Pi NoIR camera, CSI camera, infrared-only processing, or `Picamera2`.** A MAX7219 or other external display is not required for the current version.

The application must deliver reliable parking occupancy even when a plate is unreadable. HSRP-style analysis and OCR are additional capabilities; they must never be single points of failure for occupancy detection.

## Non-negotiable implementation rules

1. **Build the application, not just a proposal.** Do not stop after architecture, pseudocode, a notebook, a static dashboard, or a list of recommendations.
2. Inspect the existing repository first if one is supplied. Preserve useful working code and explain consequential changes.
3. Work in verifiable milestones. At every milestone, report files changed, commands run, actual test results, remaining failures, and next actions.
4. Never fabricate successful builds, passing tests, downloaded datasets, trained models, accuracy figures, FPS, or hardware compatibility.
5. Never replace real functionality with placeholder functions, fake detections, hardcoded dashboard counters, static test results, or simulated data presented as live data.
6. Clearly separate **PASS**, **WARN**, **FAIL**, **SKIPPED**, and **NOT RUN**. Simulation is not hardware validation.
7. Keep the app functional without plate OCR, without a GPU, without internet after installation, and without a connected camera (diagnostic/simulation mode).
8. When a required dependency or external dataset cannot be accessed, explain the exact blocker and provide a reproducible fallback. Do not silently invent files or labels.
9. Prefer simple, reliable components over unnecessary complexity. Do not add cloud services, remote analytics, facial recognition, identity profiling, or automatic gate-control actions.
10. Before claiming completion, run the defined tests and acceptance gates. Mark hardware-only checks `NOT RUN` if the required hardware was not physically available.

## Primary objectives

- Capture and display a live stream from a standard Zebronics USB webcam.
- Let an administrator calibrate multiple parking slots using polygon regions.
- Classify every configured active slot as `FREE`, `OCCUPIED`, or `UNKNOWN`.
- Detect relevant vehicle classes, including cars and motorcycles; support other classes only where model and dataset coverage justify them.
- Detect number-plate regions and optionally analyse visible HSRP-style cues.
- Run OCR where image quality permits, return normalized plate text with uncertainty, and associate a plate with a vehicle/slot only when evidence supports the association.
- Provide a useful, responsive, professional local GUI on Windows and the Raspberry Pi HDMI monitor.
- Save meaningful occupancy/diagnostic events locally in SQLite.
- Include a complete test suite, in-app diagnostics, report generation, reproducible dataset preparation/training/evaluation scripts, and export/deployment instructions.
- Operate locally and offline after all software and model artifacts have been installed.

## Scope boundary: HSRP claims

This is a computer-vision prototype, not an official HSRP verification service. A visible blue `IND` region, shape, colour, or OCR result cannot cryptographically prove that a plate is genuine or legally compliant. Do not label a plate fraudulent based solely on visual heuristics. Use cautious outputs such as `HSRP_STYLE_CONSISTENT`, `NON_HSRP_OR_UNCERTAIN`, and `INSUFFICIENT_IMAGE_QUALITY`.

---

# 1. Hardware, Platforms, and Runtime Boundaries

## 1.1 Development machine

- Primary development OS: Windows 10/11.
- Use a Python virtual environment and a reproducible dependency lock/requirements set.
- Use an NVIDIA GPU/CUDA for training only if present and compatible; CPU-based smoke tests must still work.
- Use Windows for coding, data inspection/annotation, model training and evaluation, test execution, export, and preparing deployment artifacts.
- Do not copy a Windows virtual environment to Linux or assume packages have identical ARM64 availability.

## 1.2 Target deployment

- Device: Raspberry Pi 5, 8 GB RAM.
- OS: supported 64-bit Raspberry Pi OS release, explicitly recorded during final testing.
- Camera: normal Zebronics USB webcam through USB/UVC/OpenCV.
- Display: attached HDMI monitor; browser kiosk mode is optional.
- Storage: initially a 32 GB microSD card; monitor disk use and document when more storage is advisable.
- Power: compatible Raspberry Pi 5 power supply.
- Network: optional for normal use. Core detection, event logging, and UI must keep working without internet.

## 1.3 Camera implementation rules

Use a dedicated camera service based on OpenCV `cv2.VideoCapture()` or another platform-compatible UVC interface if a measured compatibility problem justifies it. `Picamera2` must not be required.

Provide configurable:
- Device index or selectable device name where supported.
- Capture backend preference where applicable.
- Requested resolution and frame rate.
- Actual negotiated resolution and observed frame rate.
- Buffer size and stale-frame policy.
- Exposure, focus, white balance, and related controls only when the driver exposes them.
- Reconnect interval and timeout.
- Preview/test snapshot directory.

Probe likely device indexes (for example 0, 1, and 2) without assuming one fixed index works on both Windows and Raspberry Pi OS. Avoid opening the same camera in multiple services at the same time. One camera service owns the device and distributes frames to other components.

Include a camera diagnostic utility that lists candidates, opens a preview, reports frame size/FPS/errors, captures a test frame, and helps assess whether plates are large enough in the image. If a requested resolution cannot be set, fall back safely and show the actual setting. Never assume the webcam's advertised resolution is the resolution actually negotiated by the driver.

The camera being physically connected does not guarantee plate readability. Add an image-quality diagnostic for plate crop size, blur, glare, exposure, perspective, and occlusion. Report when placement/distance makes the task unrealistic instead of pretending software can recover detail that was never captured.

---

# 2. Architecture and Technology Selection

Use a modular Python application with a **local browser-based GUI**. FastAPI serves the backend and frontend assets; the same UI must work on Windows and Raspberry Pi. The Raspberry Pi display can open the local dashboard fullscreen, with optional Chromium kiosk setup.

Suggested technology stack (validate and pin compatible versions instead of choosing arbitrary latest versions):

- Python for services, CV, training orchestration, and application logic.
- OpenCV and NumPy for capture and image processing.
- A compact YOLO-family detector or another justified lightweight detector for vehicles/plates.
- PyTorch in the Windows training environment where appropriate.
- Benchmark ONNX Runtime and/or NCNN for ARM64 inference; select using actual Pi measurements, compatibility, output parity, and memory use.
- A configurable OCR engine such as PaddleOCR, Tesseract, or another evaluated option. Avoid forcing an unnecessarily large runtime on the Pi.
- FastAPI for REST endpoints and WebSocket or similarly suitable live-state updates.
- HTML/CSS/JavaScript for the UI. Avoid CDN dependencies at runtime; bundle required frontend assets for offline operation.
- SQLite for local events and settings metadata.
- Pydantic or equivalent schema validation and YAML/TOML/JSON for configuration.
- Pytest for unit/integration tests; add browser/UI smoke tests if an appropriate local browser automation environment is available.
- Structured logging with rotation.

Record tested Python/library versions for Windows and Pi separately. Keep training dependencies separate from deployment dependencies. Never require CUDA on the Pi.

## 2.1 Layer boundaries

Separate these concerns:

1. **Presentation:** pages, UI components, drawing tools, tables, status indicators, forms.
2. **Application services:** orchestration, state transitions, calibration workflows, and event creation.
3. **Computer vision:** capture, preprocessing, vehicle detection, plate detection, HSRP-style cues, OCR, tracking, slot association.
4. **Persistence:** SQLite repositories, migrations/schema version, event retention, settings.
5. **Infrastructure:** configuration, model loading, health monitoring, platform detection, logging, time and file handling.
6. **Testing/diagnostics:** self-tests, mocks, fixtures, reports, acceptance checks.
7. **Training/data:** data inspection, download/access instructions, label conversion, split generation, training, evaluation, model export.
8. **Deployment:** Windows setup, Pi setup, kiosk mode, service startup, logs, backup/recovery.

The GUI must not execute SQL throughout components or implement detector algorithms. No giant monolithic file. Use typed interfaces and dependency injection (or equally clear seams) so services can be replaced by mocks in tests.

## 2.2 Data flow

`Zebronics USB webcam → camera owner/capture worker → frame validation/quality checks → vehicle/slot detection → plate localization (optional) → HSRP-style cue analysis (optional) → OCR (optional) → temporal smoothing/tracking → slot state service → event repository → dashboard/API`

Plate OCR must be optional and must not block the main occupancy loop. Keep frame flow bounded; discard stale frames where needed rather than allowing latency to grow without limit.

---

# 3. Configuration, State, and Application Lifecycle

## 3.1 Central configuration

Create validated, versioned configuration with safe defaults. Support a documented precedence order, for example: built-in defaults → config file → environment variables → explicitly persisted UI settings. Make the final effective configuration inspectable in diagnostics, excluding secrets.

Configuration must cover:
- Camera selection, resolution, FPS, timeout, retries, and supported controls.
- Parking slot definitions and calibration dimensions.
- Detector model paths, class maps, input sizes, thresholds, runtime choice.
- Occupancy matching thresholds, smoothing, hysteresis, and timing rules.
- Plate detection, crop quality, OCR normalization, and confidence settings.
- Database path, retention policies, and export options.
- UI preferences and masking defaults.
- Log level and rotation.
- Diagnostic/test settings.
- Host/port, local-only default, optional LAN access policy.

Validate values at startup and when edited. Reject malformed or unsafe config updates with a useful error. Keep configuration migration support when the schema changes, or provide an explicit migration warning and backup path.

## 3.2 Single source of truth

Use one authoritative service each for the current camera status, model status, occupancy state, slot calibration, application health, config, and persistence. UI counters and page summaries must consume these services, not maintain independent conflicting state.

## 3.3 Lifecycle

Implement clear startup, ready/degraded, shutdown, and failure states. Cleanly release the camera, stop workers, close database connections, stop streaming clients, and flush logs at shutdown. Handle OS signals and Windows console shutdown where applicable. Do not leave orphan worker threads or a locked camera after an exception.

---

# 4. Parking-Slot Calibration and Occupancy

## 4.1 Slot calibration UI

Implement an interactive calibration page that allows the user to:
- Select a camera and capture a reference frame.
- Draw a polygon by clicking points.
- Edit vertices, move/rename/delete slots, and undo changes.
- Assign an ID such as `S01`, a slot type, a display label, and enabled/disabled status.
- Validate minimum polygon area, self-intersection, duplicate IDs, out-of-frame points, and potential overlaps.
- Save a calibration revision, preview overlays, export/import the configuration, and revert to a prior version.

Store coordinates normalized to the reference resolution and preserve the reference frame size, camera/config fingerprint, and calibration version. Scale consistently when actual capture resolution changes. Warn if a materially different camera view or resolution invalidates calibration; do not silently reuse bad coordinates.

## 4.2 Occupancy baseline

Start with a pretrained lightweight vehicle detector mapped to manually configured slot polygons. Use the overlap of relevant vehicle detections with slot regions, detection centre, polygon intersection, and tracking evidence as appropriate. Evaluate a slot-crop classifier as an alternative or complementary signal if the real camera geometry needs it. Do not assume a single method is universally best.

Avoid relying only on pixel difference from an empty reference frame; shadows, lighting, reflections, rain, pedestrians, and camera movement make it unreliable.

## 4.3 Slot state machine

Every enabled slot must have one explicit state:
- `FREE`: sufficient evidence supports vacancy.
- `OCCUPIED`: sufficient evidence supports a vehicle being present.
- `UNKNOWN`: evidence is insufficient, contradictory, stale, or unavailable.

Use configurable confidence thresholds, hysteresis, and temporal smoothing. Require stable evidence over a time/frame window before changing stable state. Expose the current state confidence/quality and reason for `UNKNOWN` where feasible.

If the camera disconnects or frames go stale, do **not** set slots to `FREE`. Mark them `UNKNOWN` or report state as stale according to documented policy. The UI must distinguish the last known state from a currently verified state.

The occupied/free/unknown counters must sum correctly to active slots and derive from the same state service. Unknown must never be counted as free.

## 4.4 Vehicle/slot association

Handle vehicles partly visible, straddling adjacent slots, motorcycles, duplicate detections, entry/exit motion, people crossing, occlusion, and vehicles outside slot boundaries. Do not mark multiple adjacent slots occupied from one vehicle unless overlap and tracking evidence justifies it. Make association thresholds configurable and record diagnostic evidence for difficult cases.

Support a vehicle track or short-lived internal identifier for temporal association, but do not use face recognition or attempt to identify drivers/people. Vehicle track IDs should not be treated as permanent identity.

---

# 5. Plate Detection, HSRP-Style Analysis, and OCR

This is a separate optional pipeline built on top of occupancy, not a prerequisite for it.

## 5.1 Plate detection

Use a small detector trained or fine-tuned on verified plate bounding boxes. Support car and two-wheeler plates and additional classes only if the training/evaluation data supports them. Ensure plate detection works from a frame or video file as well as from the live camera.

Store box, score, crop bounds, source frame timestamp, model version, and crop quality. Handle multiple candidates and duplicate boxes. Do not send every frame to OCR; select stable, high-quality candidate crops and use bounded worker queues.

## 5.2 HSRP-style cues

When image quality permits, analyse cues such as:
- Shape/aspect ratio and orientation.
- Background/character contrast and approximate colour.
- Possible blue `IND` panel/mark where applicable.
- Visible character arrangement.
- Perspective, glare, blur, occlusion, crop size, and focus.
- OCR quality and geometric consistency.

These cues are configurable and explanatory, not universal mandatory rules. Legitimate plates and images vary. If there is not an annotated verified HSRP dataset, implement a transparent, experimental cue report rather than a purported trained authenticity model.

Use structured outcomes such as:
- `PLATE_DETECTED`
- `HSRP_STYLE_CONSISTENT`
- `NON_HSRP_OR_UNCERTAIN`
- `INSUFFICIENT_IMAGE_QUALITY`
- `NOT_ASSESSED`

Record the visible cues supporting the outcome. Never display “genuine”, “fake”, “fraudulent”, or legal validation as a definitive conclusion from these heuristics.

## 5.3 OCR pipeline

For suitable plate crops:
1. Validate crop dimensions and quality.
2. Correct orientation and perspective where evidence allows.
3. Preserve aspect ratio during resize; avoid aggressive resampling that invents detail.
4. Benchmark grayscale, contrast enhancement, denoising, and thresholding variants rather than applying every transformation blindly.
5. Run the selected OCR engine.
6. Preserve raw OCR text separately from normalized text.
7. Apply configurable, explainable registration-format checks.
8. Report confidence only as the engine's estimate, not as proof of identity or authenticity.
9. Avoid silently guessing uncertain characters. If a character is unclear, retain uncertainty rather than invent a registration number.
10. Return `UNREADABLE` or `UNCERTAIN` when confidence or image quality is insufficient.

Consider common character confusions only as suggestions for ranking candidate OCR outputs, never automatic unverified rewriting. Keep an audit record of normalization decisions.

## 5.4 Independence rule

The state decision matrix must follow this principle:
- No vehicle evidence + no other occupancy evidence → potentially `FREE`, if the frame is valid and slot assessment is reliable.
- Vehicle evidence + no visible plate → `OCCUPIED`, plate `NOT_VISIBLE`.
- Vehicle evidence + detected plate + failed OCR → `OCCUPIED`, plate `UNREADABLE`.
- Vehicle evidence + plate + plausible OCR → `OCCUPIED`, display recognition as a candidate with confidence/quality.
- Missing/stale camera or conflicting low-quality evidence → `UNKNOWN` where appropriate.

OCR success must never be required to determine occupancy.

## 5.5 Recognition metrics

Report separately:
- Plate-detection precision/recall and mAP where applicable.
- Plate crop quality/coverage rate.
- Character error rate (CER).
- Exact full-registration match rate after documented normalization.
- OCR abstention/unreadable rate.
- False positive rate on non-plate hard negatives.
- Performance by vehicle type, distance, illumination, blur, and occlusion where sample size permits.
- Mean and percentile latency on Windows and Raspberry Pi.

Use a held-out test set captured with the actual Zebronics webcam. Do not claim high accuracy using only training metrics or public datasets that differ from the deployment view.

---

# 6. Datasets, Annotation, Licensing, and Reproducible Training

Create `docs/DATASETS.md` with a dataset inventory. All external datasets below are candidates that must be inspected, not assumed valid or automatically downloaded. Check current availability, dataset card, licence, permitted use, annotation schema, and access requirements before use. Record the verification date and exact downloaded version. Do not use a dataset if its licensing or data provenance is unclear.

## 6.1 Parking occupancy candidates

**PKLot**  
https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/  
Candidate for free/occupied parking-space baselines and parking-region annotation parsing. Verify current terms, annotation formats, and dataset version.

**CNRPark+EXT**  
https://cnrpark.it/  
Candidate for parking-patch classification and variation across cameras/conditions. Verify access and licence conditions before using it.

These datasets are for baseline training/benchmarking, not a guarantee of performance on the Zebronics camera. Where possible, split by camera/session/day to reduce leakage from similar adjacent frames.

## 6.2 General vehicle-detection candidates

**COCO**  
https://cocodataset.org/  
Use a compatible pretrained model with relevant vehicle classes as the initial baseline. Do not download/retrain the full dataset unless necessary.

**India Driving Dataset (IDD)**  
https://idd.insaan.iiit.ac.in/  
Potentially useful for Indian road-scene diversity. Confirm available classes/annotations and terms; add missing labels only through verified annotation.

Do not fine-tune a vehicle detector just to say it was trained. First benchmark a pretrained baseline; fine-tune only when real footage identifies a specific failure.

## 6.3 Indian plate detection/recognition candidates

Treat these as candidates, not guaranteed downloads. Confirm the current page, version, labels, licence, and whether transcriptions are included:

1. Kaggle — Indian License Plates with Labels:  
   https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels
2. DataCluster Labs — Indian Licence Plate Image Dataset:  
   https://github.com/datacluster-labs/Indian-Licence-Plate-Image-Dataset
3. DataCluster Labs — Indian Number Plates Dataset:  
   https://github.com/datacluster-labs/Indian-Number-Plates-Dataset
4. Roboflow Universe — Indian License Plate Detection:  
   https://universe.roboflow.com/atria-university/indian-license-plate-detection-6tmbr-ixo3p

Verify whether the data provides plate bounding boxes, full-text transcriptions, character labels, HSRP-specific labels, train/test splits, and legal permission for the intended use. A plate detector requires boxes; OCR fine-tuning requires correctly transcribed crops; an HSRP-style model requires meaningful HSRP labels. Do not confuse these tasks.

## 6.4 Optional non-Indian plate data

**UFPR-ALPR**: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/  
**CCPD**: https://github.com/detectRecog/CCPD

These may be useful for generic plate localization research where terms permit, but their Brazilian/Chinese plate formats must not be treated as representative Indian OCR or HSRP validation. Clearly separate them from Indian-format evaluation.

## 6.5 Custom dataset from the actual webcam

Build a local capture/annotation workflow for the actual camera placement. This is essential because public datasets cannot guarantee the same viewpoint, plate size, optics, compression, and lighting.

With appropriate permission and privacy safeguards, collect representative still images or sampled frames covering:
- Cars, motorcycles/scooters and supported commercial vehicles.
- Front/rear plate viewpoints where relevant.
- Multiple realistic distances and positions.
- Daylight, shadow, cloudy conditions, indoor lighting, and night only if relevant to the setup.
- Blur, perspective, glare, mild occlusion, and partially visible plates.
- Empty slots, occupied slots with no visible plate, and vehicles outside slots.
- Non-plate rectangular signs, stickers, grilles, reflections, text, and other hard negatives.
- Unreadable and uncertain plate cases.

Annotate only what can be reliably observed:
- Plate bounding box.
- Slot region/slot ID where determinable.
- Vehicle category where supported.
- Ground-truth plate transcription only when legible and appropriate to record.
- Readability/quality, blur, glare, occlusion, and distance/scale indicators if feasible.
- Plate style: `HSRP_STYLE`, `OTHER_STYLE`, or `UNCERTAIN`, only when evidence supports the label.
- Optional visibly observable features such as `IND_REGION_VISIBLE`.

Do not infer legal status from image appearance. Do not store unnecessary personal data. Keep public demonstration screenshots redacted and raw data retention configurable. Use access-controlled local storage.

## 6.6 Split and leakage prevention

Create explicit train/validation/test manifests. Do not randomly split neighbouring video frames across sets. Split custom clips by capture session and, where feasible, vehicle/recording group. Deduplicate exact/near-duplicate images when practical. Keep final test clips unseen during training and threshold tuning. Document known leakage risks and mitigation.

## 6.7 Dataset validation inventory

Before a training run, report for every dataset:
- Link/source and verification date.
- Access/download status.
- Exact version and checksum where possible.
- Actual image/annotation counts (after inspecting data).
- Annotation schema/classes.
- Whether full plate transcriptions are present.
- License and required attribution.
- Allowed use and known restrictions.
- Corrupt/missing files and duplicates.
- Conversion steps and resulting split counts.
- Why it is suitable for the specific training objective.

Never invent counts or claim that a dataset has been downloaded if it has not.

## 6.8 Dataset tooling

Implement functioning scripts/commands with useful help, path checks, progress feedback, and clear exit codes. Suggested commands (adjust names only if the equivalent documented functions exist):

```bash
python scripts/inspect_dataset.py --config configs/datasets.yaml
python scripts/prepare_parking_dataset.py --config configs/datasets.yaml
python scripts/prepare_plate_dataset.py --config configs/datasets.yaml
python scripts/validate_annotations.py --config configs/datasets.yaml
python scripts/create_splits.py --config configs/datasets.yaml
python scripts/train_vehicle_detector.py --config configs/training.yaml
python scripts/train_plate_detector.py --config configs/training.yaml
python scripts/evaluate_models.py --config configs/evaluation.yaml
python scripts/export_deployment_models.py --config configs/deployment.yaml
```

No script may be a placeholder. If training is not feasible because data/access is absent, the script must produce an actionable, truthful error.

## 6.9 Staged model-training strategy

1. **Baseline first:** validate a pretrained vehicle detector, available plate detector, and OCR engine on representative camera footage.
2. **Parking occupancy:** compare vehicle-to-slot mapping and slot-crop classification with held-out data; fine-tune only if needed.
3. **Plate detection:** fine-tune on verified Indian plate boxes if the baseline misses realistic plates.
4. **OCR:** benchmark existing OCR on transcribed, held-out Indian plate crops before considering fine-tuning.
5. **HSRP-style assessment:** train a classifier only if enough verified, appropriately labelled examples exist; otherwise use an explainable experimental cue module and state the limitation.
6. **Deployment optimization:** export selected models and compare runtime/model parity on the target architecture.

For every experiment save config, seed, software versions, dataset manifest, checkpoints, best validation model, metric outputs, confusion matrix where appropriate, and model provenance. Use reproducible seeds where supported.

---

# 7. Model Registry, Export, and Runtime Compatibility

Maintain a model manifest that records for each model:
- Model ID/name and purpose.
- Version and creation date.
- Source/checkpoint identity and hash.
- License/attribution obligations.
- Training dataset manifest/version.
- Input dimensions, preprocessing, class names, normalization, and output decoding.
- Model file path, size, checksum, and export format.
- Runtime and tested architecture/platform.
- Evaluation metrics and test-set description.
- Known limitations.

Do not load arbitrary model files silently. Validate manifest and file hash if provided. Fail safely with a clear UI error if a model is missing, corrupt, incompatible, or has changed unexpectedly. Support a documented rollback to the last known-good model where practical.

Benchmark ONNX Runtime and NCNN (or a justified compatible alternative) on Raspberry Pi. Compare output consistency against the source model as well as latency and memory. Do not assume the fastest Windows runtime is best on ARM64. Measure cold-start time, p50/p95 inference latency, end-to-end FPS, peak resident memory, CPU load, temperature/throttling indicators where available, and stability over an extended run.

Use a configurable inference resolution, frame skipping/adaptive cadence, bounded queues, and optional crop-based OCR. Never promise a fixed real-time FPS until measured on the actual Pi.

---

# 8. GUI and Product Experience

Build a coherent, polished, functional GUI; it must not be a static mock-up. All numbers and states are backed by the real service layer, and all forms/actions have validation, loading, success/failure feedback, and correct persistence behaviour.

## 8.1 Visual system

Use a professional dark technical theme: deep navy/charcoal, restrained cyan/blue accents, clear typography, consistent spacing and component sizes. Use green for `FREE`, red for `OCCUPIED`, amber for `UNKNOWN`/warnings, and neutral shades for inactive controls. Maintain adequate contrast, readable type, keyboard focus indication, and usable controls. Keep animations subtle and optional.

Support desktop layouts including 1920×1080, smaller browser windows, and the Pi monitor. Use the actual browser size responsively rather than hardcoding pixel positions. Do not rely on an internet-hosted font, icon, or script at runtime; vendor dependencies or use system-safe fallbacks.

## 8.2 Navigation and pages

Implement these pages with a consistent navigation system:

1. **Overview:** occupancy summary, recent events, readiness status, camera preview, uptime and key metrics.
2. **Live Monitor:** large live stream, slot overlays, optional detection/plate overlays, state confidence, camera freshness, pause/resume preview, independently controlled processing stop/start.
3. **Parking Calibration:** capture reference frame, draw/edit polygons, type/ID, overlap validation, preview, version history, import/export.
4. **Vehicle & Plate Events:** search, date/slot/event filters, pagination, state-change history, optional privacy-safe export, default plate masking.
5. **Model Management:** detector/OCR model versions, load status, compatibility, checksums, thresholds, manifest view, reload/rollback with confirmation.
6. **Camera Settings:** available indexes/devices, actual resolution/FPS, supported property settings, camera test, reconnect.
7. **System Diagnostics:** quick/full self-test, per-test output, progress, failures/warnings/skips, logs, runtime health, report export.
8. **Settings:** occupancy thresholds, smoothing, retention, privacy, UI, logs, host access, startup, config import/export.
9. **Dataset & Training Information:** dataset inventory, licences, annotation readiness, training runs/metrics, deployed model and provenance. Do not put expensive training controls on the Pi unless intentionally supported; the Pi is primarily an inference target.

## 8.3 Dashboard state

Show total active slots, free, occupied, unknown, camera status, last valid frame time, actual capture dimensions/FPS, inference FPS, latency, application uptime, detector/OCR readiness, and recent genuine state-change events. Include a feature-availability matrix so the operator can see what is working even if optional OCR or a model is absent.

Clicking a slot should show its current state, why it was assigned, state age/staleness, associated candidate vehicle/plate if any, and the last relevant event. Do not claim confidence calibration if it has not been measured; label engine scores accurately.

Plate numbers are masked by default in ordinary UI/event lists. Full visibility requires an explicit settings/authorization choice. Do not use plate text as a heading or public-facing display unless configured for a controlled demo.

## 8.4 Preview versus processing

Distinguish pausing the visual preview from stopping inference/state updates. Controls must be labelled accurately and not imply that one has stopped when only the other has paused. Provide a clear stop/start processing action with state confirmation.

## 8.5 Empty, error, and loading states

Every page must handle first launch, empty database, no slots, missing model, camera unavailable, offline operation, loading, server disconnection, stale feed, and operation failure. Use helpful explanations and safe recovery actions. Never leave a blank page or spinner indefinitely.

---

# 9. Automated Tests and In-App Self-Diagnostics

Implement both (A) a conventional pytest suite and (B) a user-accessible diagnostics service/UI. These are different systems and both are required.

## 9.1 Startup health checks

Run lightweight, non-destructive checks for:
- Python/platform compatibility.
- Required directories and config validity.
- Config schema and effective values.
- Data/log write permissions.
- Database creation/open/read/write and schema version.
- Camera enumeration and optional test capture.
- Model manifest and file existence/readability/hash.
- Model runtime compatibility and lightweight load/inference if safe.
- Required dependencies.
- Frontend/API readiness.
- Disk space and basic resource availability.
- Version compatibility and migration needs.

Classify results as `PASS`, `WARN`, `FAIL`, or `SKIPPED`. Use `NOT RUN` for hardware acceptance tests that have not actually been performed. Distinguish critical failures from optional features: missing OCR must not prevent core occupancy if the occupancy path is working. If there is no camera, allow diagnostics/simulation mode but do not claim live monitoring is operational.

## 9.2 One-click self-test

Add a **Run Self-Test** control with live progress and cancellation. Tests should include, as appropriate:
- Environment check.
- Configuration validation.
- File/directory permissions.
- Database integrity plus temporary read/write test.
- Camera enumeration and capture (optional if unavailable).
- Frame validity/preprocessing.
- Slot polygon geometry and serialization.
- Occupancy state machine/hysteresis.
- Model file validation and test inference when present.
- OCR normalisation and uncertain-result handling.
- API and frontend health.
- Logging/rotation.
- Resource check.

Each individual test records an ID, human name, description, status, duration, actual output, exception details when safe, and corrective action. Long tests run asynchronously and cannot freeze the GUI. Cancellation must clean temporary files, camera resources, workers, and transactions.

## 9.3 Three testing levels

- **Quick self-test:** safe startup/on-demand checks; lightweight and fast.
- **Full diagnostics:** camera, pipeline, model inference, database, API, and longer checks.
- **Developer suite:** unit/integration/system tests via terminal and CI.

Document working commands equivalent to:

```bash
python -m app.diagnostics --quick
python -m app.diagnostics --full
pytest -q
```

Do not run expensive training, full benchmarks, large downloads, or destructive tests automatically at startup.

## 9.4 Offline test requirements

The ordinary test suite must run without a physical webcam, internet, trained model, GPU, cloud service, or live parking lot. Use mock cameras, synthetic frames, fake inference adapters, test SQLite databases, temporary directories, and short locally stored fixture clips.

At minimum test:
- Configuration loading, validation, precedence, and migration handling.
- Camera frames, missing camera, failed reads, stale frames, reconnection, resource cleanup.
- Image resizing/preprocessing.
- Polygon geometry, scaling, invalid/overlapping regions, calibration import/export.
- State transitions, hysteresis, smoothing, duplicate event suppression.
- Multiple vehicle-to-slot associations and boundary edge cases.
- Plate crop transformation and quality gating.
- OCR text normalization, uncertain/unreadable results, character confusion policy.
- Confirmation that plate/OCR failure does not force `FREE`.
- Repository transactions, retention, privacy masking, deletion/export.
- API schemas and error responses.
- Model-manifest/hash/load errors and runtime fallback.
- Logging, log rotation, and shutdown.
- UI state/API integration and WebSocket reconnection where testable.
- Offline operation.

Test failure cases: missing/invalid config, invalid polygons, missing/corrupt model, unsupported runtime, camera disconnect mid-processing, corrupt frames, locked/unwritable DB, insufficient disk, OCR unavailable, invalid UI updates, inference exception, client disconnect, worker cancellation, shutdown during active test.

Failures should become explicit diagnostics and safe `UNKNOWN`/degraded states, not crashes or false free results.

## 9.5 Simulation mode

Implement clearly isolated synthetic scenarios: all free, all occupied, mixed, unknown slot, missing detector, missing OCR, camera disconnected, delayed/slow inference, duplicate transitions, and a vehicle overlapping adjacent slots.

Show a persistent `SIMULATION MODE` banner. Keep simulation data/events separate from production by default, preferably using a dedicated demo database. Never insert simulated values into production history without an explicit isolated demo selection. Every report must identify simulation mode.

## 9.6 Reports

Generate timestamped JSON and human-readable HTML or Markdown reports under a path like `data/test_reports/<timestamp>/`. Each report contains application/test version, timestamp, platform/Python/dependency versions, tests by status, durations, errors/stack traces where useful, camera/model availability, overall result, and corrective guidance. A skipped/not-run test must never count as pass.

---

# 10. Database, Events, Logs, and Data Retention

Use SQLite locally with an explicit schema version and migration strategy. Use parameterized queries and transactions. Keep access behind repositories/services rather than scattering SQL through the app.

Store structures for:
- Parking slot/configuration versions.
- Occupancy state-change events.
- Plate detection/OCR events.
- Camera/model/application health events.
- Self-test runs and report references.
- Model registry/deployment metadata where useful.

Occupancy event fields should include slot ID, previous/current state, timestamp, evidence/score metadata, model/config version, camera health/staleness, and associated short-lived track ID where applicable.

Plate event fields should include timestamp, slot association if supported, raw and normalized text only when retention is enabled, OCR engine/model version, confidence, crop-quality status, HSRP-style outcome and visible cues, and review status where appropriate.

Do not write every frame to the database. Debounce repeated identical events; store meaningful transitions and optionally sampled diagnostic snapshots with retention limits. Make retention/deletion configurable. Avoid holding database writes on the inference loop. Handle locks, disk-full errors, migration failure, and recovery clearly. Provide backup/export and restore guidance.

Use structured logs with rotation and appropriate redaction. Do not log full plate strings, credentials, or sensitive paths by default. The UI should show a readable last error plus reference to details in logs, without exposing raw tracebacks to an ordinary user.

---

# 11. API, Security, and Privacy

- Bind to localhost by default. LAN exposure must be opt-in and documented.
- If LAN access is enabled, supply reasonable administrative access controls/authentication and explain firewall considerations. Do not imply localhost-only security applies to all network settings.
- Validate all API inputs and outputs. Apply request size limits where applicable and constrain user-controlled paths to intended data directories.
- Protect state-changing administration endpoints appropriately.
- Keep secrets out of source/config examples and logs.
- Do not upload frames, plate numbers, datasets, or database records to third-party services; do not implement telemetry by default.
- Do not add face/person identification.
- Make privacy defaults conservative: masked plate UI, local processing, configurable retention, explicit deletion and export.
- Obtain permission for custom data capture; document the dataset's provenance/consent and avoid collecting unrelated people or unnecessary personal information.
- Explain that registration numbers may be personal/sensitive operational data and should be retained only when necessary.
- Document limitations and use policy in `docs/PRIVACY.md`.

---

# 12. Performance, Reliability, and Recoverability

Design for CPU-based inference on Raspberry Pi 5. Use lightweight models, configurable input resolution, optional frame skipping/adaptive inference frequency, bounded queues, sensible worker separation, and OCR only on selected stable candidate crops. Do not spawn unbounded threads/processes or copy full-resolution images repeatedly.

Implement:
- Dedicated capture worker and safe camera ownership.
- Freshness timestamps and stale-frame detection.
- Reconnection with capped/backoff retry rather than tight loops.
- Graceful optional-feature degradation.
- Safe exception boundaries and structured logs.
- Runtime monitoring for CPU/memory/disk and temperature/throttling where accessible.
- Health states for camera, detector, plate detector, OCR, database, frontend, and application.
- Recovery actions and clean startup/shutdown.
- Protection against memory growth during long operation.

Set a realistic configurable target, but report actual measured metrics rather than guarantees. Include a soak test (for example, a configurable multi-hour run) and record memory trend, dropped frames, reconnects, errors, and temperature/resource behaviour on the real Pi.

---

# 13. Repository Structure and Engineering Hygiene

Create a repository similar to the following, adapting only when there is a documented benefit:

```text
smart-parking-hsrp/
├── app/
│   ├── main.py
│   ├── config/
│   ├── camera/
│   ├── detection/
│   ├── parking/
│   ├── plates/
│   ├── ocr/
│   ├── tracking/
│   ├── application/
│   ├── database/
│   ├── api/
│   ├── diagnostics/
│   ├── services/
│   ├── infrastructure/
│   └── utils/
├── web/
│   ├── templates/
│   ├── static/
│   └── tests/
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_parking_dataset.py
│   ├── prepare_plate_dataset.py
│   ├── validate_annotations.py
│   ├── create_splits.py
│   ├── train_vehicle_detector.py
│   ├── train_plate_detector.py
│   ├── evaluate_models.py
│   └── export_deployment_models.py
├── configs/
│   ├── windows.yaml
│   ├── raspberry_pi.yaml
│   ├── datasets.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── deployment.yaml
├── models/
│   ├── README.md
│   └── manifest.json
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   ├── splits/
│   ├── demo/
│   └── test_reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── system/
│   ├── fixtures/
│   └── conftest.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASETS.md
│   ├── TRAINING.md
│   ├── WINDOWS_SETUP.md
│   ├── RASPBERRY_PI_SETUP.md
│   ├── CAMERA_CALIBRATION.md
│   ├── TESTING_DIAGNOSTICS.md
│   ├── TROUBLESHOOTING.md
│   ├── PRIVACY.md
│   ├── SECURITY.md
│   └── ACCEPTANCE_TESTS.md
├── logs/                    # runtime-only; do not commit logs
├── requirements-windows.txt
├── requirements-pi.txt
├── pyproject.toml
├── .gitignore
├── LICENSES.md
└── README.md
```

Engineering hygiene:
- Add type hints and docstrings at public module/service boundaries.
- Use consistent formatting/linting and optionally static type checks; document the commands.
- Add `.gitignore` for virtual environments, caches, large datasets, model artifacts as appropriate, logs, temporary frames, local DBs, secrets, and private captures.
- Keep generated output out of source directories unless explicitly intended.
- Add an application/version field and a database schema version.
- Use relative/configured paths resolved safely, not machine-specific absolute paths.
- Include a clear licence file for project code and a `LICENSES.md` for third-party dependencies and dataset/model attribution. Do not claim rights over external data/models.
- Keep secrets and local camera clips out of Git.

---

# 14. Windows Development Workflow

Provide a verified clean-install process:
1. Install supported Python version and prerequisites.
2. Create and activate a virtual environment.
3. Install Windows dependencies from a pinned/reproducible requirements or lock file.
4. Run configuration validation and quick diagnostics.
5. Run unit/integration tests.
6. Run synthetic mode with no webcam.
7. Detect and test the Zebronics webcam.
8. Capture reference frames and calibrate slots.
9. Run a recorded video through the pipeline.
10. Benchmark model/threshold alternatives and prepare dataset manifests.
11. Train/fine-tune only when needed.
12. Evaluate and export deployment artifacts.

Document exact working commands once the actual entry points are implemented. Do not leave a setup guide with guessed package versions or untested commands. Include a PowerShell-friendly path where appropriate and quote paths that may contain spaces.

Suggested launch command (adapt to the code, then verify it):

```powershell
python -m app.main --config configs/windows.yaml
```

Do not assume the user has a particular GPU; detect it and report availability. Full training may require more compute/storage than app testing, so keep those tasks separate and estimate resource requirements after inspecting selected datasets/models.

---

# 15. Raspberry Pi Deployment Workflow

Provide a complete, verified deployment process for a clean supported 64-bit Raspberry Pi OS installation:
- OS/Python/runtime compatibility check.
- USB webcam discovery and actual capture verification.
- Pi-specific installation instructions for ARM64 packages.
- Transfer of model artifact, manifest, configuration, and optional demo video from Windows.
- Verify hashes after transfer where available.
- Install inference-only dependencies, not the full training stack.
- Initialize/migrate local SQLite database.
- Run quick diagnostics and start the dashboard.
- Test on HDMI monitor and configure optional kiosk mode.
- Configure a systemd unit only after manual launch works, with sane working directory, environment, restart policy, and logs.
- Document stop/start/restart/status/log commands and clean shutdown.
- Test boot recovery, camera disconnect/reconnect, missing model, database errors, and offline operation.
- Include backup/restore, calibration export, and model rollback instructions.

Never copy the Windows `.venv` to the Pi. The target runtime must not require CUDA or a discrete NVIDIA GPU. Model format selection is made using tested Pi results, not assumptions.

---

# 16. Test Plan and Acceptance Gates

Create a manual checklist and automate as many checks as possible. Record actual evidence/command output for each gate.

## Gate A — Clean Windows installation
- Fresh environment installs using documented steps.
- Application starts with no undocumented dependencies or manual code edits.
- Quick diagnostic output is accurate.

## Gate B — No-camera / offline mode
- Application opens without hardware or internet.
- Simulation is clearly labelled and isolated.
- Offline tests pass or report specific failures.
- No simulated data is represented as live data.

## Gate C — Real webcam
- Zebronics camera is detected.
- Negotiated dimensions and observed FPS are displayed.
- Preview frames update and are not stale.
- Device disconnect triggers correct degraded status and reconnect recovery.

## Gate D — Calibration and occupancy
- Slots can be created/edited/saved/reloaded/exported/imported.
- Coordinate scaling works across supported resolutions.
- Invalid polygons are rejected or warned about.
- Real recorded and live footage produce defensible FREE/OCCUPIED/UNKNOWN transitions.
- Camera loss never marks all slots free.
- OCR failure does not change occupied to free.

## Gate E — Plate pipeline
- Plate detection and OCR can be individually disabled.
- Plate crop and quality metadata are available.
- OCR returns uncertain/unreadable outcomes appropriately.
- Candidate OCR is not presented as legal/authentic identity verification.
- Metrics are measured on a held-out set captured with the actual camera.

## Gate F — GUI integration
- Every navigation item works.
- Controls perform advertised actions.
- Loading, empty, warning, and failure states are handled.
- Dashboard counters match the state service.
- Self-test progress does not freeze the interface.
- Plate text is masked by default.
- Responsive display and keyboard usability are checked.

## Gate G — Database, diagnostics, and recovery
- Event records are created only for meaningful state changes.
- Retention/deletion/export works as documented.
- JSON and human-readable test reports are generated.
- Tests that are skipped or not run are not counted as pass.
- Database/model/camera errors are visible and actionable.
- Workers and the camera close cleanly on shutdown.

## Gate H — Raspberry Pi real-hardware acceptance
- Fresh Pi setup follows the documented instructions.
- Webcam, inference runtime, database, and local dashboard work on the actual Pi 5 8 GB.
- The HDMI monitor displays the GUI fullscreen.
- FPS, p50/p95 latency, RAM, CPU, disk and thermal behaviour are measured.
- Long-run test, camera reconnect, reboot/startup, and offline operation are checked.

If physical camera/Pi/model/data is unavailable, mark that gate `NOT RUN` with the specific missing prerequisite. A mock test may validate logic but cannot pass a hardware gate.

## Definition of done

The project is finished only when:
- The complete source repository and docs exist.
- It runs on Windows with the Zebronics USB webcam.
- It supports no-camera diagnostic/simulation mode.
- Slot calibration needs no source-code edits.
- Occupancy works independently of plate OCR.
- GUI data comes from live services, not hardcoded values.
- State changes persist locally and duplicate events are controlled.
- Self-tests and developer tests produce truthful reports.
- The Pi deployment path is reproducible and tested on the real target before the project is called Pi-validated.
- Model/dataset provenance, licence, metric and known limitations are documented.
- Offline operation works after installation and transferring required assets.

---

# 17. Final Deliverables

Deliver all of the following, with no empty placeholders:

1. Complete executable source repository.
2. Professional responsive dashboard with functioning navigation/actions.
3. Working USB webcam capture and a diagnostic camera test utility.
4. Interactive slot calibration with validation and import/export.
5. FREE/OCCUPIED/UNKNOWN state engine and temporal smoothing.
6. Vehicle detection and vehicle-to-slot association.
7. Independent plate detector, HSRP-style cue assessment, and OCR integration.
8. Robust error/degraded paths and offline mode.
9. Local SQLite data model, event history, retention, export, and migrations.
10. Startup health checks and one-click full diagnostic page.
11. pytest unit/integration/system tests with mock hardware.
12. Simulation mode with explicit banner and isolated data.
13. JSON plus human-readable self-test reports.
14. Dataset inventory, licence/provenance checks, annotation conversion, split and validation scripts.
15. Reproducible training/evaluation/export scripts and run manifests.
16. Model registry/manifest, hashes, model versioning and rollback guidance.
17. Windows setup/development guide.
18. Raspberry Pi installation, runtime, kiosk, service, recovery and offline guide.
19. Camera-calibration, testing, troubleshooting, security, privacy and acceptance-test documentation.
20. A small permitted demo/fixture set or instructions to create it; do not include large/unlicensed/private datasets.
21. Exact commands to install, run, test, diagnose, train, evaluate, export, deploy, and recover.
22. A release checklist recording what was actually tested and what remains `NOT RUN`.

Do not claim the deliverable includes pretrained weights or dataset files unless they really are present and distributable. When license/size prevents bundling, provide the manifest and exact acquisition instructions instead.

---

# 18. Required Implementation Order

Work incrementally. Finish and verify the early milestones before spending time on advanced model training.

### Milestone 0 — Repository and engineering baseline
- Inspect existing code if any.
- Create repository skeleton, config schema, logging, version metadata, docs outline, test harness, and environment files.
- Decide and document compatible dependencies for Windows/Pi.
- Add CI-friendly offline tests.

### Milestone 1 — Camera and app shell
- Start the FastAPI app/UI.
- Implement one camera owner, device scan/test, live stream/preview, no-camera state, and reconnect logic.
- Run quick diagnostics and mock-camera tests.

### Milestone 2 — Calibration
- Implement drawing/editing/persistence/import/export and coordinate scaling.
- Validate geometry and add tests.

### Milestone 3 — Occupancy MVP
- Add baseline pretrained vehicle detection and slot association.
- Implement state machine, hysteresis, confidence, UNKNOWN and stale-camera behaviour.
- Test against recorded footage and edge cases.

### Milestone 4 — Real dashboard and events
- Finish live/overview screens, counters, slot details, event history, SQLite persistence, and configuration UI.
- Ensure all counts/events are real service data.

### Milestone 5 — Diagnostics and hardening
- Complete all three test levels, simulation mode, cancellation, reports, failure injection tests, logging and graceful shutdown.

### Milestone 6 — Plate pipeline
- Add plate detection, crop quality checks, HSRP-style cue reporting and OCR as independent optional services.
- Establish actual camera limits and evaluate results on annotated holdout footage.

### Milestone 7 — Dataset/training quality
- Validate candidate dataset availability/licences and labels.
- Build data preparation, annotation checks, group-aware splitting, training manifests and evaluation reports.
- Fine-tune only components shown to need it.

### Milestone 8 — Pi optimization and deployment
- Export models and compare runtime formats on actual hardware.
- Install inference environment, configure HDMI/kiosk, service startup, logs and recovery.
- Perform Gate H if hardware is available.

### Milestone 9 — Release readiness
- Run all tests from a clean environment.
- Verify docs and commands.
- Review privacy/security and dependency/data licences.
- Produce release notes, limitations, known issues, and test status matrix.

For each milestone, report:
- Scope completed.
- Files created/modified.
- Exact commands run.
- Actual test results and report paths.
- Manual checks completed.
- Open bugs/risks.
- Next milestone.

Do not build every advanced feature before achieving an end-to-end MVP.

---

# 19. First Response and Coding-Agent Behaviour

Begin by inspecting the repository and available runtime. Then provide a concise implementation checklist and repository map, followed immediately by actual implementation of Milestone 0 and Milestone 1. Do not wait for another approval unless a truly blocking decision cannot be reasonably defaulted.

When making design choices, prefer explicit, documented defaults and record trade-offs. Ask questions only when a missing fact genuinely blocks implementation. Otherwise proceed and provide settings for tunable variables.

At every stage:
- Tell the truth about what was executed and verified.
- Separate completed, partially completed, and planned work.
- Preserve actual logs and test outputs.
- Never mark skipped tests as passes.
- Fix regressions before moving on.
- Do not leave broken imports, unimplemented routes, nonfunctional buttons, empty test files, placeholder model calls, or README commands that fail.

**Core engineering principle:** build a useful MVP first, make it testable, prove it on real footage, then improve accuracy and performance. The final product should be understandable, recoverable, privacy-conscious, reproducible, and honest about uncertainty—not merely visually impressive.
