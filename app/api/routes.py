"""FastAPI REST routes and MJPEG streaming endpoints."""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import cv2
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.camera.device_scanner import scan_camera_devices
from app.camera.quality_gate import QualityGate
from app.config.loader import get_config, save_config
from app.config.schema import RootConfig
from app.database.connection import DatabaseManager
from app.database.repository import ParkingRepository
from app.diagnostics.health import DiagnosticsRunner
from app.infrastructure.paths import paths
from app.infrastructure.platform_info import get_platform_info
from app.parking.calibration_service import CalibrationService
from app.parking.occupancy_service import OccupancyService
from app.services.camera_service import CameraService
from app.parking.allocation_service import (
    AllocationService,
    SlotUnavailableError,
    ConcurrencyConflictError,
    DuplicateAllocationError,
)

router = APIRouter()

# Global service references (set during app lifespan)
camera_service: Optional[CameraService] = None
repository: Optional[ParkingRepository] = None
calibration_service: Optional[CalibrationService] = None
occupancy_service: Optional[OccupancyService] = None
allocation_service: Optional[AllocationService] = None
detector = None
start_time: float = time.time()


class ScenarioRequest(BaseModel):
    scenario: str


class SlotsUpdateRequest(BaseModel):
    reference_width: int
    reference_height: int
    slots: List[Dict[str, Any]]


class AllocateSlotRequest(BaseModel):
    user_id: str
    slot_id: Optional[str] = None
    zone: Optional[str] = None
    slot_type: str = "car"
    vehicle_plate: Optional[str] = None
    vehicle_type: str = "car"
    duration_minutes: Optional[int] = 30


class ReleaseSlotRequest(BaseModel):
    reservation_code: str


class CancelReservationRequest(BaseModel):
    reservation_code: str
    reason: Optional[str] = "User cancelled"



@router.get("/health")
def get_health():
    """System health check and runtime metrics."""
    uptime_sec = round(time.time() - start_time, 1)
    cam_status = camera_service.get_status() if camera_service else {"state": "STOPPED"}
    return {
        "status": "healthy" if (cam_status.get("is_healthy", False) or cam_status.get("is_mock", False)) else "degraded",
        "uptime_sec": uptime_sec,
        "platform": get_platform_info(),
        "camera": cam_status,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/camera/status")
def get_camera_status():
    """Returns current camera capture metrics."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized.")
    return camera_service.get_status()


@router.get("/camera/devices")
def get_camera_devices():
    """Scans and lists available hardware camera indexes."""
    return {"devices": scan_camera_devices(max_devices_to_check=2)}


@router.post("/camera/reconnect")
def reconnect_camera():
    """Triggers camera hardware reconnection."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized.")
    success = camera_service.reconnect()
    return {"success": success, "status": camera_service.get_status()}


@router.post("/camera/pause_preview")
def pause_preview():
    """Pauses visual video feed preview without interrupting inference."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized.")
    camera_service.pause_preview()
    return {"preview_paused": True}


@router.post("/camera/resume_preview")
def resume_preview():
    """Resumes visual video feed preview."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized.")
    camera_service.resume_preview()
    return {"preview_paused": False}


@router.post("/camera/toggle_processing")
def toggle_processing():
    """Toggles inference processing loop on or off."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized.")
    active = camera_service.toggle_processing()
    return {"processing_active": active}


@router.post("/camera/set_scenario")
def set_synthetic_scenario(req: ScenarioRequest):
    """Sets synthetic scenario (normal, all_free, all_occupied, mixed, camera_loss, straddling)."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized.")
    camera_service.set_scenario(req.scenario)
    return {"scenario": req.scenario}


@router.get("/camera/test_frame")
def test_camera_frame():
    """Grabs a frame, evaluates image quality metrics, and returns diagnostic results."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized.")
    frame, meta = camera_service.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available to evaluate.")

    gate = QualityGate()
    assessment = gate.assess_frame(frame)
    return {
        "assessment": assessment.__dict__,
        "metadata": meta.__dict__ if meta else {},
    }


def _stream_generator():
    """Yields multipart JPEG frames with real-time slot and detection overlays without duplicate transmission."""
    last_sent_ts = 0.0
    while True:
        if camera_service is not None:
            frame, meta = camera_service.get_latest_frame()
            if frame is not None and meta is not None:
                # Only encode and transmit when a fresh frame has arrived or preview is paused
                if meta.timestamp > last_sent_ts or camera_service.preview_paused:
                    t_deliv_0 = time.perf_counter()
                    overlay_fn = occupancy_service.render_overlay if occupancy_service else None
                    jpeg_bytes = camera_service.generate_mjpeg_frame(overlay_renderer=overlay_fn)
                    if jpeg_bytes:
                        last_sent_ts = meta.timestamp
                        t_deliv_1 = time.perf_counter()
                        try:
                            from app.infrastructure.telemetry import telemetry
                            telemetry.record_video_delivery((t_deliv_1 - t_deliv_0) * 1000.0)
                        except Exception:
                            pass
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                        )
        time.sleep(0.015)  # Responsive check without busy-waiting or frame duplication


@router.get("/stream/video_feed")
def get_video_stream():
    """Live MJPEG video stream with visual detection & bay overlays."""
    return StreamingResponse(
        _stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/slots")
def get_parking_slots():
    """Returns authoritative calibrated parking bays merged with live occupancy states."""
    calib = calibration_service.load_calibration() if calibration_service else {}
    if not calib:
        calib_path = paths.resolve("data/calibration.json")
        if calib_path.exists():
            try:
                calib = json.loads(calib_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read calibration: {e}")
        else:
            calib = {"slots": [], "reference_width": 1280, "reference_height": 720, "version": 1}

    calib_slots = calib.get("slots", [])
    ref_w = calib.get("reference_width", 1280)
    ref_h = calib.get("reference_height", 720)
    version = calib.get("version", 1)

    snap = occupancy_service.get_latest_snapshot() if occupancy_service else None
    snap_slots = snap.get("slots", []) if snap else []
    snap_map = {(s.get("id") or s.get("slot_id")): s for s in snap_slots if (s.get("id") or s.get("slot_id"))}

    # Fetch active reservations to enforce authoritative operational status
    active_res_by_slot = {}
    if allocation_service:
        try:
            with allocation_service.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT slot_id, reservation_code, vehicle_plate, status FROM slot_reservations WHERE status = 'ACTIVE';"
                )
                for r in cursor.fetchall():
                    active_res_by_slot[r["slot_id"]] = dict(r)
        except Exception:
            pass

    merged_slots = []
    for s in calib_slots:
        slot_data = dict(s)
        sid = s.get("id")
        live_info = snap_map.get(sid)
        is_enabled = s.get("enabled", True)
        res_info = active_res_by_slot.get(sid)

        if not is_enabled:
            slot_data["state"] = "DISABLED"
            slot_data["physical_state"] = "DISABLED"
            slot_data["confidence"] = 1.0
            slot_data["occupied"] = False
            slot_data["reservation_status"] = "OUT_OF_SERVICE"
        elif live_info:
            phys_state = live_info.get("state", "UNKNOWN")
            slot_data["physical_state"] = phys_state
            slot_data["confidence"] = live_info.get("confidence", 0.0)
            slot_data["occupied"] = (phys_state == "OCCUPIED")
            slot_data["active_reservation"] = res_info.get("reservation_code") if res_info else None
            if res_info:
                slot_data["reservation_status"] = "OCCUPIED_AUTHORIZED" if phys_state == "OCCUPIED" else "RESERVED"
                slot_data["state"] = "OCCUPIED" if phys_state == "OCCUPIED" else "RESERVED"
            else:
                slot_data["reservation_status"] = "OCCUPIED_UNAUTHORIZED" if phys_state == "OCCUPIED" else ("AVAILABLE" if phys_state == "FREE" else "UNKNOWN")
                slot_data["state"] = phys_state
        else:
            # Safe fail-closed: missing live observations MUST be marked UNKNOWN, never assumed FREE
            slot_data["state"] = "UNKNOWN"
            slot_data["physical_state"] = "UNKNOWN"
            slot_data["confidence"] = 0.0
            slot_data["occupied"] = False
            slot_data["reservation_status"] = "UNKNOWN"

        merged_slots.append(slot_data)

    merged_summary = {
        "total": len(merged_slots),
        "free": len([s for s in merged_slots if s["state"] == "FREE"]),
        "occupied": len([s for s in merged_slots if s["state"] == "OCCUPIED"]),
        "reserved": len([s for s in merged_slots if s["state"] == "RESERVED"]),
        "unknown": len([s for s in merged_slots if s["state"] in ("UNKNOWN", "DISABLED")]),
    }

    return {
        "version": version,
        "reference_width": ref_w,
        "reference_height": ref_h,
        "summary": merged_summary,
        "slots": merged_slots,
    }


@router.post("/slots")
def save_parking_slots(req: SlotsUpdateRequest):
    """Saves updated parking bay calibration and reloads occupancy and allocation engines."""
    if calibration_service is not None:
        try:
            _, warnings, payload = calibration_service.save_calibration(
                slots=req.slots,
                ref_width=req.reference_width,
                ref_height=req.reference_height,
            )
            if occupancy_service is not None:
                occupancy_service.reload_slots()
            if allocation_service is not None:
                allocation_service.sync_slots_from_calibration(req.slots)
            return {"success": True, "slots_count": len(req.slots), "warnings": warnings}
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

    calib_path = paths.resolve("data/calibration.json")
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "reference_width": req.reference_width,
        "reference_height": req.reference_height,
        "updated_at": datetime.now().isoformat(),
        "slots": req.slots,
    }
    calib_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if occupancy_service is not None:
        occupancy_service.reload_slots()
    if allocation_service is not None:
        allocation_service.sync_slots_from_calibration(req.slots)
    return {"success": True, "slots_count": len(req.slots), "warnings": []}


@router.get("/events/occupancy")
def get_occupancy_events(limit: int = 50, slot_id: Optional[str] = None):
    """Returns recent occupancy state changes."""
    if not repository:
        return {"events": []}
    events = repository.get_recent_occupancy_events(limit=limit, slot_id=slot_id)
    return {"events": events}


@router.get("/events/plates")
def get_plate_events(limit: int = 50):
    """Returns recent license plate detection events."""
    if not repository:
        return {"events": []}
    plates = repository.get_recent_plate_events(limit=limit)
    return {"events": plates}


@router.post("/diagnostics/run")
def trigger_diagnostics(run_type: str = "quick"):
    """Triggers in-app self-diagnostics."""
    runner = DiagnosticsRunner()
    report = runner.run(run_type="full" if run_type == "full" else "quick")
    return report.__dict__


@router.get("/diagnostics/reports")
def list_diagnostic_reports():
    """Lists saved diagnostic reports."""
    report_dir = paths.resolve("data/test_reports")
    if not report_dir.exists():
        return {"reports": []}
    files = sorted(report_dir.glob("*.json"), reverse=True)
    reports = []
    for f in files[:20]:
        try:
            content = json.loads(f.read_text(encoding="utf-8"))
            reports.append({
                "report_id": content.get("report_id"),
                "timestamp": content.get("timestamp"),
                "run_type": content.get("run_type"),
                "overall_status": content.get("overall_status"),
                "summary": content.get("summary"),
            })
        except Exception:
            pass
    return {"reports": reports}


@router.get("/diagnostics/telemetry")
def get_system_telemetry():
    """Returns real-time telemetry metrics covering all 7 measurement rates and stage latencies."""
    from app.infrastructure.telemetry import telemetry
    return telemetry.get_metrics()


@router.post("/diagnostics/telemetry/export")
def export_telemetry_benchmark():
    """Exports machine-readable benchmark report to data/test_reports/telemetry_benchmark.json."""
    from app.infrastructure.telemetry import telemetry
    out_path = telemetry.export_benchmark_report()
    return {"success": True, "report_path": out_path, "metrics": telemetry.get_metrics()}


@router.get("/models")
def get_registered_models():
    """Returns registered model artifacts and metadata with actual file existence check."""
    manifest_path = paths.models_manifest_file
    if not manifest_path.exists():
        return {"models": []}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for m in data.get("models", []):
            m_path = paths.resolve(m.get("file_path", ""))
            m["file_exists"] = m_path.exists()
            m["status"] = "READY" if m_path.exists() else "NOT_TRAINED"
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read model manifest: {e}")


@router.get("/config")
def get_application_config():
    """Returns current active configuration."""
    cfg = get_config()
    return cfg.model_dump()


# =========================================================================
# Virtual Parking Allocation & Concurrency Endpoints (Phase 7)
# =========================================================================

@router.post("/parking/allocate")
def allocate_parking_slot(req: AllocateSlotRequest):
    """
    Transactionally allocates an available parking slot.
    Guarantees duplicate-allocation prevention and strictly refuses OCCUPIED or UNKNOWN slots.
    """
    if not allocation_service:
        raise HTTPException(status_code=503, detail="Allocation service not initialized.")

    phys_map = {}
    if occupancy_service:
        snap = occupancy_service.get_latest_snapshot()
        for s in snap.get("slots", []):
            phys_map[s["id"]] = s

    try:
        receipt = allocation_service.allocate_slot(
            user_id=req.user_id,
            slot_id=req.slot_id,
            zone=req.zone,
            slot_type=req.slot_type,
            vehicle_plate=req.vehicle_plate,
            vehicle_type=req.vehicle_type,
            duration_minutes=req.duration_minutes,
            current_physical_states=phys_map,
        )
        return receipt
    except SlotUnavailableError as sue:
        raise HTTPException(status_code=404, detail=str(sue))
    except (ConcurrencyConflictError, DuplicateAllocationError) as cce:
        raise HTTPException(status_code=409, detail=str(cce))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Allocation error: {str(e)}")


@router.post("/parking/release")
def release_parking_slot(req: ReleaseSlotRequest):
    """Releases an active parking reservation."""
    if not allocation_service:
        raise HTTPException(status_code=503, detail="Allocation service not initialized.")
    try:
        return allocation_service.release_slot(req.reservation_code)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.post("/parking/cancel")
def cancel_parking_reservation(req: CancelReservationRequest):
    """Cancels an active parking reservation."""
    if not allocation_service:
        raise HTTPException(status_code=503, detail="Allocation service not initialized.")
    try:
        return allocation_service.cancel_reservation(req.reservation_code, reason=req.reason or "User cancelled")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.get("/parking/availability")
def get_parking_availability():
    """
    Returns authentic live parking availability counts calculated from
    the live slot registry and physical state engine. Removes arbitrary counters.
    """
    if not allocation_service:
        raise HTTPException(status_code=503, detail="Allocation service not initialized.")

    snap = occupancy_service.get_latest_snapshot() if occupancy_service else None
    return allocation_service.get_availability_summary(snap)


@router.get("/parking/audit")
def get_parking_audit_trail(limit: int = 50, slot_id: Optional[str] = None):
    """Returns reservation and allocation audit log history."""
    if not allocation_service:
        raise HTTPException(status_code=503, detail="Allocation service not initialized.")
    return {"audit_trail": allocation_service.get_audit_trail(limit=limit, slot_id=slot_id)}


@router.post("/parking/sync_slots")
def sync_calibrated_slots():
    """Synchronizes calibrated parking slots into database table."""
    if not allocation_service or not calibration_service:
        raise HTTPException(status_code=503, detail="Services not initialized.")
    calib = calibration_service.load_calibration()
    count = allocation_service.sync_slots_from_calibration(calib.get("slots", []))
    return {"success": True, "synchronized_slots": count}


# =========================================================================
# AI Model Lab & Continuous Improvement Endpoints (Section 13)
# =========================================================================

class ReviewCandidateRequest(BaseModel):
    frame_id: str
    action: str  # ACCEPT, REJECT, CORRECT
    corrected_class: Optional[str] = None


class PromoteCandidateRequest(BaseModel):
    experiment_id: str
    force: bool = False
    min_mAP50_improvement: float = 0.0


@router.get("/lab/overview")
def get_ai_lab_overview():
    """Returns authoritative AI Model Lab overview including active model, checksum integrity, and candidates."""
    from app.models.registry import model_registry

    active = model_registry.get_active_model()
    reg = model_registry.load_registry()
    candidates = reg.get("candidates", [])
    prev_stable = reg.get("previous_stable_model")

    # Load latest benchmark if exists
    bench_p = paths.data_dir / "test_reports" / "vehicle_edge_benchmark.json"
    latest_bench = None
    if bench_p.exists():
        try:
            b_data = json.loads(bench_p.read_text(encoding="utf-8"))
            latest_bench = b_data.get("benchmarks", [{}])[-1]
        except Exception:
            pass

    return {
        "active_model": active,
        "previous_stable_model": prev_stable,
        "total_candidates": len(candidates),
        "latest_candidate": candidates[-1] if candidates else None,
        "latest_benchmark": latest_bench,
        "rollback_available": prev_stable is not None,
    }


@router.get("/lab/experiments")
def get_lab_experiments():
    """Returns registered candidate training experiments and history."""
    from app.models.registry import model_registry

    reg = model_registry.load_registry()
    return {
        "active_model": reg.get("active_model"),
        "candidates": reg.get("candidates", []),
        "history": reg.get("history", []),
    }


@router.get("/lab/comparison")
def get_lab_model_comparison(candidate_id: Optional[str] = None):
    """Returns side-by-side metric comparison between active production baseline and selected candidate."""
    from app.models.registry import model_registry
    from scripts.compare_vehicle_models import compare_metrics

    reg = model_registry.load_registry()
    candidates = reg.get("candidates", [])
    if not candidates:
        return {"error": "No experimental candidates registered."}

    cand = next((c for c in candidates if c.get("experiment_id") == candidate_id), candidates[-1])
    active = reg.get("active_model", {})

    base_m = active.get("metrics", {})
    cand_m = cand.get("test_metrics") or cand.get("validation_metrics", {})

    comparison = compare_metrics(base_m, cand_m)
    return {
        "candidate_id": cand.get("experiment_id"),
        "candidate_status": cand.get("status"),
        "comparison": comparison,
    }


@router.get("/lab/error_gallery")
def get_lab_error_gallery():
    """Lists reviewed false positives and false negatives from latest evaluation error gallery."""
    gallery_dir = paths.data_dir / "test_reports" / "error_gallery"
    fp_dir = gallery_dir / "false_positives"
    fn_dir = gallery_dir / "false_negatives"

    fp_files = [{"filename": f.name, "url": f"/api/lab/gallery_image/false_positives/{f.name}"} for f in sorted(fp_dir.glob("*.jpg"))[:20]] if fp_dir.exists() else []
    fn_files = [{"filename": f.name, "url": f"/api/lab/gallery_image/false_negatives/{f.name}"} for f in sorted(fn_dir.glob("*.jpg"))[:20]] if fn_dir.exists() else []

    return {
        "total_false_positives": len(fp_files),
        "total_false_negatives": len(fn_files),
        "false_positives": fp_files,
        "false_negatives": fn_files,
    }


@router.get("/lab/gallery_image/{category}/{filename}")
def get_gallery_image(category: str, filename: str):
    """Serves error gallery visual crops securely."""
    if category not in ["false_positives", "false_negatives"]:
        raise HTTPException(status_code=400, detail="Invalid gallery category.")
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename or not (safe_name.lower().endswith((".jpg", ".jpeg", ".png"))):
        raise HTTPException(status_code=400, detail="Invalid filename format.")
    base_dir = (paths.data_dir / "test_reports" / "error_gallery" / category).resolve()
    img_p = (base_dir / safe_name).resolve()
    if not img_p.is_relative_to(base_dir) or not img_p.exists():
        raise HTTPException(status_code=404, detail="Gallery crop not found.")
    return FileResponse(str(img_p), media_type="image/jpeg")


@router.get("/lab/candidate_image/{filename}")
def get_candidate_image(filename: str):
    """Serves mined candidate image securely."""
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename or not (safe_name.lower().endswith((".jpg", ".jpeg", ".png"))):
        raise HTTPException(status_code=400, detail="Invalid filename format.")
    base_dir = (paths.data_dir / "active_learning" / "candidates" / "images").resolve()
    img_p = (base_dir / safe_name).resolve()
    if not img_p.is_relative_to(base_dir) or not img_p.exists():
        raise HTTPException(status_code=404, detail="Candidate image not found.")
    return FileResponse(str(img_p), media_type="image/jpeg")


@router.get("/lab/dataset_health")
def get_lab_dataset_health():
    """Returns dataset health metrics, class distribution, and validation status."""
    audit_p = paths.data_dir / "test_reports" / "dataset_validation_audit.json"
    if audit_p.exists():
        try:
            return json.loads(audit_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    from scripts.inspect_vehicle_dataset import inspect_dataset_statistics
    yaml_p = paths.resolve("data/vehicles.yaml")
    return inspect_dataset_statistics(yaml_p)


@router.post("/lab/capture_frame")
def trigger_manual_frame_capture(reason: str = "MANUAL_OPERATOR_CAPTURE"):
    """Captures current camera frame into the active learning review queue."""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service unavailable.")
    frame, _ = camera_service.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No active camera frame available.")

    from app.detection.data_capture import data_capture_manager
    success, fid = data_capture_manager.capture_candidate_frame(frame=frame, reason=reason, force=True)
    if not success:
        raise HTTPException(status_code=400, detail=f"Capture failed: {fid}")
    return {"success": True, "frame_id": fid}


@router.get("/lab/candidates")
def get_lab_candidates(limit: int = 50, status: Optional[str] = None):
    """Returns pending or reviewed candidate frames in active learning queue."""
    from app.detection.data_capture import data_capture_manager
    return {"candidates": data_capture_manager.get_candidate_queue(limit=limit, status=status)}


@router.post("/lab/review_candidate")
def review_lab_candidate(req: ReviewCandidateRequest):
    """Human-in-the-loop review of candidate frames (ACCEPT, REJECT, CORRECT)."""
    from app.detection.data_capture import data_capture_manager
    success = data_capture_manager.review_candidate(
        frame_id=req.frame_id,
        action=req.action,
        corrected_class=req.corrected_class,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Candidate frame not found.")
    return {"success": True, "frame_id": req.frame_id, "action": req.action}


@router.post("/lab/promote")
def promote_candidate_endpoint(req: PromoteCandidateRequest):
    """Promotes candidate model to production with gate verification."""
    from app.models.registry import model_registry
    success, msg = model_registry.promote_candidate(
        experiment_id=req.experiment_id,
        force=req.force,
        min_mAP50_improvement=req.min_mAP50_improvement,
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@router.post("/lab/rollback")
def rollback_model_endpoint():
    """Rolls back production model to previous stable version."""
    from app.models.registry import model_registry
    success, msg = model_registry.rollback_model()
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@router.get("/lab/report/{report_name}")
def get_lab_markdown_report(report_name: str):
    """Returns formatted markdown research reports (baseline, splits, readiness)."""
    valid_reports = {
        "baseline": paths.docs_dir / "vehicle_detection" / "BASELINE_AUDIT.md",
        "splits": paths.docs_dir / "vehicle_detection" / "DATASET_AND_SPLIT_AUDIT.md",
        "readiness": paths.docs_dir / "vehicle_detection" / "DEPLOYMENT_READINESS.md",
    }
    if report_name not in valid_reports:
        raise HTTPException(status_code=404, detail="Report not found.")
    target_p = valid_reports[report_name]
    if not target_p.exists():
        raise HTTPException(status_code=404, detail="Report file not generated yet.")
    return {"report_name": report_name, "content": target_p.read_text(encoding="utf-8")}

