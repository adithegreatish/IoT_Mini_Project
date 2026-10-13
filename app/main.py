"""Main application entry point for the Smart Parking & HSRP Management System."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
import time
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api import routes
from app.config.loader import load_config, get_config
from app.config.schema import RootConfig
from app.database.connection import DatabaseManager
from app.database.repository import ParkingRepository
from app.detection.vehicle_detector import VehicleDetector
from app.infrastructure.logging_setup import setup_logging
from app.infrastructure.paths import paths
from app.parking.calibration_service import CalibrationService
from app.parking.occupancy_service import OccupancyService
from app.parking.allocation_service import AllocationService
from app.services.camera_service import CameraService

logger = logging.getLogger("smart_parking.main")

_inference_running = False
_inference_thread = None


def _background_inference_loop(cam_service: CameraService, occ_service: OccupancyService, interval_ms: int):
    """Background processing loop running detection and occupancy updates."""
    interval = max(0.02, interval_ms / 1000.0)
    logger.info(f"Background inference worker started (cadence: {interval_ms} ms).")
    while _inference_running:
        loop_start = time.time()
        if cam_service.processing_active:
            frame, meta = cam_service.get_latest_frame()
            is_healthy = cam_service._provider.is_healthy() if cam_service._provider else False
            is_stale = meta.is_stale if meta else True
            occ_service.process_frame(frame, is_camera_healthy=is_healthy, is_camera_stale=is_stale)

        elapsed = time.time() - loop_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and graceful shutdown."""
    global _inference_running, _inference_thread

    cfg = get_config()
    setup_logging(cfg.logging, cfg.privacy)
    paths.ensure_directories()
    paths.log_paths()

    logger.info("=" * 60)
    logger.info(f"Starting {cfg.app.name} v{cfg.app.version} [{cfg.app.env}]")
    logger.info(f"Host: {cfg.app.host}:{cfg.app.port}")
    logger.info(f"Root: {paths.root}")
    logger.info("=" * 60)

    # 1. Database
    db_mgr = DatabaseManager(cfg.database.db_path)
    db_mgr.initialize_schema()
    repo = ParkingRepository(db_mgr, allow_raw_retention=cfg.privacy.allow_raw_retention)
    routes.repository = repo

    # 2. Calibration & Geometry
    calib_service = CalibrationService(cfg.parking.calibration_file, db_manager=db_mgr)
    routes.calibration_service = calib_service

    # 2b. Virtual Parking Allocation & Concurrency
    alloc_service = AllocationService(db_mgr)
    alloc_service.sync_slots_from_calibration(calib_service.load_calibration().get("slots", []))
    routes.allocation_service = alloc_service

    # 3. Vehicle Detector
    detector = VehicleDetector(cfg.detection)
    routes.detector = detector

    # 4. Occupancy Engine
    occ_service = OccupancyService(cfg.parking, calib_service, detector, repository=repo)
    routes.occupancy_service = occ_service

    # 5. Camera Capture Worker
    cam_service = CameraService(cfg.camera)
    cam_service.start()
    routes.camera_service = cam_service

    # 6. Start Background Inference Thread
    _inference_running = True
    _inference_thread = threading.Thread(
        target=_background_inference_loop,
        args=(cam_service, occ_service, cfg.detection.inference_interval_ms),
        daemon=True,
        name="InferenceWorker",
    )
    _inference_thread.start()

    yield

    # Clean shutdown
    logger.info("Shutting down application services...")
    _inference_running = False
    if _inference_thread and _inference_thread.is_alive():
        _inference_thread.join(timeout=1.0)
    if routes.camera_service:
        routes.camera_service.stop()
        routes.camera_service = None
    logger.info("Shutdown complete.")


def create_app(config: RootConfig = None) -> FastAPI:
    """Creates and configures FastAPI application instance."""
    app = FastAPI(
        title="Smart Parking & HSRP System",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Mount API routes
    app.include_router(routes.router, prefix="/api")

    # Mount Static Web Assets
    static_dir = paths.resolve("web/static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Serve Root Technical Dashboard
    @app.get("/")
    async def serve_index():
        index_file = paths.resolve("web/templates/index.html")
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Smart Parking & HSRP System Active"}

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Smart Parking Management System")
    parser.add_argument("--config", type=str, default="configs/windows.yaml", help="Path to YAML configuration profile")
    args = parser.parse_args()

    cfg = load_config(args.config)
    uvicorn.run(
        "app.main:app",
        host=cfg.app.host,
        port=cfg.app.port,
        reload=False,
        log_level=cfg.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
