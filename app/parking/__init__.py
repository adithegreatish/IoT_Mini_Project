"""Parking slot geometry, calibration, and occupancy state machine."""

from app.parking.geometry import (
    validate_slot_polygon,
    calculate_overlap_ratio,
    check_slot_overlaps,
    normalize_points,
    denormalize_points,
)
from app.parking.calibration_service import CalibrationService
from app.parking.state_machine import SlotState, SlotStateMachine
from app.parking.occupancy_service import OccupancyService

__all__ = [
    "validate_slot_polygon",
    "calculate_overlap_ratio",
    "check_slot_overlaps",
    "normalize_points",
    "denormalize_points",
    "CalibrationService",
    "SlotState",
    "SlotStateMachine",
    "OccupancyService",
]
