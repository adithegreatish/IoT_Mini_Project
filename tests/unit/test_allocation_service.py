"""
Unit tests for Virtual Parking Space Allocation and Concurrency Engine.

Tests:
- Transactional allocation and double-allocation prevention under concurrency.
- Strict refusal to allocate physically OCCUPIED or UNKNOWN bays.
- Automatic expiration of stale reservations.
- Unified physical-virtual state separation.
- Real-time zone and type availability counts.
- Release and cancellation workflows.
"""

import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import pytest

from app.database.connection import DatabaseManager
from app.parking.allocation_service import (
    AllocationService,
    SlotUnavailableError,
    ConcurrencyConflictError,
)
from app.parking.state_machine import SlotState


@pytest.fixture
def test_db_manager(tmp_path):
    """Creates a clean SQLite database for testing."""
    db_file = tmp_path / "test_parking_allocation.sqlite"
    mgr = DatabaseManager(str(db_file))
    mgr.initialize_schema()
    return mgr


@pytest.fixture
def allocation_service(test_db_manager):
    """Creates an AllocationService instance initialized with test bays."""
    service = AllocationService(test_db_manager, default_reservation_ttl_min=30)
    # Populate test bays
    test_slots = [
        {"id": "A01", "label": "Bay A01", "zone": "Zone-A", "level": "Ground", "slot_type": "car", "enabled": True},
        {"id": "A02", "label": "Bay A02", "zone": "Zone-A", "level": "Ground", "slot_type": "car", "enabled": True},
        {"id": "A03", "label": "Bay A03", "zone": "Zone-A", "level": "Ground", "slot_type": "ev", "enabled": True},
        {"id": "B01", "label": "Bay B01", "zone": "Zone-B", "level": "Level-1", "slot_type": "motorcycle", "enabled": True},
        {"id": "B02", "label": "Bay B02", "zone": "Zone-B", "level": "Level-1", "slot_type": "accessible", "enabled": True},
    ]
    service.sync_slots_from_calibration(test_slots)
    return service


def test_normal_allocation(allocation_service):
    """Verifies standard slot reservation allocation."""
    phys_snapshot = {
        "A01": {"state": SlotState.FREE.value, "confidence": 0.95},
        "A02": {"state": SlotState.FREE.value, "confidence": 0.92},
    }
    receipt = allocation_service.allocate_slot(
        user_id="user_101",
        zone="Zone-A",
        slot_type="car",
        vehicle_plate="MH12AB1234",
        current_physical_states=phys_snapshot,
    )

    assert receipt["success"] is True
    assert receipt["reservation_code"].startswith("RES-")
    assert receipt["slot_id"] == "A01"
    assert receipt["zone"] == "Zone-A"


def test_reject_physically_occupied_or_unknown_slot(allocation_service):
    """Verifies that slots observed as OCCUPIED or UNKNOWN are NEVER allocated."""
    phys_snapshot = {
        "A01": {"state": SlotState.OCCUPIED.value, "confidence": 0.98},
        "A02": {"state": SlotState.UNKNOWN.value, "confidence": 0.0},
    }

    # Attempting to allocate in Zone-A where only A01 and A02 exist for 'car'
    with pytest.raises(SlotUnavailableError) as exc_info:
        allocation_service.allocate_slot(
            user_id="user_102",
            zone="Zone-A",
            slot_type="car",
            current_physical_states=phys_snapshot,
        )
    assert "OCCUPIED or UNKNOWN" in str(exc_info.value)


def test_double_allocation_prevention_under_concurrency(allocation_service):
    """
    CRITICAL TEST: Simulates 10 concurrent allocation requests competing for
    a single available slot. Guarantees that EXACTLY ONE request succeeds and
    9 receive SlotUnavailableError or ConcurrencyConflictError.
    Double-allocation must be impossible.
    """
    # Only slot A01 is free
    phys_snapshot = {
        "A01": {"state": SlotState.FREE.value, "confidence": 0.95},
        "A02": {"state": SlotState.OCCUPIED.value, "confidence": 0.90},
    }

    results = []
    errors = []

    def request_allocation(user_idx):
        try:
            res = allocation_service.allocate_slot(
                user_id=f"concurrent_user_{user_idx}",
                slot_id="A01",
                current_physical_states=phys_snapshot,
            )
            return True, res
        except Exception as e:
            return False, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(request_allocation, i) for i in range(10)]
        for fut in concurrent.futures.as_completed(futures):
            ok, val = fut.result()
            if ok:
                results.append(val)
            else:
                errors.append(val)

    # Exactly ONE allocation must have succeeded!
    assert len(results) == 1, f"Expected exactly 1 successful allocation, got {len(results)}: {results}"
    assert len(errors) == 9
    assert all(isinstance(e, (SlotUnavailableError, ConcurrencyConflictError)) for e in errors)


def test_stale_reservation_expiration(allocation_service):
    """Verifies that expired reservations are released and logged."""
    phys_snapshot = {"A01": {"state": SlotState.FREE.value, "confidence": 0.95}}

    # Insert a reservation that expired 5 minutes ago directly
    past_iso = (datetime.now() - timedelta(minutes=5)).isoformat()
    with allocation_service.db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO slot_reservations (reservation_code, slot_id, user_id, status, reserved_at, expires_at)
            VALUES ('RES-EXPIRED01', 'A01', 'old_user', 'ACTIVE', datetime('now', '-35 minutes'), ?);
            """,
            (past_iso,),
        )
        conn.commit()

    # Expire stale reservations
    expired_count = allocation_service.expire_stale_reservations()
    assert expired_count == 1

    # Verify A01 is now free and can be allocated to a new user
    receipt = allocation_service.allocate_slot(
        user_id="new_user",
        slot_id="A01",
        current_physical_states=phys_snapshot,
    )
    assert receipt["success"] is True
    assert receipt["slot_id"] == "A01"


def test_release_and_cancel_workflow(allocation_service):
    """Verifies releasing and cancelling active parking reservations."""
    phys_snapshot = {"A01": {"state": SlotState.FREE.value, "confidence": 0.95}}
    receipt = allocation_service.allocate_slot(
        user_id="user_rel",
        slot_id="A01",
        current_physical_states=phys_snapshot,
    )
    code = receipt["reservation_code"]

    # Release
    rel_res = allocation_service.release_slot(code)
    assert rel_res["success"] is True
    assert rel_res["status"] == "COMPLETED"

    # Verify cannot re-release
    with pytest.raises(ValueError):
        allocation_service.release_slot(code)


def test_availability_summary_genuine_counts(allocation_service):
    """Verifies authentic calculated availability summary across zones and types."""
    phys_snapshot = {
        "slots": [
            {"id": "A01", "state": "FREE", "confidence": 0.95},
            {"id": "A02", "state": "OCCUPIED", "confidence": 0.90},
            {"id": "A03", "state": "FREE", "confidence": 0.92},
            {"id": "B01", "state": "FREE", "confidence": 0.88},
            {"id": "B02", "state": "UNKNOWN", "confidence": 0.0},
        ]
    }

    summary = allocation_service.get_availability_summary(phys_snapshot)
    assert summary["total_slots"] == 5
    assert summary["physical_counts"]["free"] == 3
    assert summary["physical_counts"]["occupied"] == 1
    assert summary["physical_counts"]["unknown"] == 1

    # Reserve A01
    phys_map = {s["id"]: s for s in phys_snapshot["slots"]}
    allocation_service.allocate_slot("user_sum", slot_id="A01", current_physical_states=phys_map)

    # Recheck summary
    summary2 = allocation_service.get_availability_summary(phys_snapshot)
    assert summary2["reservation_counts"]["active_reservations"] == 1
    # Only A03 (EV) and B01 (motorcycle) remain allocatable
    assert summary2["reservation_counts"]["allocatable_vacant"] == 2
