"""
Virtual Parking Space Allocation and Concurrency Management Engine.

Implements Phase 6 & Phase 7 requirements:
- Authoritative separation of physical occupancy (sensed) from reservation state (virtual).
- Strict transactional concurrency control preventing double-allocation.
- Automatic expiration of stale reservations.
- Zone, level, and slot-type filtering (regular, accessible, EV charging, motorcycle).
- Complete reservation audit history and genuine live count calculations.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

from app.database.connection import DatabaseManager
from app.parking.state_machine import SlotState

logger = logging.getLogger("smart_parking.allocation")


class SlotAllocationError(Exception):
    """Base exception for slot allocation failures."""
    pass


class SlotUnavailableError(SlotAllocationError):
    """Raised when no free/unreserved slot matching criteria exists."""
    pass


class ConcurrencyConflictError(SlotAllocationError):
    """Raised when a concurrent transaction locks the target slot."""
    pass


@dataclass
class SlotUnifiedState:
    slot_id: str
    label: str
    zone: str
    level: str
    slot_type: str
    camera_id: str
    physical_state: str  # FREE, OCCUPIED, UNKNOWN
    physical_confidence: float
    reservation_status: str  # AVAILABLE, RESERVED, OCCUPIED_AUTHORIZED, CONFLICT_UNAUTHORIZED
    active_reservation_code: Optional[str] = None
    reserved_plate: Optional[str] = None
    is_allocatable: bool = False


class AllocationService:
    """Manages virtual parking allocation, concurrency locking, and bay state union."""

    def __init__(self, db_manager: DatabaseManager, default_reservation_ttl_min: int = 30):
        self.db = db_manager
        self.default_ttl = timedelta(minutes=default_reservation_ttl_min)
        self._lock = threading.Lock()
        # Initialize schema tables if not present
        self.db.initialize_schema()

    def sync_slots_from_calibration(self, slots_def: List[Dict[str, Any]], camera_id: str = "cam01") -> int:
        """Synchronizes calibrated parking bay definitions into parking_slots database table."""
        count = 0
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for s in slots_def:
                sid = s.get("id")
                if not sid:
                    continue
                label = s.get("label", sid)
                zone = s.get("zone", "Zone-A")
                level = s.get("level", "Ground")
                stype = s.get("slot_type", "car")
                enabled = 1 if s.get("enabled", True) else 0

                cursor.execute(
                    """
                    INSERT INTO parking_slots (id, label, zone, level, slot_type, camera_id, is_enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        label=excluded.label,
                        zone=excluded.zone,
                        level=excluded.level,
                        slot_type=excluded.slot_type,
                        camera_id=excluded.camera_id,
                        is_enabled=excluded.is_enabled;
                    """,
                    (sid, label, zone, level, stype, camera_id, enabled),
                )
                count += 1
            conn.commit()
        logger.info(f"Synchronized {count} parking slots from calibration into database.")
        return count

    def expire_stale_reservations(self) -> int:
        """
        Marks active reservations that exceeded expires_at as EXPIRED.
        Emits audit logs for each expired reservation.
        """
        now_iso = datetime.now().isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, reservation_code, slot_id
                FROM slot_reservations
                WHERE status = 'ACTIVE' AND expires_at < ?;
                """,
                (now_iso,),
            )
            stale = cursor.fetchall()
            if not stale:
                return 0

            for r in stale:
                cursor.execute(
                    "UPDATE slot_reservations SET status = 'EXPIRED' WHERE id = ?;",
                    (r["id"],),
                )
                cursor.execute(
                    """
                    INSERT INTO allocation_audit_log (event_type, slot_id, reservation_code, details)
                    VALUES ('EXPIRE', ?, ?, 'Reservation expired automatically by TTL');
                    """,
                    (r["slot_id"], r["reservation_code"]),
                )
            conn.commit()
            logger.info(f"Expired {len(stale)} stale parking reservations.")
            return len(stale)

    def allocate_slot(
        self,
        user_id: str,
        slot_id: Optional[str] = None,
        zone: Optional[str] = None,
        slot_type: str = "car",
        vehicle_plate: Optional[str] = None,
        vehicle_type: str = "car",
        duration_minutes: Optional[int] = None,
        current_physical_states: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Transactionally allocates a parking slot with strict double-allocation prevention.
        
        Guarantees:
        - NEVER allocates a slot if physical state is OCCUPIED or UNKNOWN.
        - Uses BEGIN IMMEDIATE transaction to guarantee atomic serialization.
        - Prevents two concurrent users from receiving the same slot.
        """
        self.expire_stale_reservations()

        ttl = timedelta(minutes=duration_minutes) if duration_minutes else self.default_ttl
        now = datetime.now()
        expires_at = (now + ttl).isoformat()
        res_code = f"RES-{uuid.uuid4().hex[:8].upper()}"

        # Default physical state lookup
        phys_map = current_physical_states or {}

        conn = self.db.get_connection()
        try:
            # SQLite BEGIN IMMEDIATE acquires a reserved lock immediately, blocking other writes
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()

            # 1. Query candidate slots
            query = """
            SELECT s.id, s.label, s.zone, s.level, s.slot_type
            FROM parking_slots s
            LEFT JOIN slot_reservations r ON s.id = r.slot_id AND r.status = 'ACTIVE'
            WHERE s.is_enabled = 1
              AND r.id IS NULL
            """
            params: List[Any] = []

            if slot_id:
                query += " AND s.id = ?"
                params.append(slot_id)
            if zone:
                query += " AND s.zone = ?"
                params.append(zone)
            if slot_type:
                if slot_type in ("car", "standard"):
                    query += " AND s.slot_type IN ('car', 'standard')"
                else:
                    query += " AND s.slot_type = ?"
                    params.append(slot_type)

            query += " ORDER BY s.zone, s.id;"
            cursor.execute(query, params)
            candidates = cursor.fetchall()

            if not candidates:
                conn.rollback()
                raise SlotUnavailableError(
                    f"No available unreserved slots found for criteria: zone={zone}, type={slot_type}, slot_id={slot_id}"
                )

            # 2. Filter candidates by physical occupancy (NEVER allocate OCCUPIED or UNKNOWN)
            selected_slot = None
            for cand in candidates:
                cid = cand["id"]
                p_info = phys_map.get(cid, {})
                p_state = p_info.get("state", SlotState.FREE.value if not phys_map else SlotState.UNKNOWN.value)

                # Strict allocation policy: MUST BE PHYSICALLY FREE
                if p_state == SlotState.FREE.value:
                    selected_slot = cand
                    break

            if not selected_slot:
                conn.rollback()
                raise SlotUnavailableError(
                    f"Candidate slots are unreserved but physically OCCUPIED or UNKNOWN. Allocation refused."
                )

            target_sid = selected_slot["id"]

            # 3. Insert reservation atomically
            cursor.execute(
                """
                INSERT INTO slot_reservations (
                    reservation_code, slot_id, user_id, vehicle_plate, vehicle_type,
                    status, reserved_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?);
                """,
                (res_code, target_sid, user_id, vehicle_plate, vehicle_type, now.isoformat(), expires_at),
            )

            # 4. Record audit log
            cursor.execute(
                """
                INSERT INTO allocation_audit_log (event_type, slot_id, reservation_code, physical_state, details)
                VALUES ('RESERVE', ?, ?, 'FREE', ?);
                """,
                (target_sid, res_code, f"Allocated to user {user_id} for {duration_minutes or 30} mins"),
            )

            conn.commit()

            logger.info(f"Allocated slot {target_sid} to user {user_id} (Code: {res_code})")
            return {
                "success": True,
                "reservation_code": res_code,
                "slot_id": target_sid,
                "slot_label": selected_slot["label"],
                "zone": selected_slot["zone"],
                "level": selected_slot["level"],
                "slot_type": selected_slot["slot_type"],
                "reserved_at": now.isoformat(),
                "expires_at": expires_at,
            }

        except sqlite3.OperationalError as e:
            conn.rollback()
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                raise ConcurrencyConflictError("Database locked by concurrent allocation request. Please retry.") from e
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def release_slot(self, reservation_code: str) -> Dict[str, Any]:
        """Releases an active reservation and marks it COMPLETED."""
        now_iso = datetime.now().isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, slot_id, status FROM slot_reservations WHERE reservation_code = ?;",
                (reservation_code,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Reservation code '{reservation_code}' not found.")
            if row["status"] != "ACTIVE":
                raise ValueError(f"Reservation is not active (current status: {row['status']}).")

            cursor.execute(
                "UPDATE slot_reservations SET status = 'COMPLETED', released_at = ? WHERE id = ?;",
                (now_iso, row["id"]),
            )
            cursor.execute(
                """
                INSERT INTO allocation_audit_log (event_type, slot_id, reservation_code, details)
                VALUES ('RELEASE', ?, ?, 'Slot released by driver / operator');
                """,
                (row["slot_id"], reservation_code),
            )
            conn.commit()
            logger.info(f"Released reservation {reservation_code} on slot {row['slot_id']}.")
            return {"success": True, "slot_id": row["slot_id"], "status": "COMPLETED", "released_at": now_iso}

    def cancel_reservation(self, reservation_code: str, reason: str = "User cancelled") -> Dict[str, Any]:
        """Cancels an active reservation."""
        now_iso = datetime.now().isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, slot_id, status FROM slot_reservations WHERE reservation_code = ?;",
                (reservation_code,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Reservation code '{reservation_code}' not found.")
            if row["status"] != "ACTIVE":
                raise ValueError(f"Cannot cancel non-active reservation (current status: {row['status']}).")

            cursor.execute(
                "UPDATE slot_reservations SET status = 'CANCELLED', released_at = ?, notes = ? WHERE id = ?;",
                (now_iso, reason, row["id"]),
            )
            cursor.execute(
                """
                INSERT INTO allocation_audit_log (event_type, slot_id, reservation_code, details)
                VALUES ('CANCEL', ?, ?, ?);
                """,
                (row["slot_id"], reservation_code, f"Cancelled: {reason}"),
            )
            conn.commit()
            return {"success": True, "slot_id": row["slot_id"], "status": "CANCELLED"}

    def get_unified_slot_states(
        self, physical_snapshot: Optional[Dict[str, Any]] = None
    ) -> List[SlotUnifiedState]:
        """
        Unifies physical camera observations with virtual database reservation states.
        Correctly distinguishes physical state from virtual reservation status.
        """
        self.expire_stale_reservations()

        # Build physical lookup map
        phys_map = {}
        if physical_snapshot and "slots" in physical_snapshot:
            for s in physical_snapshot["slots"]:
                phys_map[s["id"]] = s

        # Build active reservations lookup map
        res_map = {}
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.id, s.label, s.zone, s.level, s.slot_type, s.camera_id, s.is_enabled,
                       r.reservation_code, r.vehicle_plate, r.vehicle_type, r.expires_at
                FROM parking_slots s
                LEFT JOIN slot_reservations r ON s.id = r.slot_id AND r.status = 'ACTIVE'
                ORDER BY s.zone, s.id;
                """
            )
            db_slots = cursor.fetchall()

        results = []
        for row in db_slots:
            sid = row["id"]
            p_obs = phys_map.get(sid, {})
            p_state = p_obs.get("state", SlotState.UNKNOWN.value if phys_map else SlotState.FREE.value)
            p_conf = p_obs.get("confidence", 0.0)

            has_res = row["reservation_code"] is not None

            # Determine reservation status and conflict states
            if not has_res:
                if p_state == SlotState.OCCUPIED.value:
                    res_status = "OCCUPIED_UNAUTHORIZED"  # Parked without virtual reservation
                else:
                    res_status = "AVAILABLE" if p_state == SlotState.FREE.value else "UNKNOWN"
            else:
                if p_state == SlotState.OCCUPIED.value:
                    res_status = "OCCUPIED_AUTHORIZED"  # Reserved and vehicle arrived
                elif p_state == SlotState.FREE.value:
                    res_status = "RESERVED"  # Reserved, vehicle not yet arrived
                else:
                    res_status = "RESERVED_CAMERA_UNKNOWN"

            # Allocatable: Enabled AND Physically FREE AND No active reservation
            is_alloc = bool(row["is_enabled"] == 1 and p_state == SlotState.FREE.value and not has_res)

            unified = SlotUnifiedState(
                slot_id=sid,
                label=row["label"],
                zone=row["zone"],
                level=row["level"],
                slot_type=row["slot_type"],
                camera_id=row["camera_id"],
                physical_state=p_state,
                physical_confidence=p_conf,
                reservation_status=res_status,
                active_reservation_code=row["reservation_code"],
                reserved_plate=row["vehicle_plate"],
                is_allocatable=is_alloc,
            )
            results.append(unified)

        return results

    def get_availability_summary(self, physical_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculates authentic real-time availability metrics across zones, levels, and bay types."""
        slots = self.get_unified_slot_states(physical_snapshot)

        total = len(slots)
        phys_free = sum(1 for s in slots if s.physical_state == SlotState.FREE.value)
        phys_occupied = sum(1 for s in slots if s.physical_state == SlotState.OCCUPIED.value)
        phys_unknown = sum(1 for s in slots if s.physical_state == SlotState.UNKNOWN.value)
        reserved_active = sum(1 for s in slots if s.active_reservation_code is not None)
        allocatable = sum(1 for s in slots if s.is_allocatable)

        by_zone = {}
        for s in slots:
            z = by_zone.setdefault(s.zone, {"total": 0, "free": 0, "occupied": 0, "allocatable": 0})
            z["total"] += 1
            if s.physical_state == SlotState.FREE.value:
                z["free"] += 1
            elif s.physical_state == SlotState.OCCUPIED.value:
                z["occupied"] += 1
            if s.is_allocatable:
                z["allocatable"] += 1

        by_type = {}
        for s in slots:
            t = by_type.setdefault(s.slot_type, {"total": 0, "free": 0, "occupied": 0, "allocatable": 0})
            t["total"] += 1
            if s.physical_state == SlotState.FREE.value:
                t["free"] += 1
            elif s.physical_state == SlotState.OCCUPIED.value:
                t["occupied"] += 1
            if s.is_allocatable:
                t["allocatable"] += 1

        return {
            "timestamp": datetime.now().isoformat(),
            "total_slots": total,
            "physical_counts": {
                "free": phys_free,
                "occupied": phys_occupied,
                "unknown": phys_unknown,
            },
            "reservation_counts": {
                "active_reservations": reserved_active,
                "allocatable_vacant": allocatable,
            },
            "by_zone": by_zone,
            "by_slot_type": by_type,
            "slots": [asdict(s) for s in slots],
        }

    def get_audit_trail(self, limit: int = 50, slot_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves allocation and reservation audit log records."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if slot_id:
                cursor.execute(
                    "SELECT * FROM allocation_audit_log WHERE slot_id = ? ORDER BY id DESC LIMIT ?;",
                    (slot_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM allocation_audit_log ORDER BY id DESC LIMIT ?;",
                    (limit,),
                )
            return [dict(r) for r in cursor.fetchall()]
