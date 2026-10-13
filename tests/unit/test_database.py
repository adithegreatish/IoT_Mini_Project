"""Unit tests for SQLite database connection and repository."""

from pathlib import Path

from app.database.connection import DatabaseManager, SCHEMA_VERSION
from app.database.repository import ParkingRepository


def test_database_initialization_and_migration(tmp_path: Path):
    db_file = tmp_path / "test.sqlite"
    mgr = DatabaseManager(str(db_file))
    mgr.initialize_schema()

    assert db_file.exists()
    with mgr.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_migrations;")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == SCHEMA_VERSION


def test_repository_occupancy_event_insertion(tmp_path: Path):
    db_file = tmp_path / "test.sqlite"
    mgr = DatabaseManager(str(db_file))
    mgr.initialize_schema()
    repo = ParkingRepository(mgr)

    # First transition FREE -> OCCUPIED
    eid = repo.record_occupancy_event(
        slot_id="S01",
        previous_state="FREE",
        new_state="OCCUPIED",
        confidence=0.88,
        track_id="trk-1",
        model_version="v1",
    )
    assert eid > 0

    # Redundant identical transition should be debounced
    eid_dup = repo.record_occupancy_event(
        slot_id="S01",
        previous_state="OCCUPIED",
        new_state="OCCUPIED",
        confidence=0.88,
    )
    assert eid_dup == 0

    events = repo.get_recent_occupancy_events(limit=10)
    assert len(events) == 1
    assert events[0]["slot_id"] == "S01"
    assert events[0]["new_state"] == "OCCUPIED"


def test_repository_plate_event_masking(tmp_path: Path):
    db_file = tmp_path / "test.sqlite"
    mgr = DatabaseManager(str(db_file))
    mgr.initialize_schema()
    repo = ParkingRepository(mgr, allow_raw_retention=False)

    repo.record_plate_event(
        slot_id="S01",
        raw_plate="DL01CA9999",
        confidence=0.92,
        crop_quality="EXCELLENT",
        hsrp_outcome="HSRP_STYLE_CONSISTENT",
        visible_cues={"ind_region": True, "aspect_ratio": 4.2},
    )

    plates = repo.get_recent_plate_events(limit=10)
    assert len(plates) == 1
    assert plates[0]["raw_plate"] is None  # raw retention false
    assert plates[0]["masked_plate"] == "DL01****99"
    assert plates[0]["visible_cues"]["ind_region"] is True
