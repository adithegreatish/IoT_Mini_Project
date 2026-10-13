-- Database Schema for Smart Parking & HSRP Management System
-- Schema Version: 1

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS calibration_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    reference_width INTEGER NOT NULL,
    reference_height INTEGER NOT NULL,
    camera_fingerprint TEXT,
    slots_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS occupancy_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id TEXT NOT NULL,
    previous_state TEXT NOT NULL,
    new_state TEXT NOT NULL,
    confidence REAL NOT NULL,
    track_id TEXT,
    model_version TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_occupancy_slot_time ON occupancy_events(slot_id, timestamp);

CREATE TABLE IF NOT EXISTS plate_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id TEXT,
    raw_plate TEXT,
    masked_plate TEXT NOT NULL,
    confidence REAL NOT NULL,
    crop_quality TEXT,
    hsrp_outcome TEXT,
    visible_cues_json TEXT,
    ocr_engine TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_plate_events_time ON plate_events(timestamp);

CREATE TABLE IF NOT EXISTS diagnostic_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_sec REAL NOT NULL,
    total_tests INTEGER NOT NULL,
    passed_tests INTEGER NOT NULL,
    failed_tests INTEGER NOT NULL,
    report_path TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Virtual Parking Allocation & Concurrency Schema (Version 2)
CREATE TABLE IF NOT EXISTS parking_slots (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    zone TEXT DEFAULT 'Zone-A',
    level TEXT DEFAULT 'Ground',
    slot_type TEXT DEFAULT 'car',
    camera_id TEXT DEFAULT 'cam01',
    is_enabled INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS slot_reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reservation_code TEXT UNIQUE NOT NULL,
    slot_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    vehicle_plate TEXT,
    vehicle_type TEXT DEFAULT 'car',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    reserved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    checked_in_at TIMESTAMP,
    released_at TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (slot_id) REFERENCES parking_slots(id)
);

CREATE INDEX IF NOT EXISTS idx_reservations_slot ON slot_reservations(slot_id, status);
CREATE INDEX IF NOT EXISTS idx_reservations_code ON slot_reservations(reservation_code);

CREATE TABLE IF NOT EXISTS allocation_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    slot_id TEXT NOT NULL,
    reservation_code TEXT,
    physical_state TEXT,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_allocation_audit_time ON allocation_audit_log(timestamp);

