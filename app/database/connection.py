"""SQLite connection management, WAL mode, and schema migrations."""

import logging
from pathlib import Path
import sqlite3
from typing import Optional
from app.infrastructure.paths import paths

logger = logging.getLogger("smart_parking.database")

SCHEMA_VERSION = 2


class DatabaseManager:
    """Manages thread-safe SQLite connection lifecycle and migrations."""

    def __init__(
        self,
        db_path: str = "data/parking_events.sqlite",
        cache_size_mb: int = 8,
        mmap_size_mb: int = 16,
    ):
        self.db_path = paths.resolve(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_size_mb = max(2, cache_size_mb)
        self.mmap_size_mb = max(0, mmap_size_mb)

    def get_connection(self) -> sqlite3.Connection:
        """Creates and configures a new SQLite connection with WAL journal mode and bounded memory."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        # WAL mode provides concurrent read performance and crash resilience
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        # Negative cache_size in SQLite requests approximate cache in KiB (-8192 = 8 MiB)
        cache_kib = self.cache_size_mb * 1024
        conn.execute(f"PRAGMA cache_size = -{cache_kib};")
        mmap_bytes = self.mmap_size_mb * 1024 * 1024
        conn.execute(f"PRAGMA mmap_size = {mmap_bytes};")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    def initialize_schema(self) -> None:
        """Applies schema migrations if not already up to date."""
        schema_file = Path(__file__).resolve().parent / "schema.sql"
        if not schema_file.exists():
            raise FileNotFoundError(f"Database schema file not found at '{schema_file}'")

        schema_sql = schema_file.read_text(encoding="utf-8")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(schema_sql)

            # Check applied migration version
            cursor.execute("SELECT MAX(version) FROM schema_migrations;")
            row = cursor.fetchone()
            current_version = row[0] if (row and row[0] is not None) else 0

            if current_version < SCHEMA_VERSION:
                cursor.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?);",
                    (SCHEMA_VERSION,),
                )
                conn.commit()
                logger.info(f"Database migrated to schema version {SCHEMA_VERSION}.")
            else:
                logger.debug(f"Database is currently at schema version {current_version}.")

    def vacuum(self) -> None:
        """Executes VACUUM to reclaim space on SD card or disk."""
        with self.get_connection() as conn:
            conn.execute("VACUUM;")
            logger.info("Database VACUUM completed.")
