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

    def __init__(self, db_path: str = "data/parking_events.sqlite"):
        self.db_path = paths.resolve(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Creates and configures a new SQLite connection with WAL journal mode."""
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
        # High-performance RAM cache and memory-mapped I/O (up to 7GB buffer headroom)
        conn.execute("PRAGMA cache_size = -524288;")  # 512 MB memory cache
        conn.execute("PRAGMA mmap_size = 1073741824;") # 1 GB memory-mapped I/O
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
