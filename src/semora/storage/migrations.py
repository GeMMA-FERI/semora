"""SQLite schema migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply every packaged migration that has not yet been recorded."""
    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        applied = {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for path in sorted(MIGRATIONS_DIR.glob("*.sql"), key=migration_version):
            version = migration_version(path)
            if version in applied:
                continue
            connection.executescript(path.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (version, path.name),
            )


def migration_version(path: Path) -> int:
    """Extract the numeric prefix from a migration filename."""
    prefix = path.name.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError as exc:
        raise ValueError(
            f"Migration file must start with a numeric prefix: {path.name}"
        ) from exc
