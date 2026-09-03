"""SQLite connection configuration."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def open_connection(path: str | Path) -> sqlite3.Connection:
    """Open a Semora SQLite connection with required pragmas and row access."""
    path_string = str(path)
    if path_string == ":memory:":
        connection = sqlite3.connect(":memory:")
    else:
        database_path = Path(path_string)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
