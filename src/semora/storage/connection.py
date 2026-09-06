"""SQLite connection configuration."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def open_connection(path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a Semora SQLite connection with required pragmas and row access."""
    path_string = str(path)
    if path_string == ":memory:":
        if read_only:
            raise ValueError("An in-memory database cannot be opened read-only.")
        connection = sqlite3.connect(":memory:")
    else:
        database_path = Path(path_string)
        if read_only:
            database_uri = database_path.resolve(strict=True).as_uri() + "?mode=ro"
            connection = sqlite3.connect(database_uri, uri=True)
        else:
            database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA foreign_keys = ON")
    if read_only:
        connection.execute("PRAGMA query_only = ON")
    return connection
