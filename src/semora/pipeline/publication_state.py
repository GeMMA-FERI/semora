"""Invalidate a serving pointer before an additive writer changes stored data."""

from __future__ import annotations

from pathlib import Path

from semora.storage.connection import open_connection


def invalidate_publication(database_path: str | Path) -> int | None:
    """Commit an epoch advance before a separate output transaction begins.

    Older archives without migration 019 have no serving pointer and need no
    invalidation. Call under the pipeline maintenance lock, outside any other
    SQLite transaction. Extra advances are safe; they require republishing.
    """
    connection = open_connection(database_path)
    try:
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_publication_state'"
        ).fetchone()
        if present is None:
            return None
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE pipeline_publication_state SET epoch=epoch+1 WHERE state_id=1")
            row = connection.execute("SELECT epoch FROM pipeline_publication_state WHERE state_id=1").fetchone()
            if row is None:
                raise ValueError("Publication state row is missing.")
            return int(row[0])
    finally:
        connection.close()
