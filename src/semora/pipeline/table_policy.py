"""Batch-scoped table exclusion receipts for derived retrieval text."""

from __future__ import annotations

import sqlite3

EXCLUDE_HTML_TABLES = "exclude_html_tables_v1"


def table_policy(connection: sqlite3.Connection, batch_id: str) -> str | None:
    """Return active policy; refuse reads of a partially repaired batch."""
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_batch_table_policies'"
    ).fetchone() is None:
        return None
    row = connection.execute(
        "SELECT policy,state FROM pipeline_batch_table_policies WHERE batch_id=?", (batch_id,)
    ).fetchone()
    if row is None:
        return None
    if row["state"] != "active":
        raise ValueError("Batch table repair is incomplete; finish it before indexing.")
    if row["policy"] != EXCLUDE_HTML_TABLES:
        raise ValueError("Unsupported batch table policy.")
    return row["policy"]
