"""Canonical source identity across historical adoption and later batches."""

from __future__ import annotations

import json
import sqlite3


def has_identity_registry(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_source_identities'"
    ).fetchone() is not None


def metadata_identifiers(metadata: dict) -> set[str]:
    record = metadata.get("Record", {})
    values = record.get("identifier", []) if isinstance(record, dict) else []
    values = values if isinstance(values, list) else [values]
    identifiers = set()
    for value in values:
        text = value.get("#text") if isinstance(value, dict) else value
        if isinstance(text, str) and text.casefold().startswith(("urn:", "gf-urn:")):
            identifiers.add(text.casefold())
    return identifiers


def identity_owners(conn: sqlite3.Connection | None) -> dict[str, set[str]]:
    """Read registered identities, with a read-only fallback for legacy issues."""
    owners: dict[str, set[str]] = {}
    if conn is None:
        return owners
    registered = has_identity_registry(conn)
    if registered:
        for row in conn.execute("SELECT newspaper_id,identifier_key FROM pipeline_source_identities"):
            if row["identifier_key"] is not None:
                owners.setdefault(row["identifier_key"], set()).add(row["newspaper_id"])
    missing = (
        " WHERE NOT EXISTS (SELECT 1 FROM pipeline_source_identities s WHERE s.newspaper_id=n.newspaper_id)"
        if registered else ""
    )
    for row in conn.execute("SELECT n.newspaper_id,n.metadata_json,n.urn FROM newspapers n" + missing):
        metadata = json.loads(row["metadata_json"] or "{}")
        if not isinstance(metadata, dict):
            raise ValueError("Stored issue metadata must be a JSON object.")
        keys = metadata_identifiers(metadata)
        if row["urn"]:
            keys.add(row["urn"].casefold())
        for key in keys:
            owners.setdefault(key, set()).add(row["newspaper_id"])
    ledger = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_batch_issues'"
    ).fetchone()
    if ledger:
        for row in conn.execute(
            "SELECT newspaper_id,source_identifier_key FROM pipeline_batch_issues "
            "WHERE source_identifier_key IS NOT NULL"
        ):
            owners.setdefault(row["source_identifier_key"], set()).add(row["newspaper_id"])
    return owners


def register_identity(conn: sqlite3.Connection, newspaper_id: str, identifier: str | None) -> None:
    """Store identity inside the caller's corpus/adoption transaction."""
    if not conn.in_transaction:
        raise ValueError("Identity registration requires an active transaction.")
    key = identifier.casefold() if identifier else None
    previous = conn.execute(
        "SELECT identifier_key FROM pipeline_source_identities WHERE newspaper_id=?", (newspaper_id,)
    ).fetchone()
    if previous and previous[0] is not None:
        if key is not None and previous[0] != key:
            raise ValueError("Existing source identity changed; explicit identity repair is required.")
        return
    if previous:
        conn.execute(
            "UPDATE pipeline_source_identities SET identifier_key=?,source_identifier=? WHERE newspaper_id=?",
            (key, identifier, newspaper_id),
        )
    else:
        conn.execute(
            "INSERT INTO pipeline_source_identities(newspaper_id,identifier_key,source_identifier) VALUES (?,?,?)",
            (newspaper_id, key, identifier),
        )
