"""Prevent legacy builders from accepting an unpublished additive corpus."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from semora.storage.connection import open_connection


def reject_pipeline_batches(database_path: str | Path) -> None:
    """Inspect an existing database without creating or migrating it.

    Phase-one batches populate the corpus only. Legacy index completion and
    snapshot rules cannot establish coverage for these appended records.
    """
    if str(database_path) == ":memory:":
        return
    path = Path(database_path)
    if not path.is_file():
        return
    with closing(open_connection(path, read_only=True)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pipeline_batches'"
        ).fetchone()
        if exists is None:
            return
        if connection.execute("SELECT 1 FROM pipeline_batches LIMIT 1").fetchone() is None:
            return
    raise ValueError(
        "This corpus contains additive pipeline batches and is not ready for legacy indexing. "
        "The current pipeline ingests corpus records only; additive retrieval stages and "
        "publication are not implemented yet. Keep retrieval readers stopped, retain your "
        "backup, and finish the additive index migration before building or publishing indexes."
    )
