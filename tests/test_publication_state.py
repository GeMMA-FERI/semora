"""The serving revision must become stale before a failed index write."""

import sqlite3

import pytest

from semora.pipeline.publication_state import invalidate_publication


def test_invalidation_survives_a_later_failed_output_transaction(tmp_path):
    path = tmp_path / "corpus.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE pipeline_publication_state (state_id INTEGER PRIMARY KEY, epoch INTEGER NOT NULL);"
            "INSERT INTO pipeline_publication_state VALUES (1, 7);"
            "CREATE TABLE output (value TEXT NOT NULL);"
        )

    assert invalidate_publication(path) == 8
    with pytest.raises(RuntimeError), sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO output VALUES ('unpublished')")
        raise RuntimeError("simulated interruption")

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT epoch FROM pipeline_publication_state").fetchone()[0] == 8
        assert connection.execute("SELECT COUNT(*) FROM output").fetchone()[0] == 0


def test_legacy_archive_without_publication_state_is_unchanged(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE old_data (value TEXT)")
    assert invalidate_publication(path) is None
