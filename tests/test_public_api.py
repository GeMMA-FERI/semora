from __future__ import annotations

import sqlite3

import pytest

from semora import Article, Database, Newspaper, Run


def test_public_storage_api_and_repositories() -> None:
    database = Database(":memory:")
    try:
        database.initialize()
        database.runs.add(Run(run_id="run-1", run_type="test"))
        database.documents.add_newspaper(
            Newspaper(
                newspaper_id="newspaper-1",
                run_id="run-1",
                content="source",
            )
        )
        database.documents.add_article(
            Article(
                article_id="article-1",
                run_id="run-1",
                newspaper_id="newspaper-1",
                title="Title",
                content="Text",
            )
        )
        database.documents.set_article_validity("article-1", is_valid=True)

        rows = database.documents.list_articles(valid_only=True)
        assert [row["article_id"] for row in rows] == ["article-1"]
        assert [row["version"] for row in database.conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        )] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    finally:
        database.close()


def test_read_only_database_allows_queries_but_not_writes_or_migrations(tmp_path) -> None:
    database_path = tmp_path / "semora.sqlite"
    writable = Database(database_path)
    try:
        writable.initialize()
        writable.insert_run(Run(run_id="run-1", run_type="test"))
    finally:
        writable.close()

    database = Database(database_path, read_only=True)
    try:
        assert database.read_only is True
        assert database.conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert database.conn.execute("SELECT run_type FROM runs").fetchone()[0] == "test"
        with pytest.raises(RuntimeError, match="read-only"):
            database.initialize()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            database.conn.execute("DELETE FROM runs")
    finally:
        database.close()


def test_read_only_database_does_not_create_a_missing_database(tmp_path) -> None:
    database_path = tmp_path / "missing" / "semora.sqlite"

    with pytest.raises(FileNotFoundError):
        Database(database_path, read_only=True)

    assert not database_path.parent.exists()
