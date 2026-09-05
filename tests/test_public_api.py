from __future__ import annotations

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
        )] == [1, 2, 3, 4, 5, 6, 7]
    finally:
        database.close()
