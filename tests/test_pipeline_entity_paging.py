from __future__ import annotations

import sqlite3
from contextlib import closing

from semora.pipeline.entities import _ARTICLE_PAGE
from tests.test_pipeline_ner import ingest


def test_entity_pages_use_ordered_article_key_scan(tmp_path):
    database = ingest(tmp_path, "batch", {
        "Paper/a.md": "First article.",
        "Paper/b.md": "Second article.",
        "Paper/c.md": "Third article.",
    })
    ingest(tmp_path, "other", {"Paper/outside.md": "Outside the batch."})
    with closing(sqlite3.connect(database)) as connection:
        plan = [row[3] for row in connection.execute(
            "EXPLAIN QUERY PLAN " + _ARTICLE_PAGE, ("run", "batch", "", 1),
        )]
        expected = [row[0] for row in connection.execute(
            "SELECT a.article_id FROM articles a JOIN pipeline_batch_issues b USING(newspaper_id) "
            "WHERE b.batch_id=? AND a.is_valid=1 AND TRIM(a.content)<>'' ORDER BY a.article_id", ("batch",),
        )]
        cursor = ""
        actual = []
        while rows := connection.execute(_ARTICLE_PAGE, ("run", "batch", cursor, 1)).fetchall():
            actual.extend(row[0] for row in rows)
            cursor = rows[-1][0]
    assert actual == expected and len(actual) == 3
    assert any("SEARCH a USING INDEX" in step and "(article_id>?)" in step for step in plan), plan
    assert not any("USE TEMP B-TREE" in step for step in plan), plan
