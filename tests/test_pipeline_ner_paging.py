from __future__ import annotations

import sqlite3
from contextlib import closing

from semora.pipeline import ner
from tests.test_pipeline_ner import ingest


def test_pending_pages_use_bounded_article_key_scan(tmp_path):
    database = ingest(tmp_path, "batch", {
        "Paper/a.md": "First article.",
        "Paper/b.md": "Second article.",
        "Paper/c.md": "Third article.",
    })
    ingest(tmp_path, "other", {"Paper/outside.md": "Outside the batch."})
    query = (ner._ROWS + " AND s.article_id IS NULL AND a.article_id>? AND a.article_id<=? "
             "ORDER BY a.article_id LIMIT ?")
    with closing(sqlite3.connect(database)) as connection:
        plan = [row[3] for row in connection.execute(
            "EXPLAIN QUERY PLAN " + query, ("run", "batch", "", "~", 1),
        )]
        expected = [row[0] for row in connection.execute(
            "SELECT a.article_id FROM articles a JOIN pipeline_batch_issues b USING(newspaper_id) "
            "WHERE b.batch_id=? AND a.is_valid=1 AND TRIM(a.content)<>'' ORDER BY a.article_id", ("batch",),
        )]
        cursor = ""
        actual = []
        while True:
            rows = connection.execute(query, ("run", "batch", cursor, "~", 1)).fetchall()
            if not rows:
                break
            actual.extend(row[0] for row in rows)
            cursor = rows[-1][0]
    assert actual == expected and len(actual) == 3
    assert any("SEARCH a USING INDEX" in step and "(article_id>? AND article_id<?)" in step for step in plan), plan
    assert not any("USE TEMP B-TREE" in step for step in plan), plan
