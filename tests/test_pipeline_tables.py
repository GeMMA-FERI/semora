from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing

from semora.pipeline.batches import run_batch
from semora.pipeline.lexical import index_batch_lexical
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.pipeline.tables import repair_batch_tables, table_repair_plan


class WordTokenizer:
    def __call__(self, text, **_kwargs):
        import re

        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


def test_repair_only_table_article_and_resume(tmp_path):
    root = tmp_path / "incoming"
    root.mkdir()
    content = (
        "# Table article\nBefore table. <table><tr><td>secretcell</td></tr></table> After table.\n"
        "# Plain article\nPlain text stays.\n"
    )
    markdown = root / "one.md"
    markdown.write_text(content, encoding="utf-8")
    metadata = root / "one_metadata.json"
    metadata.write_text(json.dumps({"Record": {"source": "Test", "date": "1925 02 18"}}), encoding="utf-8")
    manifest = BatchManifest("batch-a", (BatchIssue(
        relative_path="one.md", markdown_path="one.md", metadata_path="one_metadata.json",
        markdown_sha256=hashlib.sha256(markdown.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
    ),))
    database = tmp_path / "semora.sqlite"
    run_batch(manifest, root, database, exclude_tables=False, tokenizer=WordTokenizer())
    index_batch_lexical(database, "batch-a")
    with closing(sqlite3.connect(database)) as conn:
        before = conn.execute(
            "SELECT article_id,COUNT(*) FROM chunks GROUP BY article_id ORDER BY article_id"
        ).fetchall()
        plain_id = conn.execute("SELECT article_id FROM articles WHERE title='Plain article'").fetchone()[0]
    assert table_repair_plan(database, "batch-a")["table_articles"] == 1
    result = repair_batch_tables(database, "batch-a", tokenizer=WordTokenizer())
    assert result["repaired_articles"] == 1
    with closing(sqlite3.connect(database)) as conn:
        after = conn.execute(
            "SELECT article_id,COUNT(*) FROM chunks GROUP BY article_id ORDER BY article_id"
        ).fetchall()
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%secretcell%'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM article_fts WHERE article_fts MATCH 'secretcell'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pipeline_article_table_repairs").fetchone()[0] == 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert dict(before)[plain_id] == dict(after)[plain_id]
    assert index_batch_lexical(database, "batch-a")["added_articles"] == 0
