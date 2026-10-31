"""Plan and repair table-bearing articles in one already ingested batch."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

from semora.corpus import indexer
from semora.corpus.spans import LineMap, SourceArticle, exclude_html_tables
from semora.pipeline.lexical import repair_table_lexical
from semora.pipeline.publication_state import invalidate_publication
from semora.pipeline.table_policy import EXCLUDE_HTML_TABLES
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database
from semora.storage.connection import open_connection


def _candidates(connection: sqlite3.Connection, batch_id: str) -> list[sqlite3.Row]:
    rows = connection.execute(
        "SELECT a.article_id,a.newspaper_id,a.title,a.content,a.run_id,a.char_start,a.char_end,"
        "a.line_start,a.line_end FROM articles a "
        "JOIN pipeline_batch_issues b ON b.newspaper_id=a.newspaper_id "
        "WHERE b.batch_id=? AND a.is_valid=1 AND instr(lower(a.content),'table')>0 "
        "ORDER BY a.newspaper_id,a.article_id", (batch_id,),
    )
    return [row for row in rows if exclude_html_tables(row["content"]) != row["content"]]


def _guard_downstream(
    connection: sqlite3.Connection, batch_id: str, semantic_path: Path, candidates: list[sqlite3.Row],
) -> None:
    for table, join in (
        ("embeddings", "JOIN chunks c ON c.chunk_id=x.chunk_id JOIN articles a ON a.article_id=c.article_id"),
        ("chunk_fts_map", "JOIN chunks c ON c.chunk_id=x.chunk_id JOIN articles a ON a.article_id=c.article_id"),
        (
            "openai_embedding_batch_items",
            "JOIN chunks c ON c.chunk_id=x.chunk_id JOIN articles a ON a.article_id=c.article_id",
        ),
        ("article_ner_state", "JOIN articles a ON a.article_id=x.article_id"),
        ("article_entities", "JOIN articles a ON a.article_id=x.article_id"),
    ):
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            if connection.execute(
                f"SELECT 1 FROM {table} x {join} JOIN pipeline_batch_issues b ON b.newspaper_id=a.newspaper_id "
                "WHERE b.batch_id=? LIMIT 1", (batch_id,),
            ).fetchone():
                raise ValueError(f"Batch already has {table} outputs; table repair must precede downstream indexing.")
    if semantic_path.is_file():
        with closing(open_connection(semantic_path, read_only=True)) as semantic:
            if semantic.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_chunks'"
            ).fetchone():
                for article in candidates:
                    for chunk in connection.execute(
                        "SELECT chunk_id FROM chunks WHERE article_id=?", (article["article_id"],),
                    ):
                        if semantic.execute(
                            "SELECT 1 FROM semantic_chunks WHERE chunk_id=?", (chunk["chunk_id"],),
                        ).fetchone():
                            raise ValueError("Table article already has semantic vectors; repair is unsafe.")


def table_repair_plan(
    database_path: str | Path, batch_id: str, *, semantic_dir: str | Path | None = None,
) -> dict:
    with closing(open_connection(database_path, read_only=True)) as connection:
        batch = connection.execute("SELECT state FROM pipeline_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if batch is None or batch["state"] != "ingested":
            raise ValueError("Table repair requires an ingested batch.")
        policy_row = connection.execute(
            "SELECT policy,state FROM pipeline_batch_table_policies WHERE batch_id=?", (batch_id,)
        ).fetchone() if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_batch_table_policies'"
        ).fetchone() else None
        candidates = _candidates(connection, batch_id)
        semantic_root = Path(semantic_dir) if semantic_dir is not None else Path(database_path).parent / "semantic"
        _guard_downstream(connection, batch_id, semantic_root / "mapping.sqlite", candidates)
        has_receipts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_article_table_repairs'"
        ).fetchone()
        repaired = connection.execute(
            "SELECT COUNT(*) FROM pipeline_article_table_repairs WHERE batch_id=?", (batch_id,),
        ).fetchone()[0] if has_receipts else 0
        return {
            "batch_id": batch_id, "policy": policy_row["state"] if policy_row else "legacy",
            "table_articles": len(candidates),
            "already_repaired": repaired, "remaining_articles": len(candidates) - repaired,
        }


def repair_batch_tables(
    database_path: str | Path, batch_id: str, *, tokenizer: Any = None,
    semantic_dir: str | Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    path = Path(database_path).resolve(strict=True)
    semantic_root = Path(semantic_dir).resolve() if semantic_dir is not None else path.parent / "semantic"
    with ExitStack() as locks:
        locks.enter_context(SemanticBuildLock(path.with_name(path.name + ".pipeline.lock"), "pipeline"))
        locks.enter_context(SemanticBuildLock(path.with_name(path.name + ".ner.lock"), "NER"))
        locks.enter_context(SemanticBuildLock(semantic_root / ".vectors.lock", "vector"))
        database = Database(path)
        try:
            database.initialize()
            connection = database.conn
            batch = connection.execute(
                "SELECT state,config_json FROM pipeline_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None or batch["state"] != "ingested":
                raise ValueError("Table repair requires an ingested batch.")
            receipt = connection.execute(
                "SELECT state FROM pipeline_batch_table_policies WHERE batch_id=?", (batch_id,),
            ).fetchone()
            if receipt and receipt["state"] == "active":
                raise ValueError("Batch already uses table exclusion.")
            candidates = _candidates(connection, batch_id)
            _guard_downstream(connection, batch_id, semantic_root / "mapping.sqlite", candidates)
            settings = json.loads(batch["config_json"])["chunking"]
            if tokenizer is None and candidates:
                tokenizer = indexer._load_tokenizer(
                    settings["model_id"], revision=settings.get("tokenizer_revision"),
                )
            invalidate_publication(path)
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO pipeline_batch_table_policies(batch_id,policy,state) "
                    "VALUES (?,?,'repairing')", (batch_id, EXCLUDE_HTML_TABLES),
                )
            completed = 0
            cached_newspaper_id = None
            line_map = None
            for row in candidates:
                if connection.execute(
                    "SELECT 1 FROM pipeline_article_table_repairs WHERE article_id=?", (row["article_id"],),
                ).fetchone():
                    continue
                if row["newspaper_id"] != cached_newspaper_id:
                    newspaper = connection.execute(
                        "SELECT content FROM newspapers WHERE newspaper_id=?", (row["newspaper_id"],),
                    ).fetchone()
                    line_map = LineMap(newspaper["content"])
                    cached_newspaper_id = row["newspaper_id"]
                assert line_map is not None
                old_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE article_id=?", (row["article_id"],),
                ).fetchone()[0]
                chunking = connection.execute(
                    "SELECT DISTINCT chunking_run_id FROM chunks WHERE article_id=?", (row["article_id"],),
                ).fetchall()
                if len(chunking) != 1:
                    raise ValueError("Table article must have one recorded chunking run.")
                source = SourceArticle(
                    title=row["title"], content=row["content"], article_index=0,
                    char_start=row["char_start"], char_end=row["char_end"],
                    content_char_start=row["char_end"] - len(row["content"]),
                    line_start=row["line_start"], line_end=row["line_end"],
                )
                chunks = indexer._token_chunks(
                    source, article_id=row["article_id"], run_id=row["run_id"],
                    chunking_run_id=chunking[0][0], tokenizer=tokenizer,
                    token_count=settings["token_count"], token_overlap=settings["token_overlap"],
                    line_map=line_map, exclude_tables=True,
                )
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    current = connection.execute(
                        "SELECT content FROM articles WHERE article_id=?", (row["article_id"],),
                    ).fetchone()
                    if current is None or current["content"] != row["content"]:
                        raise ValueError("Article changed during table repair.")
                    connection.execute("DELETE FROM chunks WHERE article_id=?", (row["article_id"],))
                    connection.executemany(
                        "INSERT INTO chunks(chunk_id,run_id,article_id,chunking_run_id,chunk_index,method,text,"
                        "char_start,char_end,line_start,line_end) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        [(
                            c.chunk_id,c.run_id,c.article_id,c.chunking_run_id,c.chunk_index,c.method,c.text,
                            c.char_start,c.char_end,c.line_start,c.line_end,
                        ) for c in chunks],
                    )
                    repair_table_lexical(connection, row["article_id"])
                    connection.execute(
                        "UPDATE pipeline_issue_stages SET output_count=output_count+? "
                        "WHERE newspaper_id=? AND stage='chunks'",
                        (len(chunks) - old_count, row["newspaper_id"]),
                    )
                    connection.execute(
                        "INSERT INTO pipeline_article_table_repairs(article_id,batch_id,policy,previous_chunks,"
                        "replacement_chunks) VALUES (?,?,?,?,?)",
                        (row["article_id"], batch_id, EXCLUDE_HTML_TABLES, old_count, len(chunks)),
                    )
                completed += 1
                if progress is not None and completed % 100 == 0:
                    progress({"event": "table_repair_checkpoint", "completed": completed, "total": len(candidates)})
            with connection:
                connection.execute(
                    "UPDATE pipeline_batch_table_policies SET state='active',updated_at=CURRENT_TIMESTAMP "
                    "WHERE batch_id=?", (batch_id,),
                )
            return {"batch_id": batch_id, "table_articles": len(candidates), "repaired_articles": completed}
        finally:
            database.close()
