"""Small transactional repository for article-level entity extraction."""

import hashlib
import json
import sqlite3

from .base import validate


def register_run(connection, run_id, config):
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    with connection:
        connection.execute(
            "INSERT OR IGNORE INTO ner_runs(ner_run_id, config_json) VALUES (?, ?)", (run_id, encoded)
        )
        saved = connection.execute("SELECT config_json FROM ner_runs WHERE ner_run_id = ?", (run_id,)).fetchone()
        if saved[0] != encoded:
            raise ValueError("NER run configuration changed. Use a different --run-id for different models/settings.")


def save_batch(connection: sqlite3.Connection, run_id, batch, *, batch_id=None, table_policy=None):
    """Validate first; commit completion and all mentions in a single transaction."""
    pairs = list(zip(batch.rows, batch.prediction.entities, strict=True))
    for row, entities in pairs:
        validate(row["content"], entities)
    added_articles = added_entities = 0
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        if batch_id is not None:
            policy_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_batch_table_policies'"
            ).fetchone()
            policy = connection.execute(
                "SELECT policy, state FROM pipeline_batch_table_policies WHERE batch_id=?", (batch_id,)
            ).fetchone() if policy_table else None
            if (policy is None and table_policy is not None) or (
                policy is not None and (policy[0] != table_policy or policy[1] != 'active')
            ):
                raise RuntimeError("Batch table policy changed during NER extraction.")
        for row, entities in pairs:
            current = connection.execute(
                "SELECT content, is_valid FROM articles WHERE article_id = ?", (row["article_id"],)
            ).fetchone()
            if current is None or current[0] != row["content"] or current[1] != 1:
                raise RuntimeError(f"Article changed during NER extraction: {row['article_id']}. Rerun to resume.")
            if batch_id is not None and connection.execute(
                "SELECT 1 FROM articles a JOIN pipeline_batch_issues i ON i.newspaper_id=a.newspaper_id "
                "JOIN pipeline_batches b ON b.batch_id=i.batch_id "
                "WHERE a.article_id=? AND i.batch_id=? AND b.state='ingested'",
                (row["article_id"], batch_id),
            ).fetchone() is None:
                raise RuntimeError("Batch membership changed during NER extraction.")
            digest = hashlib.sha256(row["content"].encode()).hexdigest()
            payload = [(i, e.text, e.label, e.start, e.end, e.score) for i, e in enumerate(entities)]
            state = connection.execute(
                "SELECT content_sha256, entity_count FROM article_ner_state "
                "WHERE ner_run_id = ? AND article_id = ?", (run_id, row["article_id"]),
            ).fetchone()
            stored = [tuple(item) for item in connection.execute(
                "SELECT entity_index, text, label, char_start, char_end, score FROM article_entities "
                "WHERE ner_run_id = ? AND article_id = ? ORDER BY entity_index", (run_id, row["article_id"]),
            )]
            if state is not None:
                if tuple(state) != (digest, len(entities)) or stored != payload:
                    raise RuntimeError(f"Conflicting NER result for article: {row['article_id']}")
                continue
            if stored:
                raise RuntimeError(f"Orphan NER entities for article: {row['article_id']}")
            connection.execute(
                "INSERT INTO article_ner_state(ner_run_id, article_id, content_sha256, entity_count) "
                "VALUES (?, ?, ?, ?)",
                (run_id, row["article_id"], digest, len(entities)),
            )
            connection.executemany(
                "INSERT INTO article_entities "
                "(ner_run_id, article_id, entity_index, text, label, char_start, char_end, score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, row["article_id"], *item) for item in payload],
            )
            added_articles += 1
            added_entities += len(entities)
    return added_articles, added_entities


def completed_count(connection, run_id):
    return connection.execute(
        "SELECT COUNT(*) FROM article_ner_state WHERE ner_run_id = ?", (run_id,)
    ).fetchone()[0]
