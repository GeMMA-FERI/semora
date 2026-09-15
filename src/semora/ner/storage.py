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


def save_batch(connection: sqlite3.Connection, run_id, batch):
    """Validate first; commit completion and all mentions in a single transaction."""
    pairs = list(zip(batch.rows, batch.prediction.entities, strict=True))
    for row, entities in pairs:
        validate(row["content"], entities)
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        for row, entities in pairs:
            current = connection.execute(
                "SELECT content, is_valid FROM articles WHERE article_id = ?", (row["article_id"],)
            ).fetchone()
            if current is None or current[0] != row["content"] or current[1] != 1:
                raise RuntimeError(f"Article changed during NER extraction: {row['article_id']}. Rerun to resume.")
            connection.execute(
                "INSERT INTO article_ner_state(ner_run_id, article_id, content_sha256, entity_count) "
                "VALUES (?, ?, ?, ?)",
                (run_id, row["article_id"], hashlib.sha256(row["content"].encode()).hexdigest(), len(entities)),
            )
            connection.executemany(
                "INSERT INTO article_entities "
                "(ner_run_id, article_id, entity_index, text, label, char_start, char_end, score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, row["article_id"], i, e.text, e.label, e.start, e.end, e.score)
                 for i, e in enumerate(entities)],
            )


def completed_count(connection, run_id):
    return connection.execute(
        "SELECT COUNT(*) FROM article_ner_state WHERE ner_run_id = ?", (run_id,)
    ).fetchone()[0]
