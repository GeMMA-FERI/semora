"""Historical NER adoption records evidence without replaying extraction."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from semora.ner.base import Entity, Prediction
from semora.ner.indexing import build_ner_index
from semora.pipeline.ner_adoption import adopt_ner_baseline
from semora.storage import Database


class Model:
    metadata = {"adapter": "fixture", "version": 1}

    def prepare(self, texts):
        return texts

    def predict_prepared(self, texts, _batch_size):
        return Prediction([[Entity(0, 3, "PER", "Ana")] if text.startswith("Ana") else [] for text in texts])


def legacy(tmp_path):
    path = tmp_path / "indexes" / "semora.sqlite"
    database = Database(path)
    try:
        database.initialize()
        with database.conn:
            database.conn.execute("INSERT INTO runs(run_id,run_type) VALUES ('source','fixture')")
            database.conn.execute("INSERT INTO newspapers(newspaper_id,run_id,content) VALUES ('issue','source','')")
            database.conn.executemany(
                "INSERT INTO articles(article_id,run_id,newspaper_id,content,is_valid) VALUES (?,'source','issue',?,1)",
                [("a", "Ana goes home."), ("b", "No entities here.")],
            )
    finally:
        database.close()
    build_ner_index(path, model=Model(), device="cpu", pipeline_depth=1)
    return path


def rows(path):
    with closing(sqlite3.connect(path)) as conn:
        state = conn.execute(
            "SELECT ner_run_id,article_id,content_sha256,entity_count FROM article_ner_state ORDER BY article_id"
        ).fetchall()
        mentions = conn.execute(
            "SELECT ner_run_id,article_id,entity_index,text,label,char_start,char_end,score "
            "FROM article_entities ORDER BY article_id,entity_index"
        ).fetchall()
    return state, mentions


def test_ner_adoption_preserves_outputs_and_retry_uses_receipt(tmp_path, monkeypatch):
    database = legacy(tmp_path)
    before = rows(database)
    from semora.ner import adapters

    monkeypatch.setattr(adapters, "load_adapter", lambda *_a, **_k: pytest.fail("model loaded during adoption"))
    first = adopt_ner_baseline(database)
    second = adopt_ner_baseline(database)

    assert first["eligible"] and not first["adoption"]["already_adopted"]
    assert second["adoption"]["already_adopted"]
    assert first["evidence_fingerprint"] == second["evidence_fingerprint"]
    assert first["adoption"]["model_identity"] == {"adapter": "fixture", "version": 1}
    assert rows(database) == before
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_ner_adoptions").fetchone()[0] == 1


def test_ineligible_ner_baseline_cannot_write_adoption_marker(tmp_path):
    database = legacy(tmp_path)
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute("UPDATE article_ner_state SET content_sha256='wrong' WHERE article_id='a'")

    with pytest.raises(ValueError, match="ineligible"):
        adopt_ner_baseline(database)

    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_ner_adoptions").fetchone()[0] == 0
