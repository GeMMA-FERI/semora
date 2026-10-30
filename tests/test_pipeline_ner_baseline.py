from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.ner.base import Entity, Prediction
from semora.ner.indexing import build_ner_index
from semora.pipeline.ner_baseline import audit_ner_baseline
from semora.storage import Database


class Model:
    metadata = {"adapter": "fixture", "revision": None, "version": 1}

    def prepare(self, texts):
        return texts

    def predict_prepared(self, texts, _batch_size):
        result = []
        for text in texts:
            entities = []
            for name, label in (("Ana", "PER"), ("Maribor", "LOC"), ("Zürich", "LOC")):
                start = text.find(name)
                if start >= 0:
                    entities.append(Entity(start, start + len(name), label, name, None if name == "Ana" else 0.9))
            result.append(entities)
        return Prediction(result)

    def close(self):
        pass


def _legacy(tmp_path: Path, *, extra_whitespace: bool = False) -> Path:
    path = tmp_path / "indexes" / "semora.sqlite"
    database = Database(path)
    try:
        database.initialize()
        with database.conn:
            database.conn.execute("INSERT INTO runs(run_id,run_type) VALUES ('source','fixture')")
            database.conn.execute(
                "INSERT INTO newspapers(newspaper_id,run_id,content) VALUES ('newspaper','source','')"
            )
            records = [
                ("a", "Ana\x00 gre v Maribor.", 1), ("b", "Brez imen.", 1), ("c", "Zürich?", 1),
                ("invalid", "Ana", 0), ("empty", "   ", 1),
            ]
            if extra_whitespace:
                records.append(("whitespace", "\t\n", 1))
            database.conn.executemany(
                "INSERT INTO articles(article_id,run_id,newspaper_id,content,is_valid) "
                "VALUES (?,'source','newspaper',?,?)", records,
            )
    finally:
        database.close()
    build_ner_index(path, model=Model(), device="cpu", batch_articles=2, pipeline_depth=1)
    return path


def _edit(path, sql, parameters=()):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(sql, parameters)


def test_audit_is_read_only_preserves_zero_mentions_unicode_and_controls(tmp_path, monkeypatch):
    database = _legacy(tmp_path)
    before = database.read_bytes()
    from semora.ner import indexing
    monkeypatch.setattr(indexing, "load_adapter", lambda *_a, **_k: pytest.fail("model loaded during audit"))
    events = []

    report = audit_ner_baseline(database, progress=events.append)

    assert report["eligible"], report["errors"]
    assert report["membership_complete"] and report["coverage_complete"]
    assert report["counts"]["eligible_articles"] == report["counts"]["verified_articles"] == 3
    assert report["counts"]["mentions"] == report["counts"]["verified_mentions"] == 3
    assert report["counts"]["zero_mention_articles"] == 1
    assert report["model_evidence"]["recorded_revision"] is None
    assert not report["compatibility_verified"]
    assert len(report["config_sha256"]) == len(report["evidence_fingerprint"]) == 64
    assert database.read_bytes() == before
    assert events[-1] == {"event": "ner_baseline_articles_audited", "checked": 3, "total": 3}


def test_eligibility_matches_producer_sql_trim_rule(tmp_path):
    report = audit_ner_baseline(_legacy(tmp_path, extra_whitespace=True))
    assert report["eligible"], report["errors"]
    assert report["counts"]["eligible_articles"] == 4
    assert report["counts"]["zero_mention_articles"] == 2


@pytest.mark.parametrize("sql,reason", [
    ("UPDATE article_ner_state SET content_sha256='wrong' WHERE article_id='a'", "content SHA-256"),
    ("UPDATE article_ner_state SET entity_count=entity_count+1 WHERE article_id='a'", "mention count"),
    ("UPDATE article_entities SET entity_index=9 WHERE article_id='a' AND entity_index=0", "contiguous"),
    ("UPDATE article_entities SET char_end=999 WHERE article_id='a' AND entity_index=0", "outside current"),
    ("UPDATE article_entities SET text='Wrong' WHERE article_id='a' AND entity_index=0", "text disagrees"),
    ("UPDATE article_entities SET label='' WHERE article_id='a' AND entity_index=0", "label is empty"),
    ("UPDATE article_entities SET score=1e999 WHERE article_id='a' AND entity_index=0", "score must be finite"),
    ("UPDATE article_entities SET score='not-a-number' WHERE article_id='a' AND entity_index=0",
     "score must be finite"),
])
def test_corrupt_completion_or_mentions_fail_closed(tmp_path, sql, reason):
    database = _legacy(tmp_path)
    _edit(database, sql)

    report = audit_ner_baseline(database)

    assert not report["eligible"]
    assert not report["coverage_complete"]
    assert any(reason in error["reason"] for error in report["errors"])


def test_missing_zero_mention_completion_prevents_adoption_eligibility(tmp_path):
    database = _legacy(tmp_path)
    _edit(database, "DELETE FROM article_ner_state WHERE article_id='b'")

    report = audit_ner_baseline(database)

    assert not report["eligible"]
    assert report["counts"]["missing_completion"] == 1
    assert report["counts"]["mentions"] == 3


@pytest.mark.parametrize("article_id", ["invalid", "empty", "missing"])
def test_extra_completion_membership_fails_closed(tmp_path, article_id):
    database = _legacy(tmp_path)
    _edit(database,
          "INSERT INTO article_ner_state(ner_run_id,article_id,content_sha256,entity_count) "
          "VALUES ('classla-sl',?, ?,0)", (article_id, hashlib.sha256(b"Ana").hexdigest()))

    report = audit_ner_baseline(database)

    assert not report["eligible"]
    assert report["counts"]["extra_completion"] == 1


def test_mentions_without_completion_are_detected(tmp_path):
    database = _legacy(tmp_path)
    _edit(database, """
        INSERT INTO article_entities(ner_run_id,article_id,entity_index,text,label,char_start,char_end)
        VALUES ('classla-sl','missing',0,'A','PER',0,1)
    """)

    report = audit_ner_baseline(database)

    assert not report["eligible"]
    assert report["counts"]["orphan_mentions"] == 1


def test_audit_only_reads_the_selected_run(tmp_path):
    database = _legacy(tmp_path)
    build_ner_index(database, run_id="other-run", model=Model(), device="cpu")

    report = audit_ner_baseline(database)

    assert report["eligible"], report["errors"]
    assert report["counts"]["completed_articles"] == 3
    assert report["counts"]["mentions"] == 3
    assert not audit_ner_baseline(database, run_id="absent")["eligible"]


def test_recorded_checkpoint_hash_is_evidence_not_full_runtime_compatibility(tmp_path):
    database = _legacy(tmp_path)
    with closing(sqlite3.connect(database)) as conn:
        config = json.loads(conn.execute(
            "SELECT config_json FROM ner_runs WHERE ner_run_id='classla-sl'"
        ).fetchone()[0])
    config["model"] = {
        "adapter": "classla", "revision": "main", "model_sha256": "a" * 64,
        "source_sha256": "b" * 64, "classla_version": "2.2.2",
        "ner_config": {"model_path": "C:/historical/cache/model.pt"},
    }
    _edit(database, "UPDATE ner_runs SET config_json=? WHERE ner_run_id='classla-sl'", (json.dumps(config),))

    report = audit_ner_baseline(database)

    assert report["eligible"], report["errors"]
    assert report["model_evidence"]["recorded_model_sha256"] == "a" * 64
    assert report["model_evidence"]["recorded_source_sha256"] == "b" * 64
    assert report["model_evidence"]["recorded_tokenizer_sha256"] is None
    assert report["model_evidence"]["recorded_revision"] == "main"
    assert not report["compatibility_verified"]
    assert not report["model_evidence"]["resource_paths_verified"]


@pytest.mark.parametrize("alteration", ["missing_model", "unknown_field", "bad_hash"])
def test_malformed_run_configuration_is_not_inferred_or_ignored(tmp_path, alteration):
    database = _legacy(tmp_path)
    with closing(sqlite3.connect(database)) as conn:
        config = json.loads(conn.execute("SELECT config_json FROM ner_runs").fetchone()[0])
    if alteration == "missing_model":
        del config["model"]
    elif alteration == "unknown_field":
        config["unknown_pipeline"] = True
    else:
        config["model"]["model_sha256"] = "wrong"
    _edit(database, "UPDATE ner_runs SET config_json=?", (json.dumps(config),))

    assert not audit_ner_baseline(database)["eligible"]


def test_missing_database_does_not_create_files(tmp_path):
    target = tmp_path / "missing" / "semora.sqlite"
    report = audit_ner_baseline(target)
    assert not report["eligible"]
    assert not target.parent.exists()


def test_corruption_report_is_bounded_without_hiding_total_error_count(tmp_path):
    database = _legacy(tmp_path)
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.executemany("""
            INSERT INTO article_entities(ner_run_id,article_id,entity_index,text,label,char_start,char_end)
            VALUES ('classla-sl','b',?,'B','',0,1)
        """, [(index,) for index in range(60)])
        conn.execute("UPDATE article_ner_state SET entity_count=60 WHERE article_id='b'")
    before = database.read_bytes()

    report = audit_ner_baseline(database)

    assert not report["eligible"]
    assert report["error_count"] == 60
    assert len(report["errors"]) == 50
    assert database.read_bytes() == before


def test_mentions_are_read_with_bounded_queries_not_per_article(tmp_path, monkeypatch):
    database = _legacy(tmp_path)
    from semora.pipeline import ner_baseline
    original = ner_baseline.open_connection
    queries = []

    def traced(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_trace_callback(queries.append)
        return connection

    monkeypatch.setattr(ner_baseline, "open_connection", traced)

    report = audit_ner_baseline(database)

    assert report["eligible"], report["errors"]
    entity_queries = [query for query in queries
                      if query.lstrip().upper().startswith("SELECT") and "article_entities" in query]
    assert len(entity_queries) == 2  # One count plus one ordered streaming cursor.


@pytest.mark.parametrize("article_id", ["0-before-first", "between-a-and-b", "z-after-last"])
def test_streaming_merge_detects_orphans_at_every_cursor_boundary(tmp_path, article_id):
    database = _legacy(tmp_path)
    _edit(database, """
        INSERT INTO article_entities(ner_run_id,article_id,entity_index,text,label,char_start,char_end)
        VALUES ('classla-sl',?,0,'A','PER',0,1)
    """, (article_id,))

    report = audit_ner_baseline(database)

    assert not report["eligible"]
    assert report["counts"]["orphan_mentions"] == 1
