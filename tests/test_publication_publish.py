"""A failed gate preserves the old pointer; successful retry is idempotent."""

import json
import sqlite3

from semora.pipeline.publication_publish import publish_batch


def _paths(tmp_path):
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    database = indexes / "semora.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE pipeline_publication_state (state_id INTEGER PRIMARY KEY, epoch INTEGER NOT NULL);"
            "INSERT INTO pipeline_publication_state VALUES (1, 2);"
            "CREATE TABLE pipeline_corpus_identity (identity_id INTEGER PRIMARY KEY, corpus_id TEXT NOT NULL);"
            "INSERT INTO pipeline_corpus_identity VALUES (1, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');"
        )
    semantic, entities = indexes / "semantic", indexes / "entities"
    semantic.mkdir()
    entities.mkdir()
    return database, semantic, entities


def _candidate():
    return {
        "format": "semora.pipeline.publication.v1",
        "corpus_id": "a" * 32,
        "epoch": 2,
        "semantic": {"generation": "c" * 64, "chunks": 3, "index_sha256": "d" * 64},
        "entities": {"generation": "e" * 32, "ner_run_id": "classla-sl", "manifest_sha256": "f" * 64},
    }


def test_publish_replaces_stale_pointer_once_after_gate(tmp_path, monkeypatch):
    database, semantic, entities = _paths(tmp_path)
    pointer = database.parent / "publication.json"
    old = {**_candidate(), "publication_id": "1" * 32, "epoch": 1}
    pointer.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(
        "semora.pipeline.publication_publish.audit_publication",
        lambda *_args, **_kwargs: {"eligible": True, "candidate": _candidate()},
    )
    first = publish_batch(database, "batch-a", semantic, "c" * 64, entities)
    second = publish_batch(database, "batch-a", semantic, "c" * 64, entities)
    saved = json.loads(pointer.read_text(encoding="utf-8"))
    assert first["published"] and not first["already_published"]
    assert second["already_published"] and second["publication_id"] == first["publication_id"]
    assert saved["epoch"] == 2 and saved["publication_id"] == first["publication_id"]


def test_failed_gate_preserves_previous_pointer(tmp_path, monkeypatch):
    database, semantic, entities = _paths(tmp_path)
    pointer = database.parent / "publication.json"
    pointer.write_text("old-pointer", encoding="utf-8")
    monkeypatch.setattr(
        "semora.pipeline.publication_publish.audit_publication",
        lambda *_args, **_kwargs: {"eligible": False, "candidate": None},
    )
    result = publish_batch(database, "batch-a", semantic, "c" * 64, entities)
    assert not result["published"]
    assert pointer.read_text(encoding="utf-8") == "old-pointer"
