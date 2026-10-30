"""Serving loads one pinned revision and fails closed after a writer starts."""

import json
import sqlite3

import pytest

from semora.retrieval.engine import SearchEngine


def _corpus(tmp_path):
    directory = tmp_path / "indexes"
    directory.mkdir()
    path = directory / "semora.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE pipeline_publication_state (state_id INTEGER PRIMARY KEY, epoch INTEGER NOT NULL);"
            "INSERT INTO pipeline_publication_state VALUES (1, 4);"
            "CREATE TABLE pipeline_corpus_identity (identity_id INTEGER PRIMARY KEY, corpus_id TEXT NOT NULL);"
            "INSERT INTO pipeline_corpus_identity VALUES (1, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');"
        )
    return path


def _pointer():
    return {
        "format": "semora.pipeline.publication.v1",
        "publication_id": "b" * 32,
        "corpus_id": "a" * 32,
        "epoch": 4,
        "semantic": {"generation": "c" * 64, "chunks": 3, "index_sha256": "d" * 64},
        "entities": {"generation": "e" * 32, "ner_run_id": "classla-sl", "manifest_sha256": "f" * 64},
    }


def test_reader_rejects_missing_or_stale_pointer_after_writer(tmp_path):
    path = _corpus(tmp_path)
    with pytest.raises(ValueError, match="without a serving publication"):
        SearchEngine(path)
    (path.parent / "publication.json").write_text(json.dumps(_pointer()), encoding="utf-8")
    engine = SearchEngine(path, semantic_dir=path.parent / "semantic")
    try:
        assert engine.publication["publication_id"] == "b" * 32
        assert engine._semantic_files()["index"] == (
            path.parent / "semantic" / "generations" / ("c" * 64) / "index.faiss"
        )
    finally:
        engine.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE pipeline_publication_state SET epoch=5")
    with pytest.raises(ValueError, match="stale"):
        SearchEngine(path)


def test_reader_rejects_pointer_for_another_corpus(tmp_path):
    path = _corpus(tmp_path)
    pointer = _pointer()
    pointer["corpus_id"] = "9" * 32
    (path.parent / "publication.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="different corpus"):
        SearchEngine(path)
