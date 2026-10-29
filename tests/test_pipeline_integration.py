from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing

import pytest

from semora.corpus import indexer
from semora.pipeline.baseline import adopt_baseline
from semora.pipeline.batches import batch_status, run_batch
from semora.pipeline.lexical import index_batch_lexical
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.retrieval.indexing import build_bm25_index, build_lemma_index


class Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class Lemmatizer:
    def __init__(self):
        self.calls = 0

    def lemmatize_many(self, texts):
        self.calls += 1
        return [text.casefold().replace("horses", "horse") for text in texts]


def _legacy(tmp_path, monkeypatch):
    source = tmp_path / "old-input"
    source.mkdir()
    markdown = source / "old.md"
    markdown.write_text("# Archive\nHorses drove wagons.\n", encoding="utf-8")
    database = tmp_path / "old-data" / "semora.sqlite"
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda _model: Tokenizer())
    indexer.ingest_corpus(source, database, token_count=3, token_overlap=1)
    build_bm25_index(database)
    build_lemma_index(database, lemmatizer=Lemmatizer())
    markdown.unlink()
    return database


def _incoming(tmp_path):
    source = tmp_path / "new-input"
    source.mkdir()
    markdown = b"# Arrival\nHorses pulled carriages.\n"
    metadata = json.dumps({"Record": {
        "source": "Straža", "date": "1925 02 18", "identifier": "GF-URN:new-issue",
    }}).encode("utf-8")
    (source / "new.md").write_bytes(markdown)
    (source / "new_metadata.json").write_bytes(metadata)
    issue = BatchIssue(
        "Straza/new.md", "new.md", "new_metadata.json",
        hashlib.sha256(markdown).hexdigest(), hashlib.sha256(metadata).hexdigest(), "GF-URN:new-issue",
    )
    return source, BatchManifest("new-batch", (issue,))


def test_adopt_relocate_append_and_index_without_repeating_historical_nlp(tmp_path, monkeypatch):
    original = _legacy(tmp_path, monkeypatch)
    with closing(sqlite3.connect(original)) as conn:
        before = conn.execute("SELECT * FROM articles").fetchall()
        lemmas = conn.execute("SELECT * FROM article_lemmas").fetchall()
    adopted = adopt_baseline(original)
    assert adopted["adoption"]["lexical_markers_added"] == {"surface": 1, "lemma": 1}
    assert adopted["adoption"]["chunk_markers_added"] == 0
    database = tmp_path / "relocated" / "archive.sqlite"
    database.parent.mkdir()
    original.rename(database)
    source, manifest = _incoming(tmp_path)
    ingested = run_batch(manifest, source, database, token_count=3, token_overlap=1, tokenizer=Tokenizer())
    assert ingested["lexical"]["surface"]["pending_articles"] == 1
    surface = index_batch_lexical(database, manifest.batch_id)
    model = Lemmatizer()
    lemma = index_batch_lexical(database, manifest.batch_id, stage="lemma", lemmatizer=model)
    assert surface["added_articles"] == lemma["added_articles"] == 1
    assert model.calls == 1
    index_batch_lexical(database, manifest.batch_id, stage="lemma", lemmatizer=model)
    assert model.calls == 1
    with closing(sqlite3.connect(database)) as conn:
        old_id = before[0][0]
        assert conn.execute("SELECT * FROM articles WHERE article_id=?", (old_id,)).fetchall() == before
        assert conn.execute("SELECT * FROM article_lemmas WHERE article_id=?", (old_id,)).fetchall() == lemmas
        new_id = conn.execute("SELECT article_id FROM articles WHERE article_id<>?", (old_id,)).fetchone()[0]
        surface_ids = {r[0] for r in conn.execute(
            "SELECT m.article_id FROM article_fts JOIN article_fts_map m ON m.fts_id=article_fts.rowid "
            "WHERE article_fts MATCH 'carriages'"
        )}
        lemma_ids = {r[0] for r in conn.execute(
            "SELECT m.article_id FROM article_lemma_fts JOIN article_fts_map m ON m.fts_id=article_lemma_fts.rowid "
            "WHERE article_lemma_fts MATCH 'horse'"
        )}
        assert new_id in surface_ids and {old_id, new_id} <= lemma_ids
        assert conn.execute("SELECT urn FROM newspapers WHERE relative_path='Straza/new.md'").fetchone()[0] is None
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    status = batch_status(database, manifest.batch_id)
    assert status["lexical"]["lemma"]["pending_articles"] == 0
    assert not status["retrieval_ready"]


def test_adopted_articles_do_not_hide_incompatible_historical_chunks(tmp_path, monkeypatch):
    database = _legacy(tmp_path, monkeypatch)
    adopt_baseline(database)
    source, manifest = _incoming(tmp_path)
    before = database.read_bytes()
    with pytest.raises(ValueError, match="Historical chunk configuration differs"):
        run_batch(manifest, source, database, token_count=5, token_overlap=1, tokenizer=Tokenizer())
    assert database.read_bytes() == before
