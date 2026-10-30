from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.corpus import indexer
from semora.pipeline import semantic
from semora.pipeline.baseline import adopt_baseline
from semora.pipeline.batches import run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.pipeline.semantic import index_batch_vectors, read_store_config, store_corpus_identity
from semora.pipeline.semantic_adoption import adopt_semantic_baseline
from semora.retrieval.semantic_store import read_semantic_build_spec
from semora.retrieval.semantic_vectors import build_semantic_vectors

np = pytest.importorskip("numpy")


class Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class Model:
    def __init__(self):
        self.texts = []

    def encode_document(self, texts, **_kwargs):
        self.texts.extend(texts)
        return np.array([[len(text) % 5 + 1, 1, 2, 3] for text in texts], dtype="float32")

    encode = encode_document


def _legacy(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus" / "Paper"
    corpus.mkdir(parents=True)
    (corpus / "old.md").write_text("# First\nOne two three four.\n# Second\nFive six seven eight.\n", encoding="utf-8")
    database = tmp_path / "indexes" / "semora.sqlite"
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda *_a, **_k: Tokenizer())
    indexer.ingest_corpus(corpus.parent, database, token_count=3, token_overlap=1)
    root = database.parent / "semantic"
    build_semantic_vectors(database, root, dimensions=4, shard_size=2, model=Model())
    spec = read_semantic_build_spec(root)
    assert spec is not None
    (root / "manifest.json").write_text(
        json.dumps(
            {
                **spec.as_dict(),
                "normalized": True,
                "metric": "cosine_via_inner_product",
                "chunks": spec.valid_chunks,
                "vectors_complete": True,
                "index_complete": False,
            }
        ),
        encoding="utf-8",
    )
    return database, root


def _legacy_artifacts(root: Path) -> dict:
    with closing(sqlite3.connect(root / "mapping.sqlite")) as conn:
        return {
            "shards": conn.execute("SELECT * FROM semantic_shards ORDER BY shard_index").fetchall(),
            "mappings": conn.execute("SELECT * FROM semantic_chunks ORDER BY vector_id").fetchall(),
            "state": conn.execute("SELECT * FROM semantic_build_state").fetchall(),
            "manifest": (root / "manifest.json").read_bytes(),
            "vectors": {path.name: path.read_bytes() for path in (root / "vectors").glob("*.f16")},
        }


def _forbid_models(monkeypatch):
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda *_a, **_k: pytest.fail("tokenizer loaded"))
    monkeypatch.setattr(semantic, "_load_embedding_model", lambda *_a, **_k: pytest.fail("embedding model loaded"))


def test_adoption_preserves_old_vectors_mapping_and_defaults_without_models(tmp_path, monkeypatch):
    database, root = _legacy(tmp_path, monkeypatch)
    before = _legacy_artifacts(root)
    _forbid_models(monkeypatch)

    adopted = adopt_semantic_baseline(database, root)
    retry = adopt_semantic_baseline(database, root)

    assert adopted["eligible"] and retry["eligible"]
    assert adopted["declared_configuration_match"]
    assert not adopted["compatibility_verified"]
    assert adopted["adoption"]["input_hashes_added"] == len(before["mappings"])
    assert adopted["adoption"]["vector_files_rewritten"] == 0
    assert not adopted["adoption"]["historical_embedding_execution_verified"]
    assert retry["adoption"]["input_hashes_added"] == 0
    assert _legacy_artifacts(root) == before
    with closing(sqlite3.connect(database)) as conn:
        corpus_id = conn.execute("SELECT corpus_id FROM pipeline_corpus_identity").fetchone()[0]
    with closing(sqlite3.connect(root / "mapping.sqlite")) as conn:
        assert store_corpus_identity(conn) == corpus_id == adopted["adoption"]["corpus_id"]
        assert read_store_config(conn)["model_revision"] is None
        assert read_store_config(conn)["chunking"]["tokenizer_revision"] is None
        assert conn.execute("SELECT COUNT(*) FROM pipeline_semantic_adoptions").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_new_batch_after_adoption_embeds_only_new_chunks(tmp_path, monkeypatch):
    database, root = _legacy(tmp_path, monkeypatch)
    adopt_baseline(database)
    before = _legacy_artifacts(root)
    adopt_semantic_baseline(database, root)
    inputs = tmp_path / "new-inputs"
    inputs.mkdir()
    markdown = inputs / "new.md"
    metadata = inputs / "new_metadata.json"
    markdown.write_text("# New article\nA new train travels north every morning.\n", encoding="utf-8")
    metadata.write_text(json.dumps({"Record": {"source": "New paper"}}), encoding="utf-8")
    manifest = BatchManifest(
        "new-batch",
        (
            BatchIssue(
                relative_path="New/new.md",
                markdown_path="new.md",
                metadata_path="new_metadata.json",
                markdown_sha256=hashlib.sha256(markdown.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
            ),
        ),
    )
    run_batch(manifest, inputs, database, tokenizer=Tokenizer(), token_count=3, token_overlap=1)
    with closing(sqlite3.connect(database)) as conn:
        expected = [
            row[0]
            for row in conn.execute(
                "SELECT c.text FROM chunks c JOIN articles a USING(article_id) "
                "JOIN pipeline_batch_issues b USING(newspaper_id) WHERE b.batch_id='new-batch'"
            )
        ]
    _forbid_models(monkeypatch)
    model = Model()

    appended = index_batch_vectors(database, "new-batch", root, model=model, dimensions=4, shard_size=2)

    assert appended["complete"]
    assert appended["added_chunks"] == len(expected) == len(model.texts)
    assert sorted(model.texts) == sorted(expected)
    after = _legacy_artifacts(root)
    assert after["mappings"][: len(before["mappings"])] == before["mappings"]
    assert after["state"] == before["state"]
    assert after["manifest"] == before["manifest"]
    for name, content in before["vectors"].items():
        assert after["vectors"][name] == content


def test_corrupt_vector_file_rejects_before_store_or_input_adoption(tmp_path, monkeypatch):
    database, root = _legacy(tmp_path, monkeypatch)
    file = next((root / "vectors").glob("*.f16"))
    file.write_bytes(file.read_bytes()[:-1])
    before = _legacy_artifacts(root)

    with pytest.raises(ValueError, match="Semantic baseline audit failed"):
        adopt_semantic_baseline(database, root)

    assert _legacy_artifacts(root) == before
    with closing(sqlite3.connect(root / "mapping.sqlite")) as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='pipeline_semantic_store'").fetchone()
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='pipeline_semantic_inputs'").fetchone()


def test_existing_mapping_bound_to_other_corpus_is_not_rebound(tmp_path, monkeypatch):
    database, root = _legacy(tmp_path, monkeypatch)
    adopt_semantic_baseline(database, root)
    before = _legacy_artifacts(root)
    with closing(sqlite3.connect(root / "mapping.sqlite")) as conn, conn:
        conn.execute("UPDATE pipeline_semantic_store SET corpus_id=?", ("f" * 32,))

    with pytest.raises(ValueError, match="different logical corpus"):
        adopt_semantic_baseline(database, root)

    assert _legacy_artifacts(root) == before
    with closing(sqlite3.connect(root / "mapping.sqlite")) as conn:
        assert store_corpus_identity(conn) == "f" * 32


def test_existing_input_conflict_rolls_back_other_new_input_hashes(tmp_path, monkeypatch):
    database, root = _legacy(tmp_path, monkeypatch)
    adopt_semantic_baseline(database, root)
    before = _legacy_artifacts(root)
    with closing(sqlite3.connect(root / "mapping.sqlite")) as conn, conn:
        last = conn.execute("SELECT chunk_id FROM semantic_chunks ORDER BY vector_id DESC LIMIT 1").fetchone()[0]
        conn.execute("DELETE FROM pipeline_semantic_inputs WHERE chunk_id<>?", (last,))
        conn.execute("UPDATE pipeline_semantic_inputs SET input_sha256='bad'")

    with pytest.raises(ValueError, match="Existing semantic input hash differs"):
        adopt_semantic_baseline(database, root)

    assert _legacy_artifacts(root) == before
    with closing(sqlite3.connect(root / "mapping.sqlite")) as conn:
        assert conn.execute("SELECT chunk_id,input_sha256 FROM pipeline_semantic_inputs").fetchall() == [(last, "bad")]


def test_corpus_mapping_alias_rejected_before_migration(tmp_path, monkeypatch):
    _database, root = _legacy(tmp_path, monkeypatch)
    mapping = root / "mapping.sqlite"
    before = mapping.read_bytes()

    with pytest.raises(ValueError, match="different files"):
        adopt_semantic_baseline(mapping, root)

    assert mapping.read_bytes() == before
