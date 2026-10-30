from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.corpus import indexer
from semora.pipeline.semantic_baseline import audit_semantic_baseline
from semora.retrieval.semantic_store import SemanticBuildSpec, read_semantic_build_spec
from semora.retrieval.semantic_vectors import build_semantic_vectors

np = pytest.importorskip("numpy")


class WordTokenizer:
    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class TinyEmbedding:
    def encode_document(self, texts, **_kwargs):
        return np.array([[len(text) % 5 + 1, 1, 2, 3] for text in texts], dtype="float32")

    encode = encode_document


def _legacy(tmp_path: Path, monkeypatch, *, revisions: bool = False) -> tuple[Path, Path]:
    corpus = tmp_path / "old-input" / "corpus" / "Paper"
    corpus.mkdir(parents=True)
    (corpus / "a.md").write_text("# First\nOne two three four.\n# Second\nFive six seven eight.\n", encoding="utf-8")
    (corpus / "b.md").write_text("# Third\nNine ten eleven twelve.\n", encoding="utf-8")
    database = tmp_path / "indexes" / "semora.sqlite"
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda *_a, **_k: WordTokenizer())
    indexer.ingest_corpus(corpus.parent, database, token_count=3, token_overlap=1)
    if revisions:
        with closing(sqlite3.connect(database)) as conn, conn:
            config = json.loads(conn.execute("SELECT config_json FROM chunking_runs").fetchone()[0])
            config["tokenizer_revision"] = "tokenizer-revision-1"
            conn.execute("UPDATE chunking_runs SET config_json=?", (json.dumps(config),))
    semantic = database.parent / "semantic"
    build_semantic_vectors(
        database,
        semantic,
        dimensions=4,
        shard_size=2,
        model=TinyEmbedding(),
        model_revision="model-revision-1" if revisions else None,
    )
    spec = read_semantic_build_spec(semantic)
    assert spec is not None
    manifest = {
        **spec.as_dict(),
        "normalized": True,
        "metric": "cosine_via_inner_product",
        "chunks": spec.valid_chunks,
        "vectors_complete": True,
        "index_complete": False,
    }
    (semantic / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return database, semantic


def _edit_mapping(semantic, sql, parameters=()):
    with closing(sqlite3.connect(semantic / "mapping.sqlite")) as conn, conn:
        conn.execute(sql, parameters)


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_audit_is_read_only_and_preserves_unknown_revision_evidence(tmp_path: Path, monkeypatch) -> None:
    database, semantic = _legacy(tmp_path, monkeypatch)
    shutil.rmtree(tmp_path / "old-input")
    before = _snapshot(tmp_path)
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda *_a, **_k: pytest.fail("tokenizer loaded"))
    events = []

    report = audit_semantic_baseline(database, semantic, progress=events.append)

    assert report["eligible"], report["errors"]
    assert report["declared_configuration_match"]
    assert not report["compatibility_verified"]
    assert not report["recorded_revisions"]
    assert report["model"]["model_revision"] is None
    assert not report["chunking"]["tokenization_completeness_verified"]
    assert not report["embedding_input_link_verified"]
    assert not report["faiss"]["identity_verified"]
    assert not report["faiss"]["adopted"]
    assert report["counts"]["eligible_chunks"] == report["counts"]["verified_chunks"] == 6
    assert report["counts"]["mapped_chunks"] == report["counts"]["verified_vectors"] == 6
    assert len(report["shards"]) == 3
    assert len(report["evidence_fingerprint"]) == 64
    assert _snapshot(tmp_path) == before
    assert events[-1] == {"event": "semantic_shards_audited", "checked": 3, "total": 3}


def test_recorded_revisions_do_not_prove_exact_compatibility_or_execution(tmp_path, monkeypatch):
    database, semantic = _legacy(tmp_path, monkeypatch, revisions=True)

    report = audit_semantic_baseline(database, semantic)

    assert report["eligible"], report["errors"]
    assert not report["compatibility_verified"]
    assert report["recorded_revisions"]
    assert report["model"]["model_revision"] == "model-revision-1"
    assert report["chunking"]["tokenizer_revision"] == "tokenizer-revision-1"
    assert not report["embedding_input_link_verified"]


def test_audit_survives_data_root_relocation(tmp_path, monkeypatch):
    database, semantic = _legacy(tmp_path / "original", monkeypatch)
    before = audit_semantic_baseline(database, semantic)
    moved = tmp_path / "moved"
    shutil.move(str(database.parent), moved)

    after = audit_semantic_baseline(moved / "semora.sqlite", moved / "semantic")

    assert after["eligible"], after["errors"]
    assert before["evidence_fingerprint"] == after["evidence_fingerprint"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM semantic_chunks WHERE vector_id=0",
        "UPDATE semantic_chunks SET shard_offset=0 WHERE vector_id=1",
        "UPDATE semantic_chunks SET shard_index=999 WHERE vector_id=1",
        "UPDATE semantic_shards SET first_vector_id=99 WHERE shard_index=1",
        "UPDATE semantic_build_state SET vectors_complete=0",
        "UPDATE semantic_build_state SET spec_fingerprint='wrong'",
        "UPDATE semantic_shards SET dtype='float32' WHERE shard_index=0",
        "UPDATE semantic_shards SET file_name='../outside.f16' WHERE shard_index=0",
    ],
)
def test_malformed_mapping_or_shard_ledger_fails_closed(tmp_path, monkeypatch, sql):
    database, semantic = _legacy(tmp_path, monkeypatch)
    _edit_mapping(semantic, sql)

    report = audit_semantic_baseline(database, semantic)

    assert not report["eligible"]
    assert report["error_count"] > 0


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE chunks SET text='wrong chunk text' WHERE rowid=(SELECT MIN(rowid) FROM chunks)",
        "UPDATE chunks SET char_start=char_start+1 WHERE rowid=(SELECT MIN(rowid) FROM chunks)",
        "UPDATE chunks SET line_end=line_end+1 WHERE rowid=(SELECT MIN(rowid) FROM chunks)",
        "UPDATE articles SET is_valid=0 WHERE rowid=(SELECT MIN(rowid) FROM articles)",
        "UPDATE chunks SET chunk_index=99 WHERE rowid=(SELECT MIN(rowid) FROM chunks)",
    ],
)
def test_malformed_corpus_chunk_evidence_fails_closed(tmp_path, monkeypatch, sql):
    database, semantic = _legacy(tmp_path, monkeypatch)
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute(sql)

    report = audit_semantic_baseline(database, semantic)

    assert not report["eligible"]
    assert report["errors"]


@pytest.mark.parametrize("kind", ["checksum", "nan", "nonunit", "truncated"])
def test_vector_bytes_are_checked_beyond_recorded_counts(tmp_path, monkeypatch, kind):
    database, semantic = _legacy(tmp_path, monkeypatch)
    path = semantic / "vectors" / "shard-000000.f16"
    values = bytearray(path.read_bytes())
    if kind == "truncated":
        values = values[:-1]
    elif kind == "nan":
        values[:2] = np.array([np.nan], dtype="<f2").tobytes()
    elif kind == "nonunit":
        values[:8] = np.array([0, 0, 0, 0], dtype="<f2").tobytes()
    else:
        values[0] ^= 1
    path.write_bytes(values)
    if kind in {"nan", "nonunit"}:
        _edit_mapping(
            semantic, "UPDATE semantic_shards SET sha256=? WHERE shard_index=0", (hashlib.sha256(values).hexdigest(),)
        )

    report = audit_semantic_baseline(database, semantic)

    assert not report["eligible"]
    assert report["errors"]


def test_unknown_manifest_metadata_is_not_silently_ignored(tmp_path, monkeypatch):
    database, semantic = _legacy(tmp_path, monkeypatch)
    path = semantic / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["unexpected_version"] = "new behavior"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_semantic_baseline(database, semantic)

    assert not report["eligible"]
    assert any("manifest fields" in error for error in report["errors"])


def test_missing_artifacts_audit_never_creates_directories(tmp_path):
    report = audit_semantic_baseline(tmp_path / "absent" / "semora.sqlite", tmp_path / "semantic")
    assert not report["eligible"]
    assert not (tmp_path / "absent").exists()
    assert not (tmp_path / "semantic").exists()


def test_invalid_newspaper_identity_cannot_be_adopted_through_valid_child_chunks(tmp_path, monkeypatch):
    database, semantic = _legacy(tmp_path, monkeypatch)
    with closing(sqlite3.connect(database)) as conn, conn:
        old_id = conn.execute("SELECT newspaper_id FROM newspapers ORDER BY newspaper_id LIMIT 1").fetchone()[0]
        conn.execute("UPDATE newspapers SET newspaper_id='newspaper_wrong' WHERE newspaper_id=?", (old_id,))
        conn.execute("UPDATE articles SET newspaper_id='newspaper_wrong' WHERE newspaper_id=?", (old_id,))

    report = audit_semantic_baseline(database, semantic)

    assert not report["eligible"]
    assert any("newspaper ID disagrees with canonical path" in error for error in report["errors"])


def test_invalid_article_identity_rejected_even_when_chunk_ids_and_spec_are_recomputed(tmp_path, monkeypatch):
    database, semantic = _legacy(tmp_path, monkeypatch)
    replacements = []
    with closing(sqlite3.connect(database)) as conn, conn:
        old_id = conn.execute("SELECT article_id FROM articles ORDER BY article_id LIMIT 1").fetchone()[0]
        conn.execute("UPDATE articles SET article_id='article_wrong' WHERE article_id=?", (old_id,))
        for chunk_id, chunk_index, start, end in conn.execute(
            "SELECT chunk_id,chunk_index,char_start,char_end FROM chunks WHERE article_id=?",
            (old_id,),
        ).fetchall():
            new_id = indexer._stable_id("chunk", "article_wrong", str(chunk_index), str(start), str(end))
            conn.execute("UPDATE chunks SET article_id='article_wrong',chunk_id=? WHERE chunk_id=?", (new_id, chunk_id))
            replacements.append((new_id, chunk_id))
        bounds = conn.execute("SELECT MIN(chunk_id),MAX(chunk_id) FROM chunks").fetchone()
    with closing(sqlite3.connect(semantic / "mapping.sqlite")) as conn, conn:
        conn.executemany("UPDATE semantic_chunks SET chunk_id=? WHERE chunk_id=?", replacements)
        values = json.loads(conn.execute("SELECT spec_json FROM semantic_build_state").fetchone()[0])
        values.pop("format")
        values.update(first_chunk_id=bounds[0], last_chunk_id=bounds[1])
        spec = SemanticBuildSpec(**values)
        conn.execute(
            "UPDATE semantic_build_state SET spec_json=?,spec_fingerprint=?,last_chunk_id=?",
            (json.dumps(spec.as_dict()), spec.fingerprint, bounds[1]),
        )
    path = semantic / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(spec.as_dict())
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_semantic_baseline(database, semantic)

    assert not report["eligible"]
    assert any("article ID disagrees with canonical path" in error for error in report["errors"])
