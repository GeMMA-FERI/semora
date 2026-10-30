from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from semora.pipeline import semantic
from semora.pipeline.batches import run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.pipeline.semantic import index_batch_vectors, read_store_config


class Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class Model:
    def __init__(self):
        self.calls = []

    def encode_document(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.array([[len(text), 1.0, 2.0] for text in texts], dtype="float32")


def ingest(root: Path, batch_id: str, *names: str) -> Path:
    inputs = root / "incoming"
    entries = []
    for name in names:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Title {Path(name).stem}\nA train travels to Ljubljana each morning.\n", encoding="utf-8")
        metadata = path.with_name(path.stem + "_metadata.json")
        metadata.write_text(json.dumps({"Record": {"source": "Straža", "date": "1925 02 18"}}), encoding="utf-8")
        entries.append(
            BatchIssue(
                relative_path=name,
                markdown_path=name,
                metadata_path=metadata.relative_to(inputs).as_posix(),
                markdown_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
            )
        )
    database = root / "indexes" / "semora.sqlite"
    run_batch(
        BatchManifest(batch_id, tuple(entries)), inputs, database, tokenizer=Tokenizer(), token_count=3, token_overlap=1
    )
    return database


def build(database: Path, batch: str, directory: Path, **kwargs):
    return index_batch_vectors(database, batch, directory, dimensions=2, batch_size=2, shard_size=2, **kwargs)


def mapping(directory: Path) -> list[tuple]:
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        return connection.execute("SELECT * FROM semantic_chunks ORDER BY vector_id").fetchall()


def test_append_two_batches_keeps_old_ids_bytes_and_uses_declared_defaults(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "indexes" / "semantic"
    model = Model()
    first = build(database, "batch-a", directory, model=model)
    old = mapping(directory)
    old_files = {path.name: path.read_bytes() for path in (directory / "vectors").glob("*.f16")}
    ingest(tmp_path, "batch-b", "Straza/b.md")

    second = build(database, "batch-b", directory, model=model)
    retry = build(database, "batch-b", directory)

    assert first["complete"] and second["complete"] and retry["added_chunks"] == 0
    assert old == mapping(directory)[: len(old)]
    assert second["added_chunks"] > 0
    assert not second["retrieval_ready"] and not second["publication_ready"]
    assert not second["compatibility_verified"]
    for name, data in old_files.items():
        assert (directory / "vectors" / name).read_bytes() == data
        vectors = np.frombuffer(data, dtype="<f2").reshape(-1, 2)
        np.testing.assert_allclose(np.linalg.norm(vectors.astype("float32"), axis=1), 1.0, atol=0.002)
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        config = read_store_config(connection)
        assert config["model_id"] == "google/embeddinggemma-300m" and config["model_revision"] is None
        assert config["chunking"]["tokenizer_revision"] is None
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_new_chunk_hash_before_old_checkpoint_is_not_missed(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    build(database, "batch-a", directory, model=Model())
    old = mapping(directory)
    ingest(tmp_path, "batch-b", "Straza/b.md", "Straza/c.md")
    with closing(sqlite3.connect(database)) as connection:
        new_ids = [
            row[0]
            for row in connection.execute(
                "SELECT c.chunk_id FROM chunks c JOIN articles a USING(article_id) "
                "JOIN pipeline_batch_issues b USING(newspaper_id) WHERE batch_id='batch-b'"
            )
        ]
    assert min(new_ids) < max(row[1] for row in old)

    result = build(database, "batch-b", directory, model=Model())

    assert result["added_chunks"] == len(new_ids)
    assert {row[1] for row in mapping(directory)}.issuperset(new_ids)


@pytest.mark.parametrize("after_publish", [False, True])
def test_pending_shard_recovery_needs_no_model_after_publication_crash(
    tmp_path: Path,
    monkeypatch,
    after_publish,
) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    publish = semantic._publish_file

    def crash(temporary, final):
        if after_publish:
            publish(temporary, final)
        raise RuntimeError("publication interrupted")

    monkeypatch.setattr(semantic, "_publish_file", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        index_batch_vectors(database, "batch-a", directory, dimensions=2, shard_size=100, model=Model())
    assert mapping(directory) == []
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_semantic_pending_shards").fetchone()[0] == 1

    monkeypatch.setattr(semantic, "_publish_file", publish)
    monkeypatch.setattr(semantic, "_load_embedding_model", lambda *_a, **_k: pytest.fail("Do not reload model"))
    result = index_batch_vectors(database, "batch-a", directory, dimensions=2, shard_size=100)

    assert result["complete"] and result["recovered_chunks"] > 0 and result["added_chunks"] == 0
    assert len(list((directory / "vectors").glob("*.f16"))) == 1


def test_mapping_transaction_failure_preserves_published_file_and_recovers_once(tmp_path: Path, monkeypatch) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    original = semantic._finish_pending

    def install_failure(connection, vectors, row, config):
        connection.execute(
            "CREATE TRIGGER fail_mapping BEFORE INSERT ON semantic_chunks "
            "BEGIN SELECT RAISE(ABORT,'mapping failure'); END"
        )
        connection.commit()
        return original(connection, vectors, row, config)

    monkeypatch.setattr(semantic, "_finish_pending", install_failure)
    with pytest.raises(sqlite3.IntegrityError, match="mapping failure"):
        index_batch_vectors(database, "batch-a", directory, dimensions=2, shard_size=100, model=Model())
    assert mapping(directory) == []
    published = next((directory / "vectors").glob("*.f16"))
    before = published.read_bytes()
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        connection.execute("DROP TRIGGER fail_mapping")
    monkeypatch.setattr(semantic, "_finish_pending", original)

    result = index_batch_vectors(database, "batch-a", directory, dimensions=2, shard_size=100)

    assert result["complete"] and result["recovered_chunks"] == len(mapping(directory))
    assert published.read_bytes() == before


def test_existing_unjournaled_shard_is_never_overwritten(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    vectors = directory / "vectors"
    vectors.mkdir(parents=True)
    protected = vectors / "shard-000000.f16"
    protected.write_bytes(b"unowned source bytes")

    with pytest.raises(ValueError, match="audit/adoption"):
        build(database, "batch-a", directory, model=Model())

    assert protected.read_bytes() == b"unowned source bytes"
    assert not (directory / "mapping.sqlite").exists()


def test_retry_rejects_changed_chunk_source_and_corrupt_bytes(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    build(database, "batch-a", directory, model=Model())
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE chunks SET text=text || ' changed'")
        connection.commit()
    with pytest.raises(ValueError, match="Mapped chunk source differs"):
        build(database, "batch-a", directory)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE chunks SET text=substr(text,1,length(text)-8)")
        connection.commit()
    shard = next((directory / "vectors").glob("*.f16"))
    shard.write_bytes(bytes(shard.stat().st_size))
    with pytest.raises(ValueError, match="shard bytes"):
        build(database, "batch-a", directory)


@pytest.mark.parametrize("bad", ["zero", "nan", "shape"])
def test_invalid_embedding_output_never_commits_shards(tmp_path: Path, bad: str) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"

    class InvalidModel:
        def encode_document(self, texts, **kwargs):
            value = np.zeros((len(texts), 2), dtype="float32")
            if bad == "nan":
                value[:] = np.nan
            return value[:, :1] if bad == "shape" else value

    with pytest.raises(ValueError, match="vector"):
        build(database, "batch-a", directory, model=InvalidModel())
    assert mapping(directory) == []
    assert list((directory / "vectors").glob("*.f16")) == []


def test_model_revision_and_chunking_compatibility_are_separate_from_runtime_tuning(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    build(database, "batch-a", directory, model=Model())

    with pytest.raises(ValueError, match="revision/dimensions/chunking changed"):
        build(database, "batch-a", directory, model_revision="known-different")
    assert index_batch_vectors(database, "batch-a", directory, dimensions=2, batch_size=1, shard_size=99)["complete"]


def test_complete_batch_relocates_without_loading_model(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "source"
    database = ingest(root, "batch-a", "Straza/a.md")
    directory = root / "indexes" / "semantic"
    build(database, "batch-a", directory, model=Model())
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    monkeypatch.setattr(semantic, "_load_embedding_model", lambda *_a, **_k: pytest.fail("No pending embeddings"))

    result = build(relocated / "indexes" / "semora.sqlite", "batch-a", relocated / "indexes" / "semantic")

    assert result["complete"] and result["added_chunks"] == 0
    assert mapping(directory) == mapping(relocated / "indexes" / "semantic")


def test_mapping_cannot_alias_corpus_by_path_or_hardlink(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    other = tmp_path / "aliased"
    other.mkdir()
    try:
        os.link(database, other / "mapping.sqlite")
    except OSError as exc:
        pytest.skip(f"Host does not permit hardlinks: {exc}")
    before = database.read_bytes()

    with pytest.raises(ValueError, match="cannot alias"):
        build(database, "batch-a", other, model=Model())

    assert database.read_bytes() == before


def test_mapping_cannot_alias_corpus_path(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    alias = tmp_path / "same" / "mapping.sqlite"
    alias.parent.mkdir()
    shutil.copyfile(database, alias)
    before = alias.read_bytes()

    with pytest.raises(ValueError, match="cannot alias"):
        build(alias, "batch-a", alias.parent, model=Model())

    assert alias.read_bytes() == before


def test_legacy_mapping_is_not_implicitly_adopted(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    directory.mkdir()
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        connection.execute("CREATE TABLE semantic_chunks(vector_id INTEGER PRIMARY KEY,chunk_id TEXT)")
        connection.commit()

    with pytest.raises(ValueError, match="explicit audit/adoption"):
        build(database, "batch-a", directory, model=Model())


def test_new_batch_overlapping_old_chunks_embeds_only_new_inputs(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    first = build(database, "batch-a", directory, model=Model())
    ingest(tmp_path, "batch-b", "Straza/a.md", "Straza/b.md")

    second = build(database, "batch-b", directory, model=Model())

    assert second["skipped_chunks"] == first["completed_chunks"]
    assert second["added_chunks"] == second["eligible_chunks"] - first["completed_chunks"]


def test_journal_insert_failure_removes_only_current_unpublished_temporary(tmp_path: Path, monkeypatch) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    prepare = semantic._prepare_shard

    def fail_journal(connection, *args):
        connection.execute(
            "CREATE TRIGGER fail_intent BEFORE INSERT ON pipeline_semantic_pending_shards "
            "BEGIN SELECT RAISE(ABORT,'journal failure'); END"
        )
        connection.commit()
        return prepare(connection, *args)

    monkeypatch.setattr(semantic, "_prepare_shard", fail_journal)
    with pytest.raises(sqlite3.IntegrityError, match="journal failure"):
        build(database, "batch-a", directory, model=Model())

    assert list((directory / "vectors").glob("*.tmp")) == []
    assert mapping(directory) == []
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_semantic_pending_shards").fetchone()[0] == 0


def test_startup_cleans_only_reserved_unjournaled_temporary_namespace(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    build(database, "batch-a", directory, model=Model())
    managed = directory / "vectors" / (".pipeline-000999-" + "a" * 32 + ".tmp")
    unknown = directory / "vectors" / ".unknown-user-data.tmp"
    managed.write_bytes(b"interrupted unpublished bytes")
    unknown.write_bytes(b"preserve this unrelated file")
    before = {path.name: path.read_bytes() for path in (directory / "vectors").glob("*.f16")}

    result = build(database, "batch-a", directory)

    assert result["removed_unjournaled_temporaries"] == 1
    assert not managed.exists() and unknown.read_bytes() == b"preserve this unrelated file"
    assert before == {path.name: path.read_bytes() for path in (directory / "vectors").glob("*.f16")}


def test_older_copy_with_same_corpus_identity_cannot_drop_mapped_chunks(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    build(database, "batch-a", directory, model=Model())
    fork = tmp_path / "fork.sqlite"
    shutil.copyfile(database, fork)
    ingest(tmp_path, "batch-b", "Straza/b.md")
    build(database, "batch-b", directory, model=Model())
    before = mapping(directory)

    with pytest.raises(ValueError, match="Mapped historical chunks"):
        build(fork, "batch-a", directory, model=Model())

    assert mapping(directory) == before


def test_incomplete_chunk_stage_is_rejected_before_creating_vector_store(tmp_path: Path) -> None:
    database = ingest(tmp_path, "batch-a", "Straza/a.md")
    directory = tmp_path / "semantic"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM chunks WHERE chunk_id=(SELECT MIN(chunk_id) FROM chunks)")
        connection.commit()

    with pytest.raises(ValueError, match="chunk stage count differs"):
        build(database, "batch-a", directory, model=Model())

    assert not (directory / "mapping.sqlite").exists()
