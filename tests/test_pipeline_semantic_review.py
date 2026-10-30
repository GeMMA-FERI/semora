"""Independent additive semantic target-isolation regressions."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.pipeline.batches import run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest


def _write_issue(root: Path, relative_path: str, identifier: str) -> BatchIssue:
    markdown = b"# Article\nThe original newspaper text.\n"
    metadata = json.dumps({"Record": {"source": "Paper", "identifier": identifier}}).encode("utf-8")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(markdown)
    metadata_path = path.with_name(path.stem + "_metadata.json")
    metadata_path.write_bytes(metadata)
    return BatchIssue(
        relative_path=relative_path,
        markdown_path=relative_path,
        metadata_path=metadata_path.relative_to(root).as_posix(),
        markdown_sha256=hashlib.sha256(markdown).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        source_identifier=identifier,
    )


def test_semantic_store_rejects_a_different_logical_archive(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    from semora.pipeline.semantic import index_batch_vectors

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}

    class Model:
        def __init__(self):
            self.calls = 0

        def encode_document(self, texts, **_kwargs):
            self.calls += 1
            return np.ones((len(texts), 2), dtype="float32")

    databases = []
    for name in ("first", "unrelated"):
        source = tmp_path / name / "incoming"
        database = tmp_path / name / "indexes" / "semora.sqlite"
        issue = _write_issue(source, f"Paper/{name}.md", f"GF-URN:{name}")
        run_batch(BatchManifest(name, (issue,)), source, database, tokenizer=Tokenizer())
        databases.append(database)
    directory = tmp_path / "shared-semantic"
    index_batch_vectors(databases[0], "first", directory, dimensions=2, model=Model())
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        before = connection.execute("SELECT * FROM semantic_chunks ORDER BY vector_id").fetchall()
    other_model = Model()

    with pytest.raises(ValueError):
        index_batch_vectors(databases[1], "unrelated", directory, dimensions=2, model=other_model)

    assert other_model.calls == 0
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        assert connection.execute("SELECT * FROM semantic_chunks ORDER BY vector_id").fetchall() == before


def test_semantic_store_rejects_restored_archive_missing_committed_mapping_members(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    from semora.pipeline.semantic import index_batch_vectors

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}

    class Model:
        def __init__(self):
            self.calls = 0

        def encode_document(self, texts, **_kwargs):
            self.calls += 1
            return np.ones((len(texts), 2), dtype="float32")

    original = tmp_path / "original" / "semora.sqlite"
    restored = tmp_path / "restored" / "semora.sqlite"
    directory = tmp_path / "semantic"

    def ingest(database: Path, batch: str) -> None:
        source = tmp_path / "incoming" / batch
        issue = _write_issue(source, f"Paper/{batch}.md", f"GF-URN:{batch}")
        run_batch(BatchManifest(batch, (issue,)), source, database, tokenizer=Tokenizer())

    ingest(original, "first")
    index_batch_vectors(original, "first", directory, dimensions=2, model=Model())
    restored.parent.mkdir(parents=True)
    # Backup preserves the logical corpus UUID. It predates the second batch.
    with closing(sqlite3.connect(original)) as source, closing(sqlite3.connect(restored)) as target:
        source.backup(target)
    ingest(original, "second")
    index_batch_vectors(original, "second", directory, dimensions=2, model=Model())
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        before = connection.execute("SELECT * FROM semantic_chunks ORDER BY vector_id").fetchall()
    ingest(restored, "third")
    model = Model()

    with pytest.raises(ValueError):
        index_batch_vectors(restored, "third", directory, dimensions=2, model=model)

    assert model.calls == 0
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        assert connection.execute("SELECT * FROM semantic_chunks ORDER BY vector_id").fetchall() == before
