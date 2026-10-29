from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semora.ner.indexing import build_ner_index
from semora.ner.names import build_entity_names
from semora.pipeline.guards import reject_pipeline_batches
from semora.retrieval.indexing import (
    build_bm25_index,
    build_lemma_index,
    build_semantic_index,
)
from semora.retrieval.semantic_vectors import build_semantic_vectors


def make_database(path: Path, *, batches: bool, populated: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('unchanged')")
        if batches:
            connection.execute("CREATE TABLE pipeline_batches (batch_id TEXT PRIMARY KEY)")
            if populated:
                connection.execute("INSERT INTO pipeline_batches VALUES ('batch-a')")


@pytest.mark.parametrize("batches", [False, True])
def test_guard_leaves_legacy_or_empty_batch_database_unchanged(tmp_path: Path, batches: bool) -> None:
    path = tmp_path / "semora.sqlite"
    make_database(path, batches=batches)
    before = path.read_bytes()

    reject_pipeline_batches(path)

    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_guard_does_not_create_missing_database_or_parent(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "semora.sqlite"

    reject_pipeline_batches(path)
    reject_pipeline_batches(":memory:")

    assert not path.parent.exists()


@pytest.mark.parametrize("builder", [
    build_bm25_index,
    build_lemma_index,
    build_semantic_index,
    build_semantic_vectors,
    build_ner_index,
])
def test_legacy_builder_rejects_batches_before_any_writes_or_model_loading(tmp_path: Path, builder) -> None:
    path = tmp_path / "semora.sqlite"
    # Deliberately not an initialized corpus: the guard must run before schema
    # initialization, stage logging, file locks, and model loading.
    make_database(path, batches=True, populated=True)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="additive pipeline batches.*not ready for legacy indexing"):
        builder(path)

    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_entity_builder_rejects_before_creating_generation_or_loading_spellfix(tmp_path: Path) -> None:
    path = tmp_path / "indexes" / "semora.sqlite"
    make_database(path, batches=True, populated=True)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="additive pipeline batches"):
        build_entity_names(tmp_path)

    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_bm25_without_batches_still_reaches_legacy_builder(tmp_path: Path) -> None:
    path = tmp_path / "semora.sqlite"

    assert build_bm25_index(path) == 0

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM article_fts_map").fetchone()[0] == 0
