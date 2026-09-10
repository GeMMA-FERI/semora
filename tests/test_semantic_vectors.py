from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from semora.retrieval.semantic_store import SemanticVectorStore
from semora.retrieval.semantic_vectors import build_semantic_vectors
from semora.storage import Article, Chunk, ChunkingRun, Database, Newspaper, Run


class FakeEmbeddingModel:
    def encode(self, texts: list[str], **_kwargs) -> np.ndarray:
        return self.encode_document(texts)

    def encode_document(self, texts: list[str], **_kwargs) -> np.ndarray:
        return np.asarray(
            [[float(index + 1), 1.0, 9.0] for index, _text in enumerate(texts)],
            dtype="float32",
        )


def _database(path: Path, count: int = 5) -> Path:
    database = Database(path)
    try:
        database.initialize()
        database.insert_run(Run(run_id="run", run_type="corpus"))
        database.insert_newspaper(Newspaper(newspaper_id="paper", run_id="run", content="source"))
        database.insert_article(
            Article(
                article_id="article",
                run_id="run",
                newspaper_id="paper",
                title="Article",
                content="source",
                is_valid=True,
            )
        )
        database.insert_chunking_run(
            ChunkingRun(
                chunking_run_id="token-256-64",
                run_id="run",
                method="token",
                config={"method": "token", "token_count": 256, "token_overlap": 64},
            )
        )
        database.insert_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{index:03d}",
                    run_id="run",
                    article_id="article",
                    chunking_run_id="token-256-64",
                    chunk_index=index,
                    method="token",
                    text=f"text {index}",
                )
                for index in range(count)
            ]
        )
    finally:
        database.close()
    return path


def test_semantic_vectors_write_normalized_float16_shards_and_resume(tmp_path: Path) -> None:
    database_path = _database(tmp_path / "semora.sqlite")
    output_dir = tmp_path / "semantic"
    first = build_semantic_vectors(
        database_path,
        output_dir,
        dimensions=2,
        batch_size=2,
        shard_size=2,
        max_chunks=3,
        model=FakeEmbeddingModel(),
    )
    assert first.indexed_chunks == 3
    assert first.added_chunks == 3
    assert first.complete is False
    assert sorted(path.name for path in (output_dir / "vectors").glob("*.f16")) == [
        "shard-000000.f16",
        "shard-000001.f16",
    ]

    second = build_semantic_vectors(
        database_path,
        output_dir,
        dimensions=2,
        batch_size=2,
        shard_size=2,
        model=FakeEmbeddingModel(),
    )
    assert second.indexed_chunks == 5
    assert second.added_chunks == 2
    assert second.complete is True
    vectors = np.fromfile(output_dir / "vectors" / "shard-000000.f16", dtype="float16").reshape(-1, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors.astype("float32"), axis=1), 1.0, atol=1e-3)

    no_op = build_semantic_vectors(
        database_path,
        output_dir,
        dimensions=2,
        model=FakeEmbeddingModel(),
    )
    assert no_op.added_chunks == 0


def test_semantic_vectors_reject_changed_corpus_on_resume(tmp_path: Path) -> None:
    database_path = _database(tmp_path / "semora.sqlite")
    output_dir = tmp_path / "semantic"
    build_semantic_vectors(
        database_path,
        output_dir,
        dimensions=2,
        max_chunks=2,
        model=FakeEmbeddingModel(),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE articles SET is_valid = 0")

    with pytest.raises(ValueError, match="exactly one valid chunking run"):
        build_semantic_vectors(
            database_path,
            output_dir,
            dimensions=2,
            model=FakeEmbeddingModel(),
        )


def test_semantic_vector_mapping_matches_stable_chunk_order(tmp_path: Path) -> None:
    database_path = _database(tmp_path / "semora.sqlite", count=3)
    output_dir = tmp_path / "semantic"
    build_semantic_vectors(
        database_path,
        output_dir,
        dimensions=2,
        shard_size=3,
        model=FakeEmbeddingModel(),
    )
    database = Database(database_path)
    try:
        from semora.retrieval.semantic_vectors import _semantic_build_spec

        spec = _semantic_build_spec(
            database,
            database_path=database_path,
            model_id="google/embeddinggemma-300m",
            model_revision=None,
            dimensions=2,
        )
    finally:
        database.close()
    with SemanticVectorStore(output_dir, spec) as store:
        assert store.chunk_ids([2, 0, 1]) == ["chunk-002", "chunk-000", "chunk-001"]
