from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semora.retrieval.semantic_store import SemanticBuildSpec, SemanticVectorStore


def _spec(*, dimensions: int = 256) -> SemanticBuildSpec:
    return SemanticBuildSpec(
        model_id="google/embeddinggemma-300m",
        model_revision="revision-1",
        dimensions=dimensions,
        database="semora.sqlite",
        chunking_run_id="token-256-64",
        chunking={"method": "token", "token_count": 256, "token_overlap": 64},
        valid_chunks=3,
        first_chunk_id="chunk-a",
        last_chunk_id="chunk-c",
    )


def test_semantic_store_checkpoints_shards_and_resolves_numeric_ids(tmp_path: Path) -> None:
    with SemanticVectorStore(tmp_path / "semantic", _spec()) as store:
        assert store.state.indexed_chunks == 0
        store.record_shard(
            shard_index=0,
            file_name="shard-000000.f16",
            chunk_ids=["chunk-a", "chunk-b"],
            dimensions=256,
            dtype="float16",
            byte_size=1024,
            sha256="a" * 64,
        )
        assert store.state.indexed_chunks == 2
        assert store.state.last_chunk_id == "chunk-b"
        assert store.state.next_shard_index == 1
        assert store.state.vectors_complete is False
        assert store.chunk_ids([1, 0, 99]) == ["chunk-b", "chunk-a", None]

        store.record_shard(
            shard_index=1,
            file_name="shard-000001.f16",
            chunk_ids=["chunk-c"],
            dimensions=256,
            dtype="float16",
            byte_size=512,
            sha256="b" * 64,
        )
        assert store.state.vectors_complete is True

    with sqlite3.connect(tmp_path / "semantic" / "mapping.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM semantic_chunks").fetchone()[0] == 3


def test_semantic_store_reopens_matching_build_and_rejects_changed_spec(tmp_path: Path) -> None:
    directory = tmp_path / "semantic"
    with SemanticVectorStore(directory, _spec()) as store:
        store.record_shard(
            shard_index=0,
            file_name="shard-000000.f16",
            chunk_ids=["chunk-a"],
            dimensions=256,
            dtype="float16",
            byte_size=512,
            sha256="a" * 64,
        )

    with SemanticVectorStore(directory, _spec()) as resumed:
        assert resumed.state.indexed_chunks == 1
        assert resumed.state.last_chunk_id == "chunk-a"

    with pytest.raises(ValueError, match="configuration or corpus changed"):
        SemanticVectorStore(directory, _spec(dimensions=128))


def test_semantic_store_rejects_out_of_order_or_duplicate_shards(tmp_path: Path) -> None:
    with SemanticVectorStore(tmp_path / "semantic", _spec()) as store:
        with pytest.raises(ValueError, match="Expected semantic shard 0"):
            store.record_shard(
                shard_index=1,
                file_name="shard-000001.f16",
                chunk_ids=["chunk-a"],
                dimensions=256,
                dtype="float16",
                byte_size=512,
                sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="unique chunk IDs"):
            store.record_shard(
                shard_index=0,
                file_name="shard-000000.f16",
                chunk_ids=["chunk-a", "chunk-a"],
                dimensions=256,
                dtype="float16",
                byte_size=1024,
                sha256="a" * 64,
            )
