from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semora.retrieval.engine import _read_faiss_index
from semora.retrieval.semantic_faiss import FaissBuildConfig, build_faiss_index
from semora.retrieval.semantic_store import SemanticBuildSpec, SemanticVectorStore


class FakeFlat:
    def __init__(self, dimensions: int) -> None:
        self.d = dimensions


class FakeIdMap:
    def __init__(self, inner: FakeFlat) -> None:
        self.d = inner.d
        self.ids: list[int] = []

    @property
    def ntotal(self) -> int:
        return len(self.ids)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert vectors.shape == (len(ids), self.d)
        self.ids.extend(int(value) for value in ids)


class FakeFaiss:
    IndexFlatIP = FakeFlat
    IndexIDMap2 = FakeIdMap

    def __init__(self) -> None:
        self.files: dict[str, list[int]] = {}

    def write_index(self, index: FakeIdMap, path: str) -> None:
        Path(path).write_text(json.dumps(index.ids), encoding="utf-8")

    def read_index(self, path: str) -> FakeIdMap:
        index = FakeIdMap(FakeFlat(2))
        index.ids = json.loads(Path(path).read_text(encoding="utf-8"))
        return index


def _vectors(directory: Path, *, complete: bool = True) -> None:
    spec = SemanticBuildSpec(
        model_id="model",
        model_revision="revision",
        dimensions=2,
        database="semora.sqlite",
        chunking_run_id="chunks",
        chunking={"token_count": 256, "token_overlap": 64},
        valid_chunks=4,
        first_chunk_id="chunk-0",
        last_chunk_id="chunk-3",
    )
    with SemanticVectorStore(directory, spec) as store:
        shard_count = 2 if complete else 1
        for shard_index in range(shard_count):
            vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float16")
            file_name = f"shard-{shard_index:06d}.f16"
            path = store.vectors_dir / file_name
            vectors.tofile(path)
            store.record_shard(
                shard_index=shard_index,
                file_name=file_name,
                chunk_ids=[f"chunk-{shard_index * 2}", f"chunk-{shard_index * 2 + 1}"],
                dimensions=2,
                dtype="float16",
                byte_size=path.stat().st_size,
                sha256="0" * 64,
            )


def test_flat_faiss_build_is_published_and_resumable(tmp_path: Path) -> None:
    target = tmp_path / "semantic"
    _vectors(target)
    faiss = FakeFaiss()
    first = build_faiss_index(
        target,
        config=FaissBuildConfig(index_type="flat"),
        checkpoint_shards=1,
        faiss_module=faiss,
    )
    assert first.indexed_chunks == 4
    assert first.added_chunks == 4
    assert first.index_complete is True
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "semora.semantic.v2"
    assert manifest["faiss"]["index_type"] == "flat"
    assert manifest["chunks"] == 4

    resumed = build_faiss_index(
        target,
        config=FaissBuildConfig(index_type="flat", nprobe=64),
        faiss_module=faiss,
    )
    assert resumed.indexed_chunks == 4
    assert resumed.added_chunks == 0


def test_faiss_requires_complete_vectors_by_default(tmp_path: Path) -> None:
    target = tmp_path / "semantic"
    _vectors(target, complete=False)
    with pytest.raises(ValueError, match="vector shards are incomplete"):
        build_faiss_index(
            target,
            config=FaissBuildConfig(index_type="flat"),
            faiss_module=FakeFaiss(),
        )

    stats = build_faiss_index(
        target,
        config=FaissBuildConfig(index_type="flat"),
        allow_partial=True,
        faiss_module=FakeFaiss(),
    )
    assert stats.indexed_chunks == 2
    assert stats.index_complete is False


def test_rebuild_faiss_keeps_vector_shards(tmp_path: Path) -> None:
    target = tmp_path / "semantic"
    _vectors(target)
    faiss = FakeFaiss()
    build_faiss_index(
        target,
        config=FaissBuildConfig(index_type="flat"),
        faiss_module=faiss,
    )

    stats = build_faiss_index(
        target,
        config=FaissBuildConfig(index_type="flat", pq_bits=4),
        rebuild=True,
        faiss_module=faiss,
    )

    assert stats.added_chunks == 4
    assert (target / "mapping.sqlite").is_file()
    assert len(list((target / "vectors").glob("*.f16"))) == 2


def test_ivfpq_configuration_requires_divisible_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        FaissBuildConfig(index_type="ivfpq", pq_m=32).validate(386)


def test_real_ivfpq_index_can_be_trained_persisted_and_searched(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    target = tmp_path / "semantic"
    dimensions = 8
    count = 256
    spec = SemanticBuildSpec(
        model_id="model",
        model_revision="revision",
        dimensions=dimensions,
        database="semora.sqlite",
        chunking_run_id="chunks",
        chunking={"token_count": 256, "token_overlap": 64},
        valid_chunks=count,
        first_chunk_id="chunk-000",
        last_chunk_id="chunk-255",
    )
    random = np.random.default_rng(42)
    vectors = random.normal(size=(count, dimensions)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with SemanticVectorStore(target, spec) as store:
        path = store.vectors_dir / "shard-000000.f16"
        vectors.astype("float16").tofile(path)
        store.record_shard(
            shard_index=0,
            file_name=path.name,
            chunk_ids=[f"chunk-{index:03d}" for index in range(count)],
            dimensions=dimensions,
            dtype="float16",
            byte_size=path.stat().st_size,
            sha256="0" * 64,
        )

    stats = build_faiss_index(
        target,
        config=FaissBuildConfig(
            index_type="ivfpq",
            nlist=4,
            pq_m=2,
            pq_bits=4,
            train_samples=count,
            nprobe=4,
        ),
    )

    assert stats.index_complete is True
    index = _read_faiss_index(faiss, target / "index.faiss")
    scores, ids = index.search(vectors[:1], 5)
    assert scores.shape == (1, 5)
    assert ids.shape == (1, 5)
    assert all(0 <= int(vector_id) < count for vector_id in ids[0])
