"""Compact, resumable FAISS indexes built from durable vector shards."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from semora.retrieval.semantic_store import SemanticBuildSpec, SemanticShard, SemanticVectorStore

DEFAULT_INDEX_TYPE = "ivfpq"
DEFAULT_NLIST = 16_384
DEFAULT_PQ_M = 32
DEFAULT_PQ_BITS = 8
DEFAULT_TRAIN_SAMPLES = 1_000_000
DEFAULT_NPROBE = 32


@dataclass(frozen=True)
class FaissBuildConfig:
    index_type: str = DEFAULT_INDEX_TYPE
    nlist: int = DEFAULT_NLIST
    pq_m: int = DEFAULT_PQ_M
    pq_bits: int = DEFAULT_PQ_BITS
    train_samples: int = DEFAULT_TRAIN_SAMPLES
    nprobe: int = DEFAULT_NPROBE

    def validate(self, dimensions: int) -> None:
        if self.index_type not in {"flat", "ivfpq"}:
            raise ValueError("index_type must be 'flat' or 'ivfpq'.")
        if min(self.nlist, self.pq_m, self.pq_bits, self.train_samples, self.nprobe) <= 0:
            raise ValueError("FAISS index parameters must be positive.")
        if self.index_type == "ivfpq" and dimensions % self.pq_m:
            raise ValueError("Embedding dimensions must be divisible by pq_m.")

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class FaissIndexStats:
    index_type: str
    indexed_chunks: int
    added_chunks: int
    vector_chunks: int
    vectors_complete: bool
    index_complete: bool


def build_faiss_index(
    output_dir: str | Path,
    *,
    config: FaissBuildConfig | None = None,
    allow_partial: bool = False,
    checkpoint_shards: int = 1,
    rebuild: bool = False,
    faiss_module: Any = None,
) -> FaissIndexStats:
    """Build or resume a FAISS index without recomputing embeddings."""
    if checkpoint_shards <= 0:
        raise ValueError("checkpoint_shards must be positive.")
    faiss = faiss_module if faiss_module is not None else _import_faiss()
    target = Path(output_dir).resolve()
    if rebuild:
        _clear_faiss_files(target)
    spec = _read_vector_spec(target)
    active_config = config or FaissBuildConfig()
    active_config.validate(spec.dimensions)
    with SemanticVectorStore(target, spec) as store:
        state = store.state
        if not state.vectors_complete and not allow_partial:
            raise ValueError("Semantic vector shards are incomplete; finish them or pass allow_partial=True.")
        shards = store.shards()
        _validate_shards(target, spec, state.indexed_chunks, shards)
        build_fingerprint = _build_fingerprint(spec, active_config)
        checkpoint_path = target / ".index.build.faiss"
        checkpoint_state_path = target / ".index.build.json"
        index = _load_or_create_index(
            faiss,
            target,
            spec,
            active_config,
            shards,
            build_fingerprint,
            checkpoint_path,
            checkpoint_state_path,
        )
        starting_chunks = int(index.ntotal)
        if starting_chunks > state.indexed_chunks:
            raise ValueError("FAISS checkpoint contains more vectors than the vector store.")
        pending = _pending_shards(shards, starting_chunks)
        with tqdm(
            total=state.indexed_chunks,
            initial=starting_chunks,
            desc="Building FAISS index",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            since_checkpoint = 0
            for shard in pending:
                vectors = _read_shard(target, shard, spec.dimensions, dtype="float32")
                ids = _vector_ids(shard.first_vector_id, shard.vector_count)
                index.add_with_ids(vectors, ids)
                progress.update(shard.vector_count)
                since_checkpoint += 1
                if since_checkpoint >= checkpoint_shards:
                    _write_checkpoint(faiss, index, checkpoint_path)
                    _write_build_state(checkpoint_state_path, build_fingerprint, int(index.ntotal))
                    since_checkpoint = 0
            if since_checkpoint or not checkpoint_path.is_file():
                _write_checkpoint(faiss, index, checkpoint_path)
                _write_build_state(checkpoint_state_path, build_fingerprint, int(index.ntotal))
        if hasattr(index, "nprobe"):
            index.nprobe = active_config.nprobe
            _write_checkpoint(faiss, index, checkpoint_path)
        index_complete = state.vectors_complete and int(index.ntotal) == spec.valid_chunks
        _publish_index(
            faiss,
            index,
            target,
            spec,
            active_config,
            build_fingerprint,
            vectors_complete=state.vectors_complete,
            index_complete=index_complete,
        )
        return FaissIndexStats(
            index_type=active_config.index_type,
            indexed_chunks=int(index.ntotal),
            added_chunks=int(index.ntotal) - starting_chunks,
            vector_chunks=state.indexed_chunks,
            vectors_complete=state.vectors_complete,
            index_complete=index_complete,
        )


def _load_or_create_index(
    faiss: Any,
    target: Path,
    spec: SemanticBuildSpec,
    config: FaissBuildConfig,
    shards: list[SemanticShard],
    build_fingerprint: str,
    checkpoint_path: Path,
    checkpoint_state_path: Path,
) -> Any:
    if checkpoint_path.is_file() or checkpoint_state_path.is_file():
        if not checkpoint_path.is_file() or not checkpoint_state_path.is_file():
            raise ValueError("FAISS checkpoint files are incomplete; explicitly rebuild the FAISS stage.")
        checkpoint_state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
        if checkpoint_state.get("build_fingerprint") != build_fingerprint:
            raise ValueError("FAISS configuration changed; explicitly rebuild the FAISS stage.")
        index = faiss.read_index(str(checkpoint_path))
        indexed_chunks = int(index.ntotal)
        recorded_chunks = int(checkpoint_state.get("indexed_chunks", -1))
        if indexed_chunks < recorded_chunks:
            raise ValueError("FAISS checkpoint and build state disagree.")
        _pending_shards(shards, indexed_chunks)
        if indexed_chunks != recorded_chunks:
            _write_build_state(checkpoint_state_path, build_fingerprint, indexed_chunks)
        return index
    if config.index_type == "flat":
        return faiss.IndexIDMap2(faiss.IndexFlatIP(spec.dimensions))
    if sum(shard.vector_count for shard in shards) < config.nlist:
        raise ValueError("IVF-PQ requires at least nlist vector samples; use a smaller nlist for this pilot.")
    quantizer = faiss.IndexFlatIP(spec.dimensions)
    index = faiss.IndexIVFPQ(
        quantizer,
        spec.dimensions,
        config.nlist,
        config.pq_m,
        config.pq_bits,
        faiss.METRIC_INNER_PRODUCT,
    )
    training = _training_sample(target, shards, spec.dimensions, config.train_samples)
    index.train(training)
    index.nprobe = config.nprobe
    return index


def _training_sample(
    target: Path,
    shards: list[SemanticShard],
    dimensions: int,
    requested: int,
) -> Any:
    import numpy as np

    total = sum(shard.vector_count for shard in shards)
    sample_count = min(requested, total)
    selected = np.linspace(0, total - 1, num=sample_count, dtype="int64")
    sample = np.empty((sample_count, dimensions), dtype="float32")
    destination = 0
    for shard in shards:
        lower = int(np.searchsorted(selected, shard.first_vector_id, side="left"))
        upper = int(np.searchsorted(selected, shard.first_vector_id + shard.vector_count, side="left"))
        if lower == upper:
            continue
        vectors = _read_shard(target, shard, dimensions, dtype="float16")
        offsets = selected[lower:upper] - shard.first_vector_id
        sample[destination : destination + len(offsets)] = vectors[offsets]
        destination += len(offsets)
    if destination != sample_count:
        raise ValueError("Could not sample the expected number of training vectors.")
    return sample


def _read_shard(
    target: Path,
    shard: SemanticShard,
    dimensions: int,
    *,
    dtype: str,
) -> Any:
    import numpy as np

    source = np.memmap(
        target / "vectors" / shard.file_name,
        mode="r",
        dtype="float16",
        shape=(shard.vector_count, dimensions),
    )
    return np.asarray(source, dtype=dtype)


def _vector_ids(first: int, count: int) -> Any:
    import numpy as np

    return np.arange(first, first + count, dtype="int64")


def _pending_shards(shards: list[SemanticShard], indexed_chunks: int) -> list[SemanticShard]:
    pending: list[SemanticShard] = []
    cursor = 0
    for shard in shards:
        if shard.first_vector_id != cursor:
            raise ValueError("Semantic shards are not contiguous.")
        end = cursor + shard.vector_count
        if cursor >= indexed_chunks:
            pending.append(shard)
        elif cursor < indexed_chunks < end:
            raise ValueError("FAISS checkpoint ends in the middle of a vector shard.")
        cursor = end
    if indexed_chunks > cursor:
        raise ValueError("FAISS checkpoint exceeds the vector shard ledger.")
    return pending


def _validate_shards(
    target: Path,
    spec: SemanticBuildSpec,
    indexed_chunks: int,
    shards: list[SemanticShard],
) -> None:
    if not shards or sum(shard.vector_count for shard in shards) != indexed_chunks:
        raise ValueError("Semantic shard ledger does not match the vector build state.")
    for shard in shards:
        path = target / "vectors" / shard.file_name
        expected_size = shard.vector_count * spec.dimensions * 2
        if (
            shard.dimensions != spec.dimensions
            or shard.dtype != "float16"
            or shard.byte_size != expected_size
            or not path.is_file()
            or path.stat().st_size != expected_size
        ):
            raise ValueError(f"Semantic vector shard is missing or malformed: {shard.file_name}")
    _pending_shards(shards, 0)


def _read_vector_spec(target: Path) -> SemanticBuildSpec:
    import sqlite3

    database_path = target / "mapping.sqlite"
    if not database_path.is_file():
        raise FileNotFoundError(f"Semantic vector state is missing: {database_path}")
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT spec_json FROM semantic_build_state WHERE state_id = 1").fetchone()
    if row is None:
        raise ValueError("Semantic vector build specification is missing.")
    values = json.loads(str(row[0]))
    values.pop("format", None)
    return SemanticBuildSpec(**values)


def _build_fingerprint(spec: SemanticBuildSpec, config: FaissBuildConfig) -> str:
    build_config = config.as_dict()
    build_config.pop("nprobe")
    value = json.dumps(
        {"spec_fingerprint": spec.fingerprint, "faiss": build_config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _write_checkpoint(faiss: Any, index: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    faiss.write_index(index, str(temporary))
    _fsync_file(temporary)
    temporary.replace(path)


def _write_build_state(path: Path, fingerprint: str, indexed_chunks: int) -> None:
    _write_json_atomic(
        path,
        {"build_fingerprint": fingerprint, "indexed_chunks": indexed_chunks},
    )


def _publish_index(
    faiss: Any,
    index: Any,
    target: Path,
    spec: SemanticBuildSpec,
    config: FaissBuildConfig,
    build_fingerprint: str,
    *,
    vectors_complete: bool,
    index_complete: bool,
) -> None:
    final_index = target / "index.faiss"
    temporary = target / ".index.faiss.tmp"
    faiss.write_index(index, str(temporary))
    _fsync_file(temporary)
    temporary.replace(final_index)
    _write_json_atomic(
        target / "manifest.json",
        {
            **spec.as_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "normalized": True,
            "metric": "cosine_via_inner_product",
            "chunks": int(index.ntotal),
            "vectors_complete": vectors_complete,
            "index_complete": index_complete,
            "faiss": config.as_dict(),
            "build_fingerprint": build_fingerprint,
        },
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as source:
        os.fsync(source.fileno())


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Semantic indexing requires the 'retrieval' extra.") from exc
    return faiss


def _clear_faiss_files(target: Path) -> None:
    for name in (
        ".index.build.faiss",
        ".index.build.json",
        "index.faiss",
        "manifest.json",
        "..index.build.faiss.tmp",
        "..index.build.json.tmp",
        ".index.faiss.tmp",
        ".manifest.json.tmp",
    ):
        path = target / name
        if path.is_file():
            path.unlink()
