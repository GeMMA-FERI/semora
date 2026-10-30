"""Build immutable FAISS generations without replacing the serving snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

from semora.pipeline.semantic import (
    read_store_config,
    store_corpus_identity,
    validate_corpus_membership,
)
from semora.retrieval.semantic_faiss import (
    FaissBuildConfig,
    _import_faiss,
    _read_shard,
    _training_sample,
)
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.retrieval.semantic_store import SemanticShard
from semora.storage.connection import open_connection

GENERATION_FORMAT = "semora.semantic.faiss-generation.v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _child(directory: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid FAISS artifact filename.")
    result = directory / name
    if result.is_symlink() or result.resolve().parent != directory.resolve():
        raise ValueError("FAISS artifact escapes its generation directory.")
    return result


def _sync(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        handle = os.open(path, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        elif os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _shards(connection: sqlite3.Connection, root: Path, config: dict) -> list[SemanticShard]:
    shards = [
        SemanticShard(**dict(row)) for row in connection.execute("SELECT * FROM semantic_shards ORDER BY shard_index")
    ]
    cursor = 0
    for number, shard in enumerate(shards):
        file = _child(root / "vectors", shard.file_name)
        if (
            shard.shard_index != number
            or shard.first_vector_id != cursor
            or shard.vector_count <= 0
            or shard.dimensions != config["dimensions"]
            or shard.dtype != "float16"
            or shard.byte_size != shard.vector_count * shard.dimensions * 2
            or not file.is_file()
            or file.stat().st_size != shard.byte_size
            or _file_hash(file) != shard.sha256
        ):
            raise ValueError("Semantic shard ledger/file checksum mismatch.")
        cursor += shard.vector_count
    mapped = connection.execute("SELECT COUNT(*),MIN(vector_id),MAX(vector_id) FROM semantic_chunks").fetchone()
    if mapped[0] != cursor or (cursor and (mapped[1] != 0 or mapped[2] != cursor - 1)):
        raise ValueError("Semantic vector IDs are not a contiguous immutable range.")
    if connection.execute(
        "SELECT 1 FROM semantic_chunks c LEFT JOIN semantic_shards s USING(shard_index) "
        "WHERE s.shard_index IS NULL OR c.shard_offset<0 OR c.shard_offset>=s.vector_count "
        "OR c.vector_id<>s.first_vector_id+c.shard_offset LIMIT 1"
    ).fetchone():
        raise ValueError("Semantic shard offsets disagree with vector IDs.")
    if connection.execute("SELECT 1 FROM pipeline_semantic_pending_shards LIMIT 1").fetchone():
        raise ValueError("Recover pending vector publication before building FAISS.")
    return shards


def _prefix(shards: list[SemanticShard], count: int) -> list[SemanticShard]:
    result = []
    cursor = 0
    for shard in shards:
        if cursor == count:
            break
        result.append(shard)
        cursor += shard.vector_count
    if cursor != count:
        raise ValueError("FAISS generation does not end on a vector-shard boundary.")
    return result


def _ledger_hash(config: dict, shards: list[SemanticShard]) -> str:
    return _digest({"store": config, "shards": [vars(shard) for shard in shards]})


def validate_faiss_ids(faiss: Any, index: Any, count: int, dimensions: int) -> None:
    """Prove the complete vector-ID set, rather than trusting ntotal alone."""
    import numpy as np

    if int(index.ntotal) != count or int(index.d) != dimensions or not index.is_trained:
        raise ValueError("FAISS shape/training state differs from the vector ledger.")
    if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
        raise ValueError("FAISS index must use inner product for normalized vectors.")
    seen = np.zeros(count, dtype=bool)

    def record(ids):
        values = np.asarray(ids, dtype="int64")
        if values.size and (
            np.any(values < 0)
            or np.any(values >= count)
            or np.unique(values).size != values.size
            or np.any(seen[values])
        ):
            raise ValueError("FAISS contains duplicate or unknown vector IDs.")
        seen[values] = True

    if hasattr(index, "id_map"):
        ids = faiss.vector_to_array(index.id_map)
        for start in range(0, len(ids), 100000):
            record(ids[start : start + 100000])
    else:
        ivf = faiss.extract_index_ivf(index)
        for number in range(ivf.nlist):
            size = ivf.invlists.list_size(number)
            if size:
                record(faiss.rev_swig_ptr(ivf.invlists.get_ids(number), size))
    if not bool(seen.all()):
        raise ValueError("FAISS is missing recorded vector IDs.")


def _validate_index_kind(faiss, index, config):
    if config.index_type == "flat":
        if not hasattr(index, "id_map") or not isinstance(faiss.downcast_index(index.index), faiss.IndexFlat):
            raise ValueError("FAISS index type differs from the flat generation configuration.")
    elif (
        not hasattr(index, "pq")
        or index.nlist != config.nlist
        or index.pq.M != config.pq_m
        or index.pq.nbits != config.pq_bits
    ):
        raise ValueError("FAISS index type/quantization differs from its generation configuration.")


def _new_index(faiss, root, store, shards, config):
    if config.index_type == "flat":
        return faiss.IndexIDMap2(faiss.IndexFlatIP(store["dimensions"]))
    total = sum(shard.vector_count for shard in shards)
    required = max(config.nlist, 1 << config.pq_bits)
    if min(total, config.train_samples) < required:
        raise ValueError("Too few training vectors for this IVF-PQ configuration; use flat for a small pilot.")
    index = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(store["dimensions"]),
        store["dimensions"],
        config.nlist,
        config.pq_m,
        config.pq_bits,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(_training_sample(root, shards, store["dimensions"], config.train_samples))
    return index


def _checkpoint(faiss, index, build: Path, fingerprint: str, shard_count: int) -> None:
    name = f"checkpoint-{int(index.ntotal)}-{uuid.uuid4().hex}.faiss"
    target = _child(build, name)
    faiss.write_index(index, str(target))
    _sync(target)
    _write_json(
        build / "checkpoint.json",
        {
            "fingerprint": fingerprint,
            "index_file": name,
            "index_sha256": _file_hash(target),
            "vector_count": int(index.ntotal),
            "shard_count": shard_count,
        },
        replace=True,
    )
    _cleanup_checkpoints(build, keep=name)


def _cleanup_checkpoints(build: Path, *, keep: str | None = None) -> None:
    # Only reserved names inside this private build are eligible for cleanup.
    for path in build.glob("checkpoint-*.faiss"):
        if re.fullmatch(r"checkpoint-[0-9]+-[0-9a-f]{32}\.faiss", path.name) and path.name != keep:
            _child(build, path.name).unlink()
    _sync_directory(build)


def _load_generation(
    faiss, directory: Path, store: dict, shards: list[SemanticShard], config: FaissBuildConfig, corpus_id: str
):
    manifest = json.loads(_child(directory, "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != GENERATION_FORMAT
        or manifest["store_config"] != store
        or manifest.get("corpus_id") != corpus_id
    ):
        raise ValueError("FAISS generation has incompatible store metadata.")
    recorded = dict(manifest["faiss"])
    requested = config.as_dict()
    recorded.pop("nprobe", None)
    requested.pop("nprobe", None)
    if recorded != requested:
        raise ValueError("FAISS generation configuration changed.")
    prefix = _prefix(shards, manifest["chunks"])
    if manifest["vector_fingerprint"] != _ledger_hash(store, prefix):
        raise ValueError("FAISS generation vector prefix changed.")
    expected = _digest({"corpus_id": corpus_id, "vectors": manifest["vector_fingerprint"], "faiss": manifest["faiss"]})
    if manifest.get("generation") != expected or directory.name != expected:
        raise ValueError("FAISS generation identity is inconsistent.")
    path = _child(directory, "index.faiss")
    if _file_hash(path) != manifest["index_sha256"]:
        raise ValueError("FAISS generation checksum mismatch.")
    index = faiss.read_index(str(path))
    validate_faiss_ids(faiss, index, manifest["chunks"], store["dimensions"])
    _validate_index_kind(faiss, index, config)
    return index, manifest


def build_semantic_generation(
    database_path: str | Path,
    semantic_dir: str | Path,
    *,
    config: FaissBuildConfig | None = None,
    checkpoint_shards: int = 10,
    faiss_module: Any = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Append to a verified prior index or train once; publish only a private generation."""
    if type(checkpoint_shards) is not int or checkpoint_shards <= 0:
        raise ValueError("Checkpoint shard count must be positive.")
    source, root = Path(database_path).resolve(strict=True), Path(semantic_dir).resolve(strict=True)
    mapping = root / "mapping.sqlite"
    if mapping.resolve() == source or source.samefile(mapping):
        raise ValueError("Corpus and semantic mapping must be different files.")
    with ExitStack() as stack:
        stack.enter_context(SemanticBuildLock(source.with_name(source.name + ".pipeline.lock"), "pipeline"))
        stack.enter_context(SemanticBuildLock(root / ".vectors.lock", "vector"))
        with closing(open_connection(mapping, read_only=True)) as connection:
            connection.execute("ATTACH DATABASE ? AS corpus", (source.as_uri() + "?mode=ro",))
            connection.execute("BEGIN")
            store = read_store_config(connection)
            if store is None:
                raise ValueError("Audit/adopt legacy vectors before building additive FAISS generations.")
            identity = connection.execute(
                "SELECT corpus_id FROM corpus.pipeline_corpus_identity WHERE identity_id=1"
            ).fetchone()
            if identity is None or identity[0] != store_corpus_identity(connection):
                raise ValueError("Semantic vectors belong to a different logical corpus.")
            validate_corpus_membership(connection)
            shards = _shards(connection, root, store)
            corpus_id = identity[0]
        active = config or FaissBuildConfig()
        active.validate(store["dimensions"])
        if active.index_type == "ivfpq" and active.nprobe > active.nlist:
            raise ValueError("FAISS nprobe cannot exceed nlist.")
        faiss = faiss_module if faiss_module is not None else _import_faiss()
        vectors = sum(shard.vector_count for shard in shards)
        fingerprint = _digest(
            {"corpus_id": corpus_id, "vectors": _ledger_hash(store, shards), "faiss": active.as_dict()}
        )
        generations = root / "generations"
        if generations.is_symlink():
            raise ValueError("Generation directory must not be a symlink.")
        generations.mkdir(exist_ok=True)
        final = generations / fingerprint
        build = generations / f".build-{fingerprint}"
        if final.is_symlink() or build.is_symlink():
            raise ValueError("Generation directory must not be a symlink.")
        if final.exists():
            _, manifest = _load_generation(faiss, final, store, shards, active, corpus_id)
            if manifest["chunks"] != vectors:
                raise ValueError("Completed generation does not cover the current vector ledger.")
            return {
                "generation": fingerprint,
                "generation_dir": str(final),
                "chunks": vectors,
                "added_vectors": 0,
                "index_complete": True,
                "retrieval_ready": False,
            }
        build.mkdir(exist_ok=True)
        pointer = build / "checkpoint.json"
        start = 0
        if pointer.exists():
            saved = json.loads(_child(build, "checkpoint.json").read_text(encoding="utf-8"))
            if saved["fingerprint"] != fingerprint:
                raise ValueError("FAISS checkpoint fingerprint changed.")
            path = _child(build, saved["index_file"])
            if _file_hash(path) != saved["index_sha256"]:
                raise ValueError("FAISS checkpoint checksum mismatch.")
            index = faiss.read_index(str(path))
            validate_faiss_ids(faiss, index, saved["vector_count"], store["dimensions"])
            _validate_index_kind(faiss, index, active)
            start = len(_prefix(shards, saved["vector_count"]))
            if start != saved["shard_count"]:
                raise ValueError("FAISS checkpoint shard boundary changed.")
            _cleanup_checkpoints(build, keep=saved["index_file"])
        else:
            candidates = []
            for directory in generations.iterdir():
                if directory.is_dir() and re.fullmatch(r"[0-9a-f]{64}", directory.name):
                    candidate = json.loads(_child(directory, "manifest.json").read_text(encoding="utf-8"))
                    requested = active.as_dict()
                    recorded = dict(candidate.get("faiss", {}))
                    requested.pop("nprobe", None)
                    recorded.pop("nprobe", None)
                    if (
                        candidate.get("corpus_id") == corpus_id
                        and candidate.get("store_config") == store
                        and recorded == requested
                        and candidate.get("chunks", vectors + 1) <= vectors
                    ):
                        candidates.append((candidate["chunks"], directory))
            if candidates:
                _, previous = max(candidates)
                index, manifest = _load_generation(faiss, previous, store, shards, active, corpus_id)
                start = len(_prefix(shards, manifest["chunks"]))
            else:
                index = _new_index(faiss, root, store, shards, active)
        initial = int(index.ntotal)
        for number, shard in enumerate(shards[start:], start + 1):
            import numpy as np

            index.add_with_ids(
                _read_shard(root, shard, store["dimensions"], dtype="float32"),
                np.arange(shard.first_vector_id, shard.first_vector_id + shard.vector_count, dtype="int64"),
            )
            if number % checkpoint_shards == 0 or number == len(shards):
                _checkpoint(faiss, index, build, fingerprint, number)
            if progress:
                progress({"event": "faiss_generation_progress", "indexed_vectors": int(index.ntotal), "total": vectors})
        if hasattr(index, "nprobe"):
            index.nprobe = active.nprobe
        validate_faiss_ids(faiss, index, vectors, store["dimensions"])
        output = _child(build, "index.faiss")
        # A prior crash may have left this unpublished output. Replace only this
        # build's private file; completed generation files are never modified.
        temporary = _child(build, f"index-{uuid.uuid4().hex}.tmp")
        faiss.write_index(index, str(temporary))
        _sync(temporary)
        os.replace(temporary, output)
        manifest = {
            "format": GENERATION_FORMAT,
            "corpus_id": corpus_id,
            "generation": fingerprint,
            "store_config": store,
            "chunks": vectors,
            "faiss": active.as_dict(),
            "vector_fingerprint": _ledger_hash(store, shards),
            "index_sha256": _file_hash(output),
            "index_complete": True,
            "retrieval_ready": False,
        }
        _write_json(build / "manifest.json", manifest, replace=True)
        # Once both complete outputs exist, an interruption may repeat the last
        # private checkpoint but cannot expose an incomplete serving generation.
        pointer.unlink(missing_ok=True)
        _cleanup_checkpoints(build)
        os.rename(build, final)
        _sync_directory(generations)
        return {
            "generation": fingerprint,
            "generation_dir": str(final),
            "chunks": vectors,
            "added_vectors": vectors - initial,
            "index_complete": True,
            "retrieval_ready": False,
        }
