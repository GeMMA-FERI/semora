"""Resumable, sharded document embedding for large semantic indexes."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from semora.corpus import DEFAULT_MODEL_ID
from semora.retrieval.semantic_store import SemanticBuildSpec, SemanticVectorStore
from semora.storage import Database

DEFAULT_DIMENSIONS = 256
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_SHARD_SIZE = 100_000


@dataclass(frozen=True)
class SemanticVectorStats:
    available_chunks: int
    target_chunks: int
    indexed_chunks: int
    added_chunks: int
    complete: bool


def build_semantic_vectors(
    database_path: str | Path = "indexes/semora.sqlite",
    output_dir: str | Path = "indexes/semantic",
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str | None = None,
    dimensions: int = DEFAULT_DIMENSIONS,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_chunks: int | None = None,
    device: str | None = None,
    model: Any = None,
) -> SemanticVectorStats:
    """Embed valid chunks into atomic float16 shards and resume from checkpoints."""
    if dimensions <= 0 or batch_size <= 0 or shard_size <= 0:
        raise ValueError("Dimensions, batch size, and shard size must be positive.")
    if max_chunks is not None and max_chunks < 0:
        raise ValueError("max_chunks must be non-negative.")

    database = Database(database_path)
    try:
        database.initialize()
        spec = _semantic_build_spec(
            database,
            database_path=database_path,
            model_id=model_id,
            model_revision=model_revision,
            dimensions=dimensions,
        )
        target_chunks = spec.valid_chunks if max_chunks is None else min(max_chunks, spec.valid_chunks)
        with SemanticVectorStore(output_dir, spec) as store:
            starting_chunks = store.state.indexed_chunks
            if starting_chunks >= target_chunks:
                return SemanticVectorStats(
                    available_chunks=spec.valid_chunks,
                    target_chunks=target_chunks,
                    indexed_chunks=starting_chunks,
                    added_chunks=0,
                    complete=store.state.vectors_complete,
                )
            embedding_model = (
                model
                if model is not None
                else _load_embedding_model(
                    model_id,
                    model_revision=model_revision,
                    dimensions=dimensions,
                    device=device,
                )
            )
            with tqdm(
                total=target_chunks,
                initial=starting_chunks,
                desc="Embedding semantic chunks",
                unit="chunk",
                dynamic_ncols=True,
            ) as progress:
                while store.state.indexed_chunks < target_chunks:
                    state = store.state
                    count = min(shard_size, target_chunks - state.indexed_chunks)
                    chunk_ids, vectors = _embed_next_shard(
                        database,
                        embedding_model,
                        after_chunk_id=state.last_chunk_id,
                        count=count,
                        batch_size=batch_size,
                        dimensions=dimensions,
                    )
                    if len(chunk_ids) != count:
                        raise RuntimeError("The semantic corpus changed or ended before the expected target.")
                    file_name = f"shard-{state.next_shard_index:06d}.f16"
                    shard_path = store.vectors_dir / file_name
                    byte_size, digest = _write_vector_shard(shard_path, vectors)
                    store.record_shard(
                        shard_index=state.next_shard_index,
                        file_name=file_name,
                        chunk_ids=chunk_ids,
                        dimensions=dimensions,
                        dtype="float16",
                        byte_size=byte_size,
                        sha256=digest,
                    )
                    progress.update(len(chunk_ids))
            finished = store.state
            return SemanticVectorStats(
                available_chunks=spec.valid_chunks,
                target_chunks=target_chunks,
                indexed_chunks=finished.indexed_chunks,
                added_chunks=finished.indexed_chunks - starting_chunks,
                complete=finished.vectors_complete,
            )
    finally:
        database.close()


def _semantic_build_spec(
    database: Database,
    *,
    database_path: str | Path,
    model_id: str,
    model_revision: str | None,
    dimensions: int,
) -> SemanticBuildSpec:
    rows = database.conn.execute(
        """
        SELECT
            chunks.chunking_run_id,
            chunking_runs.config_json,
            COUNT(*) AS valid_chunks,
            MIN(chunks.chunk_id) AS first_chunk_id,
            MAX(chunks.chunk_id) AS last_chunk_id
        FROM chunks
        JOIN chunking_runs ON chunking_runs.chunking_run_id = chunks.chunking_run_id
        JOIN articles ON articles.article_id = chunks.article_id
        WHERE articles.is_valid = 1
        GROUP BY chunks.chunking_run_id, chunking_runs.config_json
        """
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("Semantic indexing requires exactly one valid chunking run.")
    row = rows[0]
    return SemanticBuildSpec(
        model_id=model_id,
        model_revision=model_revision,
        dimensions=dimensions,
        database=Path(database_path).name,
        chunking_run_id=str(row["chunking_run_id"]),
        chunking=json.loads(row["config_json"]),
        valid_chunks=int(row["valid_chunks"]),
        first_chunk_id=str(row["first_chunk_id"]),
        last_chunk_id=str(row["last_chunk_id"]),
    )


def _load_embedding_model(
    model_id: str,
    *,
    model_revision: str | None,
    dimensions: int,
    device: str | None,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Semantic vectors require the 'retrieval' extra.") from exc
    options: dict[str, Any] = {"device": device, "truncate_dim": dimensions}
    if model_revision is not None:
        options["revision"] = model_revision
    return SentenceTransformer(model_id, **options)


def _embed_next_shard(
    database: Database,
    model: Any,
    *,
    after_chunk_id: str,
    count: int,
    batch_size: int,
    dimensions: int,
) -> tuple[list[str], Any]:
    import numpy as np

    chunk_ids: list[str] = []
    vector_batches: list[Any] = []
    cursor = after_chunk_id
    while len(chunk_ids) < count:
        limit = min(batch_size, count - len(chunk_ids))
        rows = database.conn.execute(
            """
            SELECT chunks.chunk_id, chunks.text
            FROM chunks
            JOIN articles ON articles.article_id = chunks.article_id
            WHERE articles.is_valid = 1
              AND chunks.chunk_id > ?
            ORDER BY chunks.chunk_id
            LIMIT ?
            """,
            (cursor, limit),
        ).fetchall()
        if not rows:
            break
        texts = [str(row["text"]) for row in rows]
        encode = getattr(model, "encode_document", model.encode)
        encoded = encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        matrix = np.asarray(encoded, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(rows) or matrix.shape[1] < dimensions:
            raise ValueError("Embedding model returned an unexpected vector shape.")
        matrix = matrix[:, :dimensions]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("Embedding model returned non-finite or zero vectors.")
        vector_batches.append((matrix / norms).astype("float16"))
        batch_ids = [str(row["chunk_id"]) for row in rows]
        chunk_ids.extend(batch_ids)
        cursor = batch_ids[-1]
    vectors = np.concatenate(vector_batches, axis=0) if vector_batches else np.empty((0, dimensions), dtype="float16")
    return chunk_ids, vectors


def _write_vector_shard(path: Path, vectors: Any) -> tuple[int, str]:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output:
        vectors.tofile(output)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return path.stat().st_size, digest.hexdigest()
