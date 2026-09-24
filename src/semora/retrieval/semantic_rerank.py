"""Read only selected original shard vectors and rescore with cosine similarity."""

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def rerank_candidates(
    mapping: sqlite3.Connection, directory: Path, query: np.ndarray, vector_ids: list[int]
) -> list[tuple[int, float]]:
    if not vector_ids:
        return []
    ids = list(dict.fromkeys(vector_ids))
    groups: dict[str, list] = defaultdict(list)
    for start in range(0, len(ids), 900):
        batch = ids[start : start + 900]
        rows = mapping.execute(
            f"""SELECT c.vector_id,c.shard_offset,s.file_name,s.vector_count,s.dimensions,s.dtype,s.byte_size
            FROM semantic_chunks c JOIN semantic_shards s USING(shard_index)
            WHERE c.vector_id IN ({",".join("?" for _ in batch)})""",
            batch,
        ).fetchall()
        for row in rows:
            groups[row["file_name"]].append(row)
    if sum(map(len, groups.values())) != len(ids):
        raise ValueError("Original vector mapping is incomplete; cannot rerank.")
    root = (directory / "vectors").resolve()
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(query)
    if not np.isfinite(query).all() or norm == 0:
        raise ValueError("Invalid query vector for reranking.")
    query = query / norm
    ranked = []
    for name, rows in groups.items():
        path = (root / name).resolve()
        first = rows[0]
        count, dimensions = int(first["vector_count"]), int(first["dimensions"])
        if not path.is_relative_to(root) or first["dtype"] != "float16" or dimensions != query.size:
            raise ValueError("Original vector shard does not match the index.")
        expected = count * dimensions * np.dtype("float16").itemsize
        if int(first["byte_size"]) != expected or path.stat().st_size != expected:
            raise ValueError("Original vector shard has an invalid size.")
        offsets = [int(row["shard_offset"]) for row in rows]
        if any(offset < 0 or offset >= count for offset in offsets):
            raise ValueError("Original vector shard offset is invalid.")
        # Sort by physical offset; fancy indexing copies only these rows, not the shard.
        rows = sorted(rows, key=lambda row: int(row["shard_offset"]))
        source = np.memmap(path, mode="r", dtype="float16", shape=(count, dimensions))
        try:
            vectors = np.asarray(source[[int(row["shard_offset"]) for row in rows]], dtype=np.float32)
        finally:
            source._mmap.close()
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or np.any(norms == 0):
            raise ValueError("Original shard contains invalid vectors; cannot rerank.")
        scores = (vectors / norms[:, None]) @ query
        ranked.extend((int(row["vector_id"]), float(score)) for row, score in zip(rows, scores, strict=True))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))
