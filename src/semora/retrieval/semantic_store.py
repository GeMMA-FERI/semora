"""Durable metadata and chunk mapping for resumable semantic indexes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

SEMANTIC_FORMAT = "semora.semantic.v2"


@dataclass(frozen=True)
class SemanticBuildSpec:
    model_id: str
    model_revision: str | None
    dimensions: int
    database: str
    chunking_run_id: str
    chunking: dict
    valid_chunks: int
    first_chunk_id: str
    last_chunk_id: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.database or not self.chunking_run_id:
            raise ValueError("Semantic build identifiers must not be empty.")
        if self.dimensions <= 0 or self.valid_chunks <= 0:
            raise ValueError("Semantic dimensions and valid chunk count must be positive.")
        if not self.first_chunk_id or not self.last_chunk_id:
            raise ValueError("Semantic chunk bounds must not be empty.")

    def as_dict(self) -> dict:
        return {"format": SEMANTIC_FORMAT, **asdict(self)}

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SemanticBuildState:
    indexed_chunks: int
    last_chunk_id: str
    next_shard_index: int
    vectors_complete: bool


@dataclass(frozen=True)
class SemanticShard:
    shard_index: int
    file_name: str
    first_vector_id: int
    vector_count: int
    dimensions: int
    dtype: str
    byte_size: int
    sha256: str


class SemanticVectorStore:
    """Transactional semantic build state stored beside vector shard files."""

    def __init__(self, directory: str | Path, spec: SemanticBuildSpec) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vectors_dir = self.directory / "vectors"
        self.vectors_dir.mkdir(exist_ok=True)
        self.database_path = self.directory / "mapping.sqlite"
        self.conn = sqlite3.connect(self.database_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self._initialize(spec)
        self.spec = spec

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SemanticVectorStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def state(self) -> SemanticBuildState:
        row = self.conn.execute(
            """
            SELECT indexed_chunks, last_chunk_id, next_shard_index, vectors_complete
            FROM semantic_build_state
            WHERE state_id = 1
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("Semantic build state is missing.")
        return SemanticBuildState(
            indexed_chunks=int(row["indexed_chunks"]),
            last_chunk_id=str(row["last_chunk_id"]),
            next_shard_index=int(row["next_shard_index"]),
            vectors_complete=bool(row["vectors_complete"]),
        )

    def record_shard(
        self,
        *,
        shard_index: int,
        file_name: str,
        chunk_ids: list[str],
        dimensions: int,
        dtype: str,
        byte_size: int,
        sha256: str,
    ) -> None:
        state = self.state
        if shard_index != state.next_shard_index:
            raise ValueError(f"Expected semantic shard {state.next_shard_index}, got {shard_index}.")
        if not chunk_ids or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("A semantic shard must contain unique chunk IDs.")
        if state.indexed_chunks + len(chunk_ids) > self.spec.valid_chunks:
            raise ValueError("Semantic shard exceeds the valid chunk count in the build specification.")
        if dimensions != self.spec.dimensions or dtype != "float16":
            raise ValueError("Semantic shard layout does not match the build specification.")
        if Path(file_name).name != file_name or byte_size <= 0 or len(sha256) != 64:
            raise ValueError("Semantic shard metadata is invalid.")
        first_vector_id = state.indexed_chunks
        mappings = [
            (first_vector_id + offset, chunk_id, shard_index, offset) for offset, chunk_id in enumerate(chunk_ids)
        ]
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO semantic_shards (
                    shard_index, file_name, first_vector_id, vector_count,
                    dimensions, dtype, byte_size, sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shard_index,
                    file_name,
                    first_vector_id,
                    len(chunk_ids),
                    dimensions,
                    dtype,
                    byte_size,
                    sha256,
                ),
            )
            self.conn.executemany(
                """
                INSERT INTO semantic_chunks (vector_id, chunk_id, shard_index, shard_offset)
                VALUES (?, ?, ?, ?)
                """,
                mappings,
            )
            indexed_chunks = state.indexed_chunks + len(chunk_ids)
            self.conn.execute(
                """
                UPDATE semantic_build_state
                SET indexed_chunks = ?, last_chunk_id = ?, next_shard_index = ?,
                    vectors_complete = ?, updated_at = CURRENT_TIMESTAMP
                WHERE state_id = 1
                """,
                (
                    indexed_chunks,
                    chunk_ids[-1],
                    shard_index + 1,
                    int(indexed_chunks == self.spec.valid_chunks),
                ),
            )

    def chunk_ids(self, vector_ids: list[int]) -> list[str | None]:
        if not vector_ids:
            return []
        unique = list(dict.fromkeys(vector_ids))
        placeholders = ",".join("?" for _ in unique)
        rows = self.conn.execute(
            f"SELECT vector_id, chunk_id FROM semantic_chunks WHERE vector_id IN ({placeholders})",
            unique,
        ).fetchall()
        mapping = {int(row["vector_id"]): str(row["chunk_id"]) for row in rows}
        return [mapping.get(vector_id) for vector_id in vector_ids]

    def shards(self) -> list[SemanticShard]:
        rows = self.conn.execute(
            """
            SELECT shard_index, file_name, first_vector_id, vector_count,
                   dimensions, dtype, byte_size, sha256
            FROM semantic_shards
            ORDER BY shard_index
            """
        ).fetchall()
        return [
            SemanticShard(
                shard_index=int(row["shard_index"]),
                file_name=str(row["file_name"]),
                first_vector_id=int(row["first_vector_id"]),
                vector_count=int(row["vector_count"]),
                dimensions=int(row["dimensions"]),
                dtype=str(row["dtype"]),
                byte_size=int(row["byte_size"]),
                sha256=str(row["sha256"]),
            )
            for row in rows
        ]

    def _initialize(self, spec: SemanticBuildSpec) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS semantic_build_state (
                state_id INTEGER PRIMARY KEY CHECK (state_id = 1),
                format TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                spec_fingerprint TEXT NOT NULL,
                indexed_chunks INTEGER NOT NULL DEFAULT 0,
                last_chunk_id TEXT NOT NULL DEFAULT '',
                next_shard_index INTEGER NOT NULL DEFAULT 0,
                vectors_complete INTEGER NOT NULL DEFAULT 0 CHECK (vectors_complete IN (0, 1)),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS semantic_shards (
                shard_index INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL UNIQUE,
                first_vector_id INTEGER NOT NULL,
                vector_count INTEGER NOT NULL CHECK (vector_count > 0),
                dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                dtype TEXT NOT NULL,
                byte_size INTEGER NOT NULL CHECK (byte_size > 0),
                sha256 TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS semantic_chunks (
                vector_id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                shard_index INTEGER NOT NULL,
                shard_offset INTEGER NOT NULL,
                FOREIGN KEY (shard_index) REFERENCES semantic_shards(shard_index)
            );

            CREATE INDEX IF NOT EXISTS idx_semantic_chunks_shard
            ON semantic_chunks(shard_index, shard_offset);
            """
        )
        row = self.conn.execute("SELECT spec_fingerprint FROM semantic_build_state WHERE state_id = 1").fetchone()
        if row is None:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO semantic_build_state (
                        state_id, format, spec_json, spec_fingerprint
                    ) VALUES (1, ?, ?, ?)
                    """,
                    (
                        SEMANTIC_FORMAT,
                        json.dumps(spec.as_dict(), ensure_ascii=False, sort_keys=True),
                        spec.fingerprint,
                    ),
                )
        elif str(row["spec_fingerprint"]) != spec.fingerprint:
            raise ValueError(
                "Semantic build configuration or corpus changed; use a new output directory or explicitly rebuild it."
            )


def read_semantic_build_spec(directory: str | Path) -> SemanticBuildSpec | None:
    database_path = Path(directory).resolve() / "mapping.sqlite"
    if not database_path.is_file():
        return None
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT spec_json FROM semantic_build_state WHERE state_id = 1").fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("Semantic vector build specification is missing.")
    values = json.loads(str(row[0]))
    values.pop("format", None)
    return SemanticBuildSpec(**values)
