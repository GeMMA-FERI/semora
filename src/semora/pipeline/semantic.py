"""Append-only batch embeddings with immutable shards and recoverable publication."""

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

from semora.corpus import DEFAULT_MODEL_ID
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.retrieval.semantic_vectors import _load_embedding_model
from semora.storage.connection import open_connection

STORE_FORMAT = "semora.semantic.append.v1"
_TEMPORARY_NAME = re.compile(r"\.pipeline-[0-9]{6,}-[0-9a-f]{32}\.tmp")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def chunk_input_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_chunking_config(config: dict) -> dict:
    """Compare tokenization semantics across runs without comparing run IDs."""
    allowed = {"method", "model_id", "tokenizer_revision", "token_count", "token_overlap", "source_spans"}
    if not isinstance(config, dict) or set(config).difference(allowed):
        raise ValueError("Unsupported semantic chunking configuration.")
    count, overlap = config.get("token_count"), config.get("token_overlap")
    if (
        not isinstance(config.get("model_id"), str)
        or not config["model_id"].strip()
        or type(count) is not int
        or type(overlap) is not int
        or count <= 0
        or not 0 <= overlap < count
        or config.get("source_spans") is not True
        or config.get("method", "token") != "token"
    ):
        raise ValueError("Unsupported semantic chunking configuration.")
    revision = config.get("tokenizer_revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("Tokenizer revision must be nonempty when recorded.")
    return {
        "method": "token",
        "model_id": config["model_id"],
        "tokenizer_revision": revision,
        "token_count": count,
        "token_overlap": overlap,
        "source_spans": True,
    }


def vector_config(model_id: str, model_revision: str | None, dimensions: int, chunking: dict) -> dict:
    if not isinstance(model_id, str) or not model_id.strip() or type(dimensions) is not int or dimensions <= 0:
        raise ValueError("A model identity and positive dimensions are required.")
    if model_revision is not None and (not isinstance(model_revision, str) or not model_revision.strip()):
        raise ValueError("Model revision must be nonempty when recorded.")
    return {
        "format": STORE_FORMAT,
        "model_id": model_id,
        "model_revision": model_revision,
        "dimensions": dimensions,
        "encoding": "encode_document_or_encode.default_prompt",
        "normalization": "l2-float32",
        "dtype": "float16-le",
        "chunking": canonical_chunking_config(chunking),
    }


def create_store_schema(connection: sqlite3.Connection) -> None:
    """Create compatible mapping tables without implicitly committing a caller transaction."""
    statements = (
        "CREATE TABLE IF NOT EXISTS semantic_shards (shard_index INTEGER PRIMARY KEY,file_name TEXT NOT NULL UNIQUE,"
        "first_vector_id INTEGER NOT NULL,vector_count INTEGER NOT NULL CHECK(vector_count>0),"
        "dimensions INTEGER NOT NULL CHECK(dimensions>0),dtype TEXT NOT NULL,byte_size INTEGER NOT NULL,"
        "sha256 TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS semantic_chunks (vector_id INTEGER PRIMARY KEY,chunk_id TEXT NOT NULL UNIQUE,"
        "shard_index INTEGER NOT NULL REFERENCES semantic_shards(shard_index),shard_offset INTEGER NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_semantic_chunks_shard ON semantic_chunks(shard_index,shard_offset)",
        "CREATE TABLE IF NOT EXISTS pipeline_semantic_store (store_id INTEGER PRIMARY KEY CHECK(store_id=1),"
        "format TEXT NOT NULL,config_json TEXT NOT NULL,config_sha256 TEXT NOT NULL,corpus_id TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS pipeline_semantic_inputs (chunk_id TEXT PRIMARY KEY "
        "REFERENCES semantic_chunks(chunk_id),input_sha256 TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS pipeline_semantic_pending_shards (shard_index INTEGER PRIMARY KEY,"
        "batch_id TEXT NOT NULL,file_name TEXT NOT NULL UNIQUE,temporary_file TEXT NOT NULL UNIQUE,"
        "first_vector_id INTEGER NOT NULL,vector_count INTEGER NOT NULL,dimensions INTEGER NOT NULL,"
        "dtype TEXT NOT NULL,byte_size INTEGER NOT NULL,sha256 TEXT NOT NULL,inputs_json TEXT NOT NULL,"
        "config_sha256 TEXT NOT NULL)",
    )
    for statement in statements:
        connection.execute(statement)


def read_store_config(connection: sqlite3.Connection) -> dict | None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_semantic_store'"
    ).fetchone()
    if exists is None:
        return None
    row = connection.execute(
        "SELECT format,config_json,config_sha256 FROM pipeline_semantic_store WHERE store_id=1"
    ).fetchone()
    if row is None:
        return None
    config = json.loads(row[1])
    if (
        not isinstance(config, dict)
        or row[0] != STORE_FORMAT
        or config.get("format") != STORE_FORMAT
        or chunk_input_sha256(_json(config)) != row[2]
    ):
        raise ValueError("Semantic store configuration is corrupt or unsupported.")
    expected = vector_config(config["model_id"], config["model_revision"], config["dimensions"], config["chunking"])
    if config != expected:
        raise ValueError("Unsupported semantic store configuration.")
    return config


def store_corpus_identity(connection: sqlite3.Connection) -> str | None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_semantic_store'"
    ).fetchone()
    if exists is None:
        return None
    row = connection.execute("SELECT corpus_id FROM pipeline_semantic_store WHERE store_id=1").fetchone()
    return row[0] if row else None


def _corpus_identity(connection: sqlite3.Connection) -> str:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_corpus_identity'"
    ).fetchone()
    if exists is None:
        raise ValueError("Apply the corpus identity migration before semantic processing.")
    rows = connection.execute("SELECT corpus_id FROM pipeline_corpus_identity").fetchall()
    if len(rows) != 1 or not isinstance(rows[0][0], str) or not re.fullmatch("[0-9a-f]{32}", rows[0][0]):
        raise ValueError("Corpus identity is missing or invalid.")
    return rows[0][0]


def initialize_store(connection: sqlite3.Connection, config: dict, *, corpus_identity: str) -> None:
    """Initialize a fresh store or an explicitly audited/adopted legacy store.

    Legacy adoption callers must verify artifacts and populate input hashes in
    the same transaction. This low-level helper does not certify old artifacts.
    """
    if config != vector_config(config["model_id"], config["model_revision"], config["dimensions"], config["chunking"]):
        raise ValueError("Unsupported semantic store configuration.")
    if not isinstance(corpus_identity, str) or not re.fullmatch("[0-9a-f]{32}", corpus_identity):
        raise ValueError("A valid stable corpus identity is required.")
    create_store_schema(connection)
    previous = read_store_config(connection)
    if previous is not None and store_corpus_identity(connection) != corpus_identity:
        raise ValueError("Semantic mapping belongs to a different corpus identity.")
    if previous is not None and previous != config:
        raise ValueError("Semantic model/revision/dimensions/chunking changed; use an explicit new generation.")
    if previous is None:
        connection.execute(
            "INSERT INTO pipeline_semantic_store(store_id,format,config_json,config_sha256,corpus_id) "
            "VALUES(1,?,?,?,?)",
            (STORE_FORMAT, _json(config), chunk_input_sha256(_json(config)), corpus_identity),
        )


def _batch_config(connection: sqlite3.Connection, batch_id: str) -> dict:
    row = connection.execute("SELECT * FROM pipeline_batches WHERE batch_id=?", (batch_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown batch: {batch_id}")
    count = connection.execute("SELECT COUNT(*) FROM pipeline_batch_issues WHERE batch_id=?", (batch_id,)).fetchone()[0]
    if row["state"] != "ingested" or count != row["issue_count"]:
        raise ValueError("Finish batch ingestion before embedding.")
    if connection.execute(
        "SELECT 1 FROM pipeline_batch_issues b WHERE b.batch_id=? AND NOT EXISTS "
        "(SELECT 1 FROM pipeline_issue_stages s WHERE s.newspaper_id=b.newspaper_id AND s.stage='chunks') LIMIT 1",
        (batch_id,),
    ).fetchone():
        raise ValueError("Finish/adopt chunk processing for every batch issue before embedding.")
    if connection.execute(
        "SELECT 1 FROM pipeline_batch_issues b JOIN pipeline_issue_stages s USING(newspaper_id) "
        "WHERE b.batch_id=? AND s.stage='chunks' AND s.output_count<>(SELECT COUNT(*) FROM chunks c "
        "JOIN articles a USING(article_id) WHERE a.newspaper_id=b.newspaper_id) LIMIT 1",
        (batch_id,),
    ).fetchone():
        raise ValueError("Batch chunk stage count differs from stored outputs; audit/repair is required.")
    settings = canonical_chunking_config(json.loads(row["config_json"])["chunking"])
    for run in connection.execute(
        "SELECT DISTINCT r.method,r.config_json FROM chunking_runs r JOIN chunks c USING(chunking_run_id) "
        "JOIN articles a USING(article_id) JOIN pipeline_batch_issues b USING(newspaper_id) "
        "WHERE b.batch_id=? AND a.is_valid=1",
        (batch_id,),
    ):
        if run["method"] != "token" or canonical_chunking_config(json.loads(run["config_json"])) != settings:
            raise ValueError("Batch chunks have incompatible tokenization configurations.")
    return settings


def validate_corpus_membership(connection: sqlite3.Connection) -> None:
    """Check the committed prefix against a read-only corpus attached AS corpus.

    Corpus identity alone cannot detect restoring an older copy with the same
    identity. This checks all old mapping membership/validity, without loading
    historical text or rehashing every old embedding input on each attempt.
    """
    missing = connection.execute(
        "SELECT m.chunk_id FROM semantic_chunks m LEFT JOIN corpus.chunks c ON c.chunk_id=m.chunk_id "
        "LEFT JOIN corpus.articles a ON a.article_id=c.article_id "
        "LEFT JOIN pipeline_semantic_inputs i ON i.chunk_id=m.chunk_id "
        "WHERE c.chunk_id IS NULL OR a.is_valid IS NOT 1 OR i.chunk_id IS NULL LIMIT 1"
    ).fetchone()
    if missing:
        raise ValueError("Mapped historical chunks or their input evidence are missing/invalid; audit required.")


def _store_preflight(path: Path, config: dict, corpus_identity: str, source: Path) -> None:
    if not path.is_file():
        if path.exists():
            raise ValueError("Semantic mapping path is not a file.")
        if (path.parent / "manifest.json").exists() or list((path.parent / "vectors").glob("*.f16")):
            raise ValueError("Existing semantic artifacts need explicit audit/adoption before appending.")
        return
    with closing(open_connection(path, read_only=True)) as connection:
        previous = read_store_config(connection)
        if previous is None:
            tables = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
            if (
                tables is None
                and not (path.parent / "manifest.json").exists()
                and not list((path.parent / "vectors").glob("*.f16"))
            ):
                return  # A crash before the initial transaction committed.
            raise ValueError("Legacy semantic mapping needs explicit audit/adoption before appending.")
        if previous != config:
            raise ValueError("Semantic model/revision/dimensions/chunking changed; use an explicit new generation.")
        if store_corpus_identity(connection) != corpus_identity:
            raise ValueError("Semantic mapping belongs to a different corpus identity.")
        connection.execute("ATTACH DATABASE ? AS corpus", (source.as_uri() + "?mode=ro",))
        validate_corpus_membership(connection)


_BATCH_ROWS = (
    "SELECT c.chunk_id,c.text,m.vector_id,m.shard_offset,i.input_sha256,s.* FROM corpus.chunks c "
    "JOIN corpus.articles a USING(article_id) JOIN corpus.pipeline_batch_issues b USING(newspaper_id) "
    "LEFT JOIN semantic_chunks m ON m.chunk_id=c.chunk_id "
    "LEFT JOIN semantic_shards s ON s.shard_index=m.shard_index "
    "LEFT JOIN pipeline_semantic_inputs i ON i.chunk_id=c.chunk_id WHERE b.batch_id=? AND a.is_valid=1"
)


def _coverage(connection: sqlite3.Connection, batch_id: str, vectors: Path, config: dict) -> tuple[int, int]:
    eligible = completed = 0
    checked = set()
    for row in connection.execute(_BATCH_ROWS, (batch_id,)):
        eligible += 1
        if row["vector_id"] is not None:
            if row["input_sha256"] is None or row["input_sha256"] != chunk_input_sha256(row["text"]):
                raise ValueError("Mapped chunk source differs or lacks audited input evidence; repair is required.")
            if (
                row["shard_index"] is None
                or row["dimensions"] != config["dimensions"]
                or row["dtype"] != "float16"
                or not 0 <= row["shard_offset"] < row["vector_count"]
                or row["vector_id"] != row["first_vector_id"] + row["shard_offset"]
                or row["byte_size"] != row["vector_count"] * config["dimensions"] * 2
            ):
                raise ValueError("Mapped chunk/shard layout is inconsistent; explicit repair is required.")
            if row["shard_index"] not in checked:
                _check_file(_safe_file(vectors, row["file_name"]), row["byte_size"], row["sha256"])
                checked.add(row["shard_index"])
            completed += 1
    return eligible, completed


def _safe_file(directory: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Unsafe semantic artifact filename.")
    target = directory / name
    if target.resolve().parent != directory.resolve():
        raise ValueError("Semantic artifact escapes its directory.")
    return target


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _check_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or _file_hash(path) != digest:
        raise ValueError(f"Semantic shard bytes do not match their journal: {path.name}")


def _sync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_file(temporary: Path, final: Path) -> None:
    """Publish fully flushed bytes atomically without replacing an existing shard."""
    if os.name == "nt":
        # Unlike POSIX rename(), Windows rename() refuses an existing target.
        os.rename(temporary, final)
    else:
        os.link(temporary, final)
    _sync_directory(final.parent)


def _finish_pending(connection: sqlite3.Connection, vectors: Path, row: sqlite3.Row, config: dict) -> int:
    if row["config_sha256"] != chunk_input_sha256(_json(config)):
        raise ValueError("Pending shard belongs to a different semantic configuration.")
    inputs = json.loads(row["inputs_json"])
    if (
        len(inputs) != row["vector_count"]
        or len({item[0] for item in inputs}) != len(inputs)
        or row["vector_count"] <= 0
        or row["dimensions"] != config["dimensions"]
        or row["dtype"] != "float16"
        or row["byte_size"] != row["vector_count"] * config["dimensions"] * 2
        or row["file_name"] != f"shard-{row['shard_index']:06d}.f16"
    ):
        raise ValueError("Invalid pending semantic shard membership.")
    for chunk_id, digest in inputs:
        current = connection.execute(
            "SELECT c.text,r.config_json,r.method FROM corpus.chunks c "
            "JOIN corpus.chunking_runs r USING(chunking_run_id) "
            "JOIN corpus.articles a USING(article_id) JOIN corpus.pipeline_batch_issues b USING(newspaper_id) "
            "WHERE c.chunk_id=? AND b.batch_id=? AND a.is_valid=1",
            (chunk_id, row["batch_id"]),
        ).fetchone()
        if (
            current is None
            or chunk_input_sha256(current["text"]) != digest
            or current["method"] != "token"
            or canonical_chunking_config(json.loads(current["config_json"])) != config["chunking"]
        ):
            raise ValueError("Chunk source/configuration changed during semantic publication.")
    final = _safe_file(vectors, row["file_name"])
    temporary = _safe_file(vectors, row["temporary_file"])
    if final.exists():
        _check_file(final, row["byte_size"], row["sha256"])
    else:
        _check_file(temporary, row["byte_size"], row["sha256"])
        _publish_file(temporary, final)
    if temporary.exists():
        _check_file(temporary, row["byte_size"], row["sha256"])
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        next_vector = connection.execute("SELECT COALESCE(MAX(vector_id),-1)+1 FROM semantic_chunks").fetchone()[0]
        next_shard = connection.execute("SELECT COALESCE(MAX(shard_index),-1)+1 FROM semantic_shards").fetchone()[0]
        if next_vector != row["first_vector_id"] or next_shard != row["shard_index"]:
            raise ValueError("Pending semantic shard conflicts with committed mapping order.")
        connection.execute(
            "INSERT INTO semantic_shards(shard_index,file_name,first_vector_id,vector_count,"
            "dimensions,dtype,byte_size,sha256) "
            "VALUES(?,?,?,?,?,?,?,?)",
            tuple(
                row[key]
                for key in (
                    "shard_index",
                    "file_name",
                    "first_vector_id",
                    "vector_count",
                    "dimensions",
                    "dtype",
                    "byte_size",
                    "sha256",
                )
            ),
        )
        connection.executemany(
            "INSERT INTO semantic_chunks(vector_id,chunk_id,shard_index,shard_offset) VALUES(?,?,?,?)",
            [(next_vector + offset, item[0], next_shard, offset) for offset, item in enumerate(inputs)],
        )
        connection.executemany("INSERT INTO pipeline_semantic_inputs(chunk_id,input_sha256) VALUES(?,?)", inputs)
        connection.execute("DELETE FROM pipeline_semantic_pending_shards WHERE shard_index=?", (next_shard,))
    if temporary.exists():
        temporary.unlink()
        _sync_directory(vectors)
    return len(inputs)


def _recover_pending(connection: sqlite3.Connection, vectors: Path, config: dict) -> int:
    recovered = 0
    for row in connection.execute("SELECT * FROM pipeline_semantic_pending_shards ORDER BY shard_index").fetchall():
        recovered += _finish_pending(connection, vectors, row, config)
    return recovered


def _cleanup_unjournaled_temporaries(connection: sqlite3.Connection, vectors: Path) -> int:
    """Remove only this store's reserved UUID temps lacking a committed intent."""
    journaled = {row[0] for row in connection.execute("SELECT temporary_file FROM pipeline_semantic_pending_shards")}
    removed = 0
    for path in vectors.glob(".pipeline-*.tmp"):
        if not _TEMPORARY_NAME.fullmatch(path.name) or path.name in journaled:
            continue
        checked = _safe_file(vectors, path.name)
        if checked.is_symlink() or not checked.is_file():
            raise ValueError("Reserved semantic temporary path is not an owned regular file.")
        checked.unlink()
        removed += 1
    if removed:
        _sync_directory(vectors)
    return removed


def _embed(rows: list[sqlite3.Row], model: Any, dimensions: int, batch_size: int) -> Any:
    import numpy as np

    encode = getattr(model, "encode_document", None) or getattr(model, "encode", None)
    if not callable(encode):
        raise ValueError("Embedding model must provide encode_document or encode.")
    blocks = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        matrix = np.asarray(
            encode(
                [row["text"] for row in batch],
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype="float32",
        )
        if matrix.ndim != 2 or matrix.shape[0] != len(batch) or matrix.shape[1] < dimensions:
            raise ValueError("Embedding model returned an unexpected vector shape.")
        matrix = matrix[:, :dimensions]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("Embedding model returned non-finite or zero vectors.")
        blocks.append((matrix / norms).astype("<f2"))
    return np.concatenate(blocks, axis=0)


def _prepare_shard(
    connection: sqlite3.Connection,
    vectors: Path,
    batch_id: str,
    rows: list[sqlite3.Row],
    matrix: Any,
    config: dict,
) -> None:
    shard_index = connection.execute("SELECT COALESCE(MAX(shard_index),-1)+1 FROM semantic_shards").fetchone()[0]
    first_vector = connection.execute("SELECT COALESCE(MAX(vector_id),-1)+1 FROM semantic_chunks").fetchone()[0]
    file_name = f"shard-{shard_index:06d}.f16"
    temporary_name = f".pipeline-{shard_index:06d}-{uuid.uuid4().hex}.tmp"
    temporary = _safe_file(vectors, temporary_name)
    if _safe_file(vectors, file_name).exists():
        raise ValueError("Unjournaled shard file already exists; audit it instead of overwriting.")
    journaled = created = False
    try:
        with temporary.open("xb") as output:
            created = True
            matrix.tofile(output)
            output.flush()
            os.fsync(output.fileno())
        _sync_directory(vectors)
        size, digest = temporary.stat().st_size, _file_hash(temporary)
        if size != len(rows) * config["dimensions"] * 2:
            raise ValueError("Unexpected semantic shard byte size.")
        with connection:
            connection.execute(
                "INSERT INTO pipeline_semantic_pending_shards(shard_index,batch_id,file_name,temporary_file,"
                "first_vector_id,vector_count,dimensions,dtype,byte_size,sha256,inputs_json,config_sha256) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    shard_index,
                    batch_id,
                    file_name,
                    temporary_name,
                    first_vector,
                    len(rows),
                    config["dimensions"],
                    "float16",
                    size,
                    digest,
                    _json([[r["chunk_id"], chunk_input_sha256(r["text"])] for r in rows]),
                    chunk_input_sha256(_json(config)),
                ),
            )
        journaled = True
    finally:
        if created and not journaled and temporary.exists():
            temporary.unlink()


def index_batch_vectors(
    database_path: str | Path,
    batch_id: str,
    semantic_dir: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str | None = None,
    dimensions: int = 256,
    batch_size: int = 64,
    shard_size: int = 10000,
    device: str | None = None,
    model: Any = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Append selected pending chunks; stage completion does not publish retrieval."""
    if type(batch_size) is not int or type(shard_size) is not int or min(batch_size, shard_size) <= 0:
        raise ValueError("Embedding batch and shard sizes must be positive integers.")
    source = Path(database_path).resolve(strict=True)
    directory = Path(semantic_dir).resolve()
    mapping = (directory / "mapping.sqlite").resolve()
    if mapping == source or (mapping.exists() and os.path.samefile(mapping, source)):
        raise ValueError("Semantic mapping database cannot alias the corpus database.")
    with ExitStack() as stack:
        stack.enter_context(SemanticBuildLock(source.with_name(source.name + ".pipeline.lock"), "pipeline"))
        stack.enter_context(SemanticBuildLock(directory / ".vectors.lock", "vector"))
        with closing(open_connection(source, read_only=True)) as corpus:
            chunking = _batch_config(corpus, batch_id)
            corpus_identity = _corpus_identity(corpus)
        config = vector_config(model_id, model_revision, dimensions, chunking)
        _store_preflight(mapping, config, corpus_identity, source)
        directory.mkdir(parents=True, exist_ok=True)
        vectors = directory / "vectors"
        vectors.mkdir(exist_ok=True)
        connection = sqlite3.connect(mapping.as_uri() + "?mode=rwc", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=60000")
            connection.execute("ATTACH DATABASE ? AS corpus", (source.as_uri() + "?mode=ro",))
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                initialize_store(connection, config, corpus_identity=corpus_identity)
            removed_temporaries = _cleanup_unjournaled_temporaries(connection, vectors)
            recovered = _recover_pending(connection, vectors, config)
            eligible, initial = _coverage(connection, batch_id, vectors, config)
            added = 0
            cursor = ""
            while True:
                # Local paging only. Every retry restarts from explicit batch
                # membership minus committed mappings; no global hash cursor.
                rows = connection.execute(
                    _BATCH_ROWS + " AND m.chunk_id IS NULL AND c.chunk_id>? ORDER BY c.chunk_id LIMIT ?",
                    (batch_id, cursor, shard_size),
                ).fetchall()
                if not rows:
                    break
                if model is None:
                    model = _load_embedding_model(
                        model_id, model_revision=model_revision, dimensions=dimensions, device=device
                    )
                matrix = _embed(rows, model, dimensions, batch_size)
                _prepare_shard(connection, vectors, batch_id, rows, matrix, config)
                pending = connection.execute("SELECT * FROM pipeline_semantic_pending_shards").fetchone()
                _finish_pending(connection, vectors, pending, config)
                added += len(rows)
                cursor = rows[-1]["chunk_id"]
                if progress is not None:
                    progress(
                        {
                            "event": "semantic_checkpoint",
                            "batch_id": batch_id,
                            "completed_chunks": initial + added,
                            "total": eligible,
                        }
                    )
            _, completed = _coverage(connection, batch_id, vectors, config)
            return {
                "batch_id": batch_id,
                "stage": "vectors",
                "format": STORE_FORMAT,
                "config_sha256": chunk_input_sha256(_json(config)),
                "eligible_chunks": eligible,
                "completed_chunks": completed,
                "added_chunks": added,
                "skipped_chunks": initial,
                "recovered_chunks": recovered,
                "pending_chunks": eligible - completed,
                "removed_unjournaled_temporaries": removed_temporaries,
                "complete": completed == eligible,
                "retrieval_ready": False,
                "publication_ready": False,
                "compatibility_verified": False,
                "declared_configuration_match": True,
                "revisions_recorded": model_revision is not None and chunking["tokenizer_revision"] is not None,
                "historical_input_hashes_rechecked": False,
            }
        finally:
            connection.close()
