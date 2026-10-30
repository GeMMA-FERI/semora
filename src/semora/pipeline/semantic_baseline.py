"""Read-only structural audit of legacy v2 chunk/vector artifacts.

This verifies stored data and checksums, not historical model execution. Legacy
stores contain neither embedding-input text hashes nor an explicit byte order.
Tokenization and embedding completeness cannot be reconstructed without models.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import fields
from pathlib import Path
from typing import Any

from semora.corpus.indexer import _stable_id
from semora.corpus.spans import LineMap
from semora.pipeline.baseline import _schema
from semora.pipeline.manifest import _invalid_constant, _relative_path, _unique_object
from semora.retrieval.semantic_faiss import FaissBuildConfig, _build_fingerprint
from semora.retrieval.semantic_store import SEMANTIC_FORMAT, SemanticBuildSpec
from semora.storage.connection import open_connection

_BLOCK_BYTES = 1024 * 1024
_NORM_TOLERANCE = 0.002
_ERROR_LIMIT = 50
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SPEC_KEYS = {field.name for field in fields(SemanticBuildSpec)} | {"format"}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _load_json(value: str) -> dict:
    document = json.loads(value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(document, dict):
        raise ValueError("Semantic metadata must be a JSON object")
    return document


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _spec(document: dict) -> SemanticBuildSpec:
    if set(document) != _SPEC_KEYS or document["format"] != SEMANTIC_FORMAT:
        raise ValueError("Unknown or malformed legacy semantic build specification")
    for name in ("model_id", "database", "chunking_run_id", "first_chunk_id", "last_chunk_id"):
        _string(document[name], name)
    if any(char in document["database"] for char in "/\\:"):
        raise ValueError("Legacy semantic database must be a basename, independent of the data root")
    for name in ("dimensions", "valid_chunks"):
        _integer(document[name], name, minimum=1)
    if document["dimensions"] * 2 > _BLOCK_BYTES:
        raise ValueError("Vector row exceeds the bounded audit block size")
    revision = document["model_revision"]
    if revision is not None:
        _string(revision, "model_revision")
    config = document["chunking"]
    allowed = {"model_id", "token_count", "token_overlap", "source_spans", "tokenizer_revision"}
    required = allowed - {"tokenizer_revision"}
    if not isinstance(config, dict) or required - config.keys() or config.keys() - allowed:
        raise ValueError("Unknown or incomplete chunking configuration")
    _string(config["model_id"], "chunking.model_id")
    count = _integer(config["token_count"], "token_count", minimum=1)
    overlap = _integer(config["token_overlap"], "token_overlap")
    if overlap >= count or config["source_spans"] is not True:
        raise ValueError("Chunking overlap/source-span configuration is invalid")
    if config.get("tokenizer_revision") is not None:
        _string(config["tokenizer_revision"], "tokenizer_revision")
    return SemanticBuildSpec(**{key: value for key, value in document.items() if key != "format"})


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Semantic input is not a file inside its artifact root: {path.name}")
    return resolved


def _manifest(root: Path, spec: SemanticBuildSpec) -> tuple[dict, str]:
    raw = _inside(root, root / "manifest.json").read_bytes()
    manifest = _load_json(raw.decode("utf-8"))
    required = _SPEC_KEYS | {"normalized", "metric", "chunks", "vectors_complete"}
    optional = {"created_at", "index_complete", "faiss", "build_fingerprint"}
    if required - manifest.keys() or manifest.keys() - required - optional:
        raise ValueError("Unknown or incomplete semantic manifest fields")
    if {key: manifest[key] for key in _SPEC_KEYS} != spec.as_dict():
        raise ValueError("Semantic manifest disagrees with the mapping build specification")
    if (
        manifest["normalized"] is not True
        or manifest["metric"] != "cosine_via_inner_product"
        or manifest["vectors_complete"] is not True
    ):
        raise ValueError("Legacy manifest is not a complete normalized inner-product vector store")
    if _integer(manifest["chunks"], "manifest.chunks") != spec.valid_chunks:
        raise ValueError("Manifest chunk count does not match the complete vector snapshot")
    if "index_complete" in manifest and type(manifest["index_complete"]) is not bool:
        raise ValueError("Manifest index_complete must be a boolean")
    if "build_fingerprint" in manifest and (
        not isinstance(manifest["build_fingerprint"], str) or _DIGEST.fullmatch(manifest["build_fingerprint"]) is None
    ):
        raise ValueError("Malformed FAISS build fingerprint")
    if "faiss" in manifest:
        config = manifest["faiss"]
        keys = {"index_type", "nlist", "pq_m", "pq_bits", "train_samples", "nprobe"}
        if not isinstance(config, dict) or set(config) != keys:
            raise ValueError("Unknown or incomplete FAISS configuration")
        for key in keys - {"index_type"}:
            _integer(config[key], f"faiss.{key}", minimum=1)
        active = FaissBuildConfig(**config)
        active.validate(spec.dimensions)
        if "build_fingerprint" in manifest and manifest["build_fingerprint"] != _build_fingerprint(spec, active):
            raise ValueError("FAISS build fingerprint disagrees with recorded configuration")
    return manifest, hashlib.sha256(raw).hexdigest()


def _vector_file(root: Path, shard: dict) -> tuple[str, int]:
    import numpy as np

    name = _string(shard["file_name"], "shard.file_name")
    if any(char in name for char in "/\\:") or name in {".", ".."}:
        raise ValueError("Shard file name must be a basename")
    path = _inside(root, root / "vectors" / name)
    size = shard["vector_count"] * shard["dimensions"] * 2
    if shard["dtype"] != "float16" or shard["byte_size"] != size or path.stat().st_size != size:
        raise ValueError(f"Shard dtype or byte size is inconsistent: {name}")
    if not isinstance(shard["sha256"], str) or _DIGEST.fullmatch(shard["sha256"]) is None:
        raise ValueError(f"Malformed shard SHA-256: {name}")
    digest = hashlib.sha256()
    row_bytes = shard["dimensions"] * 2
    block_bytes = max(1, _BLOCK_BYTES // row_bytes) * row_bytes
    vectors = 0
    before = path.stat()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
            if len(block) % row_bytes:
                raise ValueError(f"Truncated vector row: {name}")
            # Legacy writers used native float16.tofile(). Windows/Linux x86
            # are little-endian; this decoding assumption is explicit evidence.
            matrix = np.frombuffer(block, dtype="<f2").reshape(-1, shard["dimensions"]).astype("float32")
            if not np.isfinite(matrix).all():
                raise ValueError(f"Non-finite legacy vectors: {name}")
            norms = np.linalg.norm(matrix, axis=1)
            if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.0) > _NORM_TOLERANCE):
                raise ValueError(f"Legacy vectors are not L2 normalized within float16 tolerance: {name}")
            vectors += matrix.shape[0]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Shard changed during the audit: {name}")
    actual = digest.hexdigest()
    if actual != shard["sha256"]:
        raise ValueError(f"Shard SHA-256 mismatch: {name}")
    return actual, vectors


def _chunks(conn: sqlite3.Connection, report: dict, error: Callable[[str], None], progress) -> None:
    fingerprint = hashlib.sha256()
    verified = checked = 0
    for newspaper in conn.execute("SELECT * FROM newspapers ORDER BY newspaper_id"):
        content = newspaper["content"]
        line_map = LineMap(content)
        relative_path = newspaper["relative_path"]
        try:
            _relative_path(relative_path, "newspaper.relative_path")
            if not relative_path.endswith(".md"):
                raise ValueError("Newspaper canonical path must end in .md")
        except ValueError as exc:
            error(f"{newspaper['newspaper_id']}: {exc}")
            relative_path = ""
        if newspaper["newspaper_id"] != _stable_id("newspaper", relative_path):
            error(f"{newspaper['newspaper_id']}: newspaper ID disagrees with canonical path")
        articles = {
            row["article_id"]: row
            for row in conn.execute(
                "SELECT * FROM articles WHERE newspaper_id=? AND is_valid=1", (newspaper["newspaper_id"],)
            )
        }
        for article in articles.values():
            if (
                type(article["char_start"]) is not int
                or type(article["char_end"]) is not int
                or not 0 <= article["char_start"] < article["char_end"] <= len(content)
            ):
                error(f"{article['article_id']}: parent article source span is invalid")
                continue
            if article["article_id"] != _stable_id(
                "article",
                relative_path,
                str(article["char_start"]),
                str(article["char_end"]),
            ):
                error(f"{article['article_id']}: article ID disagrees with canonical path and source span")
            if line_map.span(article["char_start"], article["char_end"]) != (
                article["line_start"],
                article["line_end"],
            ):
                error(f"{article['article_id']}: parent article line span disagrees with source text")
        next_indices: dict[str, int] = {}
        for chunk in conn.execute(
            "SELECT c.* FROM chunks c JOIN articles a USING(article_id) "
            "WHERE a.newspaper_id=? AND a.is_valid=1 ORDER BY c.article_id,c.chunk_index,c.chunk_id",
            (newspaper["newspaper_id"],),
        ):
            checked += 1
            reasons = []
            article = articles[chunk["article_id"]]
            index = next_indices.get(chunk["article_id"], 0)
            next_indices[chunk["article_id"]] = index + 1
            if type(chunk["chunk_index"]) is not int or chunk["chunk_index"] != index:
                reasons.append("chunk indices are not contiguous from zero")
            span_names = ("char_start", "char_end", "line_start", "line_end")
            if any(type(chunk[name]) is not int for name in span_names):
                reasons.append("chunk spans are absent or non-integer")
            elif not (0 <= chunk["char_start"] < chunk["char_end"] <= len(content)):
                reasons.append("chunk character span is outside its newspaper")
            else:
                if content[chunk["char_start"] : chunk["char_end"]] != chunk["text"]:
                    reasons.append("chunk text disagrees with its source span")
                if line_map.span(chunk["char_start"], chunk["char_end"]) != (
                    chunk["line_start"],
                    chunk["line_end"],
                ):
                    reasons.append("chunk line span disagrees with source text")
                if (
                    type(article["char_end"]) is not int
                    or type(article["char_start"]) is not int
                    or not isinstance(article["content"], str)
                ):
                    reasons.append("parent article source span is absent")
                else:
                    article_content_start = article["char_end"] - len(article["content"])
                    if not (
                        article["char_start"]
                        <= article_content_start
                        <= chunk["char_start"]
                        < chunk["char_end"]
                        <= article["char_end"]
                    ):
                        reasons.append("chunk is outside parent article content")
                    if content[article_content_start : article["char_end"]] != article["content"]:
                        reasons.append("parent article content disagrees with source span")
                expected_id = _stable_id(
                    "chunk",
                    chunk["article_id"],
                    str(chunk["chunk_index"]),
                    str(chunk["char_start"]),
                    str(chunk["char_end"]),
                )
                if chunk["chunk_id"] != expected_id:
                    reasons.append("chunk ID disagrees with historical stable-ID inputs")
            if chunk["run_id"] != article["run_id"] or article["run_id"] != newspaper["run_id"]:
                reasons.append("chunk/article/newspaper corpus runs disagree")
            if chunk["method"] != "token":
                reasons.append("chunk method is not token")
            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                reasons.append("chunk text is empty or not a string")
            for reason in reasons:
                error(f"{chunk['chunk_id']}: {reason}")
            if not reasons:
                verified += 1
                fingerprint.update(
                    _json(
                        [
                            chunk["chunk_id"],
                            hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
                            chunk["article_id"],
                            chunk["char_start"],
                            chunk["char_end"],
                        ]
                    ).encode("utf-8")
                )
            if progress and checked % 10000 == 0:
                progress(
                    {
                        "event": "semantic_chunks_audited",
                        "checked": checked,
                        "total": report["counts"]["eligible_chunks"],
                    }
                )
    report["counts"]["verified_chunks"] = verified
    report["chunk_content_fingerprint"] = fingerprint.hexdigest()
    if progress:
        progress({"event": "semantic_chunks_audited", "checked": checked, "total": report["counts"]["eligible_chunks"]})


def _inspect(conn: sqlite3.Connection, root: Path, report: dict, error, progress) -> None:
    _schema(conn)
    rows = conn.execute("SELECT * FROM semantic.semantic_build_state").fetchall()
    if len(rows) != 1 or rows[0]["state_id"] != 1:
        raise ValueError("Legacy semantic build state must contain exactly singleton state 1")
    state = dict(rows[0])
    for name in ("indexed_chunks", "next_shard_index"):
        _integer(state[name], name)
    document = _load_json(state["spec_json"])
    spec = _spec(document)
    report["spec"] = spec.as_dict()
    report["spec_fingerprint"] = spec.fingerprint
    report["mapping_state"] = state
    if state["format"] != SEMANTIC_FORMAT or state["spec_fingerprint"] != spec.fingerprint:
        raise ValueError("Mapping build format or specification fingerprint is inconsistent")
    if state["vectors_complete"] != 1 or state["indexed_chunks"] != spec.valid_chunks:
        raise ValueError("Legacy semantic vectors are incomplete")
    if state["last_chunk_id"] != spec.last_chunk_id:
        raise ValueError("Mapping checkpoint does not end at the declared final chunk")
    manifest, manifest_hash = _manifest(root, spec)
    report["manifest_sha256"] = manifest_hash
    report["manifest"] = manifest
    report["faiss"]["configuration_fingerprint_verified"] = bool(
        "build_fingerprint" in manifest and "faiss" in manifest
    )
    report["model"] = {"model_id": spec.model_id, "model_revision": spec.model_revision}
    report["chunking"] = {
        "config": spec.chunking,
        "tokenizer_revision": spec.chunking.get("tokenizer_revision"),
        "tokenization_completeness_verified": False,
    }
    runs = conn.execute("""
        SELECT c.chunking_run_id,r.method,r.config_json,COUNT(*) AS chunks,
               MIN(c.chunk_id) AS first_chunk_id,MAX(c.chunk_id) AS last_chunk_id
        FROM chunks c JOIN articles a USING(article_id)
        LEFT JOIN chunking_runs r USING(chunking_run_id) WHERE a.is_valid=1
        GROUP BY c.chunking_run_id,r.method,r.config_json
    """).fetchall()
    if len(runs) != 1:
        raise ValueError("Legacy semantic snapshot must have exactly one eligible chunking run")
    run = runs[0]
    if (
        run["chunking_run_id"] != spec.chunking_run_id
        or run["method"] != "token"
        or _load_json(run["config_json"]) != spec.chunking
    ):
        raise ValueError("Stored chunking run disagrees with legacy semantic specification")
    if (run["chunks"], run["first_chunk_id"], run["last_chunk_id"]) != (
        spec.valid_chunks,
        spec.first_chunk_id,
        spec.last_chunk_id,
    ):
        raise ValueError("Eligible corpus chunks do not match the declared semantic snapshot bounds/count")
    report["declared_configuration_match"] = True
    report["counts"]["eligible_chunks"] = run["chunks"]
    missing = conn.execute("""
        SELECT COUNT(*) FROM chunks c JOIN articles a USING(article_id) WHERE a.is_valid=1
        AND NOT EXISTS(SELECT 1 FROM semantic.semantic_chunks m WHERE m.chunk_id=c.chunk_id)
    """).fetchone()[0]
    extra = conn.execute("""
        SELECT COUNT(*) FROM semantic.semantic_chunks m LEFT JOIN chunks c USING(chunk_id)
        LEFT JOIN articles a USING(article_id) WHERE c.chunk_id IS NULL OR a.article_id IS NULL OR a.is_valid IS NOT 1
    """).fetchone()[0]
    mapped = conn.execute("SELECT COUNT(*) FROM semantic.semantic_chunks").fetchone()[0]
    report["counts"].update(mapped_chunks=mapped, missing_mappings=missing, extra_mappings=extra)
    if missing or extra or mapped != spec.valid_chunks:
        error("Vector/chunk mapping membership does not exactly match eligible corpus chunks")
    orphans = conn.execute("""
        SELECT COUNT(*) FROM chunks c WHERE NOT EXISTS(SELECT 1 FROM articles a WHERE a.article_id=c.article_id)
    """).fetchone()[0]
    if orphans:
        error(f"Stored chunks without parent articles: {orphans}")
    _chunks(conn, report, error, progress)
    shards = [dict(row) for row in conn.execute("SELECT * FROM semantic.semantic_shards ORDER BY shard_index")]
    report["shards"] = []
    next_vector = 0
    for number, shard in enumerate(shards):
        try:
            for name in ("shard_index", "first_vector_id"):
                _integer(shard[name], name)
            for name in ("vector_count", "dimensions", "byte_size"):
                _integer(shard[name], name, minimum=1)
            if (
                shard["shard_index"] != number
                or shard["first_vector_id"] != next_vector
                or shard["dimensions"] != spec.dimensions
            ):
                raise ValueError("Shard indices, vector ranges or dimensions are inconsistent")
            row = conn.execute(
                """
                SELECT COUNT(*) AS count,MIN(vector_id) AS first,MAX(vector_id) AS last,
                       COUNT(DISTINCT vector_id) AS unique_vectors,COUNT(DISTINCT chunk_id) AS unique_chunks,
                       SUM(CASE WHEN typeof(vector_id)<>'integer' OR typeof(shard_index)<>'integer'
                           OR typeof(shard_offset)<>'integer' OR shard_offset<0 OR shard_offset>=?
                           OR vector_id<>?+shard_offset THEN 1 ELSE 0 END) AS invalid
                FROM semantic.semantic_chunks WHERE shard_index=?
            """,
                (shard["vector_count"], next_vector, number),
            ).fetchone()
            if (
                row["count"] != shard["vector_count"]
                or row["unique_vectors"] != row["count"]
                or row["unique_chunks"] != row["count"]
                or row["invalid"]
                or row["first"] != next_vector
                or row["last"] != next_vector + shard["vector_count"] - 1
            ):
                raise ValueError("Shard mapping rows do not match exact contiguous vector/offset ranges")
            digest, count = _vector_file(root, shard)
            report["shards"].append({**shard, "verified_sha256": digest})
            report["counts"]["verified_vectors"] += count
        except (ValueError, OSError) as exc:
            error(f"Shard {number}: {exc}")
        next_vector += _integer(shard["vector_count"], "vector_count", minimum=1)
        if progress:
            progress({"event": "semantic_shards_audited", "checked": number + 1, "total": len(shards)})
    dangling = conn.execute("""
        SELECT COUNT(*) FROM semantic.semantic_chunks m
        WHERE NOT EXISTS(SELECT 1 FROM semantic.semantic_shards s WHERE s.shard_index=m.shard_index)
    """).fetchone()[0]
    if dangling or next_vector != spec.valid_chunks or state["next_shard_index"] != len(shards):
        error("Shard ledger total/checkpoint does not cover the complete vector snapshot")
    report["counts"]["shards"] = len(shards)
    report["recorded_revisions"] = bool(spec.model_revision and spec.chunking.get("tokenizer_revision"))
    # A recorded revision can be a mutable branch (for example "main"). Even
    # immutable revision strings cannot prove legacy execution/input identity.
    report["compatibility_verified"] = False


def audit_semantic_baseline(
    database_path: str | Path,
    semantic_dir: str | Path,
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Audit legacy artifacts with read-only snapshots and bounded vector reads.

    Stop writers before using the result for adoption. No locks, migration,
    tokenizer, embedding model, or FAISS index are opened for writing here.
    """
    report: dict[str, Any] = {
        "format": "semora.semantic-baseline.v1",
        "eligible": False,
        "compatibility_verified": False,
        "recorded_revisions": False,
        "declared_configuration_match": False,
        "counts": {"eligible_chunks": 0, "mapped_chunks": 0, "verified_chunks": 0, "verified_vectors": 0},
        "errors": [],
        "error_count": 0,
        "shards": [],
        "vector_layout": {
            "dtype": "float16",
            "decoded_byte_order": "little",
            "legacy_byte_order_recorded": False,
            "normalization_tolerance": _NORM_TOLERANCE,
        },
        "faiss": {
            "adopted": False,
            "identity_verified": False,
            "evidence": "FAISS file contents and vector IDs are deferred to the generation audit.",
        },
        "embedding_input_link_verified": False,
        "limitations": [
            "Historical tokenizer execution and chunking completeness are not reconstructed.",
            "Legacy vectors have no embedding-input text hashes; current text-to-vector execution is not proven.",
            "Legacy float16 files do not record byte order; Windows/Linux little-endian decoding is assumed.",
            "Recorded model/tokenizer revisions may be absent; no revision is inferred from current caches.",
        ],
    }

    def error(reason: str) -> None:
        report["error_count"] += 1
        if len(report["errors"]) < _ERROR_LIMIT:
            report["errors"].append(reason)

    try:
        root = Path(semantic_dir).resolve(strict=True)
        mapping = _inside(root, root / "mapping.sqlite")
        with closing(open_connection(database_path, read_only=True)) as conn:
            conn.execute("ATTACH DATABASE ? AS semantic", (mapping.as_uri() + "?mode=ro",))
            conn.execute("BEGIN")
            _inspect(conn, root, report, error, progress)
    except (ValueError, OSError, sqlite3.DatabaseError, KeyError, TypeError) as exc:
        error(str(exc))
    report["eligible"] = report["error_count"] == 0
    report["compatibility_verified"] = report["eligible"] and report["compatibility_verified"]
    report["evidence_fingerprint"] = hashlib.sha256(_json(report).encode("utf-8")).hexdigest()
    return report
