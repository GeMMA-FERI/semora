"""Register structurally audited legacy vectors without recomputing embeddings."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path

from semora.pipeline.semantic import (
    chunk_input_sha256,
    initialize_store,
    read_store_config,
    store_corpus_identity,
    vector_config,
)
from semora.pipeline.semantic_baseline import audit_semantic_baseline
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database
from semora.storage.connection import open_connection


def adopt_semantic_baseline(
    database_path: str | Path,
    semantic_dir: str | Path,
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Preserve legacy vector IDs/files and record present input hashes.

    This trusts recorded legacy configuration and structurally verified files.
    Capturing input hashes now does not prove the historical embedding execution.
    Only control schema/identity may remain after a failed adoption; source text,
    old manifests and vector files are never rewritten.
    """
    source = Path(database_path).resolve(strict=True)
    root = Path(semantic_dir).resolve(strict=True)
    mapping = root / "mapping.sqlite"
    if source == mapping.resolve() or (mapping.exists() and source.samefile(mapping)):
        raise ValueError("Corpus and semantic mapping must be different files.")
    with ExitStack() as stack:
        stack.enter_context(SemanticBuildLock(source.with_name(source.name + ".pipeline.lock"), "pipeline"))
        stack.enter_context(SemanticBuildLock(root / ".vectors.lock", "vector"))
        with closing(open_connection(source, read_only=True)) as reader:
            versions = {row[0] for row in reader.execute("SELECT version FROM schema_migrations")}
            if not set(range(1, 14)).issubset(versions):
                raise ValueError("Explicitly audit/migrate historical schema before semantic adoption.")
        corpus = Database(source)
        try:
            corpus.initialize()
            with corpus.conn:
                corpus.conn.execute("BEGIN IMMEDIATE")
                corpus_id = corpus.conn.execute(
                    "SELECT corpus_id FROM pipeline_corpus_identity WHERE identity_id=1"
                ).fetchone()[0]
                report = audit_semantic_baseline(source, root, progress=progress)
                if not report["eligible"]:
                    raise ValueError("Semantic baseline audit failed: " + json.dumps(report["errors"]))
                spec = report["spec"]
                config = vector_config(spec["model_id"], spec["model_revision"], spec["dimensions"], spec["chunking"])
                with closing(open_connection(mapping)) as conn:
                    conn.execute("ATTACH DATABASE ? AS corpus", (source.as_uri() + "?mode=ro",))
                    with conn:
                        conn.execute("BEGIN IMMEDIATE")
                        previous = read_store_config(conn)
                        if previous is not None and store_corpus_identity(conn) != corpus_id:
                            raise ValueError("Semantic store belongs to a different logical corpus.")
                        initialize_store(conn, config, corpus_identity=corpus_id)
                        conn.execute(
                            "CREATE TABLE IF NOT EXISTS pipeline_semantic_adoptions "
                            "(receipt_id TEXT PRIMARY KEY,corpus_id TEXT NOT NULL,evidence_json TEXT NOT NULL,"
                            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                        )
                        added = checked = 0
                        for row in conn.execute(
                            "SELECT m.chunk_id,c.text,i.input_sha256 FROM semantic_chunks m "
                            "JOIN corpus.chunks c USING(chunk_id) LEFT JOIN pipeline_semantic_inputs i USING(chunk_id) "
                            "ORDER BY m.vector_id"
                        ):
                            digest = chunk_input_sha256(row["text"])
                            if row["input_sha256"] is not None and row["input_sha256"] != digest:
                                raise ValueError("Existing semantic input hash differs; explicit repair is required.")
                            if row["input_sha256"] is None:
                                conn.execute(
                                    "INSERT INTO pipeline_semantic_inputs(chunk_id,input_sha256) VALUES (?,?)",
                                    (row["chunk_id"], digest),
                                )
                                added += 1
                            checked += 1
                            if progress is not None and checked % 10000 == 0:
                                progress({"event": "semantic_inputs_adopted", "checked": checked})
                        if checked != report["counts"]["mapped_chunks"]:
                            raise ValueError("Semantic membership changed during adoption.")
                        conn.execute(
                            "INSERT OR IGNORE INTO pipeline_semantic_adoptions(receipt_id,corpus_id,evidence_json) "
                            "VALUES (?,?,?)",
                            (report["evidence_fingerprint"], corpus_id, json.dumps(report, sort_keys=True)),
                        )
                        report["adoption"] = {
                            "corpus_id": corpus_id,
                            "input_hashes_added": added,
                            "receipt_id": report["evidence_fingerprint"],
                            "vector_files_rewritten": 0,
                            "historical_embedding_execution_verified": False,
                        }
                        return report
        finally:
            corpus.close()
