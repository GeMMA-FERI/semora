"""Register structurally audited legacy vectors without recomputing embeddings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import ExitStack, closing
from copy import deepcopy
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
    audit_report: dict | None = None,
) -> dict:
    """Preserve legacy vector IDs/files and record present input hashes.

    This trusts recorded legacy configuration and structurally verified files.
    Capturing input hashes now does not prove the historical embedding execution.
    Only control schema/identity may remain after a failed adoption; source text,
    old manifests and vector files are never rewritten. Supplying audit_report
    trusts that the corpus and vector artifacts have not changed since its
    read-only audit; adoption still checks each mapped chunk's input hash.
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
            from semora.pipeline.publication_state import invalidate_publication
            invalidate_publication(source)
            with corpus.conn:
                corpus.conn.execute("BEGIN IMMEDIATE")
                corpus_id = corpus.conn.execute(
                    "SELECT corpus_id FROM pipeline_corpus_identity WHERE identity_id=1"
                ).fetchone()[0]
                if audit_report is None:
                    report = audit_semantic_baseline(source, root, progress=progress)
                else:
                    report = deepcopy(audit_report)
                    if (
                        not isinstance(report, dict)
                        or report.get("format") != "semora.semantic-baseline.v1"
                        or report.get("eligible") is not True
                        or report.get("error_count") != 0
                        or not isinstance(report.get("spec"), dict)
                        or not isinstance(report.get("counts"), dict)
                        or not isinstance(report.get("evidence_fingerprint"), str)
                        or report.get("manifest_sha256")
                        != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
                    ):
                        raise ValueError("Supplied semantic audit report is ineligible, invalid, or stale.")
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
                            "audit_source": "fresh_audit" if audit_report is None else "trusted_prior_report",
                        }
                        return report
        finally:
            corpus.close()
