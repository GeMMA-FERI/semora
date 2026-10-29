"""Audit and adopt historical outputs without rerunning tokenization or NLP.

The corpus audit reconstructs articles from the stored issue text. Legacy FTS
adoption instead relies on exact membership plus recorded completion evidence:
contentless FTS tables cannot reconstruct the original indexed text.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path

from semora.corpus import indexer
from semora.corpus.spans import split_articles
from semora.pipeline.batches import SPLITTER, _metadata_identifiers
from semora.pipeline.manifest import _relative_path
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database
from semora.storage.connection import open_connection
from semora.storage.metadata import normalize_newspaper_metadata
from semora.storage.migrations import MIGRATIONS_DIR, migration_version

_BASELINE_ID = "stored-articles-v1"
_ERROR_LIMIT = 50


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count(conn: sqlite3.Connection, query: str) -> int:
    return int(conn.execute(query).fetchone()[0])


def _schema(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")} if exists else set()
    known = {migration_version(path) for path in MIGRATIONS_DIR.glob("*.sql")}
    if not set(range(1, 14)).issubset(applied) or applied.difference(known):
        raise ValueError("Baseline adoption requires an audited Semora schema with migrations 001-013 applied.")


def audit_legacy_lexical(conn: sqlite3.Connection) -> dict:
    """Report exact row membership; never claim reconstructed FTS text evidence."""
    from semora.pipeline.lexical import lemma_config, lexical_schema_compatible, surface_config

    valid = _count(conn, "SELECT COUNT(*) FROM articles WHERE is_valid=1")
    mapped = _count(conn, "SELECT COUNT(*) FROM article_fts_map")
    missing_mapping = _count(conn, """
        SELECT COUNT(*) FROM articles a WHERE a.is_valid=1
        AND NOT EXISTS (SELECT 1 FROM article_fts_map m WHERE m.article_id=a.article_id)
    """)
    invalid_mapping = _count(conn, """
        SELECT COUNT(*) FROM article_fts_map m LEFT JOIN articles a USING(article_id)
        WHERE a.article_id IS NULL OR a.is_valid IS NOT 1
    """)
    missing_surface_rows = _count(conn, """
        SELECT COUNT(*) FROM article_fts_map m
        WHERE NOT EXISTS (SELECT 1 FROM article_fts f WHERE f.rowid=m.fts_id)
    """)
    extra_surface_rows = _count(conn, """
        SELECT COUNT(*) FROM article_fts f
        WHERE NOT EXISTS (SELECT 1 FROM article_fts_map m WHERE m.fts_id=f.rowid)
    """)
    surface_state = conn.execute("SELECT * FROM article_fts_state WHERE state_id=1").fetchone()
    surface_complete = bool(surface_state and surface_state["complete"] == 1)
    surface_membership = not any((missing_mapping, invalid_mapping, missing_surface_rows, extra_surface_rows))
    surface_counters = bool(surface_state and surface_state["indexed_articles"] == mapped == valid)
    surface_schema = lexical_schema_compatible(conn, "surface")
    surface = {
        "eligible": surface_complete and surface_membership and surface_counters and surface_schema,
        "schema_compatible": surface_schema,
        "config": surface_config(),
        "eligible_articles": valid,
        "mapped_articles": mapped,
        "missing_mappings": missing_mapping,
        "invalid_mappings": invalid_mapping,
        "missing_fts_rows": missing_surface_rows,
        "extra_fts_rows": extra_surface_rows,
        "exact_membership": surface_membership,
        "legacy_complete": surface_complete,
        "counters_match": surface_counters,
        "state": dict(surface_state) if surface_state else None,
        "source_text_verified": False,
        "evidence": "Exact eligible-article/rowid membership and trusted legacy completion state; "
                    "contentless FTS indexed text is not reconstructed.",
    }
    lemma_state = conn.execute("SELECT * FROM article_lemma_index_state WHERE state_id=1").fetchone()
    pipeline_type = str(lemma_state["pipeline_type"]) if lemma_state else None
    known_pipeline = bool(pipeline_type and pipeline_type.strip())
    lemma_count = _count(conn, "SELECT COUNT(*) FROM article_lemmas")
    missing_lemmas = _count(conn, """
        SELECT COUNT(*) FROM article_fts_map m
        WHERE NOT EXISTS (SELECT 1 FROM article_lemmas l WHERE l.article_id=m.article_id)
    """)
    extra_lemmas = _count(conn, """
        SELECT COUNT(*) FROM article_lemmas l
        WHERE NOT EXISTS (SELECT 1 FROM article_fts_map m WHERE m.article_id=l.article_id)
    """)
    missing_lemma_rows = _count(conn, """
        SELECT COUNT(*) FROM article_fts_map m
        WHERE NOT EXISTS (SELECT 1 FROM article_lemma_fts f WHERE f.rowid=m.fts_id)
    """)
    extra_lemma_rows = _count(conn, """
        SELECT COUNT(*) FROM article_lemma_fts f
        WHERE NOT EXISTS (SELECT 1 FROM article_fts_map m WHERE m.fts_id=f.rowid)
    """)
    wrong_pipeline = 0 if pipeline_type is None else int(conn.execute(
        "SELECT COUNT(*) FROM article_lemmas WHERE pipeline_type<>?", (pipeline_type,)
    ).fetchone()[0])
    lemma_complete = bool(lemma_state and lemma_state["complete"] == 1)
    lemma_membership = surface_membership and not any((
        missing_lemmas, extra_lemmas, missing_lemma_rows, extra_lemma_rows, wrong_pipeline,
    ))
    lemma_counters = bool(lemma_state and lemma_count == mapped == lemma_state["surface_articles"]
                          == lemma_state["processed_articles"] == lemma_state["indexed_articles"])
    lemma = {
        "eligible": surface["eligible"] and lemma_complete and lemma_membership and lemma_counters and known_pipeline
                    and lexical_schema_compatible(conn, "lemma"),
        "schema_compatible": lexical_schema_compatible(conn, "lemma"),
        "config": lemma_config(pipeline_type, None) if pipeline_type else None,
        "pipeline_type": pipeline_type,
        "historical_model_fingerprint": None,
        "materialized_articles": lemma_count,
        "missing_materialized_rows": missing_lemmas,
        "extra_materialized_rows": extra_lemmas,
        "wrong_pipeline_rows": wrong_pipeline,
        "missing_fts_rows": missing_lemma_rows,
        "extra_fts_rows": extra_lemma_rows,
        "exact_membership": lemma_membership,
        "legacy_complete": lemma_complete,
        "counters_match": lemma_counters,
        "state": dict(lemma_state) if lemma_state else None,
        "source_text_verified": False,
        "evidence": "Exact materialized lemma/mapping/rowid membership and trusted legacy completion state; "
                    "contentless FTS indexed text and historical model resources are not reconstructed.",
    }
    return {"surface": surface, "lemma": lemma}


def _audit(conn: sqlite3.Connection, progress: Callable[[dict], None] | None) -> dict:
    _schema(conn)
    errors: list[dict] = []
    error_count = 0

    def error(reason: str, newspaper_id: str | None = None) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < _ERROR_LIMIT:
            errors.append({"newspaper_id": newspaper_id, "reason": reason})

    newspapers = _count(conn, "SELECT COUNT(*) FROM newspapers")
    articles = _count(conn, "SELECT COUNT(*) FROM articles")
    chunks = _count(conn, "SELECT COUNT(*) FROM chunks")
    orphan_articles = _count(conn, """
        SELECT COUNT(*) FROM articles a
        WHERE NOT EXISTS (SELECT 1 FROM newspapers n WHERE n.newspaper_id=a.newspaper_id)
    """)
    if orphan_articles:
        error(f"Articles with no newspaper: {orphan_articles}")
    orphan_runs = _count(conn, """
        SELECT COUNT(*) FROM newspapers n
        WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id=n.run_id)
    """)
    if orphan_runs:
        error(f"Newspapers with no corpus run: {orphan_runs}")
    paths: dict[str, str] = {}
    identifiers: dict[str, str] = {}
    verified_newspapers = verified_articles = 0
    fingerprint = hashlib.sha256()
    for number, newspaper in enumerate(conn.execute("SELECT * FROM newspapers ORDER BY newspaper_id"), 1):
        newspaper_id = newspaper["newspaper_id"]
        before = error_count
        relative_path = newspaper["relative_path"]
        try:
            _relative_path(relative_path, "relative_path")
            if not relative_path.endswith(".md"):
                raise ValueError("Canonical issue path must end in .md")
        except ValueError as exc:
            error(str(exc), newspaper_id)
            relative_path = relative_path or ""
        if relative_path:
            prior = paths.setdefault(relative_path.casefold(), newspaper_id)
            if prior != newspaper_id:
                error("Canonical issue paths collide by case", newspaper_id)
            if indexer._stable_id("newspaper", relative_path) != newspaper_id:
                error("Stored newspaper ID does not match its canonical relative path", newspaper_id)
        try:
            metadata = json.loads(newspaper["metadata_json"] or "{}")
            if not isinstance(metadata, dict):
                raise ValueError("Stored metadata must be a JSON object")
            normalized = normalize_newspaper_metadata(metadata)
            for field in ("date", "publisher", "source", "rights", "title", "urn", "issue", "volume"):
                if newspaper[field] != normalized[field]:
                    error(f"Stored normalized metadata differs in {field}", newspaper_id)
            source_ids = _metadata_identifiers(metadata)
            urn = newspaper["urn"]
            if urn:
                if not isinstance(urn, str) or not urn.casefold().startswith("urn:"):
                    error("Stored normalized URN is malformed", newspaper_id)
                else:
                    if source_ids and urn.casefold() not in source_ids:
                        error("Stored normalized URN disagrees with metadata", newspaper_id)
                    source_ids.add(urn.casefold())
            if len(source_ids) > 1:
                error("Stored metadata has ambiguous source identifiers", newspaper_id)
            for source_id in source_ids:
                prior = identifiers.setdefault(source_id, newspaper_id)
                if prior != newspaper_id:
                    error("Source identifier occurs at multiple canonical paths", newspaper_id)
        except (ValueError, TypeError) as exc:
            error(f"Invalid stored issue metadata: {exc}", newspaper_id)
        expected = {
            article.article_id: article
            for article in (
                indexer._article_record(
                    source, newspaper_id=newspaper_id,
                    run_id=newspaper["run_id"], relative_path=relative_path,
                )
                for source in split_articles(newspaper["content"])
            )
        }
        seen = 0
        article_state = hashlib.sha256()
        for article in conn.execute("SELECT * FROM articles WHERE newspaper_id=? ORDER BY article_id", (newspaper_id,)):
            seen += 1
            article_state.update(_json([
                article["article_id"], article["is_valid"], article["cleaning_reason"],
            ]).encode("utf-8"))
            anticipated = expected.pop(article["article_id"], None)
            if anticipated is None:
                error("Stored article ID is absent from exact re-split output", newspaper_id)
                continue
            for field in (
                "run_id", "newspaper_id", "title", "content", "char_start", "char_end", "line_start", "line_end",
            ):
                if article[field] != getattr(anticipated, field):
                    error(f"Stored article {article['article_id']} differs in {field}", newspaper_id)
            if article["is_valid"] not in (None, 0, 1):
                error("Invalid article validity flag", newspaper_id)
        if expected:
            error(f"Missing articles from exact re-split output: {len(expected)}", newspaper_id)
        if before == error_count:
            verified_newspapers += 1
            verified_articles += seen
        fingerprint.update(_json([
            newspaper_id, relative_path, _sha(newspaper["content"]),
            _sha(newspaper["metadata_json"] or "{}"), seen, article_state.hexdigest(),
        ]).encode("utf-8"))
        if progress and (number % 1000 == 0 or number == newspapers):
            progress({"event": "baseline_articles_audited", "checked": number, "total": newspapers})
    configurations = []
    for row in conn.execute("SELECT * FROM chunking_runs ORDER BY chunking_run_id"):
        try:
            config = json.loads(row["config_json"])
        except (ValueError, TypeError):
            config = None
        configurations.append({
            "chunking_run_id": row["chunking_run_id"], "run_id": row["run_id"],
            "method": row["method"], "config": config,
            "tokenizer_revision_verified": False,
            "evidence": "Recorded configuration only; chunk outputs and tokenizer revision are not verified.",
        })
    corpus = {
        "eligible": error_count == 0,
        "total_newspapers": newspapers, "total_articles": articles, "total_chunks": chunks,
        "verified_newspapers": verified_newspapers, "verified_articles": verified_articles,
        "orphan_articles": orphan_articles, "orphan_newspaper_runs": orphan_runs,
        "config": {"splitter": SPLITTER}, "fingerprint": fingerprint.hexdigest(),
        "error_count": error_count, "errors": errors,
    }
    return {
        "format": "semora.baseline.v1", "eligible": corpus["eligible"], "corpus": corpus,
        "lexical": audit_legacy_lexical(conn),
        "chunking_configurations": configurations,
        "chunk_adoption": "deferred; no tokenization or chunk completion proof is performed",
        "retrieval_ready": False,
    }


def audit_baseline(database_path: str | Path, *, progress: Callable[[dict], None] | None = None) -> dict:
    """Audit an existing snapshot without applying migrations or loading models."""
    with closing(open_connection(database_path, read_only=True)) as conn:
        conn.execute("BEGIN")
        return _audit(conn, progress)


def adopt_baseline(database_path: str | Path, *, progress: Callable[[dict], None] | None = None) -> dict:
    """Adopt proven historical articles and eligible legacy lexical memberships.

    Retrieval readers and legacy writers must be stopped by the operator. Only
    additive schema migrations are applied; source text and NLP outputs remain
    unchanged. Audit and completion markers share one writer-reserved snapshot.
    """
    from semora.pipeline.lexical import adopt_legacy_lexical

    target = Path(database_path).resolve(strict=True)
    with ExitStack() as locks:
        locks.enter_context(SemanticBuildLock(target.with_name(target.name + ".pipeline.lock"), "pipeline"))
        locks.enter_context(SemanticBuildLock(target.with_name(target.name + ".ner.lock"), "NER"))
        locks.enter_context(SemanticBuildLock(target.parent / "semantic" / ".vectors.lock", "vector"))
        with closing(open_connection(target, read_only=True)) as existing:
            _schema(existing)
        database = Database(target)
        try:
            database.initialize()
            conn = database.conn
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                report = _audit(conn, progress)
                if not report["corpus"]["eligible"]:
                    raise ValueError("Historical article audit failed: " + _json(report["corpus"]))
                config = _json({"splitter": SPLITTER})
                added = 0
                for newspaper in conn.execute("SELECT newspaper_id,content FROM newspapers ORDER BY newspaper_id"):
                    newspaper_id = newspaper["newspaper_id"]
                    count = int(conn.execute(
                        "SELECT COUNT(*) FROM articles WHERE newspaper_id=?", (newspaper_id,),
                    ).fetchone()[0])
                    content_hash = _sha(newspaper["content"])
                    existing = conn.execute(
                        "SELECT * FROM pipeline_issue_stages WHERE newspaper_id=? AND stage='articles'", (newspaper_id,)
                    ).fetchone()
                    if existing:
                        if (existing["input_sha256"], existing["config_json"], existing["output_count"]) != (
                            content_hash, config, count,
                        ):
                            raise ValueError("Existing article-stage completion conflicts with audited baseline")
                    else:
                        conn.execute(
                            "INSERT INTO pipeline_issue_stages"
                            "(newspaper_id,stage,input_sha256,config_json,output_count) "
                            "VALUES (?,'articles',?,?,?)", (newspaper_id, content_hash, config, count),
                        )
                        added += 1
                lexical_before = {
                    row[0]: row[1]
                    for row in conn.execute("SELECT stage,COUNT(*) FROM pipeline_lexical_state GROUP BY stage")
                }
                adopt_legacy_lexical(conn, report["lexical"])
                lexical = {
                    stage: int(conn.execute(
                        "SELECT COUNT(*) FROM pipeline_lexical_state WHERE stage=?", (stage,),
                    ).fetchone()[0]) - lexical_before.get(stage, 0)
                    for stage in ("surface", "lemma")
                }
                baseline_id = _BASELINE_ID + "-" + _sha(_json(report))
                conn.execute(
                    "INSERT INTO pipeline_baselines(baseline_id,config_json,summary_json,state) "
                    "VALUES (?,?,?,'articles_adopted') ON CONFLICT(baseline_id) DO NOTHING",
                    (baseline_id, config, _json(report)),
                )
                report["adoption"] = {
                    "baseline_id": baseline_id, "article_issue_markers_added": added,
                    "lexical_markers_added": lexical, "chunk_markers_added": 0,
                }
                return report
        finally:
            database.close()
