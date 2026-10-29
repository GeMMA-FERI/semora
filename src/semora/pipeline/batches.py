"""Manifest-driven additive corpus ingestion; retrieval publication is separate.

All source files are checked before mutation. Each issue/stage commits its
output and completion marker together. Previously stored text is immutable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

from semora.corpus import indexer
from semora.corpus.spans import LineMap, SourceArticle, split_articles
from semora.pipeline.manifest import BatchIssue, BatchManifest, read_issue
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database
from semora.storage.connection import open_connection
from semora.storage.metadata import normalize_newspaper_metadata

STAGES = ("newspapers", "articles", "chunks")
SPLITTER = "semora.atx-h1.v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _read_source(input_root: str | Path, issue: BatchIssue) -> tuple[str, dict]:
    content, metadata = read_issue(input_root, issue)
    # Match Path.read_text() in legacy ingestion, preserving historical IDs/spans.
    return content.replace("\r\n", "\n").replace("\r", "\n"), metadata


def _paths(conn: sqlite3.Connection | None) -> dict[str, set[str]]:
    paths: dict[str, set[str]] = {}
    if conn is not None:
        for row in conn.execute("SELECT relative_path FROM newspapers WHERE relative_path IS NOT NULL"):
            paths.setdefault(row[0].casefold(), set()).add(row[0])
    return paths


def _metadata_identifiers(metadata: dict) -> set[str]:
    record = metadata.get("Record", {})
    values = record.get("identifier", []) if isinstance(record, dict) else []
    values = values if isinstance(values, list) else [values]
    identifiers = set()
    for value in values:
        text = value.get("#text") if isinstance(value, dict) else value
        if isinstance(text, str) and text.casefold().startswith(("urn:", "gf-urn:")):
            identifiers.add(text.casefold())
    return identifiers


def _source_identifier(issue: BatchIssue, metadata: dict) -> str | None:
    identifiers = _metadata_identifiers(metadata)
    if len(identifiers) > 1:
        raise ValueError("Ambiguous metadata source identifiers: " + issue.relative_path)
    if issue.source_identifier:
        if identifiers and issue.source_identifier.casefold() not in identifiers:
            raise ValueError("Manifest source identifier disagrees with metadata: " + issue.relative_path)
        return issue.source_identifier
    return next(iter(identifiers), None)


def _classify(
    conn: sqlite3.Connection | None,
    issue: BatchIssue,
    content: str,
    metadata: dict,
    paths: dict[str, set[str]],
    batch_id: str,
) -> tuple[str, str]:
    aliases = paths.get(issue.relative_path.casefold(), set())
    if aliases and aliases != {issue.relative_path}:
        return "conflict", "Canonical path collides by case with an existing issue."
    if conn is None:
        return "new", ""
    newspaper_id = indexer._stable_id("newspaper", issue.relative_path)
    row = conn.execute(
        "SELECT newspaper_id, content, metadata_json, relative_path FROM newspapers "
        "WHERE newspaper_id=? OR relative_path=?",
        (newspaper_id, issue.relative_path),
    ).fetchall()
    if len(row) > 1 or (row and row[0]["newspaper_id"] != newspaper_id):
        return "conflict", "Existing path has a different identity; explicit baseline adoption is required."
    urn = normalize_newspaper_metadata(metadata)["urn"]
    if (
        urn
        and conn.execute(
            "SELECT 1 FROM newspapers WHERE urn=? AND newspaper_id<>? LIMIT 1", (urn, newspaper_id)
        ).fetchone()
    ):
        return "conflict", "Source URN already exists at another canonical path."
    source_identifier = _source_identifier(issue, metadata)
    if source_identifier and _has_table(conn, "pipeline_batch_issues"):
        previous_keys = {
            row[0]
            for row in conn.execute(
                "SELECT source_identifier_key FROM pipeline_batch_issues "
                "WHERE newspaper_id=? AND source_identifier_key IS NOT NULL",
                (newspaper_id,),
            )
        }
        if previous_keys and previous_keys != {source_identifier.casefold()}:
            return "conflict", "Existing manifest source identity changed; explicit identity repair is required."
        if conn.execute(
            "SELECT 1 FROM pipeline_batch_issues WHERE source_identifier_key=? AND newspaper_id<>? LIMIT 1",
            (source_identifier.casefold(), newspaper_id),
        ).fetchone():
            return "conflict", "Source identifier already exists at another canonical path."
    if not row:
        return "new", ""
    current = row[0]
    if current["relative_path"] != issue.relative_path:
        return "conflict", "Existing identity has a different canonical path."
    if current["content"] != content:
        return "conflict", "Text replacement is not supported by additive ingestion."
    previous_metadata = json.loads(current["metadata_json"] or "{}")
    previous_identifiers = _metadata_identifiers(previous_metadata)
    next_identifiers = _metadata_identifiers(metadata)
    if previous_identifiers and previous_identifiers != next_identifiers:
        return "conflict", "Existing source identity changed; explicit identity repair is required."
    if _json(previous_metadata) != _json(metadata):
        if (
            _has_table(conn, "pipeline_batch_issues")
            and conn.execute(
                "SELECT 1 FROM pipeline_batch_issues WHERE batch_id=? AND newspaper_id=?",
                (batch_id, newspaper_id),
            ).fetchone()
        ):
            return "conflict", "A later metadata revision superseded this batch; stale replay is not allowed."
        return "metadata_changed", ""
    return "unchanged", ""


def plan_batch(
    manifest: BatchManifest,
    input_root: str | Path,
    database_path: str | Path,
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Validate every input and compare against SQLite without migrating it."""
    path = Path(database_path)
    if path.exists() and not path.is_file():
        raise ValueError("Database path is not a file.")
    conn = open_connection(path, read_only=True) if path.is_file() else None
    try:
        if conn is not None and not _has_table(conn, "newspapers"):
            raise ValueError("Target is not an initialized Semora corpus database.")
        if conn is not None:
            conn.execute("BEGIN")
        paths = _paths(conn)
        counts = dict.fromkeys(("new", "unchanged", "metadata_changed", "conflict"), 0)
        issues = []
        identifiers: set[str] = set()
        for number, issue in enumerate(manifest.issues, start=1):
            content, metadata = _read_source(input_root, issue)
            action, reason = _classify(conn, issue, content, metadata, paths, manifest.batch_id)
            source_identifier = _source_identifier(issue, metadata)
            if source_identifier and source_identifier.casefold() in identifiers:
                action, reason = "conflict", "Duplicate metadata source identifier within the batch."
            if source_identifier:
                identifiers.add(source_identifier.casefold())
            counts[action] += 1
            issues.append({"relative_path": issue.relative_path, "action": action, "reason": reason})
            if progress is not None and (number % 1000 == 0 or number == len(manifest.issues)):
                progress({"event": "batch_validation", "checked": number, "total": len(manifest.issues)})
        if conn is not None and _has_table(conn, "pipeline_batches"):
            saved = conn.execute(
                "SELECT manifest_fingerprint FROM pipeline_batches WHERE batch_id=?", (manifest.batch_id,)
            ).fetchone()
            if saved and saved[0] != manifest.fingerprint:
                raise ValueError("Batch ID already exists with a different manifest.")
        return {
            "batch_id": manifest.batch_id,
            "manifest_fingerprint": manifest.fingerprint,
            "counts": counts,
            "issues": issues,
        }
    finally:
        if conn is not None:
            conn.close()


def _config(model_id: str, token_count: int, token_overlap: int, tokenizer_revision: str | None) -> dict:
    if not model_id.strip() or token_count <= 0 or token_overlap < 0 or token_overlap >= token_count:
        raise ValueError("Invalid tokenizer model, token count or overlap.")
    if tokenizer_revision is not None and not tokenizer_revision.strip():
        raise ValueError("Tokenizer revision must be nonempty when provided.")
    return {
        "splitter": SPLITTER,
        "chunking": {
            "model_id": model_id,
            "tokenizer_revision": tokenizer_revision,
            "token_count": token_count,
            "token_overlap": token_overlap,
            "source_spans": True,
        },
    }


def run_batch(
    manifest: BatchManifest,
    input_root: str | Path,
    database_path: str | Path,
    *,
    stages: tuple[str, ...] = STAGES,
    model_id: str = indexer.DEFAULT_MODEL_ID,
    token_count: int = indexer.DEFAULT_TOKEN_COUNT,
    token_overlap: int = indexer.DEFAULT_TOKEN_OVERLAP,
    tokenizer_revision: str | None = None,
    tokenizer: Any = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Ingest selected batch stages during maintenance; never publish indexes.

    Callers must stop retrieval and legacy index writers. New coordinator runs
    are mutually exclusive and also acquire the existing semantic/NER locks.
    """
    if stages not in (STAGES[:1], STAGES[:2], STAGES):
        raise ValueError("Select a stage prefix: newspapers, then articles, then chunks.")
    config = _config(model_id, token_count, token_overlap, tokenizer_revision)
    target = Path(database_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(SemanticBuildLock(target.with_name(target.name + ".pipeline.lock"), "pipeline"))
        locks.enter_context(SemanticBuildLock(target.with_name(target.name + ".ner.lock"), "NER"))
        locks.enter_context(SemanticBuildLock(target.parent / "semantic" / ".vectors.lock", "vector"))
        plan = plan_batch(manifest, input_root, target, progress=progress)
        if plan["counts"]["conflict"]:
            raise ValueError(
                "Batch has source conflicts: " + _json([i for i in plan["issues"] if i["action"] == "conflict"])
            )
        if target.is_file():
            with closing(open_connection(target, read_only=True)) as existing:
                if not _has_table(existing, "schema_migrations") or not set(range(1, 14)).issubset(
                    row[0] for row in existing.execute("SELECT version FROM schema_migrations")
                ):
                    raise ValueError("Audit and explicitly migrate the legacy schema before additive ingestion.")
                if existing.execute("SELECT 1 FROM articles LIMIT 1").fetchone():
                    if (
                        not _has_table(existing, "pipeline_issue_stages")
                        or existing.execute(
                        "SELECT 1 FROM newspapers n WHERE EXISTS "
                        "(SELECT 1 FROM articles a WHERE a.newspaper_id=n.newspaper_id) "
                        "AND NOT EXISTS (SELECT 1 FROM pipeline_issue_stages s "
                        "WHERE s.newspaper_id=n.newspaper_id AND s.stage='articles') LIMIT 1"
                        ).fetchone()
                    ):
                        raise ValueError(
                            "Historical article outputs require baseline adoption before additive ingestion."
                        )
                if _has_table(existing, "pipeline_issue_stages"):
                    for stage, settings in (
                        ("articles", {"splitter": config["splitter"]}),
                        ("chunks", config["chunking"]),
                    ):
                        incompatible = existing.execute(
                            "SELECT 1 FROM pipeline_issue_stages WHERE stage=? AND config_json<>? LIMIT 1",
                            (stage, _json(settings)),
                        ).fetchone()
                        if stage in stages and incompatible:
                            raise ValueError(
                                f"Existing {stage} configuration differs; explicit processing migration is required."
                            )
                if "chunks" in stages:
                    _check_legacy_chunking(existing, config["chunking"])
        database = Database(target)
        try:
            database.initialize()
            conn = database.conn
            encoded = _json(config)
            saved = conn.execute(
                "SELECT config_json FROM pipeline_batches WHERE batch_id=?", (manifest.batch_id,)
            ).fetchone()
            if saved and saved[0] != encoded:
                raise ValueError("Batch processing configuration changed; resume with the original configuration.")
            with conn:
                conn.execute(
                    "INSERT INTO pipeline_batches(batch_id,manifest_fingerprint,config_json,issue_count,state) "
                    "VALUES (?,?,?,?,'running') "
                    "ON CONFLICT(batch_id) DO UPDATE SET state='running', error=NULL, updated_at=CURRENT_TIMESTAMP",
                    (manifest.batch_id, manifest.fingerprint, encoded, len(manifest.issues)),
                )
            try:
                paths = _paths(conn)
                for number, issue in enumerate(manifest.issues, start=1):
                    content, metadata = _read_source(input_root, issue)
                    with conn:
                        conn.execute("BEGIN IMMEDIATE")
                        action, reason = _classify(conn, issue, content, metadata, paths, manifest.batch_id)
                        if action == "conflict":
                            raise ValueError(reason)
                        newspaper_id = indexer._stable_id("newspaper", issue.relative_path)
                        if "newspapers" in stages:
                            _store_issue(conn, manifest.batch_id, issue, content, metadata, action)
                        elif not conn.execute(
                            "SELECT 1 FROM newspapers WHERE newspaper_id=?", (newspaper_id,)
                        ).fetchone():
                            raise ValueError("Run the newspapers stage first.")
                        elif action != "unchanged":
                            raise ValueError("Run the newspapers stage to reconcile metadata first.")
                        if "articles" in stages:
                            _articles(conn, newspaper_id, config)
                        if "chunks" in stages:
                            if not _stage(conn, newspaper_id, "chunks") and tokenizer is None:
                                tokenizer = (
                                    indexer._load_tokenizer(model_id, revision=tokenizer_revision)
                                    if tokenizer_revision is not None
                                    else indexer._load_tokenizer(model_id)
                                )
                            _chunks(conn, newspaper_id, config, tokenizer)
                    paths.setdefault(issue.relative_path.casefold(), set()).add(issue.relative_path)
                    if progress is not None and (number % 1000 == 0 or number == len(manifest.issues)):
                        progress({"event": "issue_ingested", "completed": number, "total": len(manifest.issues)})
                with conn:
                    conn.execute(
                        "UPDATE pipeline_batches SET state='ingested',updated_at=CURRENT_TIMESTAMP WHERE batch_id=?",
                        (manifest.batch_id,),
                    )
            except BaseException as exc:
                with conn:
                    conn.execute(
                        "UPDATE pipeline_batches SET state='failed',error=?,updated_at=CURRENT_TIMESTAMP "
                        "WHERE batch_id=?",
                        (str(exc), manifest.batch_id),
                    )
                raise
        finally:
            database.close()
    return batch_status(target, manifest.batch_id)


def _store_issue(
    conn: sqlite3.Connection, batch_id: str, issue: BatchIssue, content: str, metadata: dict, action: str
) -> None:
    newspaper_id = indexer._stable_id("newspaper", issue.relative_path)
    run_id = f"batch_{batch_id}"
    conn.execute("INSERT OR IGNORE INTO runs(run_id,run_type) VALUES (?, 'corpus_batch')", (run_id,))
    normalized = normalize_newspaper_metadata(metadata)
    if action == "new":
        columns = ("date", "publisher", "source", "rights", "title", "urn", "issue", "volume")
        conn.execute(
            "INSERT INTO newspapers(newspaper_id,run_id,content,metadata_json,relative_path," + ",".join(columns) + ") "
            "VALUES (" + ",".join("?" for _ in range(13)) + ")",
            (newspaper_id, run_id, content, _json(metadata), issue.relative_path, *(normalized[c] for c in columns)),
        )
    elif action == "metadata_changed":
        columns = ("date", "publisher", "source", "rights", "title", "urn", "issue", "volume")
        conn.execute(
            "UPDATE newspapers SET metadata_json=?," + ",".join(c + "=?" for c in columns) + " WHERE newspaper_id=?",
            (_json(metadata), *(normalized[c] for c in columns), newspaper_id),
        )
    source_identifier = _source_identifier(issue, metadata)
    conn.execute(
        "INSERT INTO pipeline_batch_issues(batch_id,relative_path,newspaper_id,action,content_sha256,"
        "metadata_sha256,source_identifier,source_identifier_key) "
        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(batch_id,relative_path) DO NOTHING",
        (
            batch_id,
            issue.relative_path,
            newspaper_id,
            action,
            _hash(content),
            _hash(_json(metadata)),
            source_identifier,
            source_identifier.casefold() if source_identifier else None,
        ),
    )


def _check_legacy_chunking(conn: sqlite3.Connection, requested: dict) -> None:
    """An adopted article baseline must not conceal incompatible old chunks."""
    for row in conn.execute(
        "SELECT DISTINCT r.method,r.config_json FROM chunking_runs r "
        "WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.chunking_run_id=r.chunking_run_id)"
    ):
        settings = json.loads(row["config_json"] or "{}")
        if not isinstance(settings, dict):
            raise ValueError("Historical chunk configuration is invalid; explicit migration is required.")
        settings.setdefault("tokenizer_revision", None)
        if row["method"] != "token" or settings != requested:
            raise ValueError(
                "Historical chunk configuration differs; audit tokenizer compatibility before adding chunks."
            )


def _stage(conn: sqlite3.Connection, newspaper_id: str, stage: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM pipeline_issue_stages WHERE newspaper_id=? AND stage=?", (newspaper_id, stage)
    ).fetchone()


def _pending(conn: sqlite3.Connection, newspaper_id: str, stage: str, content: str, config: dict) -> bool:
    previous = _stage(conn, newspaper_id, stage)
    if previous is None:
        return True
    if previous["input_sha256"] != _hash(content) or previous["config_json"] != _json(config):
        raise ValueError(f"Incompatible {stage} input/configuration; explicit replacement is required.")
    return False


def _done(conn: sqlite3.Connection, newspaper_id: str, stage: str, content: str, config: dict, count: int) -> None:
    conn.execute(
        "INSERT INTO pipeline_issue_stages(newspaper_id,stage,input_sha256,config_json,output_count) "
        "VALUES (?,?,?,?,?)",
        (newspaper_id, stage, _hash(content), _json(config), count),
    )


def _articles(conn: sqlite3.Connection, newspaper_id: str, config: dict) -> None:
    row = conn.execute("SELECT * FROM newspapers WHERE newspaper_id=?", (newspaper_id,)).fetchone()
    settings = {"splitter": config["splitter"]}
    if not _pending(conn, newspaper_id, "articles", row["content"], settings):
        return
    if conn.execute("SELECT 1 FROM articles WHERE newspaper_id=? LIMIT 1", (newspaper_id,)).fetchone():
        raise ValueError("Existing articles need baseline adoption before additive processing.")
    records = [
        indexer._article_record(
            source, newspaper_id=newspaper_id, run_id=row["run_id"], relative_path=row["relative_path"]
        )
        for source in split_articles(row["content"])
    ]
    conn.executemany(
        "INSERT INTO articles(article_id,run_id,newspaper_id,title,content,metadata_json,char_start,char_end,"
        "line_start,line_end,is_valid) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                a.article_id,
                a.run_id,
                a.newspaper_id,
                a.title,
                a.content,
                _json(a.metadata),
                a.char_start,
                a.char_end,
                a.line_start,
                a.line_end,
                True,
            )
            for a in records
        ],
    )
    _done(conn, newspaper_id, "articles", row["content"], settings, len(records))


def _chunks(conn: sqlite3.Connection, newspaper_id: str, config: dict, tokenizer: Any) -> None:
    newspaper = conn.execute("SELECT * FROM newspapers WHERE newspaper_id=?", (newspaper_id,)).fetchone()
    settings = config["chunking"]
    if not _pending(conn, newspaper_id, "chunks", newspaper["content"], settings):
        return
    if not _stage(conn, newspaper_id, "articles"):
        raise ValueError("Run the articles stage first; legacy articles require baseline adoption.")
    if conn.execute(
        "SELECT 1 FROM chunks JOIN articles USING(article_id) WHERE newspaper_id=? LIMIT 1", (newspaper_id,)
    ).fetchone():
        raise ValueError("Existing chunks need baseline adoption before additive processing.")
    run_id = newspaper["run_id"]
    chunking_id = indexer._stable_id("token", run_id, _json(settings))
    conn.execute(
        "INSERT OR IGNORE INTO chunking_runs(chunking_run_id,run_id,method,config_json) VALUES (?,?,?,?)",
        (chunking_id, run_id, "token", _json(settings)),
    )
    line_map = LineMap(newspaper["content"])
    count = 0
    for article in conn.execute(
        "SELECT * FROM articles WHERE newspaper_id=? ORDER BY line_start,article_id", (newspaper_id,)
    ).fetchall():
        source = SourceArticle(
            title=article["title"],
            content=article["content"],
            article_index=0,
            char_start=article["char_start"],
            char_end=article["char_end"],
            content_char_start=article["char_end"] - len(article["content"]),
            line_start=article["line_start"],
            line_end=article["line_end"],
        )
        chunks = indexer._token_chunks(
            source,
            article_id=article["article_id"],
            run_id=run_id,
            chunking_run_id=chunking_id,
            tokenizer=tokenizer,
            token_count=settings["token_count"],
            token_overlap=settings["token_overlap"],
            line_map=line_map,
        )
        conn.executemany(
            "INSERT INTO chunks(chunk_id,run_id,article_id,chunking_run_id,chunk_index,method,text,char_start,"
            "char_end,line_start,line_end) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.chunk_id,
                    c.run_id,
                    c.article_id,
                    c.chunking_run_id,
                    c.chunk_index,
                    c.method,
                    c.text,
                    c.char_start,
                    c.char_end,
                    c.line_start,
                    c.line_end,
                )
                for c in chunks
            ],
        )
        count += len(chunks)
    _done(conn, newspaper_id, "chunks", newspaper["content"], settings, count)


def batch_status(database_path: str | Path, batch_id: str) -> dict:
    """Report committed ingestion stages, without claiming retrieval readiness."""
    with closing(open_connection(database_path, read_only=True)) as conn:
        if not _has_table(conn, "pipeline_batches"):
            raise ValueError(f"Unknown batch: {batch_id}; this archive has no additive batch ledger.")
        row = conn.execute("SELECT * FROM pipeline_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown batch: {batch_id}")
        stages = {
            stage: conn.execute(
                "SELECT COUNT(*) FROM pipeline_batch_issues b JOIN pipeline_issue_stages s USING(newspaper_id) "
                "WHERE b.batch_id=? AND s.stage=?",
                (batch_id, stage),
            ).fetchone()[0]
            for stage in ("articles", "chunks")
        }
        eligible = conn.execute(
            "SELECT COUNT(*) FROM articles a JOIN pipeline_batch_issues b USING(newspaper_id) "
            "WHERE b.batch_id=? AND a.is_valid=1",
            (batch_id,),
        ).fetchone()[0]
        lexical = {}
        for stage in ("surface", "lemma"):
            recorded = (
                conn.execute(
                    "SELECT COUNT(*) FROM pipeline_lexical_state s JOIN articles a USING(article_id) "
                    "JOIN pipeline_batch_issues b USING(newspaper_id) "
                    "WHERE b.batch_id=? AND s.stage=? AND a.is_valid=1",
                    (batch_id, stage),
                ).fetchone()[0]
                if _has_table(conn, "pipeline_lexical_state")
                else 0
            )
            lexical[stage] = {
                "eligible_articles": eligible,
                "recorded_articles": recorded,
                "pending_articles": eligible - recorded,
            }
        return {
            **dict(row),
            "issues": conn.execute(
                "SELECT COUNT(*) FROM pipeline_batch_issues WHERE batch_id=?", (batch_id,)
            ).fetchone()[0],
            "stages": stages,
            "lexical": lexical,
            "ingestion_complete": row["state"] == "ingested"
            and all(count == row["issue_count"] for count in stages.values()),
            "retrieval_ready": False,
        }
