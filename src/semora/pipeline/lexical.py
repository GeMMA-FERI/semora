"""Batch-scoped additive FTS indexing with transactional completion records."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database
from semora.storage.connection import open_connection
from semora.text.lemmatization import ClasslaLemmatizer, Lemmatizer


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def lexical_input_sha256(title: str | None, content: str) -> str:
    """Fingerprint the actual title and text submitted to either lexical stage."""
    return hashlib.sha256(_json([title or "", content]).encode("utf-8")).hexdigest()


def surface_config() -> dict:
    return {"format": "semora.surface.v1", "tokenizer": "unicode61 remove_diacritics 2"}


def lexical_schema_compatible(connection: sqlite3.Connection, stage: str) -> bool:
    """Recognize the known FTS schema from actual DDL, without trusting receipts.

    This deliberately accepts only the supported two-column contentless FTS5
    layout. Unknown tokenizer options or custom layouts require explicit audit
    and migration rather than inheriting a fabricated compatibility profile.
    """
    if stage not in {"surface", "lemma"}:
        raise ValueError("Select surface or lemma lexical schema.")
    table = "article_fts" if stage == "surface" else "article_lemma_fts"
    row = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if row is None or not row[0]:
        return False

    def identifier(name: str) -> str:
        return rf'(?:{name}|"{name}"|`{name}`|\[{name}\])'

    content = r"content\s*=\s*(?:''|\"\")"
    tokenizer = r"tokenize\s*=\s*(?:'unicode61\s+remove_diacritics\s+2'|\"unicode61\s+remove_diacritics\s+2\")"
    options = rf"(?:{content}\s*,\s*{tokenizer}|{tokenizer}\s*,\s*{content})"
    ddl = (
        rf"\s*CREATE\s+VIRTUAL\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{identifier(table)}\s+USING\s+fts5\s*"
        rf"\(\s*{identifier('title')}\s*,\s*{identifier('text')}\s*,\s*{options}\s*\)\s*;?\s*"
    )
    return re.fullmatch(ddl, row[0], flags=re.IGNORECASE) is not None


def lemma_config(pipeline_type: str = "default", model_fingerprint: str | None = None) -> dict:
    if not isinstance(pipeline_type, str) or not pipeline_type.strip():
        raise ValueError("CLASSLA pipeline type must be nonempty.")
    if model_fingerprint is not None and (
        not isinstance(model_fingerprint, str) or not model_fingerprint.strip()
    ):
        raise ValueError("Model fingerprint must be nonempty when provided.")
    # None is explicit unknown legacy evidence, never a hash inferred from a
    # currently installed model/cache. Runtime paths and batching are not identity.
    return {
        "format": "semora.lemma.v1", "pipeline_type": pipeline_type,
        "model_fingerprint": model_fingerprint,
    }


def _register_config(connection: sqlite3.Connection, stage: str, config: dict) -> None:
    """Register compatibility once per stage/attempt, within its write transaction."""
    encoded = _json(config)
    previous = connection.execute(
        "SELECT config_json FROM pipeline_lexical_configs WHERE stage=?", (stage,)
    ).fetchone()
    if previous is not None:
        if previous["config_json"] != encoded:
            raise ValueError("Existing lexical configuration differs; explicit reindexing is required.")
        return
    connection.execute(
        "INSERT INTO pipeline_lexical_configs(stage,config_json) VALUES (?,?)", (stage, encoded)
    )


def _record_completion(connection: sqlite3.Connection, stage: str, row: sqlite3.Row, config: dict) -> None:
    digest = lexical_input_sha256(row["title"], row["content"])
    encoded = _json(config)
    saved = connection.execute(
        "SELECT input_sha256,config_json FROM pipeline_lexical_state WHERE article_id=? AND stage=?",
        (row["article_id"], stage),
    ).fetchone()
    if saved is not None:
        if saved["input_sha256"] != digest or saved["config_json"] != encoded:
            raise ValueError("Existing lexical input/configuration differs; explicit reindexing is required.")
        return
    # The composite foreign key enforces registration/compatibility without a
    # separate configuration lookup for every historical or newly indexed row.
    connection.execute(
        "INSERT INTO pipeline_lexical_state(article_id,stage,input_sha256,config_json) VALUES (?,?,?,?)",
        (row["article_id"], stage, digest, encoded),
    )


def adopt_legacy_lexical(connection: sqlite3.Connection, evidence: dict) -> dict:
    """Record audited legacy coverage inside the caller's adoption transaction.

    The caller must audit exact membership/mapping and trusted legacy completion
    state in the same transaction. Contentless FTS does not reconstruct the
    original text; the evidence must retain that limitation in the baseline.
    """
    if not connection.in_transaction:
        raise ValueError("Legacy lexical adoption requires an active audited transaction.")
    counts = {"surface": 0, "lemma": 0}
    for stage in counts:
        report = evidence.get(stage, {})
        if not report.get("eligible"):
            continue
        if not lexical_schema_compatible(connection, stage):
            raise ValueError("Unsupported lexical FTS schema/tokenizer; explicit schema migration is required.")
        config = report.get("config")
        expected = surface_config() if stage == "surface" else lemma_config(
            config.get("pipeline_type", "") if isinstance(config, dict) else ""
        )
        if config != expected:
            raise ValueError("Unsupported legacy lexical configuration evidence.")
        _register_config(connection, stage, config)
        join = "" if stage == "surface" else "JOIN article_lemmas l ON l.article_id=a.article_id"
        rows = connection.execute(
            "SELECT a.article_id,a.title,a.content FROM articles a "
            "JOIN article_fts_map m ON m.article_id=a.article_id " + join + " WHERE a.is_valid=1"
        )
        for row in rows:
            _record_completion(connection, stage, row, config)
            counts[stage] += 1
    return counts


_BATCH_ROWS = (
    "SELECT a.article_id,a.title,a.content,m.fts_id,l.article_id AS lemma_article_id,"
    "l.pipeline_type AS lemma_pipeline_type,"
    "s.input_sha256,s.config_json,ss.input_sha256 AS surface_input_sha256,"
    "ss.config_json AS surface_config_json,f.rowid AS surface_fts_id,lf.rowid AS lemma_fts_id "
    "FROM articles a "
    "JOIN pipeline_batch_issues b ON b.newspaper_id=a.newspaper_id "
    "LEFT JOIN article_fts_map m ON m.article_id=a.article_id "
    "LEFT JOIN article_lemmas l ON l.article_id=a.article_id "
    "LEFT JOIN pipeline_lexical_state s ON s.article_id=a.article_id AND s.stage=? "
    "LEFT JOIN pipeline_lexical_state ss ON ss.article_id=a.article_id AND ss.stage='surface' "
    "LEFT JOIN article_fts f ON f.rowid=m.fts_id "
    "LEFT JOIN article_lemma_fts lf ON lf.rowid=m.fts_id "
    "WHERE b.batch_id=? AND a.is_valid=1"
)


def _validate_batch(connection: sqlite3.Connection, batch_id: str) -> None:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_batches'"
    ).fetchone() is None:
        raise ValueError("No registered batches exist; audit/adopt the baseline and ingest a batch first.")
    batch = connection.execute("SELECT * FROM pipeline_batches WHERE batch_id=?", (batch_id,)).fetchone()
    if batch is None:
        raise ValueError(f"Unknown batch: {batch_id}")
    count = connection.execute(
        "SELECT COUNT(*) FROM pipeline_batch_issues WHERE batch_id=?", (batch_id,)
    ).fetchone()[0]
    if batch["state"] != "ingested" or count != batch["issue_count"]:
        raise ValueError("Finish batch ingestion before lexical indexing.")
    if connection.execute(
        "SELECT 1 FROM pipeline_batch_issues b WHERE b.batch_id=? AND NOT EXISTS "
        "(SELECT 1 FROM pipeline_issue_stages s WHERE s.newspaper_id=b.newspaper_id "
        "AND s.stage='articles') LIMIT 1", (batch_id,),
    ).fetchone():
        raise ValueError("Finish batch article extraction before lexical indexing.")


def _validate_existing(connection: sqlite3.Connection, batch_id: str, stage: str, config: dict) -> tuple[int, int]:
    encoded = _json(config)
    registered = connection.execute(
        "SELECT config_json FROM pipeline_lexical_configs WHERE stage=?", (stage,)
    ).fetchone()
    if registered is not None and registered["config_json"] != encoded:
        raise ValueError("Existing lexical configuration differs; explicit reindexing is required.")
    eligible = completed = 0
    for row in connection.execute(_BATCH_ROWS, (stage, batch_id)):
        eligible += 1
        output_exists = row["fts_id"] is not None if stage == "surface" else row["lemma_article_id"] is not None
        if stage == "lemma":
            if row["surface_input_sha256"] is None or row["fts_id"] is None or row["surface_fts_id"] is None:
                raise ValueError("Finish/adopt surface indexing before lemma indexing.")
            if (
                row["surface_input_sha256"] != lexical_input_sha256(row["title"], row["content"])
                or row["surface_config_json"] != _json(surface_config())
            ):
                raise ValueError("Surface input/configuration differs; explicit reindexing is required.")
        if row["input_sha256"] is None:
            if output_exists:
                raise ValueError("Existing lexical outputs require audited baseline adoption before indexing.")
            continue
        if row["input_sha256"] != lexical_input_sha256(row["title"], row["content"]) or row["config_json"] != encoded:
            raise ValueError("Existing lexical input/configuration differs; explicit reindexing is required.")
        if not output_exists:
            raise ValueError("Completed lexical output is missing; explicit repair is required.")
        if stage == "lemma" and row["lemma_pipeline_type"] != config["pipeline_type"]:
            raise ValueError("Materialized lemma configuration differs; explicit repair is required.")
        if row["surface_fts_id" if stage == "surface" else "lemma_fts_id"] is None:
            raise ValueError("Completed FTS row is missing; explicit repair is required.")
        completed += 1
    return eligible, completed


def _validate_stage_preflight(connection: sqlite3.Connection, stage: str) -> None:
    if not lexical_schema_compatible(connection, stage):
        raise ValueError("Unsupported lexical FTS schema/tokenizer; explicit schema migration is required.")
    has_registry = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_lexical_configs'"
    ).fetchone()
    registered = has_registry and connection.execute(
        "SELECT 1 FROM pipeline_lexical_configs WHERE stage=?", (stage,)
    ).fetchone()
    if registered:
        return
    outputs = ("article_fts_map", "article_fts") if stage == "surface" else ("article_lemmas", "article_lemma_fts")
    # Before the first stage registration, every existing output needs explicit
    # baseline evidence, including rows outside the selected new batch. Once
    # registered, transactional completion/FK invariants maintain compatibility.
    if any(connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() for table in outputs):
        raise ValueError("Existing lexical outputs require audited baseline adoption before indexing.")


def _write_surface(connection: sqlite3.Connection, rows: list[sqlite3.Row], config: dict) -> None:
    next_id = connection.execute("SELECT COALESCE(MAX(fts_id),0)+1 FROM article_fts_map").fetchone()[0]
    for offset, row in enumerate(rows):
        fts_id = next_id + offset
        if connection.execute("SELECT 1 FROM article_fts WHERE rowid=?", (fts_id,)).fetchone():
            raise ValueError("Orphaned FTS row conflicts with allocation; audit/repair the index.")
        connection.execute("INSERT INTO article_fts_map(fts_id,article_id) VALUES (?,?)", (fts_id, row["article_id"]))
        connection.execute(
            "INSERT INTO article_fts(rowid,title,text) VALUES (?,?,?)", (fts_id, row["title"] or "", row["content"])
        )
        _record_completion(connection, "surface", row, config)


def _write_lemma(
    connection: sqlite3.Connection, rows: list[sqlite3.Row], config: dict, annotations: list[str],
) -> None:
    if len(annotations) != len(rows) * 2 or any(not isinstance(text, str) for text in annotations):
        raise ValueError("Lemmatizer returned invalid output or a different number of documents.")
    for offset, row in enumerate(rows):
        # Match legacy behavior for empty/zero-token annotations.
        title = annotations[offset * 2] or row["title"] or ""
        content = annotations[offset * 2 + 1] or row["content"]
        if connection.execute("SELECT 1 FROM article_lemma_fts WHERE rowid=?", (row["fts_id"],)).fetchone():
            raise ValueError("Orphaned lemma FTS row requires audit/repair.")
        connection.execute(
            "INSERT INTO article_lemmas(article_id,title,content,pipeline_type) VALUES (?,?,?,?)",
            (row["article_id"], title, content, config["pipeline_type"]),
        )
        connection.execute(
            "INSERT INTO article_lemma_fts(rowid,title,text) VALUES (?,?,?)", (row["fts_id"], title, content)
        )
        _record_completion(connection, "lemma", row, config)


def index_batch_lexical(
    database_path: str | Path, batch_id: str, *, stage: str = "surface", batch_size: int = 1000,
    lemmatizer: Lemmatizer | None = None, pipeline_type: str = "default", model_fingerprint: str | None = None,
    device: str = "cpu", resources_dir: str | Path | None = None,
    pos_batch_size: int | None = None, lemma_batch_size: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Index only pending eligible articles in an ingested batch during maintenance."""
    if stage not in {"surface", "lemma"} or batch_size <= 0:
        raise ValueError("Select surface/lemma and a positive batch size.")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("Batch ID must be nonempty.")
    config = surface_config() if stage == "surface" else lemma_config(pipeline_type, model_fingerprint)
    path = Path(database_path).resolve(strict=True)
    owned_model: ClasslaLemmatizer | None = None
    with SemanticBuildLock(path.with_name(path.name + ".pipeline.lock"), "pipeline"):
        with closing(open_connection(path, read_only=True)) as reader:
            _validate_batch(reader, batch_id)
            _validate_stage_preflight(reader, stage)
        database = Database(path)
        try:
            database.initialize()
            connection = database.conn
            eligible, initial = _validate_existing(connection, batch_id, stage, config)
            added = 0
            cursor = ""
            registered = False
            while True:
                # This cursor only pages the frozen explicit membership for this
                # attempt. It is reset on every retry; completion is per article.
                rows = connection.execute(
                    _BATCH_ROWS + " AND s.article_id IS NULL AND a.article_id>? ORDER BY a.article_id LIMIT ?",
                    (stage, batch_id, cursor, batch_size),
                ).fetchall()
                if not rows:
                    break
                annotations = []
                if stage == "lemma":
                    if lemmatizer is None:
                        owned_model = ClasslaLemmatizer(
                            pipeline_type=pipeline_type, device=device, resources_dir=resources_dir,
                            pos_batch_size=pos_batch_size, lemma_batch_size=lemma_batch_size,
                        )
                        lemmatizer = owned_model
                    annotations = lemmatizer.lemmatize_many(
                        [text for row in rows for text in (row["title"] or "", row["content"])]
                    )
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if not registered:
                        _register_config(connection, stage, config)
                    # A coordinator lock protects cooperating writers. Verify the
                    # selected inputs again before committing derived outputs.
                    for row in rows:
                        current = connection.execute(
                            "SELECT title,content,is_valid FROM articles WHERE article_id=?", (row["article_id"],)
                        ).fetchone()
                        if current is None or current["is_valid"] != 1 or lexical_input_sha256(
                            current["title"], current["content"]
                        ) != lexical_input_sha256(row["title"], row["content"]):
                            raise ValueError("Article changed during lexical processing; explicit repair is required.")
                    if stage == "surface":
                        _write_surface(connection, rows, config)
                    else:
                        _write_lemma(connection, rows, config, annotations)
                added += len(rows)
                registered = True
                cursor = rows[-1]["article_id"]
                if progress is not None:
                    progress({
                        "event": "lexical_checkpoint", "batch_id": batch_id, "stage": stage,
                        "completed_articles": initial + added, "total": eligible,
                    })
            _, completed = _validate_existing(connection, batch_id, stage, config)
            return {
                "batch_id": batch_id, "stage": stage, "eligible_articles": eligible,
                "completed_articles": completed, "added_articles": added, "skipped_articles": initial,
                "pending_articles": eligible - completed, "complete": completed == eligible,
                "retrieval_ready": False, "publication_ready": False,
            }
        finally:
            if owned_model is not None:
                owned_model.close()
            database.close()
