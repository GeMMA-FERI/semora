"""Batch-scoped NER with durable article completion and model compatibility."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

from semora.ner.adapters import load_adapter
from semora.ner.staging import iter_predictions
from semora.ner.storage import save_batch
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage.connection import open_connection

NER_FORMAT = "semora.pipeline.ner.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_MODEL_KEYS = {
    "adapter", "package", "processors", "version", "classla_version", "model", "model_id",
    "revision", "model_revision", "tokenizer_revision", "model_sha256", "source_sha256", "tokenizer_sha256",
    "max_length", "stride", "pretokenized", "precision", "aggregation", "word_id_aggregation",
}
_TUNING_KEYS = {
    "device", "use_gpu", "batch_size", "batch_articles", "pipeline_depth", "ner_pipeline_depth",
    "tokenizer_workers", "num_threads", "decoder", "ner_decoder", "sort_during_eval", "sentence_order",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime(item) for key, item in value.items()
            if isinstance(key, str) and key not in _TUNING_KEYS and key != "path" and not key.endswith("_path")
        }
    if isinstance(value, (list, tuple)):
        return [_without_runtime(item) for item in value]
    return value


def model_identity(metadata: dict) -> dict:
    """Retain recorded model/code semantics, excluding host paths and throughput tuning.

    Missing historical hashes stay missing. Paths in CLASSLA's recorded config
    cannot identify relocated resources; its model/source hashes can.
    """
    if (
        not isinstance(metadata, dict) or not isinstance(metadata.get("adapter"), str)
        or not metadata["adapter"].strip()
    ):
        raise ValueError("NER adapter metadata requires a nonempty adapter identity.")
    identity = {key: metadata[key] for key in _MODEL_KEYS if key in metadata}
    for key in ("model_sha256", "source_sha256", "tokenizer_sha256"):
        digest = identity.get(key)
        if digest is not None and (not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None):
            raise ValueError(f"NER {key} must be a lowercase SHA-256 digest when recorded.")
    if "ner_config" in metadata:
        if not isinstance(metadata["ner_config"], dict):
            raise ValueError("NER processor configuration must be an object.")
        identity["ner_config"] = _without_runtime(metadata["ner_config"])
    try:
        return json.loads(_json(identity))
    except (ValueError, TypeError) as exc:
        raise ValueError("NER model identity must contain finite JSON values.") from exc


def _table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


def _validate_batch(connection: sqlite3.Connection, batch_id: str) -> None:
    if not _table(connection, "pipeline_batches"):
        raise ValueError("No registered additive batches exist.")
    batch = connection.execute(
        "SELECT state,issue_count FROM pipeline_batches WHERE batch_id=?", (batch_id,),
    ).fetchone()
    count = connection.execute("SELECT COUNT(*) FROM pipeline_batch_issues WHERE batch_id=?", (batch_id,)).fetchone()[0]
    if batch is None or batch["state"] != "ingested" or count != batch["issue_count"]:
        raise ValueError("Finish the selected batch ingestion before NER extraction.")
    if connection.execute(
        "SELECT 1 FROM pipeline_batch_issues b WHERE b.batch_id=? AND NOT EXISTS "
        "(SELECT 1 FROM pipeline_issue_stages s WHERE s.newspaper_id=b.newspaper_id AND s.stage='articles') LIMIT 1",
        (batch_id,),
    ).fetchone():
        raise ValueError("Finish batch article extraction before NER extraction.")


def _recorded_identity(connection: sqlite3.Connection, run_id: str) -> tuple[dict | None, dict | None]:
    row = connection.execute("SELECT config_json FROM ner_runs WHERE ner_run_id=?", (run_id,)).fetchone()
    if row is None:
        if connection.execute("SELECT 1 FROM article_ner_state WHERE ner_run_id=? LIMIT 1", (run_id,)).fetchone() or (
            connection.execute("SELECT 1 FROM article_entities WHERE ner_run_id=? LIMIT 1", (run_id,)).fetchone()
        ):
            raise ValueError("NER outputs lack a registered run; explicit audit/repair is required.")
        return None, None
    try:
        config = json.loads(row[0])
        identity = model_identity(config["model"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Existing NER run configuration is malformed.") from exc
    if config.get("format") == NER_FORMAT:
        if config.get("model_identity") != identity:
            raise ValueError("Recorded pipeline NER model identity disagrees with its metadata.")
    elif config.get("format") is not None:
        raise ValueError("Unsupported NER run configuration format.")
    elif connection.execute(
        "SELECT 1 FROM article_ner_state WHERE ner_run_id=? LIMIT 1", (run_id,),
    ).fetchone() or connection.execute(
        "SELECT 1 FROM article_entities WHERE ner_run_id=? LIMIT 1", (run_id,),
    ).fetchone():
        marker = connection.execute(
            "SELECT model_identity_json FROM pipeline_ner_adoptions WHERE ner_run_id=?", (run_id,),
        ).fetchone() if _table(connection, "pipeline_ner_adoptions") else None
        if marker is None:
            raise ValueError("Historical NER outputs require explicit audited adoption before batch extraction.")
        try:
            if json.loads(marker[0]) != identity:
                raise ValueError("Adopted NER model identity differs from its recorded run.")
        except (ValueError, TypeError) as exc:
            raise ValueError("Adopted NER model identity differs from its recorded run.") from exc
    return config, identity


_MEMBERSHIP = (
    "FROM articles a "
    "JOIN pipeline_batch_issues b USING(newspaper_id) "
    "LEFT JOIN article_ner_state s ON s.article_id=a.article_id AND s.ner_run_id=? "
    "WHERE b.batch_id=? AND a.is_valid=1 AND TRIM(a.content)<>''"
)
_ROWS = "SELECT a.article_id,a.content,s.content_sha256,s.entity_count " + _MEMBERSHIP


def _coverage(connection: sqlite3.Connection, batch_id: str, run_id: str) -> tuple[int, int]:
    eligible = completed = 0
    # One streaming statement. The correlated aggregate uses the entity primary
    # key (run, article, index) and visits only mentions belonging to this batch,
    # without millions of Python queries or a full-run grouped temporary table.
    query = (
        "SELECT a.article_id,a.content,s.content_sha256,s.entity_count,"
        "(SELECT COUNT(*) FROM article_entities e WHERE e.ner_run_id=? AND e.article_id=a.article_id) AS mentions "
        + _MEMBERSHIP
    )
    for row in connection.execute(query, (run_id, run_id, batch_id)):
        eligible += 1
        if row["content_sha256"] is None:
            continue
        digest = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        if row["content_sha256"] != digest:
            raise ValueError("Completed NER input differs from the current article; explicit repair is required.")
        if type(row["entity_count"]) is not int or row["entity_count"] < 0 or row["mentions"] != row["entity_count"]:
            raise ValueError("Completed NER mention coverage is inconsistent; audit/repair the run.")
        completed += 1
    return eligible, completed


def _register(connection: sqlite3.Connection, run_id: str, metadata: dict, identity: dict) -> None:
    previous, recorded = _recorded_identity(connection, run_id)
    if previous is not None:
        if recorded != identity:
            raise ValueError("NER model identity changed; use a different run ID for incompatible models.")
        if previous.get("format") != NER_FORMAT and not connection.execute(
            "SELECT 1 FROM article_ner_state WHERE ner_run_id=? LIMIT 1", (run_id,),
        ).fetchone() and not connection.execute(
            "SELECT 1 FROM article_entities WHERE ner_run_id=? LIMIT 1", (run_id,),
        ).fetchone():
            # An unused historical run has no outputs to adopt. Mark ownership
            # before its first pipeline writes so interrupted attempts can resume.
            with connection:
                connection.execute("UPDATE ner_runs SET config_json=? WHERE ner_run_id=?", (
                    _json({**previous, "format": NER_FORMAT, "model_identity": identity}), run_id,
                ))
        return
    # Preserve full adapter metadata for audit, including its default processor
    # configuration. Only the separate identity controls compatible resumes.
    config = {"format": NER_FORMAT, "model": metadata, "model_identity": identity, "sentence_order": "input"}
    with connection:
        connection.execute("INSERT INTO ner_runs(ner_run_id,config_json) VALUES (?,?)", (run_id, _json(config)))


def index_batch_ner(
    database_path: str | Path, batch_id: str, *, run_id: str = "classla-sl", adapter: str = "classla",
    device: str = "cpu", batch_articles: int = 32, batch_size: int = 16, pipeline_depth: int = 1,
    tokenizer_workers: int = 0, ner_pipeline_depth: int = 1, ner_decoder: str = "numpy",
    resources: str | Path | None = None, model: Any = None, progress: Callable[[dict], None] | None = None,
) -> dict:
    """Extract only pending articles in one frozen batch, with atomic article outputs.

    Retrieval readers and older writers must be stopped by the caller. Injected
    adapters remain caller-owned; adapters loaded here are closed on every exit.
    """
    identifiers = (batch_id, run_id, adapter, device, ner_decoder)
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("NER batch, run, adapter, device and decoder must be nonempty strings.")
    if any(type(value) is not int or value <= 0 for value in (
        batch_articles, batch_size, pipeline_depth, ner_pipeline_depth,
    )) or type(tokenizer_workers) is not int or tokenizer_workers < 0:
        raise ValueError("NER batch sizes/depths must be positive integers and tokenizer workers nonnegative.")
    path = Path(database_path).resolve(strict=True)
    owned_model = None
    with ExitStack() as locks:
        locks.enter_context(SemanticBuildLock(path.with_name(path.name + ".pipeline.lock"), "pipeline"))
        locks.enter_context(SemanticBuildLock(path.with_name(path.name + ".ner.lock"), "NER"))
        with closing(open_connection(path)) as connection:
            _validate_batch(connection, batch_id)
            _config, recorded = _recorded_identity(connection, run_id)
            if model is not None:
                requested = model_identity(model.metadata)
                if recorded is not None and requested != recorded:
                    raise ValueError("NER model identity changed; use a different run ID for incompatible models.")
            elif recorded is not None and adapter in {"classla", "transformers"} and recorded["adapter"] != adapter:
                raise ValueError("NER adapter changed; use the matching adapter or a different run ID.")
            eligible, initial = _coverage(connection, batch_id, run_id)
            added = mentions = 0
            cursor = ""
            upper = connection.execute(
                "SELECT MAX(a.article_id) " + _MEMBERSHIP,
                (run_id, batch_id),
            ).fetchone()[0]

            def read_batch() -> list[dict]:
                nonlocal cursor
                if upper is None:
                    return []
                with closing(open_connection(path, read_only=True)) as reader:
                    rows = [dict(row) for row in reader.execute(
                        _ROWS + " AND s.article_id IS NULL AND a.article_id>? AND a.article_id<=? "
                        "ORDER BY a.article_id LIMIT ?", (run_id, batch_id, cursor, upper, batch_articles),
                    )]
                if rows:
                    cursor = rows[-1]["article_id"]
                return rows

            try:
                if eligible > initial:
                    if model is None:
                        owned_model = load_adapter(
                            adapter, device=device, resources=resources, tokenizer_workers=tokenizer_workers,
                            ner_pipeline_depth=ner_pipeline_depth, ner_decoder=ner_decoder, sentence_order="input",
                        )
                        model = owned_model
                    identity = model_identity(model.metadata)
                    _register(connection, run_id, model.metadata, identity)
                    with closing(iter_predictions(model, read_batch, batch_size, pipeline_depth)) as batches:
                        for batch in batches:
                            # save_batch rechecks current content/validity inside
                            # its write transaction, then commits mentions and
                            # completion together, including zero-entity articles.
                            save_batch(connection, run_id, batch)
                            added += len(batch.rows)
                            mentions += sum(len(entities) for entities in batch.prediction.entities)
                            if progress is not None:
                                progress({"event": "ner_checkpoint", "batch_id": batch_id, "ner_run_id": run_id,
                                          "completed_articles": initial + added, "total": eligible})
                current_eligible, completed = _coverage(connection, batch_id, run_id)
                if current_eligible != eligible or completed != eligible:
                    raise ValueError("Batch NER membership changed or has pending articles; audit/retry the batch.")
                return {
                    "batch_id": batch_id, "stage": "ner", "ner_run_id": run_id, "eligible_articles": eligible,
                    "completed_articles": completed, "added_articles": added, "skipped_articles": initial,
                    "added_entities": mentions, "pending_articles": eligible - completed, "complete": True,
                    "exhausted_snapshot": True, "retrieval_ready": False, "publication_ready": False,
                }
            finally:
                if owned_model is not None and hasattr(owned_model, "close"):
                    owned_model.close()
