"""Read-only evidence audit for one historical article-level NER run."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from semora.pipeline.baseline import _schema
from semora.pipeline.manifest import _invalid_constant, _unique_object
from semora.storage.connection import open_connection

_DIGEST = re.compile(r"[0-9a-f]{64}")
_ERROR_LIMIT = 50
_ELIGIBLE = "a.is_valid=1 AND TRIM(a.content)<>''"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _config(raw: str) -> tuple[dict, dict]:
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    keys = {"model", "device", "batch_articles", "batch_size", "ner_decoder", "sentence_order"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Unknown or incomplete legacy NER run configuration")
    if not isinstance(value["model"], dict) or not _nonempty(value["model"].get("adapter")):
        raise ValueError("NER model metadata must identify its adapter")
    if any(not _nonempty(value[key]) for key in ("device", "ner_decoder", "sentence_order")):
        raise ValueError("NER runtime/decoder/order metadata must be nonempty strings")
    if value["sentence_order"] not in {"input", "length"}:
        raise ValueError("Unknown historical NER sentence order")
    for key in ("batch_articles", "batch_size"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"NER {key} must be a positive integer")
    model = value["model"]
    recorded = [model[key] for key in ("revision", "model_revision") if model.get(key) is not None]
    if any(not _nonempty(revision) for revision in recorded):
        raise ValueError("NER model revisions are malformed or disagree")
    revisions = set(recorded)
    if len(revisions) > 1:
        raise ValueError("NER model revisions are malformed or disagree")
    for key in ("model_sha256", "source_sha256", "tokenizer_sha256"):
        digest = model.get(key)
        if digest is not None and (not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None):
            raise ValueError(f"Malformed recorded NER {key}")
    return value, {
        "adapter": model["adapter"],
        "recorded_revision": next(iter(revisions), None),
        "recorded_model_sha256": model.get("model_sha256"),
        "recorded_source_sha256": model.get("source_sha256"),
        "recorded_tokenizer_sha256": model.get("tokenizer_sha256"),
        "recorded_classla_version": model.get("classla_version"),
        "source_fingerprint_scope": "Adapter-specific; recorded CLASSLA source hashes do not establish "
                                    "complete tokenizer and Semora alignment provenance.",
        "resource_paths_verified": False,
    }


def _inspect(conn: sqlite3.Connection, report: dict, error, progress) -> None:
    _schema(conn)
    run_id = report["run_id"]
    row = conn.execute("SELECT config_json FROM ner_runs WHERE ner_run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown historical NER run: {run_id}")
    config, evidence = _config(row[0])
    report["config"] = config
    report["config_sha256"] = _sha(_json(config))
    report["model_evidence"] = evidence
    counts = report["counts"]
    counts["eligible_articles"] = conn.execute(
        f"SELECT COUNT(*) FROM articles a WHERE {_ELIGIBLE}"
    ).fetchone()[0]
    counts["completed_articles"] = conn.execute(
        "SELECT COUNT(*) FROM article_ner_state WHERE ner_run_id=?", (run_id,),
    ).fetchone()[0]
    counts["mentions"] = conn.execute(
        "SELECT COUNT(*) FROM article_entities WHERE ner_run_id=?", (run_id,),
    ).fetchone()[0]
    counts["missing_completion"] = conn.execute(
        f"SELECT COUNT(*) FROM articles a WHERE {_ELIGIBLE} "
        "AND NOT EXISTS(SELECT 1 FROM article_ner_state s WHERE s.ner_run_id=? AND s.article_id=a.article_id)",
        (run_id,),
    ).fetchone()[0]
    counts["extra_completion"] = conn.execute(
        "SELECT COUNT(*) FROM article_ner_state s LEFT JOIN articles a USING(article_id) "
        "WHERE s.ner_run_id=? AND (a.article_id IS NULL OR a.is_valid IS NOT 1 OR TRIM(a.content)='')",
        (run_id,),
    ).fetchone()[0]
    if counts["missing_completion"]:
        error(f"Eligible articles without NER completion: {counts['missing_completion']}")
    if counts["extra_completion"]:
        error(f"NER completion rows without an eligible article: {counts['extra_completion']}")
    if not counts["missing_completion"] and not counts["extra_completion"] and (
        counts["completed_articles"] != counts["eligible_articles"]
    ):
        error("NER completion row count disagrees with exact eligible membership")
    report["membership_complete"] = not counts["missing_completion"] and not counts["extra_completion"]
    fingerprint = hashlib.sha256()
    # Both primary-key prefixes are (ner_run_id, article_id). A lookahead merge
    # avoids a separate SQLite seek/query for every one of millions of articles.
    mentions = iter(conn.execute(
        "SELECT * FROM article_entities WHERE ner_run_id=? ORDER BY article_id,entity_index", (run_id,),
    ))
    pending = next(mentions, None)
    for checked, state in enumerate(conn.execute(
        "SELECT s.*,a.content,a.is_valid FROM article_ner_state s LEFT JOIN articles a USING(article_id) "
        "WHERE s.ner_run_id=? ORDER BY s.article_id", (run_id,),
    ), 1):
        before = report["error_count"]
        article_id = state["article_id"]
        while pending is not None and pending["article_id"] < article_id:
            counts["orphan_mentions"] += 1
            pending = next(mentions, None)
        text = state["content"]
        valid_source = isinstance(text, str) and state["is_valid"] == 1 and text.strip(" ") != ""
        if not valid_source:
            error("NER completion has no eligible current source", article_id)
        if valid_source and state["content_sha256"] != _sha(text):
            error("NER completion content SHA-256 disagrees with current article text", article_id)
        if type(state["entity_count"]) is not int or state["entity_count"] < 0:
            error("NER completion entity_count must be a nonnegative integer", article_id)
        found = 0
        while pending is not None and pending["article_id"] == article_id:
            entity = pending
            entity_before = report["error_count"]
            if type(entity["entity_index"]) is not int or entity["entity_index"] != found:
                error("NER mention indices must be contiguous from zero", article_id)
            start, end = entity["char_start"], entity["char_end"]
            valid_span = (type(start) is int and type(end) is int and valid_source
                          and 0 <= start < end <= len(text))
            if not valid_span:
                error("NER mention character span is outside current article text", article_id)
            elif text[start:end] != entity["text"]:
                error("NER mention text disagrees with current article source span", article_id)
            if not isinstance(entity["label"], str) or not entity["label"].strip():
                error("NER mention label is empty or not a string", article_id)
            score = entity["score"]
            if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)):
                error("NER mention score must be finite or null", article_id)
            if report["error_count"] == entity_before:
                counts["verified_mentions"] += 1
            fingerprint.update(_json([
                article_id, entity["entity_index"], entity["text"], entity["label"], start, end, score,
            ]).encode("utf-8"))
            found += 1
            pending = next(mentions, None)
        if found != state["entity_count"]:
            error("NER completion mention count disagrees with stored mentions", article_id)
        if found == 0:
            counts["zero_mention_articles"] += 1
        if report["error_count"] == before:
            counts["verified_articles"] += 1
        fingerprint.update(_json([article_id, state["content_sha256"], state["entity_count"]]).encode("utf-8"))
        if progress and (checked % 1000 == 0 or checked == counts["completed_articles"]):
            progress({"event": "ner_baseline_articles_audited", "checked": checked,
                      "total": counts["completed_articles"]})
    while pending is not None:
        counts["orphan_mentions"] += 1
        pending = next(mentions, None)
    if counts["orphan_mentions"]:
        error(f"NER mentions without a completion row: {counts['orphan_mentions']}")
    report["content_fingerprint"] = fingerprint.hexdigest()
    report["coverage_complete"] = (report["membership_complete"] and not counts["orphan_mentions"]
                                   and counts["verified_articles"] == counts["eligible_articles"])


def audit_ner_baseline(
    database_path: str | Path, *, run_id: str = "classla-sl", progress: Callable[[dict], None] | None = None,
) -> dict:
    """Inspect one NER run in a read-only snapshot without migrations or models.

    Eligibility matches the existing producer's SQL TRIM rule. Character spans
    are checked in Python, preserving Unicode and embedded control characters.
    """
    report: dict[str, Any] = {
        "format": "semora.ner-baseline.v1", "run_id": run_id, "eligible": False,
        "membership_complete": False, "coverage_complete": False, "compatibility_verified": False,
        "counts": {"eligible_articles": 0, "completed_articles": 0, "verified_articles": 0,
                   "mentions": 0, "verified_mentions": 0, "missing_completion": 0,
                   "extra_completion": 0, "orphan_mentions": 0, "zero_mention_articles": 0},
        "config": None, "model_evidence": None, "errors": [], "error_count": 0,
        "limitations": [
            "Recorded model digests/revisions do not prove historical execution or immutable runtime compatibility.",
            "Historical CLASSLA code hashes do not cover the complete tokenizer/alignment pipeline.",
            "Absolute resource paths and throughput settings are evidence, not verified compatibility identities.",
            "Entity extraction has not been rerun; zero-mention articles require explicit completion rows.",
        ],
    }

    def error(reason: str, article_id: str | None = None) -> None:
        report["error_count"] += 1
        if len(report["errors"]) < _ERROR_LIMIT:
            report["errors"].append({"article_id": article_id, "reason": reason})

    try:
        if not _nonempty(run_id):
            raise ValueError("NER run_id must be a nonempty trimmed string")
        with closing(open_connection(database_path, read_only=True)) as conn:
            conn.execute("BEGIN")
            _inspect(conn, report, error, progress)
    except (ValueError, OSError, sqlite3.DatabaseError, KeyError, TypeError) as exc:
        error(str(exc))
    report["eligible"] = report["error_count"] == 0 and report["coverage_complete"]
    report["evidence_fingerprint"] = _sha(_json(report))
    return report
