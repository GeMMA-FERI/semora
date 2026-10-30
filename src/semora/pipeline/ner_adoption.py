"""Register fully audited historical NER coverage for additive batches."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

from semora.pipeline.ner import model_identity
from semora.pipeline.ner_baseline import _config as ner_run_config
from semora.pipeline.ner_baseline import audit_ner_baseline
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _trusted_report(report: dict, run_id: str, config_json: str) -> None:
    """Check a prior report's identity, while trusting its historical row scan."""
    if not isinstance(report, dict) or report.get("format") != "semora.ner-baseline.v1":
        raise ValueError("Expected a historical NER baseline audit report.")
    if report.get("run_id") != run_id or report.get("eligible") is not True or report.get("error_count") != 0:
        raise ValueError("Prior NER audit report is ineligible or belongs to another run.")
    fingerprint = report.get("evidence_fingerprint")
    evidence = {key: value for key, value in report.items() if key != "evidence_fingerprint"}
    if not isinstance(fingerprint, str) or hashlib.sha256(_json(evidence).encode("utf-8")).hexdigest() != fingerprint:
        raise ValueError("Prior NER audit report fingerprint is invalid.")
    config, _ = ner_run_config(config_json)
    if report.get("config_sha256") != hashlib.sha256(_json(config).encode("utf-8")).hexdigest():
        raise ValueError("NER run configuration differs from the prior audit report.")


def adopt_ner_baseline(
    database_path: str | Path,
    *,
    run_id: str = "classla-sl",
    progress=None,
    audit_report: dict | None = None,
) -> dict:
    """Audit under writer exclusion, then store a one-time historical receipt.

    Existing extraction rows and model configuration are never rewritten. A
    subsequent batch extends the same NER run through transactional checkpoints.
    """
    path = Path(database_path).resolve(strict=True)
    with ExitStack() as stack:
        stack.enter_context(SemanticBuildLock(path.with_name(path.name + ".pipeline.lock"), "pipeline"))
        stack.enter_context(SemanticBuildLock(path.with_name(path.name + ".ner.lock"), "NER"))
        database = Database(path)
        try:
            database.initialize()
            conn = database.conn
            existing = conn.execute(
                "SELECT evidence_json,model_identity_json FROM pipeline_ner_adoptions WHERE ner_run_id=?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                report = json.loads(existing["evidence_json"])
                return {
                    **report,
                    "adoption": {
                        "run_id": run_id,
                        "already_adopted": True,
                        "model_identity": json.loads(existing["model_identity_json"]),
                    },
                }
            from semora.pipeline.publication_state import invalidate_publication
            invalidate_publication(path)
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                if audit_report is None:
                    report = audit_ner_baseline(path, run_id=run_id, progress=progress)
                else:
                    run = conn.execute("SELECT config_json FROM ner_runs WHERE ner_run_id=?", (run_id,)).fetchone()
                    if run is None:
                        raise ValueError("Requested NER run does not exist.")
                    _trusted_report(audit_report, run_id, run[0])
                    report = audit_report
                if not report["eligible"]:
                    raise ValueError("Historical NER baseline is ineligible; review the audit report.")
                identity = model_identity(report["config"]["model"])
                conn.execute(
                    "INSERT INTO pipeline_ner_adoptions(ner_run_id,evidence_json,model_identity_json) VALUES (?,?,?)",
                    (run_id, _json(report), _json(identity)),
                )
            return {
                **report,
                "adoption": {
                    "run_id": run_id,
                    "already_adopted": False,
                    "model_identity": identity,
                    "prior_audit_trusted_without_rescan": audit_report is not None,
                },
            }
        finally:
            database.close()
