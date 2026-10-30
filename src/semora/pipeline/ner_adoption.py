"""Register fully audited historical NER coverage for additive batches."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

from semora.pipeline.ner import model_identity
from semora.pipeline.ner_baseline import audit_ner_baseline
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def adopt_ner_baseline(
    database_path: str | Path,
    *,
    run_id: str = "classla-sl",
    progress=None,
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
                report = audit_ner_baseline(path, run_id=run_id, progress=progress)
                if not report["eligible"]:
                    raise ValueError("Historical NER baseline is ineligible; review the audit report.")
                identity = model_identity(report["config"]["model"])
                conn.execute(
                    "INSERT INTO pipeline_ner_adoptions(ner_run_id,evidence_json,model_identity_json) VALUES (?,?,?)",
                    (run_id, _json(report), _json(identity)),
                )
            return {**report, "adoption": {"run_id": run_id, "already_adopted": False, "model_identity": identity}}
        finally:
            database.close()
