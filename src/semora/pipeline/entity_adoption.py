"""Bind an audited legacy entity dictionary to one logical corpus and NER run."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import ExitStack, closing
from pathlib import Path

from semora.pipeline.entity_baseline import NORMALIZATION, _json, _manifest, audit_entity_baseline
from semora.pipeline.ner_baseline import _config as ner_run_config
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage.connection import open_connection


def _trusted_report(
    report: dict,
    run_id: str,
    manifest_hash: str,
    dictionary: Path,
    config_json: str,
) -> None:
    """Check cheap current identities and trust the prior dictionary row scan."""
    if not isinstance(report, dict) or report.get("format") != "semora.entity-baseline.v1":
        raise ValueError("Expected a historical entity baseline audit report.")
    if (report.get("run_id") != run_id or report.get("normalization") != NORMALIZATION
            or report.get("eligible") is not True or report.get("complete") is not True
            or report.get("error_count") != 0):
        raise ValueError("Prior entity audit report is incomplete or belongs to another run.")
    fingerprint = report.get("evidence_fingerprint")
    evidence = {key: value for key, value in report.items() if key != "evidence_fingerprint"}
    if not isinstance(fingerprint, str) or hashlib.sha256(_json(evidence).encode("utf-8")).hexdigest() != fingerprint:
        raise ValueError("Prior entity audit report fingerprint is invalid.")
    if report.get("manifest_sha256") != manifest_hash or report.get("dictionary_path") != str(dictionary):
        raise ValueError("Entity manifest or dictionary path differs from the prior audit report.")
    config, _ = ner_run_config(config_json)
    if report.get("ner_config_sha256") != hashlib.sha256(_json(config).encode("utf-8")).hexdigest():
        raise ValueError("NER configuration differs from the prior entity audit report.")


def adopt_entity_baseline(
    database_path: str | Path,
    entity_dir: str | Path,
    *,
    run_id: str = "classla-sl",
    spellfix_path: str | Path | None = None,
    progress=None,
    audit_report: dict | None = None,
) -> dict:
    """Audit complete legacy output, then add only control tables and a lookup index.

    The old dictionary file and entity IDs remain in place. The caller must
    stop retrieval readers and legacy writers before calling this maintenance
    operation; ordinary retry validates the existing binding without rescanning.
    """
    source = Path(database_path).resolve(strict=True)
    root = Path(entity_dir).resolve(strict=True)
    with ExitStack() as stack:
        stack.enter_context(SemanticBuildLock(source.with_name(source.name + ".pipeline.lock"), "pipeline"))
        stack.enter_context(SemanticBuildLock(source.with_name(source.name + ".ner.lock"), "NER"))
        stack.enter_context(SemanticBuildLock(root / ".build.lock", "entity dictionary"))
        with closing(open_connection(source)) as corpus:
            corpus.execute("BEGIN IMMEDIATE")
            try:
                identity = corpus.execute(
                    "SELECT corpus_id FROM pipeline_corpus_identity WHERE identity_id=1"
                ).fetchone()
                if identity is None:
                    raise ValueError("Corpus identity is missing; apply current Semora migrations first.")
                run = corpus.execute("SELECT config_json FROM ner_runs WHERE ner_run_id=?", (run_id,)).fetchone()
                if run is None:
                    raise ValueError("Requested NER run does not exist.")
                config = json.loads(run[0])
                adopted = corpus.execute(
                    "SELECT 1 FROM pipeline_ner_adoptions WHERE ner_run_id=?", (run_id,)
                ).fetchone()
                if adopted is None and config.get("format") != "semora.pipeline.ner.v1":
                    raise ValueError("Audit and adopt the historical NER run before dictionary adoption.")

                manifest, dictionary, manifest_hash = _manifest(root, run_id)
                if dictionary.samefile(source):
                    raise ValueError("Dictionary cannot alias the corpus database.")
                corpus.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_entity_bindings (
                        ner_run_id TEXT PRIMARY KEY,
                        corpus_id TEXT NOT NULL,
                        generation TEXT NOT NULL,
                        advancement_token TEXT NOT NULL,
                        pending_token TEXT
                    )
                """)
                binding = corpus.execute(
                    "SELECT * FROM pipeline_entity_bindings WHERE ner_run_id=?", (run_id,)
                ).fetchone()
                if binding is not None and (
                    binding["corpus_id"] != identity[0]
                    or binding["generation"] != manifest["generation"]
                    or binding["pending_token"] is not None
                ):
                    raise ValueError("Corpus entity binding differs or has an unfinished checkpoint.")
                with closing(open_connection(dictionary, read_only=True)) as recorded:
                    has_profile = recorded.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_entity_profile'"
                    ).fetchone()
                    if has_profile:
                        previous = recorded.execute(
                            "SELECT corpus_id,ner_run_id,normalization,advancement_token,evidence_json "
                            "FROM pipeline_entity_profile WHERE profile_id=1"
                        ).fetchone()
                        if previous is None or (
                            previous["corpus_id"],
                            previous["ner_run_id"],
                            previous["normalization"],
                        ) != (identity[0], run_id, NORMALIZATION):
                            raise ValueError("Entity dictionary is bound to a different corpus or NER profile.")
                        evidence = json.loads(previous["evidence_json"])
                        if evidence["manifest_sha256"] != manifest_hash:
                            raise ValueError("Entity manifest changed after dictionary adoption.")
                        if binding is None or binding["advancement_token"] != previous["advancement_token"]:
                            raise ValueError("Corpus and entity dictionary advancement tokens differ.")
                        return {**evidence, "adoption": {"already_adopted": True, "corpus_id": identity[0]}}

                if audit_report is None:
                    report = audit_entity_baseline(
                        source,
                        root,
                        run_id=run_id,
                        spellfix_path=spellfix_path,
                        progress=progress,
                    )
                else:
                    _trusted_report(audit_report, run_id, manifest_hash, dictionary, run[0])
                    report = audit_report
                if not report["eligible"] or not report["complete"]:
                    raise ValueError("Entity baseline is incomplete or invalid; review the read-only audit report.")
                token = binding["advancement_token"] if binding is not None else uuid.uuid4().hex
                if binding is None:
                    corpus.execute(
                        "INSERT INTO pipeline_entity_bindings VALUES (?,?,?,?,NULL)",
                        (run_id, identity[0], manifest["generation"], token),
                    )
                corpus.execute("COMMIT")
                with closing(open_connection(dictionary)) as target:
                    target.execute("BEGIN IMMEDIATE")
                    try:
                        if target.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_entity_profile'"
                        ).fetchone():
                            raise ValueError("Entity dictionary profile appeared during adoption; retry the audit.")
                        target.execute("""
                            CREATE TABLE pipeline_entity_profile (
                                profile_id INTEGER PRIMARY KEY CHECK(profile_id=1),
                                corpus_id TEXT NOT NULL,
                                ner_run_id TEXT NOT NULL,
                                normalization TEXT NOT NULL,
                                advancement_token TEXT NOT NULL,
                                audit_fingerprint TEXT NOT NULL,
                                evidence_json TEXT NOT NULL
                            )
                        """)
                        target.execute("""
                            CREATE TABLE pipeline_entity_articles (
                                article_id TEXT PRIMARY KEY,
                                content_sha256 TEXT NOT NULL,
                                expected_names_sha256 TEXT NOT NULL,
                                name_count INTEGER NOT NULL CHECK(name_count>=0)
                            )
                        """)
                        # The legacy primary key is (entity_id,article_id). New
                        # batch retries need article-first lookup without a scan.
                        target.execute(
                            "CREATE INDEX pipeline_entity_articles_lookup ON entity_articles(article_id,entity_id)"
                        )
                        target.execute(
                            "INSERT INTO pipeline_entity_profile VALUES (1,?,?,?,?,?,?)",
                            (
                                identity[0],
                                run_id,
                                NORMALIZATION,
                                token,
                                report["evidence_fingerprint"],
                                json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False),
                            ),
                        )
                        target.execute("COMMIT")
                    except BaseException:
                        if target.in_transaction:
                            target.execute("ROLLBACK")
                        raise
                return {
                    **report,
                    "adoption": {
                        "already_adopted": False,
                        "corpus_id": identity[0],
                        "prior_audit_trusted_without_rescan": audit_report is not None,
                    },
                }
            finally:
                if corpus.in_transaction:
                    corpus.execute("ROLLBACK")
