"""Bind an audited legacy entity dictionary to one logical corpus and NER run."""

from __future__ import annotations

import json
from contextlib import ExitStack, closing
from pathlib import Path

from semora.pipeline.entity_baseline import NORMALIZATION, _manifest, audit_entity_baseline
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage.connection import open_connection


def adopt_entity_baseline(
    database_path: str | Path,
    entity_dir: str | Path,
    *,
    run_id: str = "classla-sl",
    spellfix_path: str | Path | None = None,
    progress=None,
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

                _, dictionary, manifest_hash = _manifest(root, run_id)
                if dictionary.samefile(source):
                    raise ValueError("Dictionary cannot alias the corpus database.")
                with closing(open_connection(dictionary, read_only=True)) as recorded:
                    has_profile = recorded.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_entity_profile'"
                    ).fetchone()
                    if has_profile:
                        previous = recorded.execute(
                            "SELECT corpus_id,ner_run_id,normalization,evidence_json "
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
                        return {**evidence, "adoption": {"already_adopted": True, "corpus_id": identity[0]}}

                report = audit_entity_baseline(
                    source,
                    root,
                    run_id=run_id,
                    spellfix_path=spellfix_path,
                    progress=progress,
                )
                if not report["eligible"] or not report["complete"]:
                    raise ValueError("Entity baseline is incomplete or invalid; review the read-only audit report.")
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
                            "INSERT INTO pipeline_entity_profile VALUES (1,?,?,?,?,?)",
                            (
                                identity[0],
                                run_id,
                                NORMALIZATION,
                                report["evidence_fingerprint"],
                                json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False),
                            ),
                        )
                        target.execute("COMMIT")
                    except BaseException:
                        if target.in_transaction:
                            target.execute("ROLLBACK")
                        raise
                return {**report, "adoption": {"already_adopted": False, "corpus_id": identity[0]}}
            finally:
                corpus.execute("ROLLBACK")
