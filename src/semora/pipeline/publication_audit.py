"""Read-only evidence gate for a complete additive serving revision."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from semora.ner.spellfix import extension_path, load_spellfix
from semora.pipeline.batches import batch_status
from semora.pipeline.entities import _expected, _memberships
from semora.pipeline.entity_baseline import (
    NORMALIZATION,
    _dictionary_rows,
    _expected_memberships,
    _manifest,
    _schemas,
    expected_names_sha256,
)
from semora.pipeline.lexical import _validate_existing, lexical_schema_compatible
from semora.pipeline.ner import _coverage as ner_coverage
from semora.pipeline.ner import _recorded_identity
from semora.pipeline.semantic import (
    _coverage as vector_coverage,
)
from semora.pipeline.semantic import (
    read_store_config,
    store_corpus_identity,
    validate_corpus_membership,
)
from semora.pipeline.semantic_faiss import _load_generation, _shards
from semora.retrieval.semantic_faiss import FaissBuildConfig, _import_faiss
from semora.storage.connection import open_connection

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def audit_publication(
    database_path: str | Path,
    batch_id: str,
    semantic_dir: str | Path,
    semantic_generation: str,
    entity_dir: str | Path,
    *,
    run_id: str = "classla-sl",
    spellfix_path: str | Path | None = None,
    faiss_module=None,
    progress=None,
) -> dict:
    """Verify current corpus and private artifacts without changing any file.

    This deliberately streams the full dictionary and vector ledger. A sampled
    check cannot certify that an old serving revision covers the current corpus.
    """
    report: dict[str, Any] = {
        "format": "semora.pipeline.publication-audit.v1",
        "batch_id": batch_id,
        "ner_run_id": run_id,
        "semantic_generation": semantic_generation,
        "corpus_id": None,
        "epoch": None,
        "entity_generation": None,
        "entity_manifest_sha256": None,
        "candidate": None,
        "stages": {},
        "errors": [],
        "eligible": False,
        "publication_ready": False,
        "retrieval_ready": False,
    }

    def check(name: str, action) -> None:
        try:
            report["stages"][name] = action()
        except (ValueError, OSError, sqlite3.DatabaseError, KeyError, TypeError, IndexError) as exc:
            report["stages"][name] = {"complete": False}
            report["errors"].append({"stage": name, "reason": str(exc)})

    if not isinstance(batch_id, str) or not batch_id.strip() or not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Supply a nonempty batch and NER run ID.")
    if not isinstance(semantic_generation, str) or _DIGEST.fullmatch(semantic_generation) is None:
        raise ValueError("Supply the exact 64-character private FAISS generation ID.")
    source = Path(database_path).resolve(strict=True)
    semantic = Path(semantic_dir).resolve()
    entity = Path(entity_dir).resolve()
    library = Path(spellfix_path).resolve() if spellfix_path else extension_path(source.parent.parent)
    if (semantic / "mapping.sqlite").exists() and source.samefile(semantic / "mapping.sqlite"):
        raise ValueError("Corpus and semantic mapping must be different files.")

    with closing(open_connection(source, read_only=True)) as corpus:
        corpus.execute("BEGIN")
        check("ingestion", lambda: _ingestion(corpus, batch_status(source, batch_id)))
        identity = corpus.execute(
            "SELECT corpus_id FROM pipeline_corpus_identity WHERE identity_id=1"
        ).fetchone()
        if identity is None or not re.fullmatch(r"[0-9a-f]{32}", identity[0]):
            raise ValueError("Corpus identity is missing or invalid.")
        report["corpus_id"] = identity[0]
        try:
            row = corpus.execute("SELECT epoch FROM pipeline_publication_state WHERE state_id=1").fetchone()
            if row is None or type(row[0]) is not int or row[0] < 0:
                raise ValueError("Publication epoch is missing or invalid.")
            report["epoch"] = row[0]
        except sqlite3.DatabaseError as exc:
            report["errors"].append({"stage": "publication_state", "reason": str(exc)})
        for stage in ("surface", "lemma"):
            check(stage, lambda stage=stage: _lexical(corpus, batch_id, stage))
        check("ner", lambda: _ner(corpus, batch_id, run_id))

        def semantic_stage() -> dict:
            with closing(open_connection(semantic / "mapping.sqlite", read_only=True)) as mapping:
                mapping.execute("ATTACH DATABASE ? AS corpus", (source.as_uri() + "?mode=ro",))
                mapping.execute("BEGIN")
                config = read_store_config(mapping)
                if config is None or store_corpus_identity(mapping) != identity[0]:
                    raise ValueError("Semantic mapping is not adopted for this corpus.")
                validate_corpus_membership(mapping)
                eligible, completed = vector_coverage(mapping, batch_id, semantic / "vectors", config)
                if eligible != completed:
                    raise ValueError(f"Semantic vectors are incomplete for the batch: {completed}/{eligible}.")
                shards = _shards(mapping, semantic, config)
                generations = semantic / "generations"
                directory = generations / semantic_generation
                if generations.is_symlink() or directory.is_symlink() or not directory.is_dir():
                    raise ValueError("Requested private FAISS generation is missing or unsafe.")
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                faiss_config = FaissBuildConfig(**manifest["faiss"])
                faiss_config.validate(config["dimensions"])
                faiss = faiss_module if faiss_module is not None else _import_faiss()
                _, verified = _load_generation(
                    faiss, directory, config, shards, faiss_config, identity[0]
                )
                vectors = sum(shard.vector_count for shard in shards)
                if verified["chunks"] != vectors:
                    raise ValueError("Private FAISS generation does not cover the current mapping.")
                corpus_chunks = mapping.execute(
                    "SELECT COUNT(*) FROM corpus.chunks c JOIN corpus.articles a USING(article_id) "
                    "WHERE a.is_valid=1"
                ).fetchone()[0]
                if vectors != corpus_chunks:
                    raise ValueError(
                        f"Semantic mapping omits eligible corpus chunks: {vectors}/{corpus_chunks}."
                    )
                return {"complete": True, "batch_chunks": eligible, "mapped_chunks": vectors,
                        "generation": semantic_generation, "model_id": config["model_id"],
                        "dimensions": config["dimensions"],
                        "index_sha256": verified["index_sha256"]}

        check("semantic", semantic_stage)

        def entity_stage() -> dict:
            manifest, dictionary, manifest_hash = _manifest(entity, run_id)
            report["entity_generation"] = manifest["generation"]
            report["entity_manifest_sha256"] = manifest_hash
            with closing(open_connection(dictionary, read_only=True)) as names:
                load_spellfix(names, library)
                names.execute("ATTACH DATABASE ? AS corpus", (source.as_uri() + "?mode=ro",))
                names.execute("BEGIN")
                profile = names.execute(
                    "SELECT corpus_id,ner_run_id,normalization,advancement_token,evidence_json "
                    "FROM pipeline_entity_profile WHERE profile_id=1"
                ).fetchone()
                if profile is None or (profile[0], profile[1], profile[2]) != (identity[0], run_id, NORMALIZATION):
                    raise ValueError("Entity dictionary is not adopted for this corpus and NER run.")
                binding = corpus.execute(
                    "SELECT corpus_id,generation,advancement_token,pending_token "
                    "FROM pipeline_entity_bindings WHERE ner_run_id=?", (run_id,)
                ).fetchone()
                if binding is None or tuple(binding) != (
                    identity[0], manifest["generation"], profile["advancement_token"], None
                ):
                    raise ValueError("Corpus/entity dictionary advancement is unfinished or mismatched.")
                evidence = json.loads(profile["evidence_json"])
                if evidence.get("manifest_sha256") != manifest_hash:
                    raise ValueError("Entity manifest changed after adoption.")
                _schemas(names)
                counts: dict[str, int] = dict.fromkeys((
                    "names", "existing_memberships", "spellfix_rows", "orphan_name_memberships",
                    "ner_completed_articles", "ner_pending_articles", "expected_memberships",
                    "matched_memberships", "verified_memberships", "pending_memberships",
                    "unexpected_memberships", "complete_articles", "pending_articles",
                    "zero_name_articles", "orphan_ner_mentions",
                ), 0)
                errors: list[str] = []
                def error(reason, *_args, **_kwargs):
                    audit["error_count"] += 1
                    if len(errors) < 10:
                        errors.append(reason)
                audit: dict[str, Any] = {
                    "counts": counts, "manifest": dict(manifest), "run_id": run_id, "error_count": 0
                }
                # The legacy manifest is a baseline receipt. Its name count is
                # intentionally stale after append and is not a serving total.
                live_names = names.execute("SELECT COUNT(*) FROM names").fetchone()[0]
                if live_names < manifest["names"]:
                    raise ValueError("Entity dictionary has fewer names than its adopted baseline manifest.")
                audit["manifest"]["names"] = live_names
                _dictionary_rows(names, audit, error, progress)
                _expected_memberships(names, audit, error, progress)
                if errors or counts["pending_memberships"] or counts["ner_pending_articles"]:
                    raise ValueError("Entity dictionary is incomplete or inconsistent: " + "; ".join(errors[:3]))
                selected = 0
                for row in corpus.execute(
                    "SELECT a.article_id,a.content,s.content_sha256,s.entity_count FROM articles a "
                    "JOIN pipeline_batch_issues b USING(newspaper_id) "
                    "JOIN article_ner_state s ON s.article_id=a.article_id AND s.ner_run_id=? "
                    "WHERE b.batch_id=? AND a.is_valid=1 AND TRIM(a.content)<>'' ORDER BY a.article_id",
                    (run_id, batch_id),
                ):
                    expected, _ = _expected(corpus, row, run_id)
                    ledger = names.execute(
                        "SELECT content_sha256,expected_names_sha256,name_count "
                        "FROM pipeline_entity_articles WHERE article_id=?", (row["article_id"],)
                    ).fetchone()
                    if ledger is None or tuple(ledger) != (
                        row["content_sha256"], expected_names_sha256(expected), len(expected)
                    ) or _memberships(names, row["article_id"]) != {
                        key: (offset, row["content_sha256"]) for key, offset in expected.items()
                    }:
                        raise ValueError(f"Entity completion ledger differs for article {row['article_id']}.")
                    selected += 1
                return {"complete": True, "batch_articles": selected, "names": counts["names"],
                        "memberships": counts["existing_memberships"],
                        "baseline_manifest_names": manifest["names"],
                        "generation": manifest["generation"]}

        check("entities", entity_stage)
    report["eligible"] = not report["errors"] and all(
        stage.get("complete") is True for stage in report["stages"].values()
    ) and len(report["stages"]) == 6
    if report["eligible"]:
        report["candidate"] = {
            "format": "semora.pipeline.publication.v1",
            "corpus_id": report["corpus_id"],
            "epoch": report["epoch"],
            "semantic": {
                "generation": semantic_generation,
                "chunks": report["stages"]["semantic"]["mapped_chunks"],
                "index_sha256": report["stages"]["semantic"]["index_sha256"],
            },
            "entities": {
                "generation": report["entity_generation"],
                "ner_run_id": run_id,
                "manifest_sha256": report["entity_manifest_sha256"],
            },
        }
    return report


def _ingestion(corpus: sqlite3.Connection, status: dict) -> dict:
    if not status["ingestion_complete"] or status["issues"] != status["issue_count"]:
        raise ValueError("Selected batch newspaper/article/chunk ingestion is incomplete.")
    pending = corpus.execute(
        "SELECT b.batch_id FROM pipeline_batches b WHERE b.state!='ingested' "
        "OR b.issue_count<>(SELECT COUNT(*) FROM pipeline_batch_issues i WHERE i.batch_id=b.batch_id) "
        "OR EXISTS (SELECT 1 FROM pipeline_batch_issues i WHERE i.batch_id=b.batch_id "
        "AND (NOT EXISTS (SELECT 1 FROM pipeline_issue_stages s WHERE s.newspaper_id=i.newspaper_id "
        "AND s.stage='articles') OR NOT EXISTS (SELECT 1 FROM pipeline_issue_stages s "
        "WHERE s.newspaper_id=i.newspaper_id AND s.stage='chunks'))) LIMIT 1"
    ).fetchone()
    if pending is not None:
        raise ValueError(f"Another registered batch has unfinished ingestion: {pending['batch_id']}.")
    return {
        "complete": True,
        "issues": status["issues"],
        "article_stage_issues": status["stages"]["articles"],
        "chunk_stage_issues": status["stages"]["chunks"],
    }


def _lexical(corpus: sqlite3.Connection, batch_id: str, stage: str) -> dict:
    if not lexical_schema_compatible(corpus, stage):
        raise ValueError(f"Unsupported {stage} FTS schema.")
    row = corpus.execute("SELECT config_json FROM pipeline_lexical_configs WHERE stage=?", (stage,)).fetchone()
    if row is None:
        raise ValueError(f"{stage} lexical configuration is not registered.")
    eligible, completed = _validate_existing(corpus, batch_id, stage, json.loads(row[0]))
    if eligible != completed:
        raise ValueError(f"{stage} lexical index is incomplete: {completed}/{eligible}.")
    missing = corpus.execute(
        "SELECT a.article_id FROM articles a LEFT JOIN pipeline_lexical_state s "
        "ON s.article_id=a.article_id AND s.stage=? "
        "WHERE a.is_valid=1 AND s.article_id IS NULL LIMIT 1",
        (stage,),
    ).fetchone()
    if missing is not None:
        raise ValueError(f"{stage} lexical index has an unpublished article outside the selected batch.")
    return {"complete": True, "eligible_articles": eligible, "indexed_articles": completed}


def _ner(corpus: sqlite3.Connection, batch_id: str, run_id: str) -> dict:
    config, identity = _recorded_identity(corpus, run_id)
    if config is None or identity is None:
        raise ValueError("NER run has no supported recorded identity.")
    eligible, completed = ner_coverage(corpus, batch_id, run_id)
    if eligible != completed:
        raise ValueError(f"NER is incomplete for the batch: {completed}/{eligible}.")
    return {"complete": True, "eligible_articles": eligible, "completed_articles": completed}
