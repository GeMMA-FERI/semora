"""Explicit operator-trusted reuse of passed publication stages.

The earlier report did not record file snapshot watermarks. Reuse therefore
trusts the operator's unchanged-output assertion; metadata checks cannot prove
that table contents were not edited directly. This is never the default gate.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from semora.pipeline.ner import _recorded_identity
from semora.pipeline.semantic import read_store_config, store_corpus_identity
from semora.pipeline.semantic_faiss import _digest, _file_hash, _ledger_hash
from semora.retrieval.semantic_store import SemanticShard
from semora.storage.connection import open_connection


def reuse_passed_stages(
    corpus: sqlite3.Connection, batch_id: str, run_id: str, semantic: Path,
    generation: str, entities: Path, report_path: str | Path, report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate receipt/current bindings; return only NER, semantic and entities.

    Ingestion and both lexical checks are always run again. No failed stage can
    be reused. The original report bytes are identified in the resulting report.
    """
    path = Path(report_path).resolve(strict=True)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Trusted publication report exceeds 4 MiB.")
    raw = path.read_bytes()
    prior = json.loads(raw)
    if not isinstance(prior, dict) or any(prior.get(key) != expected for key, expected in (
        ("format", "semora.pipeline.publication-audit.v1"),
        ("batch_id", batch_id), ("ner_run_id", run_id),
        ("corpus_id", report["corpus_id"]), ("epoch", report["epoch"]),
        ("semantic_generation", generation), ("entity_scope", "complete"),
    )):
        raise ValueError("Trusted publication report identity/epoch/scope differs from current inputs.")
    stages = prior.get("stages", {})
    if not isinstance(stages, dict) or any(
        not isinstance(stages.get(stage), dict) or stages[stage].get("complete") is not True
        or stages[stage].get("historical_only")
        for stage in ("ner", "semantic", "entities")
    ):
        raise ValueError("Trusted publication report has an incomplete stage selected for reuse.")
    errors = prior.get("errors")
    if not isinstance(errors, list) or any(
        not isinstance(error, dict) or error.get("stage") not in {"surface", "lemma"} for error in errors
    ):
        raise ValueError("Trusted publication report contains errors outside the lexical stages.")
    ner, vectors, names = (stages[key] for key in ("ner", "semantic", "entities"))
    if (
        type(ner.get("eligible_articles")) is not int or ner["eligible_articles"] < 0
        or ner.get("completed_articles") != ner["eligible_articles"]
        or names.get("batch_articles") != ner["eligible_articles"]
        or vectors.get("generation") != generation
        or type(vectors.get("mapped_chunks")) is not int or vectors["mapped_chunks"] < 0
    ):
        raise ValueError("Trusted publication report coverage is inconsistent.")
    _recorded_identity(corpus, run_id)
    # Validate adoption, schemas and matching current advancement tokens, without
    # scanning memberships. This helper does not weaken serving scope here.
    from semora.pipeline.publication_audit import _historical_entities
    _historical_entities(corpus, entities, run_id, report["corpus_id"], report)
    if (
        prior.get("entity_generation") != report["entity_generation"]
        or prior.get("entity_manifest_sha256") != report["entity_manifest_sha256"]
        or names.get("generation") != report["entity_generation"]
    ):
        raise ValueError("Trusted entity generation/manifest differs from current dictionary.")
    directory = semantic / "generations" / generation
    if (semantic / "generations").is_symlink() or directory.is_symlink():
        raise ValueError("Trusted semantic generation must not be a symlink.")
    manifest_path = directory / "manifest.json"
    index_path = directory / "index.faiss"
    if manifest_path.is_symlink() or index_path.is_symlink():
        raise ValueError("Trusted semantic files must not be symlinks.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != expected for key, expected in (
        ("format", "semora.semantic.faiss-generation.v1"),
        ("corpus_id", report["corpus_id"]), ("generation", generation),
        ("chunks", vectors["mapped_chunks"]), ("index_sha256", vectors.get("index_sha256")),
    )):
        raise ValueError("Trusted semantic manifest differs from the passed report.")
    with closing(open_connection(semantic / "mapping.sqlite", read_only=True)) as mapping:
        config = read_store_config(mapping)
        if (config is None or store_corpus_identity(mapping) != report["corpus_id"]
                or config != manifest["store_config"]):
            raise ValueError("Trusted semantic store configuration/identity changed.")
        shards = [SemanticShard(**dict(row)) for row in mapping.execute(
            "SELECT * FROM semantic_shards ORDER BY shard_index")]
        if (
            sum(shard.vector_count for shard in shards) != vectors["mapped_chunks"]
            or _ledger_hash(config, shards) != manifest["vector_fingerprint"]
            or config["model_id"] != vectors.get("model_id")
            or config["dimensions"] != vectors.get("dimensions")
            or mapping.execute("SELECT 1 FROM pipeline_semantic_pending_shards LIMIT 1").fetchone()
        ):
            raise ValueError("Trusted semantic shard ledger changed or has pending writes.")
    if generation != _digest({"corpus_id": report["corpus_id"],
            "vectors": manifest["vector_fingerprint"], "faiss": manifest["faiss"]}):
        raise ValueError("Trusted semantic generation identity is inconsistent.")
    # Hash the compact serving index; avoid rescanning all vector files/mappings.
    if _file_hash(index_path) != vectors["index_sha256"]:
        raise ValueError("Trusted semantic serving index checksum changed.")
    report["trusted_stage_reuse"] = {
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "stages": ["ner", "semantic", "entities"],
        "assumption": ("Operator trusts passed stage outputs unchanged since this report; "
                       "bulk rows/vector files not re-audited."),
    }
    return {stage: {**stages[stage], "reused_from_trusted_report": True}
            for stage in ("ner", "semantic", "entities")}
