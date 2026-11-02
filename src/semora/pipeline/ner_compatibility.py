"""Explicit continuation of adopted historical CLASSLA NER runs."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack, closing
from pathlib import Path

from semora.execution.client import RemoteNERPool
from semora.execution.contracts import semantic_fingerprint
from semora.pipeline.ner import _json, _recorded_identity, _table, _without_runtime, model_identity
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage.connection import open_connection


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _receipt(connection, run_id):
    if not _table(connection, "pipeline_ner_compatibility"):
        return None
    row = connection.execute(
        "SELECT receipt_json FROM pipeline_ner_compatibility WHERE ner_run_id=?", (run_id,)
    ).fetchone()
    return json.loads(row[0]) if row else None


def compatible(connection, run_id, recorded, current, metadata):
    receipt = _receipt(connection, run_id)
    if receipt is None or current.get("worker_fingerprint") != receipt.get("worker_fingerprint"):
        return False
    config, historical = _recorded_identity(connection, run_id)
    adoption = connection.execute(
        "SELECT evidence_json,model_identity_json FROM pipeline_ner_adoptions WHERE ner_run_id=?", (run_id,)
    ).fetchone()
    workers = metadata.get("execution", {}).get("workers", [])
    qualified = receipt.get("worker_execution", {}).get("workers", [])
    runtime_by_url = {worker["url"]: worker["runtime"] for worker in qualified}
    if not workers or any(runtime_by_url.get(worker.get("url")) != worker.get("runtime") for worker in workers):
        raise ValueError("Remote worker endpoint or runtime differs from the qualified receipt.")
    if (
        historical != recorded
        or adoption is None
        or _digest(config) != receipt["config_sha256"]
        or _digest([adoption[0], adoption[1]]) != receipt["adoption_sha256"]
        or current.get("worker_identity") != receipt["worker_identity"]
    ):
        raise ValueError("Historical NER compatibility receipt no longer matches its evidence.")
    return True


def qualify_ner_continuation(database_path, *, run_id, execution, reference, comparison, evidence_path):
    """Record operator-supplied reference and comparison evidence for one worker identity.

    Reference metadata and identity must originate from the same loaded CLASSLA
    adapter. Comparison is an operator-supplied record of an actual sample; it
    cannot establish what was executed historically or prove general parity.
    """
    if execution.backend != "remote":
        raise ValueError("Qualification requires remote workers.")
    path = Path(database_path).resolve(strict=True)
    with ExitStack() as stack:
        stack.enter_context(SemanticBuildLock(path.with_name(path.name + ".pipeline.lock"), "pipeline"))
        stack.enter_context(SemanticBuildLock(path.with_name(path.name + ".ner.lock"), "NER"))
        connection = stack.enter_context(closing(open_connection(path)))
        config, historical = _recorded_identity(connection, run_id)
        if config is None or config.get("format") is not None or historical is None:
            raise ValueError("Qualification requires an adopted historical NER run.")
        adoption = (
            connection.execute(
                "SELECT evidence_json,model_identity_json FROM pipeline_ner_adoptions WHERE ner_run_id=?", (run_id,)
            ).fetchone()
            if _table(connection, "pipeline_ner_adoptions")
            else None
        )
        if adoption is None or json.loads(adoption[1]) != historical:
            raise ValueError("Qualification requires a matching historical adoption receipt.")
        if not isinstance(reference, dict):
            raise ValueError("Reference must be a JSON object.")
        metadata = reference.get("metadata")
        worker_identity = reference.get("identity")
        if not isinstance(metadata, dict) or not isinstance(worker_identity, dict):
            raise ValueError("Reference must contain metadata and full identity.")
        if any(not historical.get(key) for key in ("model_sha256", "source_sha256", "classla_version")):
            raise ValueError("Historical model, source and version evidence is required.")
        if model_identity(metadata) != historical:
            raise ValueError("Reference model identity does not match historical metadata.")
        if (
            worker_identity.get("classla_version") != historical["classla_version"]
            or worker_identity.get("package") != historical.get("package")
            or worker_identity.get("resources", {}).get("ner.model_path") != historical["model_sha256"]
            or _without_runtime(worker_identity.get("processor_config", {}).get("ner")) != historical.get("ner_config")
        ):
            raise ValueError("Worker model and processor identity do not match historical metadata.")
        pool = RemoteNERPool(execution)
        workers = pool.metadata.get("execution", {}).get("workers", [])
        if not workers or any(worker.get("runtime", {}).get("device") not in ("cpu", "cuda") for worker in workers):
            raise ValueError("Worker must report a qualified CPU or CUDA runtime.")
        if worker_identity != pool.metadata["worker_identity"] or reference.get("fingerprint") != semantic_fingerprint(
            worker_identity
        ):
            raise ValueError("Reference full identity does not match the current remote worker.")
        if (
            not isinstance(comparison, dict)
            or comparison.get("format") != "semora.ner-comparison.v1"
            or type(comparison.get("documents")) is not int
            or comparison["documents"] < 1
            or type(comparison.get("mismatches")) is not int
            or comparison["mismatches"] != 0
            or not isinstance(comparison.get("description"), str)
            or not comparison["description"].strip()
            or not isinstance(comparison.get("evidence_sha256"), str)
            or len(comparison["evidence_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in comparison["evidence_sha256"])
        ):
            raise ValueError("A successful measured sample comparison and evidence digest are required.")
        evidence_digest = hashlib.sha256()
        with Path(evidence_path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                evidence_digest.update(block)
        if evidence_digest.hexdigest() != comparison["evidence_sha256"]:
            raise ValueError("Comparison evidence file does not match its SHA-256 digest.")
        receipt = {
            "format": "semora.ner-continuation.v1",
            "run_id": run_id,
            "config_sha256": _digest(config),
            "adoption_sha256": _digest([adoption[0], adoption[1]]),
            "historical_identity": historical,
            "reference": {"metadata": metadata, "identity": worker_identity, "fingerprint": reference["fingerprint"]},
            "comparison": comparison,
            "worker_identity": worker_identity,
            "worker_execution": pool.metadata.get("execution"),
            "worker_fingerprint": pool.metadata["worker_fingerprint"],
            "evidence_status": "operator_supplied; historical execution not verified",
        }
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pipeline_ner_compatibility ("
                "ner_run_id TEXT PRIMARY KEY REFERENCES ner_runs(ner_run_id), "
                "receipt_json TEXT NOT NULL, qualified_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            old = _receipt(connection, run_id)
            if old is not None and old != receipt:
                raise ValueError("A different compatibility receipt already exists for this run.")
            if old is None:
                connection.execute(
                    "INSERT INTO pipeline_ner_compatibility(ner_run_id,receipt_json) VALUES (?,?)",
                    (run_id, _json(receipt)),
                )
        return receipt
