import json
import sqlite3
from contextlib import closing

import pytest

from semora.execution import ExecutionConfig
from semora.execution.contracts import semantic_fingerprint
from semora.pipeline.ner import _recorded_identity, model_identity
from semora.pipeline.ner_compatibility import compatible, qualify_ner_continuation
from tests.test_pipeline_ner import Model, ingest


def test_qualification_binds_adoption_reference_and_worker(tmp_path, monkeypatch):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana."})
    metadata = Model().metadata
    historical = model_identity(metadata)
    worker_identity = {
        "format": "semora.classla-ner.identity.v1",
        "fixture": "one",
        "classla_version": metadata["classla_version"],
        "package": metadata["package"],
        "resources": {"ner.model_path": metadata["model_sha256"]},
        "processor_config": {"ner": historical["ner_config"]},
    }
    fingerprint = semantic_fingerprint(worker_identity)
    reference = {"metadata": metadata, "identity": worker_identity, "fingerprint": fingerprint}
    evidence_path = tmp_path / "comparison.txt"
    evidence_path.write_text("fixture", encoding="utf-8")
    evidence_sha256 = __import__("hashlib").sha256(evidence_path.read_bytes()).hexdigest()
    comparison = {
        "format": "semora.ner-comparison.v1",
        "documents": 1,
        "mismatches": 0,
        "description": "fixture",
        "evidence_sha256": evidence_sha256,
    }
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "INSERT INTO ner_runs(ner_run_id,config_json) VALUES (?,?)", ("classla-sl", json.dumps({"model": metadata}))
        )
        connection.execute(
            "INSERT INTO pipeline_ner_adoptions(ner_run_id,evidence_json,model_identity_json) VALUES (?,?,?)",
            ("classla-sl", "{}", json.dumps(historical)),
        )

    class Pool:
        def __init__(self, _execution):
            self.metadata = {
                "worker_identity": worker_identity,
                "worker_fingerprint": fingerprint,
                "execution": {"workers": [{"url": "http://example", "runtime": {"device": "cuda"}}]},
            }

    monkeypatch.setattr("semora.pipeline.ner_compatibility.RemoteNERPool", Pool)
    execution = ExecutionConfig("remote", ("http://example",))
    bad_reference = {**reference, "metadata": {**metadata, "model_sha256": "c" * 64}}
    with pytest.raises(ValueError, match="historical metadata"):
        qualify_ner_continuation(
            database,
            run_id="classla-sl",
            execution=execution,
            reference=bad_reference,
            comparison=comparison,
            evidence_path=evidence_path,
        )
    with pytest.raises(ValueError, match="sample comparison"):
        qualify_ner_continuation(
            database,
            run_id="classla-sl",
            execution=execution,
            reference=reference,
            comparison={**comparison, "mismatches": 1},
            evidence_path=evidence_path,
        )
    with pytest.raises(ValueError, match="evidence file"):
        qualify_ner_continuation(
            database,
            run_id="classla-sl",
            execution=execution,
            reference=reference,
            comparison={**comparison, "evidence_sha256": "b" * 64},
            evidence_path=evidence_path,
        )
    receipt = qualify_ner_continuation(
        database,
        run_id="classla-sl",
        execution=execution,
        reference=reference,
        comparison=comparison,
        evidence_path=evidence_path,
    )
    assert receipt["worker_fingerprint"] == fingerprint
    assert (
        qualify_ner_continuation(
            database,
            run_id="classla-sl",
            execution=execution,
            reference=reference,
            comparison=comparison,
            evidence_path=evidence_path,
        )
        == receipt
    )
    current = {"worker_fingerprint": fingerprint, "worker_identity": worker_identity}
    worker_identity = {**worker_identity, "fixture": "corrected"}
    fingerprint = semantic_fingerprint(worker_identity)
    updated_reference = {"metadata": metadata, "identity": worker_identity, "fingerprint": fingerprint}
    updated_receipt = qualify_ner_continuation(
        database,
        run_id="classla-sl",
        execution=execution,
        reference=updated_reference,
        comparison=comparison,
        evidence_path=evidence_path,
    )
    assert updated_receipt["worker_fingerprint"] == fingerprint
    remote = {"execution": {"workers": [{"url": "http://example", "runtime": {"device": "cuda"}}]}}
    with closing(sqlite3.connect(database)) as connection, connection:
        _, recorded = _recorded_identity(connection, "classla-sl")
        assert (
            json.loads(connection.execute("SELECT receipt_json FROM pipeline_ner_compatibility").fetchone()[0])
            == receipt
        )
        assert (
            json.loads(connection.execute("SELECT receipt_json FROM pipeline_ner_compatibility_history").fetchone()[0])
            == updated_receipt
        )
        assert compatible(
            connection,
            "classla-sl",
            recorded,
            {"worker_fingerprint": fingerprint, "worker_identity": worker_identity},
            remote,
        )
        assert compatible(connection, "classla-sl", recorded, current, remote)
        assert not compatible(connection, "classla-sl", recorded, {**current, "worker_fingerprint": "0" * 64}, remote)
        with pytest.raises(ValueError, match="runtime"):
            compatible(
                connection,
                "classla-sl",
                recorded,
                current,
                {"execution": {"workers": [{"url": "http://example", "runtime": {"device": "cpu"}}]}},
            )
        connection.execute("UPDATE pipeline_ner_adoptions SET evidence_json='changed'")
        with pytest.raises(ValueError, match="no longer matches"):
            compatible(connection, "classla-sl", recorded, current, remote)


def test_qualified_worker_extends_adopted_run_without_rewriting_metadata(tmp_path):
    from semora.pipeline.ner import index_batch_ner
    from tests.test_pipeline_remote_ner import IDENTITY, config, worker
    from tests.test_pipeline_remote_ner import ingest as remote_ingest

    database = remote_ingest(tmp_path, {"Paper/a.md": "Ana."})
    metadata = Model().metadata
    historical = model_identity(metadata)
    old_config = json.dumps({"model": metadata})
    adoption = json.dumps(historical)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("INSERT INTO ner_runs(ner_run_id,config_json) VALUES ('classla-sl',?)", (old_config,))
        connection.execute(
            "INSERT INTO pipeline_ner_adoptions(ner_run_id,evidence_json,model_identity_json) VALUES (?,?,?)",
            ("classla-sl", "{}", adoption),
        )
    worker_identity = {
        **IDENTITY,
        "classla_version": metadata["classla_version"],
        "package": metadata["package"],
        "resources": {"ner.model_path": metadata["model_sha256"]},
        "processor_config": {"ner": historical["ner_config"]},
    }
    reference = {
        "metadata": metadata,
        "identity": worker_identity,
        "fingerprint": semantic_fingerprint(worker_identity),
    }
    evidence_path = tmp_path / "comparison.txt"
    evidence_path.write_text("fixture", encoding="utf-8")
    evidence_sha256 = __import__("hashlib").sha256(evidence_path.read_bytes()).hexdigest()
    comparison = {
        "format": "semora.ner-comparison.v1",
        "documents": 1,
        "mismatches": 0,
        "description": "fixture",
        "evidence_sha256": evidence_sha256,
    }
    with worker(worker_identity, runtime={"device": "cuda"}) as (url, _):
        execution = config(tmp_path, url)
        qualify_ner_continuation(
            database,
            run_id="classla-sl",
            execution=execution,
            reference=reference,
            comparison=comparison,
            evidence_path=evidence_path,
        )
        result = index_batch_ner(database, "batch", execution=execution)
    assert result["added_articles"] == 1
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT config_json FROM ner_runs").fetchone()[0] == old_config
        assert connection.execute("SELECT model_identity_json FROM pipeline_ner_adoptions").fetchone()[0] == adoption
