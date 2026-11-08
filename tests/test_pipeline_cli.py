from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from semora.cli.main import main
from semora.corpus import indexer
from semora.pipeline import lexical


def prepare_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "incoming"
    source.mkdir()
    markdown = b"# Example\nThe new newspaper issue.\n"
    metadata = json.dumps({"Record": {"source": "Straža", "date": "1925 02 18"}}).encode("utf-8")
    (source / "issue.md").write_bytes(markdown)
    (source / "issue_metadata.json").write_bytes(metadata)
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "semora.batch.v1",
                "batch_id": "cli-batch",
                "issues": [
                    {
                        "relative_path": "Straza/issue.md",
                        "markdown_path": "issue.md",
                        "metadata_path": "issue_metadata.json",
                        "markdown_sha256": hashlib.sha256(markdown).hexdigest(),
                        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, source


def invoke(monkeypatch, capsys, *arguments: str) -> dict:
    monkeypatch.setattr(sys, "argv", ["semora", *arguments])
    main()
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("operation,module,function,extra", [
    ("baseline", "baseline", "adopt_baseline", []),
    ("semantic", "semantic_adoption", "adopt_semantic_baseline", []),
    ("ner", "ner_adoption", "adopt_ner_baseline", []),
    ("entities", "entity_adoption", "adopt_entity_baseline", []),
])
def test_cli_adoption_passes_explicit_audit_report(
    tmp_path: Path, monkeypatch, capsys, operation: str, module: str, function: str, extra: list[str]
) -> None:
    report_path = tmp_path / "audit.json"
    report_path.write_text('{"eligible": true}', encoding="utf-8")
    received = []

    def adopt(*args, **kwargs):
        received.append(kwargs["audit_report"])
        return {"adoption": {"already_adopted": False}}

    monkeypatch.setattr(f"semora.pipeline.{module}.{function}", adopt)
    invoke(
        monkeypatch, capsys, "pipeline", operation, "adopt", "--database", str(tmp_path / "archive.sqlite"),
        "--audit-report", str(report_path), "--maintenance", *extra,
    )
    assert received == [{"eligible": True}]


def test_cli_plan_run_status_respects_explicit_paths_without_models(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "external-data" / "archive.sqlite"
    unused_root = tmp_path / "not-the-data-root"

    def unexpected_model_load(_model_id):
        raise AssertionError("newspapers-only ingestion must not load models")

    monkeypatch.setattr(indexer, "_load_tokenizer", unexpected_model_load)
    common = ["--manifest", str(manifest), "--input-root", str(source), "--database", str(database)]
    plan = invoke(monkeypatch, capsys, "--root", str(unused_root), "pipeline", "plan", *common)
    assert plan["batch_id"] == "cli-batch"
    assert plan["counts"]["new"] == 1
    assert not database.parent.exists()

    result = invoke(
        monkeypatch,
        capsys,
        "--root",
        str(unused_root),
        "pipeline",
        "run",
        *common,
        "--through",
        "newspapers",
        "--maintenance",
    )
    assert result["issues"] == 1
    assert result["stages"] == {"articles": 0, "chunks": 0}
    assert result["retrieval_ready"] is False
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT relative_path,urn FROM newspapers").fetchone() == ("Straza/issue.md", None)
        assert connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 0

    status = invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "status",
        "--database",
        str(database),
        "--batch",
        "cli-batch",
    )
    assert status["batch_id"] == "cli-batch"
    assert status["manifest_fingerprint"] == plan["manifest_fingerprint"]
    assert status["state"] == "ingested"
    assert status["retrieval_ready"] is False
    assert not unused_root.exists()


def test_cli_run_requires_maintenance_before_creating_database(tmp_path: Path, monkeypatch) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "not-created" / "semora.sqlite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semora",
            "pipeline",
            "run",
            "--manifest",
            str(manifest),
            "--input-root",
            str(source),
            "--database",
            str(database),
            "--through",
            "newspapers",
        ],
    )

    with pytest.raises(ValueError, match="--maintenance"):
        main()

    assert not database.parent.exists()


def test_cli_default_database_is_under_data_root(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, source = prepare_manifest(tmp_path)
    root = tmp_path / "data"
    result = invoke(
        monkeypatch,
        capsys,
        "--root",
        str(root),
        "pipeline",
        "run",
        "--manifest",
        str(manifest),
        "--input-root",
        str(source),
        "--through",
        "newspapers",
        "--maintenance",
    )

    assert result["issues"] == 1
    assert (root / "indexes" / "semora.sqlite").is_file()
    assert not (root / "corpus").exists()


def test_cli_report_and_input_paths_resolve_against_cwd_not_root(tmp_path: Path, monkeypatch, capsys) -> None:
    prepare_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    receipt = invoke(
        monkeypatch,
        capsys,
        "--root",
        "unused-root",
        "pipeline",
        "plan",
        "--manifest",
        "batch.json",
        "--input-root",
        "incoming",
        "--database",
        "data/archive.sqlite",
        "--output",
        "reports/plan.json",
    )

    output = tmp_path / "reports" / "plan.json"
    assert receipt == {"report_path": str(output)}
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["batch_id"] == "cli-batch"
    assert report["counts"]["new"] == 1
    assert report["issues"][0]["relative_path"] == "Straza/issue.md"
    assert list(output.parent.iterdir()) == [output]
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "unused-root").exists()


def test_cli_run_and_status_reports_use_receipts_and_separate_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "data" / "semora.sqlite"
    output = tmp_path / "reports" / "ingested.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semora",
            "pipeline",
            "run",
            "--manifest",
            str(manifest),
            "--input-root",
            str(source),
            "--database",
            str(database),
            "--through",
            "newspapers",
            "--maintenance",
            "--output",
            str(output),
        ],
    )

    main()

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"report_path": str(output)}
    progress = [json.loads(line) for line in captured.err.splitlines()]
    assert any(event["event"] == "issue_ingested" for event in progress)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["issues"] == 1
    assert report["state"] == "ingested"
    status_file = tmp_path / "reports" / "status.json"
    receipt = invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "status",
        "--database",
        str(database),
        "--batch",
        "cli-batch",
        "--output",
        str(status_file),
    )
    assert receipt == {"report_path": str(status_file)}
    assert json.loads(status_file.read_text(encoding="utf-8"))["manifest_fingerprint"] == report["manifest_fingerprint"]


def test_lexical_progress_file_tracks_committed_checkpoints(tmp_path: Path, monkeypatch, capsys) -> None:
    progress_file = tmp_path / "tasks" / "newspaper-lemma.json"

    def fake_index(database, batch, *, progress, **options):
        progress({"event": "lexical_checkpoint", "batch_id": batch, "stage": "lemma",
                  "completed_articles": 7, "total": 10})
        snapshot = json.loads(progress_file.read_text(encoding="utf-8"))
        assert snapshot["completed"] == 7
        assert snapshot["total"] == 10
        assert snapshot["state"] == "running"
        return {"completed_articles": 10, "eligible_articles": 10, "complete": True}

    monkeypatch.setattr(lexical, "index_batch_lexical", fake_index)
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "index", "--database", str(tmp_path / "semora.sqlite"),
        "--batch", "newspapers", "--stage", "lemma", "--maintenance",
        "--progress-file", str(progress_file),
    ])
    main()
    snapshot = json.loads(progress_file.read_text(encoding="utf-8"))
    assert snapshot == {
        "format": "pebble.progress.v1", "title": "Newspaper lemma indexing",
        "state": "completed", "completed": 10, "total": 10,
        "unit": "articles", "message": "",
    }


def test_semantic_progress_file_tracks_committed_shards(tmp_path: Path, monkeypatch, capsys) -> None:
    from semora.pipeline import semantic

    progress_file = tmp_path / "tasks" / "semantic.json"

    def fake_index(database, batch, semantic_dir, *, progress, **options):
        progress({"event": "semantic_start", "batch_id": batch, "completed_chunks": 3, "total": 10})
        assert json.loads(progress_file.read_text(encoding="utf-8"))["completed"] == 3
        progress({"event": "semantic_checkpoint", "batch_id": batch, "completed_chunks": 7, "total": 10})
        assert json.loads(progress_file.read_text(encoding="utf-8"))["completed"] == 7
        return {"completed_chunks": 10, "eligible_chunks": 10, "complete": True}

    monkeypatch.setattr(semantic, "index_batch_vectors", fake_index)
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "semantic", "vectors",
        "--database", str(tmp_path / "semora.sqlite"),
        "--batch", "newspapers", "--maintenance",
        "--progress-file", str(progress_file),
    ])
    main()
    assert json.loads(progress_file.read_text(encoding="utf-8")) == {
        "format": "pebble.progress.v1", "title": "Newspaper semantic vectors",
        "state": "completed", "completed": 10, "total": 10,
        "unit": "chunks", "message": "",
    }


def test_faiss_progress_file_tracks_committed_checkpoints(tmp_path: Path, monkeypatch) -> None:
    from semora.pipeline import semantic_faiss

    progress_file = tmp_path / "tasks" / "faiss.json"

    def fake_build(database, semantic_dir, *, progress, **options):
        progress({"event": "faiss_generation_start", "indexed_vectors": 4, "total": 10})
        assert json.loads(progress_file.read_text(encoding="utf-8"))["completed"] == 4
        progress({"event": "faiss_generation_checkpoint", "indexed_vectors": 8, "total": 10})
        assert json.loads(progress_file.read_text(encoding="utf-8"))["completed"] == 8
        return {"chunks": 10, "index_complete": True}

    monkeypatch.setattr(semantic_faiss, "build_semantic_generation", fake_build)
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "semantic", "faiss",
        "--database", str(tmp_path / "semora.sqlite"), "--maintenance",
        "--progress-file", str(progress_file),
    ])
    main()
    assert json.loads(progress_file.read_text(encoding="utf-8")) == {
        "format": "pebble.progress.v1", "title": "Newspaper FAISS indexing",
        "state": "completed", "completed": 10, "total": 10,
        "unit": "vectors", "message": "",
    }


@pytest.mark.parametrize("output_kind", ["existing", "database"])
def test_cli_invalid_output_prevents_run_mutation(tmp_path: Path, monkeypatch, output_kind: str) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "data" / "semora.sqlite"
    output = tmp_path / "existing-report.json" if output_kind == "existing" else database
    if output_kind == "existing":
        output.write_text("keep this report", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semora",
            "pipeline",
            "run",
            "--manifest",
            str(manifest),
            "--input-root",
            str(source),
            "--database",
            str(database),
            "--output",
            str(output),
            "--through",
            "newspapers",
            "--maintenance",
        ],
    )

    with pytest.raises(FileExistsError if output_kind == "existing" else ValueError):
        main()

    assert not database.parent.exists()
    if output_kind == "existing":
        assert output.read_text(encoding="utf-8") == "keep this report"


def test_cli_failed_baseline_audit_returns_report_and_nonzero_without_mutation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "legacy" / "semora.sqlite"
    indexer.ingest_corpus(source, database, stages=("newspapers", "articles"))
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE articles SET content='Corrupt stored article'")
        connection.commit()
    before = database.read_bytes()
    output = tmp_path / "reports" / "invalid-baseline.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semora",
            "pipeline",
            "baseline",
            "audit",
            "--database",
            str(database),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {"report_path": str(output)}
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["eligible"] is False
    assert report["corpus"]["error_count"] > 0
    assert database.read_bytes() == before
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_baselines").fetchone()[0] == 0


def test_cli_baseline_adopt_requires_maintenance_before_database_access(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "missing" / "archive.sqlite"
    monkeypatch.setattr(sys, "argv", ["semora", "pipeline", "baseline", "adopt", "--database", str(database)])

    with pytest.raises(ValueError, match="--maintenance"):
        main()

    assert not database.parent.exists()


def test_cli_semantic_audit_uses_explicit_artifact_root_and_new_report(tmp_path: Path, monkeypatch, capsys) -> None:
    from semora.pipeline import semantic_baseline

    database = tmp_path / "corpus" / "archive.sqlite"
    semantic_dir = tmp_path / "artifacts" / "semantic"
    output = tmp_path / "reports" / "audit.json"
    seen = []

    def audit(source, root, *, progress):
        seen.append((source, root))
        progress({"event": "audit_check"})
        return {"eligible": True, "format": "semora.semantic-baseline.v1"}

    monkeypatch.setattr(semantic_baseline, "audit_semantic_baseline", audit)
    receipt = invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "semantic",
        "audit",
        "--database",
        str(database),
        "--semantic-dir",
        str(semantic_dir),
        "--output",
        str(output),
    )
    assert receipt == {"report_path": str(output)}
    assert seen == [(database, semantic_dir)]
    assert json.loads(output.read_text(encoding="utf-8"))["eligible"]
    assert not database.parent.exists() and not semantic_dir.exists()


def test_cli_semantic_vectors_requires_maintenance_before_loading_model(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "data" / "archive.sqlite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semora",
            "pipeline",
            "semantic",
            "vectors",
            "--database",
            str(database),
            "--batch",
            "batch-a",
        ],
    )

    with pytest.raises(ValueError, match="--maintenance"):
        main()

    assert not database.parent.exists()


def test_cli_ner_audit_uses_explicit_database_and_report(tmp_path: Path, monkeypatch, capsys) -> None:
    from semora.pipeline import ner_baseline

    database = tmp_path / "archive.sqlite"
    output = tmp_path / "reports" / "ner-audit.json"
    seen = []

    def audit(path, *, run_id, progress):
        seen.append((path, run_id))
        progress({"event": "ner_audit_check"})
        return {"format": "semora.ner-baseline.v1", "eligible": True}

    monkeypatch.setattr(ner_baseline, "audit_ner_baseline", audit)
    receipt = invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "ner",
        "audit",
        "--database",
        str(database),
        "--run-id",
        "classla-sl",
        "--output",
        str(output),
    )
    assert receipt == {"report_path": str(output)}
    assert seen == [(database, "classla-sl")]
    assert json.loads(output.read_text(encoding="utf-8"))["eligible"]
    assert not database.exists()


def test_cli_ner_index_requires_maintenance_before_model_load(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "archive.sqlite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semora",
            "pipeline",
            "ner",
            "index",
            "--database",
            str(database),
            "--batch",
            "batch-a",
        ],
    )

    with pytest.raises(ValueError, match="--maintenance"):
        main()

    assert not database.exists()


def test_cli_entity_audit_resolves_explicit_artifact_and_spellfix_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    from semora.pipeline import entity_baseline

    database = tmp_path / "corpus" / "archive.sqlite"
    entity_dir = tmp_path / "artifacts" / "entities"
    extension = tmp_path / "extensions" / "spellfix1.dll"
    seen = []

    def audit(path, root, *, run_id, spellfix_path, progress):
        seen.append((path, root, run_id, spellfix_path))
        return {"format": "semora.entity-baseline.v1", "eligible": True, "complete": True}

    monkeypatch.setattr(entity_baseline, "audit_entity_baseline", audit)
    result = invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "entities",
        "audit",
        "--database",
        str(database),
        "--entity-dir",
        str(entity_dir),
        "--spellfix-path",
        str(extension),
        "--run-id",
        "classla-sl",
    )
    assert result["eligible"]
    assert seen == [(database, entity_dir, "classla-sl", extension)]


def test_cli_entity_index_requires_maintenance_before_dictionary_access(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "archive.sqlite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semora",
            "pipeline",
            "entities",
            "index",
            "--database",
            str(database),
            "--batch",
            "batch-a",
        ],
    )

    with pytest.raises(ValueError, match="--maintenance"):
        main()

    assert not database.exists()


def test_cli_surface_and_lemma_index_batch_without_external_models(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "data" / "archive.sqlite"
    invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "run",
        "--manifest",
        str(manifest),
        "--input-root",
        str(source),
        "--database",
        str(database),
        "--through",
        "articles",
        "--maintenance",
    )

    def unexpected_model(**_kwargs):
        raise AssertionError("surface indexing must not load CLASSLA")

    monkeypatch.setattr(lexical, "ClasslaLemmatizer", unexpected_model)
    common = ["--database", str(database), "--batch", "cli-batch", "--maintenance", "--batch-size", "1"]
    surface = invoke(monkeypatch, capsys, "pipeline", "index", *common, "--stage", "surface")
    assert surface["complete"]
    assert surface["added_articles"] == 1
    assert surface["retrieval_ready"] is False
    created = []

    class FakeLemmatizer:
        def __init__(self, **options):
            print("Backend initialization diagnostic")
            self.options = options
            self.calls = 0
            self.closed = False
            created.append(self)

        def lemmatize_many(self, texts):
            print("Backend inference diagnostic")
            self.calls += 1
            return [text.casefold() for text in texts]

        def close(self):
            self.closed = True

    monkeypatch.setattr(lexical, "ClasslaLemmatizer", FakeLemmatizer)
    lemma = invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "index",
        *common,
        "--stage",
        "lemma",
        "--classla-device",
        "cpu",
        "--classla-type",
        "default",
        "--classla-pos-batch-size",
        "4",
        "--classla-lemma-batch-size",
        "8",
        "--lemma-model-fingerprint",
        "qualified-test-model",
    )
    assert lemma["complete"]
    assert lemma["added_articles"] == 1
    assert lemma["retrieval_ready"] is False
    assert len(created) == 1 and created[0].calls == 1 and created[0].closed
    assert created[0].options["device"] == "cpu"
    assert created[0].options["pos_batch_size"] == 4
    assert created[0].options["lemma_batch_size"] == 8
    repeated = invoke(
        monkeypatch,
        capsys,
        "pipeline",
        "index",
        *common,
        "--stage",
        "lemma",
        "--lemma-model-fingerprint",
        "qualified-test-model",
    )
    assert repeated["added_articles"] == 0
    assert repeated["skipped_articles"] == 1
    assert len(created) == 1


def test_progress_replace_recovers_from_transient_windows_reader_lock(tmp_path, monkeypatch):
    from semora.pipeline import cli

    path = tmp_path / "progress.json"
    path.write_text('{"completed": 1}', encoding="utf-8")
    replace = cli.os.replace
    attempts = []

    def busy_twice(source, target):
        attempts.append(1)
        if len(attempts) <= 2:
            assert json.loads(path.read_text())["completed"] == 1
            raise PermissionError("reader still holds the destination")
        return replace(source, target)

    monkeypatch.setattr(cli.os, "replace", busy_twice)
    monkeypatch.setattr(cli.time, "sleep", lambda _delay: None)
    cli._write_progress_file(path, {"completed": 2})
    assert json.loads(path.read_text()) == {"completed": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_progress_replace_permanent_lock_keeps_previous_snapshot(tmp_path, monkeypatch):
    from semora.pipeline import cli

    path = tmp_path / "progress.json"
    path.write_text('{"completed": 1}', encoding="utf-8")
    times = iter([0.0, 4.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))

    def locked(_source, _target):
        raise PermissionError("permanent lock")

    monkeypatch.setattr(cli.os, "replace", locked)
    with pytest.raises(PermissionError, match="permanent lock"):
        cli._write_progress_file(path, {"completed": 2})
    assert json.loads(path.read_text()) == {"completed": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_ner_cli_forwards_coordinator_write_batch(monkeypatch, capsys, tmp_path):
    from semora.pipeline import ner
    received = []
    def index(*args, **kwargs):
        received.append(kwargs)
        return {"complete": True}
    monkeypatch.setattr(ner, "index_batch_ner", index)
    invoke(monkeypatch, capsys, "pipeline", "ner", "index", "--database",
           str(tmp_path / "corpus.sqlite"), "--batch", "test", "--maintenance",
           "--write-batch-articles", "512")
    assert received[0]["write_batch_articles"] == 512


def test_publication_cli_forwards_trusted_passed_stages(monkeypatch, capsys, tmp_path):
    from semora.pipeline import publication_publish
    received = []
    def publish(*args, **kwargs):
        received.append(kwargs)
        return {"eligible": True, "published": True}
    monkeypatch.setattr(publication_publish, "publish_batch", publish)
    receipt = str(tmp_path / "previous.json")
    invoke(monkeypatch, capsys, "pipeline", "publication", "--database",
           str(tmp_path / "corpus.sqlite"), "--batch", "test", "--maintenance",
           "--semantic-generation", "a" * 64, "--publish", "--trust-passed-stages-from", receipt)
    assert received[0]["trust_passed_stages_from"] == receipt
