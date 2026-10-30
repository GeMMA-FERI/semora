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
    manifest.write_text(json.dumps({
        "format": "semora.batch.v1",
        "batch_id": "cli-batch",
        "issues": [{
            "relative_path": "Straza/issue.md",
            "markdown_path": "issue.md",
            "metadata_path": "issue_metadata.json",
            "markdown_sha256": hashlib.sha256(markdown).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        }],
    }), encoding="utf-8")
    return manifest, source


def invoke(monkeypatch, capsys, *arguments: str) -> dict:
    monkeypatch.setattr(sys, "argv", ["semora", *arguments])
    main()
    return json.loads(capsys.readouterr().out)


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
        monkeypatch, capsys, "--root", str(unused_root), "pipeline", "run", *common,
        "--through", "newspapers", "--maintenance",
    )
    assert result["issues"] == 1
    assert result["stages"] == {"articles": 0, "chunks": 0}
    assert result["retrieval_ready"] is False
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT relative_path,urn FROM newspapers").fetchone() == ("Straza/issue.md", None)
        assert connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 0

    status = invoke(
        monkeypatch, capsys, "pipeline", "status", "--database", str(database), "--batch", "cli-batch",
    )
    assert status["batch_id"] == "cli-batch"
    assert status["manifest_fingerprint"] == plan["manifest_fingerprint"]
    assert status["state"] == "ingested"
    assert status["retrieval_ready"] is False
    assert not unused_root.exists()


def test_cli_run_requires_maintenance_before_creating_database(tmp_path: Path, monkeypatch) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "not-created" / "semora.sqlite"
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "run", "--manifest", str(manifest), "--input-root", str(source),
        "--database", str(database), "--through", "newspapers",
    ])

    with pytest.raises(ValueError, match="--maintenance"):
        main()

    assert not database.parent.exists()


def test_cli_default_database_is_under_data_root(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, source = prepare_manifest(tmp_path)
    root = tmp_path / "data"
    result = invoke(
        monkeypatch, capsys, "--root", str(root), "pipeline", "run", "--manifest", str(manifest),
        "--input-root", str(source), "--through", "newspapers", "--maintenance",
    )

    assert result["issues"] == 1
    assert (root / "indexes" / "semora.sqlite").is_file()
    assert not (root / "corpus").exists()


def test_cli_report_and_input_paths_resolve_against_cwd_not_root(tmp_path: Path, monkeypatch, capsys) -> None:
    prepare_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    receipt = invoke(
        monkeypatch, capsys, "--root", "unused-root", "pipeline", "plan", "--manifest", "batch.json",
        "--input-root", "incoming", "--database", "data/archive.sqlite", "--output", "reports/plan.json",
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
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "run", "--manifest", str(manifest), "--input-root", str(source),
        "--database", str(database), "--through", "newspapers", "--maintenance", "--output", str(output),
    ])

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
        monkeypatch, capsys, "pipeline", "status", "--database", str(database), "--batch", "cli-batch",
        "--output", str(status_file),
    )
    assert receipt == {"report_path": str(status_file)}
    assert json.loads(status_file.read_text(encoding="utf-8"))["manifest_fingerprint"] == report["manifest_fingerprint"]


@pytest.mark.parametrize("output_kind", ["existing", "database"])
def test_cli_invalid_output_prevents_run_mutation(tmp_path: Path, monkeypatch, output_kind: str) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "data" / "semora.sqlite"
    output = tmp_path / "existing-report.json" if output_kind == "existing" else database
    if output_kind == "existing":
        output.write_text("keep this report", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "run", "--manifest", str(manifest), "--input-root", str(source),
        "--database", str(database), "--output", str(output), "--through", "newspapers", "--maintenance",
    ])

    with pytest.raises(FileExistsError if output_kind == "existing" else ValueError):
        main()

    assert not database.parent.exists()
    if output_kind == "existing":
        assert output.read_text(encoding="utf-8") == "keep this report"


def test_cli_failed_baseline_audit_returns_report_and_nonzero_without_mutation(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    _manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "legacy" / "semora.sqlite"
    indexer.ingest_corpus(source, database, stages=("newspapers", "articles"))
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE articles SET content='Corrupt stored article'")
        connection.commit()
    before = database.read_bytes()
    output = tmp_path / "reports" / "invalid-baseline.json"
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "baseline", "audit", "--database", str(database), "--output", str(output),
    ])

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
        monkeypatch, capsys, "pipeline", "semantic", "audit", "--database", str(database),
        "--semantic-dir", str(semantic_dir), "--output", str(output),
    )
    assert receipt == {"report_path": str(output)}
    assert seen == [(database, semantic_dir)]
    assert json.loads(output.read_text(encoding="utf-8"))["eligible"]
    assert not database.parent.exists() and not semantic_dir.exists()


def test_cli_semantic_vectors_requires_maintenance_before_loading_model(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "data" / "archive.sqlite"
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "semantic", "vectors", "--database", str(database), "--batch", "batch-a",
    ])

    with pytest.raises(ValueError, match="--maintenance"):
        main()

    assert not database.parent.exists()


def test_cli_surface_and_lemma_index_batch_without_external_models(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, source = prepare_manifest(tmp_path)
    database = tmp_path / "data" / "archive.sqlite"
    invoke(
        monkeypatch, capsys, "pipeline", "run", "--manifest", str(manifest), "--input-root", str(source),
        "--database", str(database), "--through", "articles", "--maintenance",
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
        monkeypatch, capsys, "pipeline", "index", *common, "--stage", "lemma", "--classla-device", "cpu",
        "--classla-type", "default", "--classla-pos-batch-size", "4", "--classla-lemma-batch-size", "8",
        "--lemma-model-fingerprint", "qualified-test-model",
    )
    assert lemma["complete"]
    assert lemma["added_articles"] == 1
    assert lemma["retrieval_ready"] is False
    assert len(created) == 1 and created[0].calls == 1 and created[0].closed
    assert created[0].options["device"] == "cpu"
    assert created[0].options["pos_batch_size"] == 4
    assert created[0].options["lemma_batch_size"] == 8
    repeated = invoke(
        monkeypatch, capsys, "pipeline", "index", *common, "--stage", "lemma",
        "--lemma-model-fingerprint", "qualified-test-model",
    )
    assert repeated["added_articles"] == 0
    assert repeated["skipped_articles"] == 1
    assert len(created) == 1
