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
