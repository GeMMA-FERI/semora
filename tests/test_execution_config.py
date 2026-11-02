from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from semora.cli.main import main
from semora.execution import ExecutionConfig
from semora.pipeline.lexical import index_batch_lexical
from semora.pipeline.ner import index_batch_ner


@pytest.mark.parametrize("url", [
    "", "worker:8000", "ftp://worker", "http://", "http://worker:abc",
    "http://worker:0", "http://worker/path", "http://user:secret@worker",
    "http://worker?x=1", "http://worker#fragment", "http://worker /",
    "\x00http://worker", "http://worker\x7f",
])
def test_invalid_worker_urls(url: str) -> None:
    with pytest.raises(ValueError):
        ExecutionConfig("remote", (url,))


def test_execution_config_constraints() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ExecutionConfig("remote")
    with pytest.raises(ValueError, match="remote"):
        ExecutionConfig("local", ("http://worker",))
    with pytest.raises(ValueError, match="unique"):
        ExecutionConfig("remote", ("http://worker", "http://worker/"))
    with pytest.raises(ValueError, match="unique"):
        ExecutionConfig("remote", ("http://worker", "http://WORKER:80/"))
    with pytest.raises(ValueError, match="tuple"):
        ExecutionConfig("remote", ["http://worker"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend"):
        ExecutionConfig("other")  # type: ignore[arg-type]


def test_task_validation_and_conflicts() -> None:
    local = ExecutionConfig()
    local.validate_for("ner")
    local.validate_for("lemma", injected=True, local_workers=2)
    with pytest.raises(ValueError, match="task"):
        local.validate_for("surface")  # type: ignore[arg-type]
    remote = ExecutionConfig("remote", ("https://worker:8443",))
    for task in ("ner", "lemma"):
        with pytest.raises(ValueError if task == "ner" else NotImplementedError):
            remote.validate_for(task)
        with pytest.raises(ValueError, match="injected"):
            remote.validate_for(task, injected=True)
        with pytest.raises(ValueError, match="worker processes"):
            remote.validate_for(task, local_workers=2)


@pytest.mark.parametrize("task", ["ner", "lemma", "surface"])
def test_remote_pipeline_fails_before_database_access(tmp_path: Path, monkeypatch, task: str) -> None:
    database = tmp_path / "missing" / "database.sqlite"
    remote = ExecutionConfig("remote", ("http://worker",))

    def unexpected(*args, **kwargs):
        raise AssertionError("model must not be loaded")

    monkeypatch.setattr("semora.pipeline.ner.load_adapter", unexpected)
    monkeypatch.setattr("semora.pipeline.lexical.ClasslaLemmatizer", unexpected)
    error = NotImplementedError if task == "lemma" else ValueError
    with pytest.raises(error):
        if task == "ner":
            index_batch_ner(database, "batch", execution=remote)
        else:
            index_batch_lexical(database, "batch", stage=task, execution=remote)
    assert not database.parent.exists()


def test_pipeline_conflicting_injection_fails_before_database(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite"
    remote = ExecutionConfig("remote", ("http://worker",))
    with pytest.raises(ValueError, match="injected"):
        index_batch_ner(database, "batch", model=object(), execution=remote)
    with pytest.raises(ValueError, match="injected"):
        index_batch_lexical(database, "batch", stage="lemma", lemmatizer=object(), execution=remote)
    with pytest.raises(ValueError, match="worker processes"):
        index_batch_lexical(database, "batch", stage="lemma", classla_workers=2, execution=remote)
    with pytest.raises(ValueError, match="custom lemmatizer"):
        index_batch_lexical(database, "batch", stage="lemma", classla_workers=2, lemmatizer=object())
    assert not database.exists()


@pytest.mark.parametrize("task", ["ner", "lemma", "surface"])
def test_cli_remote_fails_without_output(tmp_path: Path, monkeypatch, task: str) -> None:
    database = tmp_path / "missing" / "db.sqlite"
    report = tmp_path / "report.json"
    progress = tmp_path / "progress.json"
    command = ["ner", "index"] if task == "ner" else ["index", "--stage", task, "--progress-file", str(progress)]
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", *command, "--batch", "batch", "--database", str(database),
        "--maintenance", "--backend", "remote", "--worker-url", "http://worker", "--output", str(report),
    ])
    with pytest.raises(NotImplementedError if task == "lemma" else ValueError):
        main()
    assert not database.parent.exists()
    assert not report.exists()
    assert not progress.exists()


@pytest.mark.parametrize("field,value", [
    ("connect_timeout", 0), ("read_timeout", float("inf")), ("retry_delay", -1),
    ("max_attempts", True), ("max_attempts", 11), ("token_file", "token.txt"),
])
def test_remote_runtime_validation(field, value):
    with pytest.raises(ValueError):
        ExecutionConfig("remote", ("http://worker",), **{field: value})


def test_remote_ner_configuration_and_cli(tmp_path, monkeypatch, capsys):
    token = tmp_path / "token"
    config = ExecutionConfig("remote", ("http://a:8000", "https://b"), token_file=token)
    config.validate_for("ner")
    with pytest.raises(NotImplementedError):
        config.validate_for("lemma")
    with pytest.raises(ValueError, match="remote"):
        ExecutionConfig(token_file=token)
    calls = []
    monkeypatch.setattr("semora.pipeline.ner.index_batch_ner", lambda *a, **kw: calls.append(kw) or {"complete": True})
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "ner", "index", "--batch", "b", "--maintenance",
        "--backend", "remote", "--worker-url", "http://a:8000", "--worker-url", "https://b",
        "--worker-token-file", str(token), "--worker-read-timeout", "45", "--worker-attempts", "2",
    ])
    main()
    capsys.readouterr()
    forwarded = calls[0]["execution"]
    assert forwarded.worker_urls == config.worker_urls and forwarded.token_file == token
    assert forwarded.read_timeout == 45 and forwarded.max_attempts == 2


def test_ner_cli_failed_progress_preserves_durable_count(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"

    def failed(*args, progress, **kwargs):
        progress({"event": "ner_start", "completed_articles": 4, "total": 10})
        progress({"event": "ner_checkpoint", "completed_articles": 5, "total": 10})
        raise RuntimeError("worker lost")

    monkeypatch.setattr("semora.pipeline.ner.index_batch_ner", failed)
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "ner", "index", "--batch", "b", "--maintenance", "--progress-file", str(path),
    ])
    with pytest.raises(RuntimeError, match="worker lost"):
        main()
    import json

    snapshot = json.loads(path.read_text())
    assert (snapshot["state"], snapshot["completed"], snapshot["total"]) == ("failed", 5, 10)


def test_progress_cannot_overwrite_worker_token(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("x" * 32)
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", "ner", "index", "--batch", "b", "--maintenance",
        "--backend", "remote", "--worker-url", "http://worker", "--worker-token-file", str(token),
        "--progress-file", str(token),
    ])
    with pytest.raises(ValueError, match="conflicts"):
        main()
    assert token.read_text() == "x" * 32


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("task", ["ner", "lemma"])
def test_cli_local_forwards_existing_options(monkeypatch, capsys, task: str, explicit: bool) -> None:
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {"complete": True, "completed_articles": 0, "eligible_articles": 0}

    if task == "ner":
        monkeypatch.setattr("semora.pipeline.ner.index_batch_ner", fake)
        command = ["ner", "index", "--adapter", "transformers", "--batch-size", "7"]
    else:
        monkeypatch.setattr("semora.pipeline.lexical.index_batch_lexical", fake)
        command = ["index", "--stage", "lemma", "--classla-workers", "2", "--batch-size", "7"]
    monkeypatch.setattr(sys, "argv", [
        "semora", "pipeline", *command, "--batch", "batch", "--maintenance",
        *(["--backend", "local"] if explicit else []),
    ])
    main()
    capsys.readouterr()
    kwargs = calls[0][1]
    assert kwargs["execution"] == ExecutionConfig()
    assert kwargs["batch_size"] == 7
    assert kwargs["adapter" if task == "ner" else "classla_workers"] == ("transformers" if task == "ner" else 2)


def test_local_imports_do_not_require_optional_packages_or_network() -> None:
    script = """
import builtins
import socket
original = builtins.__import__
blocked = {'classla', 'torch', 'httpx', 'requests', 'fastapi', 'uvicorn'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('optional import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def no_network(*args, **kwargs):
    raise AssertionError('unexpected network activity')
socket.socket = no_network
from semora.execution import ExecutionConfig
from semora.execution.contracts import Document, Request, semantic_fingerprint
from semora.ner.base import Entity, Prediction
from semora.pipeline.ner import index_batch_ner
from semora.pipeline.lexical import index_batch_lexical
ExecutionConfig().validate_for('ner')
ExecutionConfig().validate_for('lemma')
document = Document.create('empty', '')
request = Request('req', 'ner', semantic_fingerprint({'format': 'test.v1'}), (document,))
assert Request.from_dict(request.to_dict()) == request
assert Prediction([[Entity(0, 1, 'PER', 'A')]]).entities[0][0].text == 'A'
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
