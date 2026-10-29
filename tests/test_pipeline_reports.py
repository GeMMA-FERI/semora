from __future__ import annotations

import json
from pathlib import Path

import pytest

from semora.pipeline.reports import report_path, write_report


def test_report_uses_utf8_and_is_published_without_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "result.json"
    report = {"newspaper": "Straža", "count": 2}
    assert write_report(target, report) == target
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert list(target.parent.iterdir()) == [target]


def test_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text("keep this", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_report(target, {"new": True})
    assert target.read_text(encoding="utf-8") == "keep this"


def test_report_serialization_failure_publishes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_report(tmp_path / "report.json", {"bad": float("nan")})
    assert list(tmp_path.iterdir()) == []


def test_output_cannot_occupy_a_new_database_path(tmp_path: Path) -> None:
    target = tmp_path / "database.json"
    with pytest.raises(ValueError, match="conflicts"):
        report_path(target, protected=[target])
    assert not target.exists()


def test_relative_report_paths_resolve_against_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert report_path("result.json") == tmp_path / "result.json"
