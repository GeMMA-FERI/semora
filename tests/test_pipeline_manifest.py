from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from semora.pipeline.manifest import BatchIssue, BatchManifest, load_manifest, read_issue


def _input(root: Path, *, metadata: object = None, markdown: bytes | None = None) -> BatchIssue:
    root.mkdir(parents=True, exist_ok=True)
    text = markdown if markdown is not None else "# Straža\nA newspaper issue.\n".encode()
    data = json.dumps(metadata if metadata is not None else {
        "Record": {"source": "Straža", "identifier": ["GF-URN:abc"], "date": "1925 02 18"},
        "provenance": {"original_filename": "id-123_date-19250218.md"},
    }, ensure_ascii=False).encode()
    (root / "issue.md").write_bytes(text)
    (root / "issue_metadata.json").write_bytes(data)
    return BatchIssue(
        relative_path="Straza/GF-URN_abc/GF-URN_abc.md",
        markdown_path="issue.md",
        metadata_path="issue_metadata.json",
        markdown_sha256=hashlib.sha256(text).hexdigest(),
        metadata_sha256=hashlib.sha256(data).hexdigest(),
        source_identifier="GF-URN:abc",
    )


def _document(issue: BatchIssue) -> dict:
    return {
        "format": "semora.batch.v1", "batch_id": "import-2026-10-29",
        "issues": [dict(issue.__dict__)],
    }


def test_manifest_and_input_survive_relocation_without_dlib_urn(tmp_path: Path) -> None:
    original = tmp_path / "original"
    issue = _input(original)
    manifest_path = original / "batch.json"
    manifest_path.write_text(json.dumps(_document(issue)), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    before = read_issue(original, manifest.issues[0])
    relocated = tmp_path / "different" / "location"
    shutil.move(str(original), str(relocated))
    moved_manifest = load_manifest(relocated / "batch.json")
    assert moved_manifest.fingerprint == manifest.fingerprint
    assert read_issue(relocated, moved_manifest.issues[0]) == before
    assert before[1]["provenance"]["original_filename"] == "id-123_date-19250218.md"
    assert "URN:NBN:" not in json.dumps(before[1])


@pytest.mark.parametrize("path", [
    "../issue.md", "/issue.md", "C:/issue.md", "C:issue.md", "\\\\server\\issue.md",
    "folder\\issue.md", "folder//issue.md", "folder/./issue.md", "folder/../issue.md",
    "folder/issue.md/", "CON.md", "folder/NUL/file.md", "folder /issue.md",
    "folder./issue.md", "folder/issue?.md", "folder/issue\n.md",
])
def test_rejects_nonportable_and_noncanonical_paths(tmp_path: Path, path: str) -> None:
    issue = _input(tmp_path)
    for field in ("relative_path", "markdown_path", "metadata_path"):
        with pytest.raises(ValueError, match="path"):
            replace(issue, **{field: path})


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "G" * 64, "A" * 64, None, 123])
def test_rejects_malformed_hashes(tmp_path: Path, digest: object) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        replace(_input(tmp_path), markdown_sha256=digest)


def test_manifest_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    document = _document(_input(tmp_path))
    manifest_path = tmp_path / "batch.json"
    document["input_root"] = "C:/something"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*input_root"):
        load_manifest(manifest_path)
    del document["input_root"]
    del document["issues"][0]["metadata_sha256"]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*metadata_sha256"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("batch_id", ["", "../unsafe", "with spaces", "x" * 129, None])
def test_rejects_unsafe_batch_ids(tmp_path: Path, batch_id: object) -> None:
    with pytest.raises(ValueError, match="batch_id"):
        BatchManifest(batch_id, (_input(tmp_path),))


def test_rejects_duplicate_path_and_identifier(tmp_path: Path) -> None:
    issue = _input(tmp_path)
    with pytest.raises(ValueError, match="Duplicate canonical"):
        BatchManifest("batch", (issue, replace(issue, relative_path=issue.relative_path.upper().replace(".MD", ".md"))))
    with pytest.raises(ValueError, match="Duplicate source"):
        BatchManifest("batch", (issue, replace(issue, relative_path="Other/issue.md", source_identifier="gf-urn:ABC")))


def test_fingerprint_ignores_order_and_detects_contract_change(tmp_path: Path) -> None:
    first = _input(tmp_path)
    second = replace(first, relative_path="Other/issue.md", source_identifier=None)
    manifest = BatchManifest("batch", (first, second))
    assert manifest.fingerprint == BatchManifest("batch", (second, first)).fingerprint
    assert manifest.fingerprint != replace(manifest, batch_id="other").fingerprint
    changed = BatchManifest("batch", (replace(first, metadata_sha256="a" * 64), second))
    assert manifest.fingerprint != changed.fingerprint


def test_detects_changed_input_and_missing_sidecar(tmp_path: Path) -> None:
    issue = _input(tmp_path)
    (tmp_path / "issue.md").write_text("Changed text", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_issue(tmp_path, issue)
    issue = _input(tmp_path)
    (tmp_path / "issue_metadata.json").unlink()
    with pytest.raises(FileNotFoundError):
        read_issue(tmp_path, issue)


@pytest.mark.parametrize("markdown", [b" \n\t", b"\xff\xfe"])
def test_rejects_empty_or_non_utf8_markdown(tmp_path: Path, markdown: bytes) -> None:
    issue = _input(tmp_path, markdown=markdown)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        read_issue(tmp_path, issue)


@pytest.mark.parametrize("metadata", [[], "text", 42, True])
def test_rejects_nonobject_metadata(tmp_path: Path, metadata: object) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        read_issue(tmp_path, _input(tmp_path, metadata=metadata))


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text('{"format":"semora.batch.v1","format":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_manifest(path)


def test_rejects_symlink_to_outside_input_root(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    issue = _input(root)
    outside = tmp_path / "outside.md"
    outside.write_bytes((root / "issue.md").read_bytes())
    (root / "issue.md").unlink()
    try:
        (root / "issue.md").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    with pytest.raises(ValueError, match="escapes"):
        read_issue(root, issue)
