"""Strict input contracts independent of filesystem location and database state."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_FORMAT = "semora.batch.v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_BATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative POSIX path")
    parts = value.split("/")
    if any(
        part in {"", ".", ".."}
        or part.endswith((" ", "."))
        or part.split(".", 1)[0].casefold() in _RESERVED
        or any(ord(char) < 32 or char in '\\:*?"<>|' for char in part)
        for part in parts
    ):
        raise ValueError(f"{field} must be a canonical portable relative POSIX path: {value!r}")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class BatchIssue:
    """An issue's canonical identity path and independently located input files."""

    relative_path: str
    markdown_path: str
    metadata_path: str
    markdown_sha256: str
    metadata_sha256: str
    source_identifier: str | None = None

    def __post_init__(self) -> None:
        for field in ("relative_path", "markdown_path", "metadata_path"):
            _relative_path(getattr(self, field), field)
        for field in ("markdown_sha256", "metadata_sha256"):
            _hash(getattr(self, field), field)
        if not self.relative_path.endswith(".md") or not self.markdown_path.endswith(".md"):
            raise ValueError("relative_path and markdown_path must end in .md")
        if not self.metadata_path.endswith(".json"):
            raise ValueError("metadata_path must end in .json")
        if self.source_identifier is not None and (
            not isinstance(self.source_identifier, str)
            or not self.source_identifier.strip()
            or self.source_identifier != self.source_identifier.strip()
            or any(ord(char) < 32 for char in self.source_identifier)
        ):
            raise ValueError("source_identifier must be a nonempty trimmed string or null")


@dataclass(frozen=True)
class BatchManifest:
    batch_id: str
    issues: tuple[BatchIssue, ...]
    format: str = MANIFEST_FORMAT

    def __post_init__(self) -> None:
        if self.format != MANIFEST_FORMAT:
            raise ValueError(f"Unsupported batch manifest format: {self.format!r}")
        if not isinstance(self.batch_id, str) or _BATCH_ID.fullmatch(self.batch_id) is None:
            raise ValueError(
                "batch_id must be 1-128 ASCII letters, digits, dots, hyphens or underscores, "
                "starting with a letter or digit"
            )
        if not isinstance(self.issues, tuple) or not self.issues:
            raise ValueError("issues must be a nonempty tuple of BatchIssue records")
        paths: set[str] = set()
        identifiers: set[str] = set()
        for issue in self.issues:
            if not isinstance(issue, BatchIssue):
                raise ValueError("issues must contain only BatchIssue records")
            # Windows and Linux must interpret the same manifest without collisions.
            path_key = issue.relative_path.casefold()
            if path_key in paths:
                raise ValueError(f"Duplicate canonical issue path: {issue.relative_path}")
            paths.add(path_key)
            if issue.source_identifier is not None:
                identifier_key = issue.source_identifier.casefold()
                if identifier_key in identifiers:
                    raise ValueError(f"Duplicate source_identifier: {issue.source_identifier}")
                identifiers.add(identifier_key)

    @property
    def fingerprint(self) -> str:
        """Hash the contract, ignoring issue ordering and the manifest's location."""
        document = {
            "format": self.format,
            "batch_id": self.batch_id,
            "issues": [asdict(issue) for issue in sorted(self.issues, key=lambda issue: issue.relative_path)],
        }
        encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _keys(value: object, required: set[str], optional: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"Invalid {name} fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def load_manifest(path: str | Path) -> BatchManifest:
    """Load a strict v1 JSON manifest; this never opens or changes a database."""
    document = json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    document = _keys(document, {"format", "batch_id", "issues"}, set(), "manifest")
    if not isinstance(document["issues"], list):
        raise ValueError("issues must be a JSON array")
    required = {"relative_path", "markdown_path", "metadata_path", "markdown_sha256", "metadata_sha256"}
    issues = tuple(
        BatchIssue(**_keys(issue, required, {"source_identifier"}, "issue"))
        for issue in document["issues"]
    )
    return BatchManifest(batch_id=document["batch_id"], issues=issues, format=document["format"])


def _read_verified(root: Path, relative: str, expected_hash: str) -> bytes:
    candidate = root.joinpath(*PurePosixPath(relative).parts).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError(f"Input path escapes the input root: {relative}")
    if not candidate.is_file():
        raise ValueError(f"Input path is not a file: {relative}")
    content = candidate.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return content


def read_issue(input_root: str | Path, issue: BatchIssue) -> tuple[str, dict[str, Any]]:
    """Verify both exact byte hashes and decode the issue's text and metadata."""
    root = Path(input_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Input root is not a directory: {input_root}")
    markdown = _read_verified(root, issue.markdown_path, issue.markdown_sha256).decode("utf-8")
    if not markdown.strip():
        raise ValueError(f"Issue Markdown is empty: {issue.markdown_path}")
    metadata = json.loads(
        _read_verified(root, issue.metadata_path, issue.metadata_sha256).decode("utf-8-sig"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"Issue metadata must be a JSON object: {issue.metadata_path}")
    return markdown, metadata
