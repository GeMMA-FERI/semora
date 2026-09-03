"""Normalization helpers for source-document metadata."""

from __future__ import annotations

import re


def normalize_newspaper_metadata(metadata: dict | None) -> dict[str, str | None]:
    record = metadata.get("Record") if isinstance(metadata, dict) else None
    if not isinstance(record, dict):
        record = {}
    return {
        "date": _normalize_date(_first_text(record.get("date"))),
        "publisher": _first_text(record.get("publisher")),
        "source": _first_text(record.get("source")),
        "rights": _first_text(record.get("rights")),
        "title": _first_text(record.get("title")),
        "urn": _urn(record.get("identifier")),
        "issue": _typed_text(record.get("format"), "issue"),
        "volume": _typed_text(record.get("format"), "volume"),
    }


def _first_text(value: object) -> str | None:
    if isinstance(value, list):
        return _first_text(value[0]) if value else None
    if isinstance(value, dict):
        return _first_text(value.get("#text"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _typed_text(value: object, expected_type: str) -> str | None:
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("@format_type", "")).casefold() == expected_type.casefold():
            return _first_text(entry.get("#text"))
    return None


def _urn(value: object) -> str | None:
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("@identifier_type", "")).casefold() == "urn":
            return _first_text(entry.get("#text"))
    for entry in entries:
        text = _first_text(entry)
        if text is not None and text.casefold().startswith("urn:"):
            return text
    return None


def _normalize_date(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d{4})(?:\s+(\d{2})(?:\s+(\d{2}))?)?", value)
    return "-".join(part for part in match.groups() if part is not None) if match else value
