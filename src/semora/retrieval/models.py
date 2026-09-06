"""Stable JSON-facing retrieval records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchHit:
    newspaper: str | None
    date: str | None
    document_id: str
    relative_path: str
    score: float
    line_start: int
    line_end: int
    snippet: str
    article_title: str | None = None
    urn: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SourceExcerpt:
    newspaper: str | None
    date: str | None
    document_id: str
    urn: str | None
    relative_path: str
    line_start: int
    line_end: int
    text: str
    truncated: bool

    def as_dict(self) -> dict:
        return asdict(self)
