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
    position: str
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
    position_start: str | None
    position_end: str | None
    text: str
    truncated: bool
    has_before: bool = False
    has_after: bool = False

    def as_dict(self) -> dict:
        return asdict(self)
