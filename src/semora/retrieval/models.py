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

    def as_dict(self) -> dict:
        return asdict(self)
