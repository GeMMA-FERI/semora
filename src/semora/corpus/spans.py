"""Parse newspaper Markdown while retaining offsets into the original file."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^[ \t]{0,3}#[ \t]*(.*?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class SourceArticle:
    title: str | None
    content: str
    article_index: int
    char_start: int
    char_end: int
    content_char_start: int
    line_start: int
    line_end: int


class LineMap:
    """Translate zero-based character offsets to one-based source line numbers."""

    def __init__(self, text: str) -> None:
        self.starts = [0]
        self.starts.extend(match.end() for match in re.finditer("\n", text))

    def line_at(self, char_offset: int) -> int:
        return bisect.bisect_right(self.starts, max(0, char_offset))

    def span(self, char_start: int, char_end: int) -> tuple[int, int]:
        last_character = max(char_start, char_end - 1)
        return self.line_at(char_start), self.line_at(last_character)


def split_articles(markdown: str) -> list[SourceArticle]:
    """Split level-one Markdown articles and retain original-file spans."""
    headings = [match for match in HEADING_RE.finditer(markdown) if match.group(1).strip()]
    line_map = LineMap(markdown)
    if not headings:
        start, end = _trimmed_bounds(markdown, 0, len(markdown))
        if start == end:
            return []
        line_start, line_end = line_map.span(start, end)
        return [
            SourceArticle(
                title=None,
                content=markdown[start:end],
                article_index=0,
                char_start=start,
                char_end=end,
                content_char_start=start,
                line_start=line_start,
                line_end=line_end,
            )
        ]

    articles: list[SourceArticle] = []
    for index, heading in enumerate(headings):
        region_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        content_start, content_end = _trimmed_bounds(markdown, heading.end(), region_end)
        if content_start == content_end:
            continue
        article_start = heading.start()
        line_start, line_end = line_map.span(article_start, content_end)
        articles.append(
            SourceArticle(
                title=heading.group(1).strip(),
                content=markdown[content_start:content_end],
                article_index=len(articles),
                char_start=article_start,
                char_end=content_end,
                content_char_start=content_start,
                line_start=line_start,
                line_end=line_end,
            )
        )
    return articles


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
