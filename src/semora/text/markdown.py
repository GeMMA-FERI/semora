from __future__ import annotations

import re

MARKDOWN_IMAGE_RE = re.compile(
    r"""
    (?<!\\)!
    \[(?:\\.|[^\]\\])*\]
    \(
        [ \t]*
        (?:
            <[^>\r\n]*>
            |
            (?:\\.|[^()\r\n]|\([^()\r\n]*\))*
        )
        [ \t]*
    \)
    """,
    re.VERBOSE,
)
BLANK_LINE_WHITESPACE_RE = re.compile(r"(?m)^[ \t]+(?=\r?$)")
REPEATED_INLINE_WHITESPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")


def remove_markdown_images(text: str) -> str:
    """Remove inline Markdown images while preserving surrounding text and lines."""
    if not text:
        return text

    cleaned = MARKDOWN_IMAGE_RE.sub(" ", text)
    cleaned = BLANK_LINE_WHITESPACE_RE.sub("", cleaned)
    return REPEATED_INLINE_WHITESPACE_RE.sub(" ", cleaned)
