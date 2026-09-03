from __future__ import annotations

from semora.text import (
    NoopProcessor,
    ParagraphProcessor,
    SentenceWindowProcessor,
    build_chunk_id,
    parse_chunk_id,
)


def test_lightweight_chunkers() -> None:
    assert NoopProcessor().process("doc", "text") == [("0", "text")]
    assert ParagraphProcessor().process("doc", "one\n\ntwo") == [
        ("0", "one"),
        ("1", "two"),
    ]
    assert SentenceWindowProcessor(2, 1).process(
        "doc", "One. Two! Three?"
    ) == [("0", "One. Two!"), ("1", "Two! Three?")]


def test_chunk_id_round_trip() -> None:
    chunk_id = build_chunk_id("folder/article.md", "2")
    parsed = parse_chunk_id(chunk_id)
    assert parsed["path"] == "folder/article.md"
    assert parsed["article"] == "article"
    assert parsed["chunk"] == "2"
