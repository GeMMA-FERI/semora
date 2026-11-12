"""Regex scope equivalence and bounded work on unrelated archive rows."""

from __future__ import annotations

import re
import sqlite3
from types import MethodType, SimpleNamespace

import pytest

from semora.retrieval.engine import SearchEngine

LEGACY_SQL = """
SELECT articles.*, articles.title AS article_title,
       newspapers.source, newspapers.title AS newspaper_title,
       newspapers.date, newspapers.urn, newspapers.newspaper_id AS document_id,
       newspapers.relative_path, newspapers.content AS newspaper_content
FROM articles JOIN newspapers ON newspapers.newspaper_id=articles.newspaper_id
WHERE articles.is_valid=1
  AND (? IS NULL OR newspapers.source=? OR newspapers.title=?)
  AND (? IS NULL OR newspapers.date>=?) AND (? IS NULL OR newspapers.date<=?)
ORDER BY articles.article_id
"""


@pytest.fixture
def engine(tmp_path):
    conn = sqlite3.connect(tmp_path / "regex.sqlite")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE newspapers(newspaper_id TEXT PRIMARY KEY,source TEXT,title TEXT,date TEXT,
            urn TEXT,relative_path TEXT,content TEXT NOT NULL);
        CREATE INDEX idx_newspapers_source_date ON newspapers(source,date);
        CREATE INDEX idx_newspapers_date ON newspapers(date);
        CREATE INDEX idx_newspapers_title ON newspapers(title);
        CREATE TABLE articles(article_id TEXT PRIMARY KEY,newspaper_id TEXT,title TEXT,content TEXT,
            is_valid INTEGER,char_start INTEGER,char_end INTEGER,line_start INTEGER,line_end INTEGER);
        CREATE INDEX idx_articles_newspaper_id ON articles(newspaper_id);
        CREATE INDEX idx_articles_valid_article_id ON articles(article_id) WHERE is_valid=1;
    """)
    conn.executemany(
        "INSERT INTO newspapers VALUES(?,?,?,?,?,?,?)",
        [
            ("source", "Jutro (Ljubljana)", "Jutro", "1925-12-23", "URN:NBN:SI:doc-SOURCE", "source.md", ""),
            ("title", "Alias", "Jutro (Ljubljana)", "1925-12-23", "URN:NBN:SI:doc-TITLE", "title.md", ""),
            ("both", "Jutro (Ljubljana)", "Jutro (Ljubljana)", "1925-12-24", None, "both.md", ""),
            ("null", "Jutro (Ljubljana)", "Undated", None, None, "null.md", ""),
            ("other", "Other", "Other", "1930-01-01", None, "other.md", "mati text"),
        ],
    )

    def add(aid, issue, text, valid=1):
        previous = conn.execute("SELECT content FROM newspapers WHERE newspaper_id=?", (issue,)).fetchone()[0]
        start = len(previous)
        first_line = previous.count("\n") + 1
        conn.execute("UPDATE newspapers SET content=? WHERE newspaper_id=?", (previous + text + "\n", issue))
        conn.execute(
            "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?)",
            (
                aid,
                issue,
                "Article " + aid,
                text,
                valid,
                start,
                start + len(text),
                first_line,
                first_line + text.count("\n"),
            ),
        )

    add("z-source", "source", "Before Mati and mati.\nNext line.")
    add("a-title", "title", "MATI appears here.")
    add("m-source", "source", "The mother: mati.")
    add("x-unicode", "source", "Čudna mati živi.")
    add("b-both", "both", "Mati.")
    add("c-null", "null", "mati.")
    add("invalid", "source", "mati.", 0)
    conn.commit()
    instance = object.__new__(SearchEngine)
    instance.database = SimpleNamespace(conn=conn)
    instance._active_profile = None
    instance.last_profile = None
    yield instance
    conn.close()


def legacy(engine, pattern, limit, *, ignore_case=False, newspaper=None, date_from=None, date_to=None):
    expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    rows = engine.database.conn.execute(
        LEGACY_SQL,
        (
            newspaper,
            newspaper,
            newspaper,
            date_from,
            date_from,
            date_to,
            date_to,
        ),
    )
    matches = []
    for article in rows:
        found = expression.search(str(article["content"]))
        if found:
            row = dict(article)
            row["_focus_char"] = int(row["char_end"]) - len(str(row["content"])) + found.start()
            matches.append((row, 1.0))
            if len(matches) == limit:
                break
    return matches


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"newspaper": "Jutro (Ljubljana)"},
        {"newspaper": "Jutro"},
        {"date_from": "1925-12-23", "date_to": "1925-12-23"},
        {"newspaper": "Jutro (Ljubljana)", "date_from": "1925-12-23", "date_to": "1925-12-24"},
        {"date_to": "1925-12-23"},
        {"date_from": "1925-12-24"},
        {"newspaper": "Missing"},
    ],
)
@pytest.mark.parametrize(
    "pattern,ignore_case,limit",
    [
        ("mati", False, 10),
        ("mati", True, 2),
        (r"M.ti|mother", True, 1),
        (r"never matches", False, 10),
        (r"\bčudna\s+MATI\b", True, 10),
    ],
)
def test_filtered_regex_preserves_complete_hit_contract(engine, filters, pattern, ignore_case, limit):
    options = {"ignore_case": ignore_case, "limit": limit, "context_lines": 1, **filters}
    actual = [hit.as_dict() for hit in engine.search("regex", pattern, **options)]
    current = engine._search_regex
    engine._search_regex = MethodType(legacy, engine)
    try:
        expected = [hit.as_dict() for hit in engine.search("regex", pattern, **options)]
    finally:
        engine._search_regex = current
    assert actual == expected


def test_scoped_regex_skips_archive_history_and_materializes_only_hits(engine):
    conn = engine.database.conn
    conn.executemany(
        "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?)",
        [(f"irrelevant-{index:05}", "other", "Other", "mati text", 1, 0, 9, 1, 1) for index in range(20_000)],
    )
    conn.commit()
    hydrated = []
    materialize = engine._materialize_match

    def record(row, cache=None):
        hydrated.append(row["article_id"])
        return materialize(row, cache)

    engine._materialize_match = record
    steps = [0]

    def tick():
        steps[0] += 100
        return 0

    conn.set_progress_handler(tick, 100)
    actual = engine.search(
        "regex",
        "mati",
        newspaper="Jutro (Ljubljana)",
        date_from="1925-12-23",
        date_to="1925-12-23",
        limit=1,
        profile=True,
    )
    scoped_steps = steps[0]
    steps[0] = 0
    expected = legacy(engine, "mati", 1, newspaper="Jutro (Ljubljana)", date_from="1925-12-23", date_to="1925-12-23")
    legacy_steps = steps[0]
    conn.set_progress_handler(None, 0)
    assert len(actual) == len(expected) == 1
    assert hydrated == ["m-source"]
    assert engine.last_profile["regex_articles_examined"] == 2
    plan = "\n".join(engine.last_profile["query_plans"]["regex_candidates"])
    assert "idx_articles_newspaper_id" in plan
    assert "idx_articles_valid_article_id" not in plan
    assert scoped_steps < 3000
    assert legacy_steps > scoped_steps * 20


def test_nonmatching_scope_never_reads_full_newspaper_text(engine):
    conn = engine.database.conn
    source_reads = []

    def authorize(action, table, column, *_):
        if action == sqlite3.SQLITE_READ and table == "newspapers" and column == "content":
            source_reads.append(column)
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    assert engine.search("regex", "unmatched", newspaper="Jutro (Ljubljana)", limit=10) == []
    conn.set_authorizer(None)
    assert source_reads == []


def test_empty_scope_never_reads_article_text(engine):
    conn = engine.database.conn
    article_reads = []

    def authorize(action, table, column, *_):
        if action == sqlite3.SQLITE_READ and table == "articles" and column == "content":
            article_reads.append(column)
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    assert engine.search("regex", "mati", newspaper="Missing", limit=10) == []
    conn.set_authorizer(None)
    assert article_reads == []


def test_regex_scope_overflow_keeps_all_issues_and_global_article_order(engine):
    conn = engine.database.conn
    conn.executemany(
        "INSERT INTO newspapers VALUES(?,?,?,?,?,?,?)",
        [(f"many-{i:03}", "Many", "Many", "1925-12-23", None, "many.md", "mati") for i in range(300)],
    )
    conn.executemany(
        "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?)",
        [(f"many-{299 - i:03}", f"many-{i:03}", "Many", "mati", 1, 0, 4, 1, 1) for i in range(300)],
    )
    conn.commit()
    result = engine._search_regex("mati", 300, ignore_case=False, newspaper="Many", date_from=None, date_to=None)
    assert [row["article_id"] for row, _ in result] == [f"many-{i:03}" for i in range(300)]
