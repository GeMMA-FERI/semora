"""Qualify scoped FTS against legacy ranking on disposable corpus data."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

import semora.retrieval.engine as engine_module
from semora.retrieval.engine import SearchEngine


@pytest.fixture
def corpus(tmp_path):
    connection = sqlite3.connect(tmp_path / "corpus.sqlite")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE newspapers(newspaper_id TEXT PRIMARY KEY,source TEXT,title TEXT,date TEXT,
            urn TEXT,relative_path TEXT,content TEXT);
        CREATE INDEX idx_newspapers_source_date ON newspapers(source,date);
        CREATE INDEX idx_newspapers_date ON newspapers(date);
        CREATE INDEX idx_newspapers_title ON newspapers(title);
        CREATE TABLE articles(article_id TEXT PRIMARY KEY,newspaper_id TEXT,title TEXT,content TEXT,
            is_valid INTEGER,char_start INTEGER,char_end INTEGER,line_start INTEGER,line_end INTEGER);
        CREATE INDEX idx_articles_newspaper_id ON articles(newspaper_id);
        CREATE INDEX idx_articles_valid_article_id ON articles(article_id) WHERE is_valid=1;
        CREATE TABLE article_fts_map(fts_id INTEGER PRIMARY KEY,article_id TEXT NOT NULL UNIQUE);
        CREATE VIRTUAL TABLE article_fts USING fts5(title,text,content='',tokenize='unicode61 remove_diacritics 2');
        CREATE VIRTUAL TABLE article_lemma_fts USING fts5(
            title,text,content='',tokenize='unicode61 remove_diacritics 2');
        CREATE TABLE article_lemma_index_state(state_id INTEGER PRIMARY KEY,indexed_articles INTEGER);
        INSERT INTO article_lemma_index_state VALUES(1,100);
    """)
    issues = [
        ("target", "Jutro (Ljubljana)", "Jutro", "1925-12-23"),
        ("next", "Jutro (Ljubljana)", "Jutro", "1925-12-24"),
        ("alias", "Different source", "Jutro (Ljubljana)", "1925-12-23"),
        ("null", "Jutro (Ljubljana)", "Jutro", None),
        ("other", "Other", "Other", "1930-01-01"),
    ]
    source = "Title\nmati beseda druga appearing.\n"
    connection.executemany(
        "INSERT INTO newspapers VALUES(?,?,?,?,?,?,?)",
        [(*issue, "URN:NBN:SI:doc-" + issue[0], issue[0] + ".md", source) for issue in issues],
    )
    for index in range(80):
        issue = issues[index % len(issues)][0]
        article_id = f"a{index:05}"
        content = "mati beseda " + ("druga " * (index % 4)) + ("rare " if index % 13 == 0 else "")
        connection.execute(
            "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?)",
            (
                article_id,
                issue,
                "Title",
                content,
                int(index % 11 != 0),
                0,
                len(content),
                1,
                2,
            ),
        )
        connection.execute("INSERT INTO article_fts_map VALUES(?,?)", (index + 1, article_id))
        connection.execute(
            "INSERT INTO article_fts(rowid,title,text) VALUES(?,?,?)",
            (
                index + 1,
                "Title",
                content,
            ),
        )
        connection.execute(
            "INSERT INTO article_lemma_fts(rowid,title,text) VALUES(?,?,?)",
            (
                index + 1,
                "Title",
                content.replace("beseda", "lemma"),
            ),
        )
    connection.commit()
    engine = SearchEngine.__new__(SearchEngine)
    engine.database = SimpleNamespace(conn=connection)
    engine._active_profile = None
    engine.last_profile = None
    try:
        yield engine
    finally:
        connection.close()


def legacy(engine, table, query, limit, newspaper=None, start=None, end=None, offset=0):
    rows = engine.database.conn.execute(
        f"""
        SELECT articles.article_id,bm25({table},2.0,1.0) AS rank FROM {table}
        JOIN article_fts_map ON article_fts_map.fts_id={table}.rowid
        JOIN articles ON articles.article_id=article_fts_map.article_id
        JOIN newspapers ON newspapers.newspaper_id=articles.newspaper_id
        WHERE {table} MATCH ? AND articles.is_valid=1
        AND (? IS NULL OR newspapers.source=? OR newspapers.title=?)
        AND (? IS NULL OR newspapers.date>=?) AND (? IS NULL OR newspapers.date<=?)
        ORDER BY rank,articles.article_id LIMIT ? OFFSET ?
    """,
        (query, newspaper, newspaper, newspaper, start, start, end, end, limit, offset),
    ).fetchall()
    return [(row["article_id"], -float(row["rank"])) for row in rows]


@pytest.mark.parametrize("table", ["article_fts", "article_lemma_fts"])
@pytest.mark.parametrize(
    "query",
    [
        "mati",
        "rare",
        '"mati beseda"',
        "mati OR druga",
        "mati NOT druga",
        "mat*",
        "title:Title",
        "NEAR(mati beseda, 2)",
        "màti",
    ],
)
def test_scope_preserves_global_scores_ties_filters_and_offset(corpus, table, query):
    filters = [
        (None, None, None),
        ("Jutro (Ljubljana)", "1925-12-23", "1925-12-23"),
        ("Jutro (Ljubljana)", None, None),
        (None, "1925-12-23", "1925-12-24"),
        ("Missing", None, None),
    ]
    for newspaper, start, end in filters:
        for offset in (0, 3):
            result = corpus._search_fts(table, query, 7, newspaper, start, end, offset, focus_query=query)
            assert [(row["article_id"], score) for row, score in result] == legacy(
                corpus,
                table,
                query,
                7,
                newspaper,
                start,
                end,
                offset,
            )


def test_combined_reuses_scope_and_materializes_only_final_page(corpus, monkeypatch):
    calls = []
    original = engine_module.resolve_scope

    def resolve(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(engine_module, "resolve_scope", resolve)
    statements = []
    corpus.database.conn.set_trace_callback(statements.append)
    result = corpus._search_combined_bm25(
        "mati",
        3,
        "Jutro (Ljubljana)",
        "1925-12-23",
        "1925-12-23",
        0.5,
        2,
    )
    corpus.database.conn.set_trace_callback(None)
    assert len(calls) == 1
    surface = legacy(corpus, "article_fts", "mati", 50, "Jutro (Ljubljana)", "1925-12-23", "1925-12-23")
    lemma = legacy(corpus, "article_lemma_fts", "mati", 50, "Jutro (Ljubljana)", "1925-12-23", "1925-12-23")
    scores = dict(surface)
    for article_id, score in lemma:
        scores[article_id] = scores.get(article_id, 0) + 0.5 * score
    expected = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[2:5]
    assert [(row["article_id"], score) for row, score in result] == expected
    full_articles = [sql for sql in statements if "SELECT articles.*, articles.title AS article_title" in sql]
    full_issues = [sql for sql in statements if "SELECT content FROM newspapers WHERE newspaper_id=" in sql]
    assert len(full_articles) == 3
    assert len(full_issues) == len({row["document_id"] for row, _ in result})
    assert all("_focus_char" in row and row["newspaper_content"] for row, _ in result)


def test_overflow_uses_complete_text_first_results(corpus, monkeypatch):
    original = engine_module.resolve_scope
    monkeypatch.setattr(engine_module, "resolve_scope", lambda *a, **k: original(*a, **k, max_articles=3))
    hits = corpus.search("bm25", "mati", newspaper="Jutro (Ljubljana)", limit=50, profile=True)
    assert len(hits) > 3
    assert corpus.last_profile["fts_strategies"]["article_fts"] == "text-first"
    assert len(hits) == len(legacy(corpus, "article_fts", "mati", 50, "Jutro (Ljubljana)"))


@pytest.mark.parametrize("table", ["article_fts", "article_lemma_fts"])
def test_empty_scopes_still_reject_malformed_match(corpus, table):
    with pytest.raises(sqlite3.OperationalError, match="unterminated"):
        corpus._search_fts(table, '"bad', 10, "Missing", None, None, 0, focus_query="bad")
    assert corpus._search_fts(table, "mati", 10, "Missing", None, None, 0, focus_query="mati") == []
    corpus.database.conn.execute("DELETE FROM article_fts_map")
    with pytest.raises(sqlite3.OperationalError, match="unterminated"):
        corpus._search_fts(table, '"bad', 10, "Jutro (Ljubljana)", None, None, 0, focus_query="bad")
    corpus.database.conn.execute(f"INSERT INTO {table}({table}) VALUES('delete-all')")
    with pytest.raises(sqlite3.OperationalError, match="unterminated"):
        corpus._search_fts(table, '"bad', 10, "Missing", None, None, 0, focus_query="bad")
    assert corpus._search_fts(table, "mati", 10, "Missing", None, None, 0, focus_query="mati") == []


def test_scope_first_seek_avoids_global_common_term_work(corpus):
    connection = corpus.database.conn
    for index in range(80, 10080):
        aid = f"a{index:05}"
        connection.execute(
            "INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?)",
            (
                aid,
                "other",
                "Title",
                "mati beseda",
                1,
                0,
                11,
                1,
                2,
            ),
        )
        connection.execute("INSERT INTO article_fts_map VALUES(?,?)", (index + 1, aid))
        connection.execute(
            "INSERT INTO article_fts(rowid,title,text) VALUES(?,?,?)", (index + 1, "Title", "mati beseda")
        )
    connection.commit()
    ticks = [0]

    def tick():
        ticks[0] += 100
        return 0

    connection.set_progress_handler(tick, 100)
    expected = legacy(corpus, "article_fts", "mati", 10, "Jutro (Ljubljana)", "1925-12-23", "1925-12-23")
    legacy_steps = ticks[0]
    ticks[0] = 0
    result = corpus._search_fts(
        "article_fts", "mati", 10, "Jutro (Ljubljana)", "1925-12-23", "1925-12-23", 0, focus_query="mati"
    )
    scoped_steps = ticks[0]
    connection.set_progress_handler(None, 0)
    assert [(row["article_id"], score) for row, score in result] == expected
    assert scoped_steps < legacy_steps / 10
    corpus.search(
        "bm25", "mati", newspaper="Jutro (Ljubljana)", date_from="1925-12-23", date_to="1925-12-23", profile=True
    )
    assert corpus.last_profile["fts_strategies"]["article_fts"] == "scope-first"
    assert any(
        "VIRTUAL TABLE INDEX" in step and "=M" in step for step in corpus.last_profile["query_plans"]["article_fts"]
    )
