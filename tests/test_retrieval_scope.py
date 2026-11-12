"""Scope decisions preserve complete filtering and bound preliminary work."""

import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from semora.retrieval.scope import resolve_scope


@pytest.fixture
def connection():
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE newspapers(newspaper_id TEXT PRIMARY KEY,source TEXT,title TEXT,date TEXT);
        CREATE INDEX idx_newspapers_date ON newspapers(date);
        CREATE INDEX idx_newspapers_source_date ON newspapers(source,date);
        CREATE INDEX idx_newspapers_title ON newspapers(title);
        CREATE TABLE articles(article_id TEXT PRIMARY KEY,newspaper_id TEXT,is_valid INTEGER);
        CREATE INDEX idx_articles_newspaper_id ON articles(newspaper_id);
    """)
    connection.executemany(
        "INSERT INTO newspapers VALUES(?,?,?,?)",
        [
            ("one", "Jutro (Ljubljana)", "Jutro", "1925-12-23"),
            ("two", "Other", "Jutro (Ljubljana)", "1925-12-23"),
            ("both", "Jutro (Ljubljana)", "Jutro (Ljubljana)", "1925-12-24"),
            ("undated", "Jutro (Ljubljana)", None, None),
            ("other", "Other", "Other", "1930-01-01"),
        ],
    )
    connection.executemany(
        "INSERT INTO articles VALUES(?,?,?)",
        [
            ("a1", "one", 1),
            ("invalid", "one", 0),
            ("unknown", "one", None),
            ("a2", "two", 1),
            ("a3", "both", 1),
            ("a4", "undated", 1),
            ("a5", "other", 1),
        ],
    )
    yield connection
    connection.close()


def selected_ids(connection, scope):
    cte, parameters = scope.issues_cte()
    return {row[0] for row in connection.execute(f"WITH {cte} SELECT newspaper_id FROM selected_issues", parameters)}


@pytest.mark.parametrize(
    "newspaper,start,end,expected",
    [
        ("Jutro (Ljubljana)", "1925-12-23", "1925-12-23", {"one", "two"}),
        ("Jutro (Ljubljana)", None, None, {"one", "two", "both", "undated"}),
        (None, "1925-12-24", None, {"both", "other"}),
        (None, None, "1925-12-23", {"one", "two"}),
        ("Missing", "1925-12-23", "1925-12-23", set()),
        ("jutro (ljubljana)", None, None, set()),
    ],
)
def test_scope_preserves_exact_metadata_and_nullable_inclusive_dates(connection, newspaper, start, end, expected):
    scope = resolve_scope(connection, newspaper, start, end)
    assert scope.filtered and scope.narrow
    assert selected_ids(connection, scope) == expected
    assert len(scope.issue_ids) == len(expected)
    assert scope.article_probe_count == len(expected) + (2 if "one" in expected else 0)


def test_no_filters_do_not_probe_or_cache(connection):
    statements = []
    connection.set_trace_callback(statements.append)
    scope = resolve_scope(connection, None, None, None)
    assert not scope.filtered and not scope.narrow and scope.issue_ids is None
    assert statements == []
    assert scope.filter_sql() == ("1", ())
    with pytest.raises(FrozenInstanceError):
        scope.narrow = True


def test_issue_overflow_uses_complete_scope_not_first_probe_rows(connection):
    scope = resolve_scope(connection, "Jutro (Ljubljana)", None, None, max_issues=1)
    assert not scope.narrow and scope.issue_ids is None
    assert selected_ids(connection, scope) == {"one", "two", "both", "undated"}


def test_article_overflow_does_not_truncate_valid_articles(connection):
    connection.executemany("INSERT INTO articles VALUES(?,?,1)", [(f"more{i}", "one") for i in range(20)])
    scope = resolve_scope(connection, "Jutro (Ljubljana)", "1925-12-23", "1925-12-23", max_articles=2)
    assert not scope.narrow and scope.article_probe_count == 3
    cte, parameters = scope.issues_cte()
    count = connection.execute(
        f"WITH {cte} SELECT COUNT(*) FROM selected_issues i "
        "JOIN articles a ON a.newspaper_id=i.newspaper_id WHERE a.is_valid=1",
        parameters,
    ).fetchone()[0]
    assert count == 22


def test_scope_values_remain_parameters_and_identifiers_are_not_input(connection):
    scope = resolve_scope(connection, "Jutro' OR 1=1 --", None, None)
    assert scope.narrow and selected_ids(connection, scope) == set()
    predicate, parameters = scope.filter_sql("n")
    assert predicate == "(n.source=? OR n.title=?)"
    assert parameters == ("Jutro' OR 1=1 --",) * 2
    with pytest.raises(ValueError, match="identifier"):
        scope.issues_cte("issues; DROP TABLE articles")
    with pytest.raises(ValueError, match="identifier"):
        scope.filter_sql("n; DROP TABLE articles")


def test_narrow_probe_uses_metadata_and_article_indexes_without_scanning_other_articles(connection):
    connection.executemany("INSERT INTO articles VALUES(?,?,1)", [(f"irrelevant{i}", "other") for i in range(20_000)])
    statements = []
    steps = [0]
    connection.set_trace_callback(statements.append)

    def progress():
        steps[0] += 100
        return 0

    connection.set_progress_handler(progress, 100)
    scope = resolve_scope(connection, "Jutro (Ljubljana)", "1925-12-23", "1925-12-23")
    connection.set_progress_handler(None, 0)
    connection.set_trace_callback(None)
    assert scope.narrow and scope.article_probe_count == 4
    assert steps[0] < 1000
    assert len(statements) == 2
    issue_plan = connection.execute("EXPLAIN QUERY PLAN " + statements[0]).fetchall()
    article_plan = connection.execute("EXPLAIN QUERY PLAN " + statements[1]).fetchall()
    assert any("SEARCH newspapers USING INDEX idx_newspapers_date" in row[3] for row in issue_plan)
    assert any("SEARCH a USING COVERING INDEX idx_articles_newspaper_id" in row[3] for row in article_plan)
    assert not any("COUNT(" in statement or "content" in statement for statement in statements)


@pytest.mark.parametrize("issues,articles", [(0, 1), (901, 1), (True, 1), (1, 0), (1, 100001)])
def test_probe_bounds_are_explicit(connection, issues, articles):
    with pytest.raises(ValueError, match="probe"):
        resolve_scope(connection, None, None, None, max_issues=issues, max_articles=articles)


def test_wide_date_scope_with_missing_or_rare_paper_does_not_walk_issue_history(connection):
    connection.executemany(
        "INSERT INTO newspapers VALUES(?,?,?,?)",
        [(f"history{i}", "Common", "Common", "1925-01-01") for i in range(100_000)],
    )
    for newspaper, expected in (("Missing", set()), ("Jutro (Ljubljana)", {"one", "two", "both"})):
        steps = [0]

        def progress(steps=steps):
            steps[0] += 100
            return 0

        connection.set_progress_handler(progress, 100)
        scope = resolve_scope(connection, newspaper, "1900-01-01", "2200-01-01")
        connection.set_progress_handler(None, 0)
        assert scope.narrow
        assert selected_ids(connection, scope) == expected
        assert steps[0] < 3000


def test_source_title_overlap_and_null_source_are_deduplicated_after_probe_overflow(connection):
    scope = resolve_scope(connection, "Jutro (Ljubljana)", None, None, max_issues=1)
    connection.execute("INSERT INTO newspapers VALUES('nullsource',NULL,'Jutro (Ljubljana)',NULL)")
    assert selected_ids(connection, scope) == {"one", "two", "both", "undated", "nullsource"}


def test_large_title_bucket_outside_period_chooses_complete_fallback_without_scan(connection):
    connection.executemany(
        "INSERT INTO newspapers VALUES(?,?,?,?)",
        [(f"oldalias{i}", None, "Alias", "1800-01-01") for i in range(100_000)],
    )
    connection.executemany(
        "INSERT INTO newspapers VALUES(?,?,?,?)", [(f"dated{i}", "Other", "Other", "1925-01-01") for i in range(300)]
    )
    connection.execute("INSERT INTO newspapers VALUES('latealias',NULL,'Alias','1925-12-23')")
    steps = [0]

    def progress():
        steps[0] += 100
        return 0

    connection.set_progress_handler(progress, 100)
    scope = resolve_scope(connection, "Alias", "1900-01-01", "2200-01-01")
    connection.set_progress_handler(None, 0)
    assert not scope.narrow and scope.issue_ids is None
    assert steps[0] < 5000
    assert selected_ids(connection, scope) == {"latealias"}


def test_invalid_article_history_cannot_make_the_bounded_probe_walk_entire_issue(connection):
    connection.executemany(
        "INSERT INTO articles VALUES(?,?,0)", [(f"invalidhistory{i}", "one") for i in range(100_000)]
    )
    connection.execute("INSERT INTO articles VALUES('latevalid','one',1)")
    steps = [0]

    def progress():
        steps[0] += 100
        return 0

    connection.set_progress_handler(progress, 100)
    scope = resolve_scope(connection, "Jutro (Ljubljana)", "1925-12-23", "1925-12-23")
    connection.set_progress_handler(None, 0)
    assert not scope.narrow and scope.article_probe_count == 2049
    assert steps[0] < 20_000
    cte, parameters = scope.issues_cte()
    valid = {
        row[0]
        for row in connection.execute(
            f"WITH {cte} SELECT a.article_id FROM selected_issues i JOIN articles a "
            "ON a.newspaper_id=i.newspaper_id WHERE a.is_valid=1",
            parameters,
        )
    }
    assert valid == {"a1", "a2", "latevalid"}
