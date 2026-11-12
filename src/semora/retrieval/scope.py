"""Bounded, read-only newspaper/date scoping shared by lexical retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_ISSUES = 256
DEFAULT_MAX_ARTICLES = 2048
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _identifier(value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("Expected a SQL identifier.")
    return value


@dataclass(frozen=True)
class SearchScope:
    """A planning hint, never a truncated result set.

    ``issue_ids=None`` means the metadata probe could not prove a small scope
    within its bounds, or was unnecessary.
    An empty tuple means the complete selected issue set is empty. ``narrow``
    additionally proves all scoped article rows fit the configured threshold.
    Counting invalid rows too keeps the probe bounded with the existing issue
    index; actual retrieval still admits only valid articles.
    """

    newspaper: str | None
    date_from: str | None
    date_to: str | None
    issue_ids: tuple[str, ...] | None = None
    narrow: bool = False
    article_probe_count: int = 0

    @property
    def filtered(self) -> bool:
        return any(value is not None for value in (self.newspaper, self.date_from, self.date_to))

    def filter_sql(self, alias: str = "newspapers") -> tuple[str, tuple[Any, ...]]:
        """Emit only supplied predicates, retaining exact source-or-title semantics."""
        alias = _identifier(alias)
        clauses = []
        parameters: list[Any] = []
        if self.newspaper is not None:
            clauses.append(f"({alias}.source=? OR {alias}.title=?)")
            parameters.extend((self.newspaper, self.newspaper))
        if self.date_from is not None:
            clauses.append(f"{alias}.date>=?")
            parameters.append(self.date_from)
        if self.date_to is not None:
            clauses.append(f"{alias}.date<=?")
            parameters.append(self.date_to)
        return " AND ".join(clauses) or "1", tuple(parameters)

    def _issue_select(self) -> tuple[str, tuple[Any, ...]]:
        if self.newspaper is not None:
            dates, date_parameters = SearchScope(None, self.date_from, self.date_to).filter_sql()
            # Separate indexed aliases retain source OR title without making a
            # broad date range scan every other newspaper before a rare match.
            source = (
                "SELECT newspapers.newspaper_id FROM newspapers INDEXED BY idx_newspapers_source_date "
                f"WHERE newspapers.source=? AND {dates}"
            )
            title = (
                "SELECT newspapers.newspaper_id FROM newspapers INDEXED BY idx_newspapers_title "
                f"WHERE newspapers.title=? AND {dates}"
            )
            parameters = (self.newspaper, *date_parameters)
            return source + " UNION " + title, (*parameters, *parameters)
        predicate, parameters = self.filter_sql()
        index = " INDEXED BY idx_newspapers_date" if self.filtered else ""
        return f"SELECT newspapers.newspaper_id FROM newspapers{index} WHERE {predicate}", parameters

    def _probe_issues(self, connection: Any, maximum: int) -> tuple[str, ...] | None:
        if self.newspaper is not None and (self.date_from is not None or self.date_to is not None):
            # First sample the date range *without* the newspaper predicate.
            # The LIMIT bounds actual metadata visits, not just matching output.
            dates, parameters = SearchScope(None, self.date_from, self.date_to).filter_sql()
            rows = connection.execute(
                "SELECT newspapers.newspaper_id,newspapers.source,newspapers.title "
                f"FROM newspapers INDEXED BY idx_newspapers_date WHERE {dates} LIMIT ?",
                (*parameters, maximum + 1),
            ).fetchall()
            if len(rows) <= maximum:
                return tuple(str(row[0]) for row in rows if self.newspaper in (row[1], row[2]))
        if self.newspaper is None:
            sql, parameters = self._issue_select()
            rows = connection.execute(sql + " LIMIT ?", (*parameters, maximum + 1)).fetchall()
            return None if len(rows) > maximum else tuple(str(row[0]) for row in rows)
        dates, parameters = SearchScope(None, self.date_from, self.date_to).filter_sql()
        issue_ids: dict[str, None] = {}
        # Bound each branch before deduplication; UNION DISTINCT would otherwise
        # finish the complete compound query before its outer LIMIT applies.
        for column, index in (("source", "idx_newspapers_source_date"), ("title", "idx_newspapers_title")):
            if column == "title" and (self.date_from is not None or self.date_to is not None):
                # There is no legacy (title,date) composite index. Bound the
                # title bucket before applying dates; otherwise an empty
                # intersection could scan its entire historical issue set.
                rows = connection.execute(
                    f"SELECT newspapers.newspaper_id,newspapers.date FROM newspapers INDEXED BY {index} "
                    "WHERE newspapers.title=? LIMIT ?",
                    (self.newspaper, maximum + 1),
                ).fetchall()
                if len(rows) > maximum:
                    return None
                ids = (
                    str(row[0])
                    for row in rows
                    if (self.date_from is None or row[1] is not None and row[1] >= self.date_from)
                    and (self.date_to is None or row[1] is not None and row[1] <= self.date_to)
                )
            else:
                rows = connection.execute(
                    f"SELECT newspapers.newspaper_id FROM newspapers INDEXED BY {index} "
                    f"WHERE newspapers.{column}=? AND {dates} LIMIT ?",
                    (self.newspaper, *parameters, maximum + 1),
                ).fetchall()
                ids = (str(row[0]) for row in rows)
            issue_ids.update((issue_id, None) for issue_id in ids)
            if len(issue_ids) > maximum:
                return None
        return tuple(issue_ids)

    def issues_cte(self, name: str = "selected_issues") -> tuple[str, tuple[Any, ...]]:
        """Return one CTE declaration (without WITH) and its positional bindings.

        Known issue IDs avoid repeated metadata resolution. Overflow falls back
        to the complete indexed predicate, not the probe's first results.
        """
        name = _identifier(name)
        if self.issue_ids is None:
            sql, parameters = self._issue_select()
            return f"{name}(newspaper_id) AS MATERIALIZED ({sql})", parameters
        if not self.issue_ids:
            return f"{name}(newspaper_id) AS MATERIALIZED (SELECT NULL WHERE 0)", ()
        values = ",".join("(?)" for _ in self.issue_ids)
        return f"{name}(newspaper_id) AS MATERIALIZED (VALUES {values})", self.issue_ids


def resolve_scope(
    connection: Any,
    newspaper: str | None,
    date_from: str | None,
    date_to: str | None,
    *,
    max_issues: int = DEFAULT_MAX_ISSUES,
    max_articles: int = DEFAULT_MAX_ARTICLES,
) -> SearchScope:
    """Classify a selected scope without counting or loading archive text.

    The issue bound keeps CTE bindings below SQLite's historical 999-variable
    limit. The raw article-row threshold is an initial strategy policy, not a search
    limit: overflowing either probe selects the complete broad fallback.
    """
    if type(max_issues) is not int or not 1 <= max_issues <= 900:
        raise ValueError("Use an issue probe between 1 and 900.")
    if type(max_articles) is not int or not 1 <= max_articles <= 100_000:
        raise ValueError("Use an article probe between 1 and 100000.")
    scope = SearchScope(newspaper, date_from, date_to)
    if not scope.filtered:
        return scope
    ids = scope._probe_issues(connection, max_issues)
    if ids is None:
        return scope
    resolved = SearchScope(newspaper, date_from, date_to, issue_ids=ids)
    cte, parameters = resolved.issues_cte()
    articles = connection.execute(
        f"WITH {cte} SELECT 1 FROM selected_issues AS i "
        "CROSS JOIN articles AS a ON a.newspaper_id=i.newspaper_id LIMIT ?",
        (*parameters, max_articles + 1),
    ).fetchall()
    return SearchScope(
        newspaper,
        date_from,
        date_to,
        issue_ids=ids,
        narrow=len(articles) <= max_articles,
        article_probe_count=len(articles),
    )
