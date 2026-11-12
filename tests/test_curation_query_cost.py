"""Bounded curation list/preview reads against retained groups and large audit values."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from semora.curation import CurationStore
from semora.curation.catalogue import CurationCatalogue


@pytest.fixture
def populated(tmp_path):
    with CurationStore(tmp_path / "curation.sqlite", corpus_id="c", initialize=True) as store:
        groups = [(f"{i:032x}", f"Group {i}", "Description", i + 1, f"group {i}") for i in range(2000)]
        store.connection.executemany("INSERT INTO name_groups VALUES(?,?,?,?,?)", groups)
        store.connection.executemany(
            "INSERT INTO group_members VALUES(?,?)",
            [(f"name{i:06}-{j:03}", f"{i:032x}") for i in range(2000) for j in range(10)],
        )
        store.connection.executemany(
            "INSERT INTO group_provenance VALUES(?,?,?,?)",
            [(group[0], "human", 1, json.dumps({"note": "group proof"})) for group in groups],
        )
        store.connection.executemany(
            "INSERT INTO member_provenance VALUES(?,?,?,?)",
            [
                (f"name{i:06}-{j:03}", "human", 1, json.dumps({"note": "x" * (14000 if i < 20 else 1)}))
                for i in range(2000)
                for j in range(10)
            ],
        )
        store.connection.execute(
            "UPDATE member_provenance SET origin='autonomous',human_reviewed=0 WHERE normalized_name='name000001-000'"
        )
        store.connection.execute("DELETE FROM member_provenance WHERE normalized_name='name000002-000'")
        store.connection.execute(
            "UPDATE group_provenance SET origin='unknown',human_reviewed=0 WHERE group_id=?", (f"{3:032x}",)
        )
        store.connection.execute(
            "UPDATE member_provenance SET origin='unknown',human_reviewed=0 WHERE normalized_name LIKE 'name000003-%'"
        )
        store.connection.commit()
        yield store


def traced(store, call):
    statements = []
    steps = [0]
    store.connection.set_trace_callback(statements.append)
    store.connection.set_progress_handler(lambda: steps.__setitem__(0, steps[0] + 100) or 0, 100)
    try:
        result = call()
    finally:
        store.connection.set_trace_callback(None)
        store.connection.set_progress_handler(None, 0)
    return result, statements, steps[0]


def test_full_group_page_batches_only_selected_group_annotations(populated):
    page, statements, steps = traced(populated, lambda: populated.groups(limit=20))
    assert len(page) == 20
    assert len(statements) == 4  # One page, schema version, group annotations and joined members.
    assert steps < 20000
    assert len(json.dumps(page).encode()) > 500 * 1024  # Full legacy contract intentionally retains evidence.
    assert page[0]["members"] == [f"name000000-{j:03}" for j in range(10)]
    assert page[0]["member_provenance"]["name000000-000"]["provenance"]["note"] == "x" * 14000
    assert [(g["origin"], g["human_reviewed"]) for g in page[:4]] == [
        ("human", True),
        ("mixed", False),
        ("mixed", False),
        ("unknown", False),
    ]
    assert all("curation_changes" not in sql for sql in statements)
    assert populated.groups(after=page[-1]["group_id"], limit=1)[0]["group_id"] == f"{20:032x}"


def test_summary_page_is_compact_matches_full_provenance_and_uses_indexed_joins(populated):
    full = populated.groups(limit=20)
    page, statements, steps = traced(populated, lambda: populated.groups(limit=20, summary=True))
    assert len(statements) == 3
    assert steps < 30000
    assert len(json.dumps(page).encode()) < 10 * 1024
    assert all("provenance_json" not in sql and "curation_changes" not in sql for sql in statements)
    for summary, original in zip(page, full, strict=True):
        assert summary == {
            **{
                key: original[key]
                for key in ("group_id", "preferred_name", "description", "version", "origin", "human_reviewed")
            },
            "member_count": len(original["members"]),
        }
    plan = [row[3] for row in populated.connection.execute("EXPLAIN QUERY PLAN " + statements[-1])]
    assert any("SEARCH g USING" in step and "INDEX" in step for step in plan)
    assert any("SEARCH m USING COVERING INDEX curation_group_members" in step for step in plan)
    assert not any("SCAN g" in step or "SCAN m" in step for step in plan)


def test_preview_loads_dictionary_members_once_and_can_omit_large_provenance(populated):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE names(normalized_name TEXT PRIMARY KEY,name TEXT,article_count INTEGER)")
    connection.executemany(
        "INSERT INTO names VALUES(?,?,?)", [(f"name000001-{j:03}", f"Raw {j}", j + 1) for j in range(9)]
    )
    catalogue = CurationCatalogue(SimpleNamespace(connection=connection, generation="g", run_id="n"), populated)
    statements = []
    connection.set_trace_callback(statements.append)
    try:
        full = catalogue.preview_group(f"{1:032x}")
        statements.clear()
        light = catalogue.preview_group(f"{1:032x}", include_provenance=False)
    finally:
        connection.close()
    assert len([sql for sql in statements if "FROM names" in sql]) == 1
    assert light["members"] == full["members"]
    assert light["members"][-1]["resolved"] is False
    assert light["count_status"] == full["count_status"] == "pending"
    assert light["origin"] == "mixed" and not light["human_reviewed"]
    assert "provenance" not in light
    assert all(set(item) == {"origin", "human_reviewed"} for item in light["member_provenance"].values())
    assert full["member_provenance"]["name000001-000"]["provenance"]["note"] == "x" * 14000
    assert len(json.dumps(light).encode()) < 10 * 1024


def test_schema_one_summary_remains_unknown_without_reading_annotation_tables(populated):
    populated.connection.execute("UPDATE curation_metadata SET schema_version=1")
    populated.connection.commit()
    with CurationStore(populated.path, corpus_id="c", read_only=True) as legacy:
        full = legacy.groups(limit=1)[0]
        page, statements, _ = traced(legacy, lambda: legacy.groups(limit=1, summary=True))
        assert page[0]["origin"] == full["origin"] == "unknown"
        assert page[0]["member_count"] == len(full["members"]) == 10
        assert page[0]["human_reviewed"] is False
        assert all("member_provenance" not in sql and "group_provenance" not in sql for sql in statements)


def test_explicit_projection_flags_and_empty_keyset_pages(populated):
    assert populated.groups(after="z", summary=True) == []
    assert populated.groups(after="z") == []
    for value in (None, 1, "false"):
        with pytest.raises(ValueError, match="Boolean"):
            populated.groups(summary=value)
        with pytest.raises(ValueError, match="Boolean"):
            populated.group(f"{0:032x}", include_provenance=value)
