import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from semora.curation.frequency import (
    PAGE_SQL,
    FrequencyNotReady,
    frequency_identity,
    frequency_page,
    prepare_frequency,
)


@pytest.fixture
def source():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript("""
            CREATE TABLE names(entity_id INTEGER PRIMARY KEY,normalized_name TEXT,article_count INTEGER);
            INSERT INTO names VALUES(1,'& co.',1),(2,'#rare',1),(999,'ljubljana',430989),
                (700,'maribor',10000),(701,'celje',10000);
        """)
        yield connection


def entity(result):
    return SimpleNamespace(frequency_path=result["path"], frequency_identity=result["identity"])


def test_global_frequency_rank_and_ties_use_indexed_keyset_pages(source, tmp_path):
    result = prepare_frequency(source, tmp_path, "dictionary", batch_size=2)
    entities = entity(result)
    first = frequency_page(entities, 0, 2)
    second = frequency_page(entities, first[-1]["rank"], 2)
    assert [r["normalized_name"] for r in first + second] == ["ljubljana", "maribor", "celje", "& co."]
    assert [r["rank"] for r in first + second] == [1, 2, 3, 4]
    with closing(sqlite3.connect(result["path"])) as index:
        plan = " ".join(row[3] for row in index.execute("EXPLAIN QUERY PLAN " + PAGE_SQL, (2, 2)))
    assert "SEARCH ranked_names USING INTEGER PRIMARY KEY" in plan
    assert "TEMP B-TREE" not in plan


def test_reuse_does_not_count_or_sort_source_and_progress_is_durable(source, tmp_path):
    progress = []
    result = prepare_frequency(source, tmp_path, "dictionary", batch_size=2, progress=progress.append)
    assert [p["completed"] for p in progress] == [0, 0, 2, 4, 5, 5]
    assert progress[-1] == {"state": "completed", "completed": 5, "total": 5}
    traced = []
    source.set_trace_callback(traced.append)
    reused = prepare_frequency(source, tmp_path, "dictionary")
    assert reused == {**result, "reused": True}
    assert not any("COUNT" in sql or "FROM names" in sql for sql in traced)


def test_partial_build_is_not_published_and_retry_is_complete(source, tmp_path):
    original = source.execute("PRAGMA temp_store").fetchone()[0]

    def stop(value):
        if value["completed"] == 2:
            raise RuntimeError("operator stopped")

    with pytest.raises(RuntimeError, match="operator stopped"):
        prepare_frequency(source, tmp_path, "dictionary", batch_size=2, progress=stop)
    assert not list(tmp_path.rglob("*.sqlite"))
    assert not list(tmp_path.rglob("*.tmp-*"))
    assert source.execute("PRAGMA temp_store").fetchone()[0] == original
    result = prepare_frequency(source, tmp_path, "dictionary")
    assert len(frequency_page(entity(result), 0, 20)) == 5


def test_source_advancement_is_checked_before_cache_reuse(source, tmp_path):
    source.executescript(
        "CREATE TABLE pipeline_entity_profile(profile_id INTEGER PRIMARY KEY,advancement_token TEXT);"
        "INSERT INTO pipeline_entity_profile VALUES(1,'first');"
    )
    prepare_frequency(source, tmp_path, "dictionary", "first")
    source.execute("UPDATE pipeline_entity_profile SET advancement_token='second'")
    source.commit()
    with pytest.raises(ValueError, match="advancement changed"):
        prepare_frequency(source, tmp_path, "dictionary", "first")
    second = prepare_frequency(source, tmp_path, "dictionary", "second")
    assert second["identity"] == frequency_identity("dictionary", "second")


def test_missing_mismatched_and_incomplete_artifacts_are_rejected(source, tmp_path):
    with pytest.raises(FrequencyNotReady):
        frequency_page(SimpleNamespace(), 0, 1)
    result = prepare_frequency(source, tmp_path, "dictionary")
    entities = entity(result)
    entities.frequency_identity = "other"
    with pytest.raises(ValueError, match="does not match"):
        frequency_page(entities, 0, 1)
    with closing(sqlite3.connect(result["path"])) as broken:
        broken.execute("DELETE FROM ranked_names WHERE rank=5")
        broken.commit()
    with pytest.raises(ValueError, match="incomplete"):
        frequency_page(entity(result), 0, 1)


def test_empty_dictionary_has_a_complete_empty_order(tmp_path):
    with closing(sqlite3.connect(":memory:")) as source:
        source.execute("CREATE TABLE names(entity_id INTEGER PRIMARY KEY,normalized_name TEXT,article_count INTEGER)")
        result = prepare_frequency(source, tmp_path, "empty")
    assert result["names"] == 0
    assert frequency_page(entity(result), 0, 1) == []


def test_preparation_rejects_borrowed_temp_state_without_dropping_it(source, tmp_path):
    source.executescript("CREATE TEMP TABLE caller_state(value TEXT); INSERT INTO caller_state VALUES('keep');")
    original = source.execute("PRAGMA temp_store").fetchone()[0]
    with pytest.raises(ValueError, match="without TEMP objects"):
        prepare_frequency(source, tmp_path, "dictionary")
    assert source.execute("SELECT value FROM caller_state").fetchone()[0] == "keep"
    assert source.execute("PRAGMA temp_store").fetchone()[0] == original
    assert not list(tmp_path.rglob("*.sqlite"))
