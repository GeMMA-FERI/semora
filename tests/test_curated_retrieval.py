import uuid

import pytest

from semora.curation import CurationStore
from semora.curation.retrieval import CuratedRetrieval
from semora.ner.names import build_entity_names
from semora.retrieval import SearchEngine
from semora.storage import Database
from tests.test_entity_search import entity_root  # noqa: F401


@pytest.fixture
def curated(entity_root):  # noqa: F811
    engine = SearchEngine(entity_root / "indexes/semora.sqlite")
    with CurationStore(entity_root / "curation.sqlite", corpus_id="c", initialize=True) as store:
        created = store.mutate(
            "create_group",
            {"preferred_name": "Shared group", "members": ["Ana Novak", "Marija 000"]},
            operation_id=str(uuid.uuid4()),
            actor="admin:1",
            expected_revision=0,
        )
        yield CuratedRetrieval(engine._entities(), store), created["group_id"], entity_root
    engine.close()


def test_group_paging_deduplicates_and_preserves_source(curated):
    reader, group, _ = curated
    page = reader.articles(group, limit=7)
    seen = []
    while True:
        seen.extend(h["document_id"] for h in page["hits"])
        assert all(h["matched_name"] in ("Ana Novak", "Marija 000") for h in page["hits"])
        if page["next_after"] is None:
            break
        page = reader.articles(
            group,
            limit=7,
            after=page["next_after"],
            generation=page["generation"],
            curation_revision=page["curation_revision"],
        )
    assert len(seen) == len(set(seen)) == 110
    assert not reader.articles(group, newspaper="Other")["hits"]
    assert not reader.articles(group, date_from="1940-01-01")["hits"]
    count = reader.count_articles(group)
    assert count["article_count"] == 110
    reader.store.save_count(count)
    assert reader.preview_group(group)["article_count"] == 110


def test_revision_change_rejects_pagination_and_count(curated):
    reader, group, _ = curated
    page = reader.articles(group, limit=1)
    count = reader.count_articles(group)
    reader.store.mutate(
        "update_group",
        {"group_id": group, "preferred_name": "Changed", "members": ["Marija 000"]},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=1,
    )
    with pytest.raises(ValueError, match="changed"):
        reader.articles(
            group,
            limit=1,
            after=page["next_after"],
            generation=page["generation"],
            curation_revision=page["curation_revision"],
        )
    with pytest.raises(ValueError, match="changed"):
        reader.store.save_count(count)
    with pytest.raises(ValueError, match="requires"):
        reader.articles(group, after="a000")


def test_discovery_pending_count_and_rebuild_stable_keys(curated):
    reader, group, root = curated
    item = next(i for i in reader.discovery("Ana Novak")["entities"] if i.get("group_id") == group)
    assert item["name"] == "Shared group"
    assert item["article_count"] is None and item["count_status"] == "pending"
    reader.entities.engine.close()
    build_entity_names(root, "ner")
    engine = SearchEngine(root / "indexes/semora.sqlite")
    try:
        fresh = CuratedRetrieval(engine._entities(), reader.store)
        assert fresh.articles(group, limit=1)["hits"]
        assert fresh.count_articles(group)["article_count"] == 110
    finally:
        engine.close()


def test_stale_hashes_and_unresolved_members(curated):
    reader, group, root = curated
    db = Database(root / "indexes/semora.sqlite")
    with db.conn:
        db.conn.execute("UPDATE articles SET content='Changed' WHERE article_id='a000'")
    db.close()
    assert reader.count_articles(group)["article_count"] == 109
    reader.store.mutate(
        "update_group",
        {"group_id": group, "preferred_name": "With missing", "members": ["Ana Novak", "Not extracted"]},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=1,
    )
    assert reader.articles(group)["unresolved_members"] == ["not extracted"]
    with pytest.raises(ValueError, match="missing"):
        reader.count_articles(group)


def test_preferred_name_discovery_and_overlapping_mentions(curated):
    reader, group, root = curated
    assert reader.discovery("Shared group*")["entities"][0]["group_id"] == group
    db = Database(root / "indexes/semora.sqlite")
    with db.conn:
        db.conn.execute("INSERT INTO article_entities VALUES('ner','a000',4,'Ana','PER',14,17,NULL)")
    db.close()
    page = reader.articles(group, limit=1, after="a000", generation=reader.entities.generation, curation_revision=1)
    assert page["hits"][0]["matched_name"] == "Ana Novak"


def test_count_cache_and_inflight_count_survive_unrelated_changes(curated):
    reader, group, _ = curated
    count = reader.count_articles(group)
    reader.store.mutate(
        "create_group",
        {"preferred_name": "Other", "members": ["Unrelated"]},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=1,
    )
    reader.store.save_count(count)
    assert reader.preview_group(group)["article_count"] == 110
    with pytest.raises(ValueError, match="trailing"):
        reader.discovery("*")


def test_preferred_name_cannot_shadow_unrelated_raw_names(curated):
    reader, _, _ = curated
    created = reader.store.mutate(
        "create_group",
        {"preferred_name": "Marija 001", "members": ["Unrelated"]},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=1,
    )
    result = reader.discovery("Marija 001")["entities"]
    assert any(item.get("group_id") == created["group_id"] for item in result)
    assert any(item["name"] == "Marija 001" and not item.get("group_id") for item in result)
    prefix = reader.discovery("Marija*")
    assert any(item["name"] == "Marija 002" for item in prefix["entities"])
    assert prefix["limited"]
