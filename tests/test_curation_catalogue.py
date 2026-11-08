import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from semora.curation import CurationStore
from semora.curation.catalogue import CurationCatalogue


@pytest.fixture
def catalogue(tmp_path):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("CREATE TABLE names(normalized_name TEXT UNIQUE,name TEXT,article_count INTEGER);")
    c.executemany(
        "INSERT INTO names VALUES(?,?,?)",
        [("ljubljana", "LJUBLJANA", 5), ("ljubijana", "Ljubijana", 3), ("ljubljani", "Ljubljani", 4)],
    )
    with CurationStore(tmp_path / "curation.sqlite", corpus_id="c", initialize=True) as store:
        entities = SimpleNamespace(
            connection=c,
            generation="g",
            run_id="ner",
            search_entities=lambda q: {
                "entities": [
                    {"name": r["name"], "article_count": r["article_count"], "distance": 1}
                    for r in c.execute("SELECT * FROM names")
                ]
            },
        )
        yield CurationCatalogue(entities, store)
    c.close()


def test_candidate_reasons_rejections_and_membership(catalogue):
    result = catalogue.suggest_variants("Ljubljana")
    ocr = next(r for r in result["candidates"] if r["normalized_name"] == "ljubijana")
    assert "OCR-confusion comparison" in ocr["reasons"]
    assert "not exhaustive" in result["coverage"]
    catalogue.store.mutate(
        "reject_pair",
        {"first": "Ljubljana", "second": "Ljubijana", "reason": "Reviewed"},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=0,
    )
    assert all(r["normalized_name"] != "ljubijana" for r in catalogue.suggest_variants("Ljubljana")["candidates"])


def test_prefix_bounds_and_unresolved_members(catalogue):
    assert len(catalogue.find_names("LJ", limit=1)["names"]) == 1
    assert catalogue.find_names("LJ", limit=1)["limited"]
    created = catalogue.store.mutate(
        "create_group",
        {"preferred_name": "Ljubljana", "members": ["Ljubljana", "Missing"]},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=0,
    )
    preview = catalogue.preview_group(created["group_id"])
    assert preview["count_status"] == "pending"
    assert preview["article_count"] is None
    assert [m["resolved"] for m in preview["members"]] == [True, False]


def test_candidate_boundary(catalogue):
    for query in ("", "*foo", "ab", "x" * 201):
        with pytest.raises(ValueError):
            catalogue.suggest_variants(query)


def test_all_members_filter_even_when_not_used_as_anchors(catalogue):
    store = catalogue.store
    members = ["aaa", "bbb", "ccc", "ddd", "eee", "fff"]
    created = store.mutate(
        "create_group",
        {"preferred_name": "Example", "members": members},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=0,
    )
    store.mutate(
        "reject_pair",
        {"first": "fff", "second": "zzz", "reason": "Different"},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=1,
    )
    catalogue.entities.search_entities = lambda q: {
        "entities": [{"name": n, "article_count": 1, "distance": 1} for n in members + ["zzz"]]
    }
    assert catalogue.suggest_variants("aaa", group_id=created["group_id"])["candidates"] == []


def test_preview_pins_revision_during_concurrent_edit(catalogue, monkeypatch):
    store = catalogue.store
    store.connection.execute("PRAGMA journal_mode=WAL")
    created = store.mutate(
        "create_group",
        {"preferred_name": "Old", "members": ["Ljubljana"]},
        operation_id=str(uuid.uuid4()),
        actor="a",
        expected_revision=0,
    )
    original = store.group

    def concurrent(group_id):
        result = original(group_id)
        with CurationStore(store.path, corpus_id="c") as other:
            other.mutate(
                "update_group",
                {"group_id": group_id, "preferred_name": "New", "members": ["Ljubljani"]},
                operation_id=str(uuid.uuid4()),
                actor="b",
                expected_revision=1,
            )
        return result

    monkeypatch.setattr(store, "group", concurrent)
    result = catalogue.preview_group(created["group_id"])
    assert result["revision"] == 1
    assert result["preferred_name"] == "Old"
    assert result["members"][0]["normalized_name"] == "ljubljana"
    assert store.metadata()["revision"] == 2


def test_evidence_is_bounded_before_hash_validation(catalogue):
    c = catalogue.connection
    c.executescript("""
        CREATE TABLE entity_articles(entity_id INTEGER,article_id TEXT,char_start INTEGER,
            content_sha256 TEXT,PRIMARY KEY(entity_id,article_id));
        ALTER TABLE names ADD COLUMN entity_id INTEGER;
        UPDATE names SET entity_id=1 WHERE normalized_name='ljubljana';
        ATTACH DATABASE ':memory:' AS corpus;
        CREATE TABLE corpus.article_ner_state(article_id TEXT,ner_run_id TEXT,content_sha256 TEXT);
        CREATE TABLE corpus.articles(article_id TEXT,newspaper_id TEXT,title TEXT,content TEXT,is_valid INTEGER);
        CREATE TABLE corpus.newspapers(newspaper_id TEXT,source TEXT,date TEXT);
        CREATE TABLE corpus.article_entities(ner_run_id TEXT,article_id TEXT,char_start INTEGER,text TEXT,label TEXT);
    """)
    for i in range(45):
        article_id = f"a{i:03}"
        c.execute("INSERT INTO entity_articles VALUES(1,?,3,'hash')", (article_id,))
        c.execute("INSERT INTO corpus.article_ner_state VALUES(?,'ner',?)", (article_id, "stale" if i < 40 else "hash"))
        c.execute("INSERT INTO corpus.articles VALUES(?,'p','Title','In Ljubljana today',1)", (article_id,))
    c.execute("INSERT INTO corpus.newspapers VALUES('p','Paper','1930-01-01')")
    c.commit()
    assert catalogue.evidence("Ljubljana")["passages"] == []
    c.execute("UPDATE corpus.article_ner_state SET content_sha256='hash' WHERE article_id='a000'")
    c.execute("INSERT INTO corpus.article_entities VALUES('ner','a000',3,'Ljubljana','LOC')")
    result = catalogue.evidence("Ljubljana")
    assert result["passages"][0]["mentions"] == [{"text": "Ljubljana", "label": "LOC"}]
    assert "Ljubljana" in result["passages"][0]["passage"]
