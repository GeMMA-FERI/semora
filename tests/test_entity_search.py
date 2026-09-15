import os
from pathlib import Path

import pytest

from semora.ner.names import build_entity_names
from semora.ner.spellfix import install_spellfix
from semora.retrieval.engine import SearchEngine
from semora.storage import Database


@pytest.fixture
def entity_root(tmp_path):
    library = os.environ.get("SEMORA_TEST_SPELLFIX")
    if not library:
        pytest.skip("Set SEMORA_TEST_SPELLFIX to a trusted test DLL/SO")
    install_spellfix(tmp_path, library)
    db = Database(tmp_path / "indexes" / "semora.sqlite")
    db.initialize()
    with db.conn:
        db.conn.execute("INSERT INTO runs(run_id,run_type) VALUES ('r','test')")
        db.conn.execute("INSERT INTO ner_runs(ner_run_id,config_json) VALUES ('ner','{}')")
        for i in range(110):
            aid = f"a{i:03}"
            name = f"Marija {i:03}"
            text = name + " met Ana Novak."
            db.conn.execute(
                "INSERT INTO newspapers(newspaper_id,run_id,content,source,date,urn,relative_path) "
                "VALUES (?,'r',?,'Jutro','1934-01-01',?,?)", (aid, text, f"URN:NBN:SI:doc-{i}", f"{aid}.md"))
            db.conn.execute(
                "INSERT INTO articles(article_id,run_id,newspaper_id,content,is_valid,char_start,char_end," 
                "line_start,line_end) VALUES (?,'r',?,?,1,0,?,1,1)", (aid, aid, text, len(text)))
            db.conn.execute("INSERT INTO article_ner_state VALUES ('ner',?,'hash',3,CURRENT_TIMESTAMP)", (aid,))
            db.conn.executemany("INSERT INTO article_entities VALUES ('ner',?,?,?,?,?,?,NULL)", [
                (aid, 0, name, "PER", 0, len(name)),
                (aid, 1, name, "ORG", 0, len(name)),
                (aid, 2, "Ana Novak", "PER", text.index("Ana"), len(text)-1),
            ])
    db.close()
    build_entity_names(tmp_path, "ner", batch_size=17)
    return tmp_path


def test_real_spellfix_bounded_and_label_independent(entity_root):
    engine = SearchEngine(entity_root / "indexes" / "semora.sqlite")
    try:
        result = engine.search_entities("Marija*")
        assert len(result["entities"]) == 100
        assert result["limited"]
        assert all(e["article_count"] == 1 for e in result["entities"])
        assert all("label" not in e for e in result["entities"])
        assert engine.search_entities("Ana Novak")["entities"][0]["article_count"] == 110
        with pytest.raises(ValueError, match="trailing"):
            engine.search_entities("*Marija*")
    finally:
        engine.close()


def test_article_keyset_filters_and_stale_offsets(entity_root):
    engine = SearchEngine(entity_root / "indexes" / "semora.sqlite")
    try:
        page = engine.search_entity_articles("ANA NOVAK", limit=20)
        assert len(page["hits"]) == 20
        assert page["next_after"] == "a019"
        next_page = engine.search_entity_articles("Ana Novak", after=page["next_after"], limit=20)
        assert next_page["next_after"] == "a039"
        assert all("Ana Novak" in hit["snippet"] for hit in page["hits"])
        assert not engine.search_entity_articles("Ana Novak", newspaper="Other")["hits"]
        assert not engine.search_entity_articles("Ana Novak", date_from="1940-01-01")["hits"]
        with pytest.raises(ValueError, match="changed"):
            engine.search_entity_articles("Ana Novak", generation="old")
        db = Database(entity_root / "indexes" / "semora.sqlite")
        with db.conn:
            db.conn.execute("UPDATE articles SET content='Changed' WHERE article_id='a000'")
        db.close()
        assert engine.search_entity_articles("Ana Novak", limit=1)["hits"][0]["document_id"] == "1"
    finally:
        engine.close()


def test_checksum_rejected_before_load(tmp_path):
    source = tmp_path / "bad.dll"
    source.write_bytes(b"not executable")
    with pytest.raises(ValueError, match="checksum"):
        install_spellfix(tmp_path, source, "0" * 64)
    assert not (tmp_path / "extensions").exists()


def test_rebuild_keeps_existing_reader_usable(entity_root):
    engine = SearchEngine(entity_root / "indexes" / "semora.sqlite")
    try:
        old = engine.search_entities("Marija*")["generation"]
        new = build_entity_names(entity_root, "ner")
        assert old != new["generation"]
        assert engine.search_entities("Marija*")["generation"] == old
        assert len(list((Path(entity_root) / "indexes/entities").glob("*.sqlite"))) == 2
    finally:
        engine.close()
