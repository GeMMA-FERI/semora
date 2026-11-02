"""Transactional retries must never silently accept different NER results."""

import pytest

from semora.ner.base import Entity, Prediction
from semora.ner.staging import BatchResult
from semora.ner.storage import register_run, save_batch
from semora.storage import Database


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "corpus.sqlite")
    database.initialize()
    with database.conn:
        database.conn.execute("INSERT INTO runs(run_id, run_type) VALUES ('ingest', 'test')")
        database.conn.execute("INSERT INTO newspapers(newspaper_id, run_id, content) VALUES ('n', 'ingest', '')")
        database.conn.executemany(
            "INSERT INTO articles(article_id, run_id, newspaper_id, content, is_valid) "
            "VALUES (?, 'ingest', 'n', 'Ana', 1)", [('a',), ('b',)],
        )
    register_run(database.conn, 'ner', {})
    yield database.conn
    database.close()


def batch(*items):
    return BatchResult([{'article_id': article, 'content': 'Ana'} for article, _ in items],
                       Prediction([entities for _, entities in items]), 0, 0)


ANA = Entity(0, 3, 'PER', 'Ana', 0.5)


def test_retry_and_mixed_batch_return_only_new_counts(db):
    assert save_batch(db, 'ner', batch(('a', [ANA]))) == (1, 1)
    assert save_batch(db, 'ner', batch(('a', [ANA]), ('b', []))) == (1, 0)
    assert save_batch(db, 'ner', batch(('a', [ANA]), ('b', []))) == (0, 0)


@pytest.mark.parametrize('change', [
    "UPDATE article_ner_state SET content_sha256='wrong' WHERE article_id='a'",
    "UPDATE article_ner_state SET entity_count=2 WHERE article_id='a'",
    "UPDATE article_entities SET label='ORG' WHERE article_id='a'",
    "DELETE FROM article_entities WHERE article_id='a'",
])
def test_conflict_rolls_back_entire_batch(db, change):
    save_batch(db, 'ner', batch(('a', [ANA])))
    with db:
        db.execute(change)
    with pytest.raises(RuntimeError, match='Conflicting NER'):
        save_batch(db, 'ner', batch(('b', []), ('a', [ANA])))
    assert db.execute("SELECT COUNT(*) FROM article_ner_state WHERE article_id='b'").fetchone()[0] == 0


def test_stale_source_rolls_back_entire_batch(db):
    with db:
        db.execute("UPDATE articles SET content='Changed' WHERE article_id='a'")
    with pytest.raises(RuntimeError, match='Article changed'):
        save_batch(db, 'ner', batch(('b', []), ('a', [ANA])))
    assert db.execute("SELECT COUNT(*) FROM article_ner_state").fetchone()[0] == 0


def test_orphan_entities_are_rejected(db):
    save_batch(db, 'ner', batch(('a', [ANA])))
    db.execute('PRAGMA foreign_keys=OFF')
    with db:
        db.execute("DELETE FROM article_ner_state WHERE article_id='a'")
    with pytest.raises(RuntimeError, match='Orphan NER'):
        save_batch(db, 'ner', batch(('a', [ANA])))
    db.execute('PRAGMA foreign_keys=ON')
