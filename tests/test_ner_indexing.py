"""Production storage tests use tiny migrated databases, never the archive."""

import sqlite3
from contextlib import closing

import pytest

from semora.ner.base import Entity, Prediction
from semora.ner.indexing import build_ner_index
from semora.storage import Database


class Model:
    metadata = {"adapter": "test", "version": 1}

    def __init__(self, fail_at=None, bad_span=False):
        self.calls = 0
        self.closed = False
        self.fail_at = fail_at
        self.bad_span = bad_span

    def prepare(self, texts):
        return texts

    def predict_prepared(self, texts, batch_size):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("inference failed")
        return Prediction([
            [Entity(0, 3, "PER", "bad" if self.bad_span else "Ana")] if text.startswith("Ana") else []
            for text in texts
        ])

    def close(self):
        self.closed = True


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "semora.sqlite"
    db = Database(path)
    try:
        db.initialize()
        with db.conn:
            db.conn.execute("INSERT INTO runs(run_id, run_type) VALUES ('ingest', 'test')")
            db.conn.execute("INSERT INTO newspapers(newspaper_id, run_id, content) VALUES ('n', 'ingest', '')")
            db.conn.executemany(
                "INSERT INTO articles(article_id, run_id, newspaper_id, content, is_valid) "
                "VALUES (?, 'ingest', 'n', ?, ?)",
                [('a', 'Ana Novak', 1), ('b', 'No names.', 1), ('c', 'Ana!', 1),
                 ('invalid', 'Ana', 0), ('empty', '', 1)],
            )
    finally:
        db.close()
    return path


def run(path, model=None, **kwargs):
    return build_ner_index(path, model=model or Model(), device="cpu", batch_articles=2, **kwargs)


def counts(path):
    with closing(sqlite3.connect(path)) as db:
        return tuple(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                     for table in ('article_ner_state', 'article_entities'))


@pytest.mark.parametrize('depth', [1, 3])
def test_resume_includes_zero_entities_and_skips_invalid(corpus, depth):
    first = run(corpus, max_articles=2, pipeline_depth=depth)
    assert first['processed_articles'] == 2
    assert counts(corpus) == (2, 1)
    assert run(corpus, max_articles=2)['added_articles'] == 0
    second = run(corpus, pipeline_depth=depth)
    assert second['added_articles'] == 1
    assert second['exhausted_snapshot']
    assert counts(corpus) == (3, 2)
    model = Model()
    assert run(corpus, model)['added_articles'] == 0
    assert model.calls == 0 and model.closed


def test_failed_batch_resumes_without_losing_previous_commit(corpus):
    broken = Model(fail_at=2)
    with pytest.raises(RuntimeError, match='inference failed'):
        run(corpus, broken)
    assert broken.closed
    assert counts(corpus) == (2, 1)
    assert run(corpus)['added_articles'] == 1
    with closing(sqlite3.connect(corpus)) as db:
        assert db.execute("SELECT COUNT(*) FROM logs WHERE level='ERROR'").fetchone()[0] == 1


def test_invalid_spans_do_not_commit(corpus):
    with pytest.raises(ValueError, match='does not match'):
        run(corpus, Model(bad_span=True))
    assert counts(corpus) == (0, 0)


def test_configuration_mismatch_and_separate_runs(corpus):
    run(corpus)
    with pytest.raises(ValueError, match='configuration changed'):
        run(corpus, ner_decoder='numba')
    assert run(corpus, run_id='another')['added_articles'] == 3
    assert counts(corpus) == (6, 4)


def test_content_updates_and_deletion_invalidate_entities(corpus):
    run(corpus)
    db = Database(corpus)
    try:
        with db.conn:
            db.conn.execute("UPDATE articles SET content='No names now.' WHERE article_id='a'")
            db.conn.execute("DELETE FROM articles WHERE article_id='c'")
        assert counts(corpus) == (1, 0)
        assert run(corpus)['added_articles'] == 1
        assert counts(corpus) == (2, 0)
    finally:
        db.close()


def test_transaction_failure_rolls_back_state_and_mentions(corpus):
    with closing(sqlite3.connect(corpus)) as db:
        db.executescript("""
            CREATE TRIGGER reject_entity BEFORE INSERT ON article_entities
            BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END;
        """)
    with pytest.raises(sqlite3.IntegrityError, match='simulated write failure'):
        run(corpus)
    assert counts(corpus) == (0, 0)


def test_cli_defaults():
    from semora.cli.main import _parser

    args = _parser().parse_args(['index', 'ner', '--ner-decoder', 'numba', '--max-articles', '100'])
    assert args.max_articles == 100
    assert args.ner_decoder == 'numba'
    assert args.batch_articles == 32 and args.batch_size == 16


def test_changed_article_during_inference_rejects_entire_batch(corpus):
    class Mutating(Model):
        def predict_prepared(self, texts, batch_size):
            result = super().predict_prepared(texts, batch_size)
            with closing(sqlite3.connect(corpus)) as db:
                db.execute("UPDATE articles SET content='Changed' WHERE article_id='b'")
                db.commit()
            return result

    with pytest.raises(RuntimeError, match='Article changed'):
        run(corpus, Mutating(), pipeline_depth=1)
    assert counts(corpus) == (0, 0)
    assert run(corpus)['added_articles'] == 3


def test_prediction_count_mismatch_cannot_mark_articles_done(corpus):
    class Missing(Model):
        def predict_prepared(self, texts, batch_size):
            return Prediction([[]])

    with pytest.raises(ValueError):
        run(corpus, Missing())
    assert counts(corpus) == (0, 0)


def test_validity_change_is_reprocessed(corpus):
    run(corpus)
    db = Database(corpus)
    try:
        with db.conn:
            db.conn.execute("UPDATE articles SET is_valid=0 WHERE article_id='a'")
        assert counts(corpus) == (2, 1)
        assert run(corpus)['added_articles'] == 0
        with db.conn:
            db.conn.execute("UPDATE articles SET is_valid=1 WHERE article_id='a'")
        assert run(corpus)['added_articles'] == 1
    finally:
        db.close()


def test_second_builder_is_rejected(corpus):
    from semora.retrieval.semantic_lock import SemanticBuildLock

    with SemanticBuildLock(corpus.with_name(corpus.name + '.ner.lock'), 'NER'):
        with pytest.raises(RuntimeError, match='already using'):
            run(corpus)
