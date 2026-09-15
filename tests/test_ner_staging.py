import sqlite3
import threading
import time
from contextlib import closing

import pytest

from semora.diagnostics.ner.base import Prediction
from semora.diagnostics.ner.staging import prediction_batches


def sample(tmp_path):
    path = tmp_path / "sample.sqlite"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE articles (position INTEGER, article_id TEXT, content TEXT)")
        db.executemany("INSERT INTO articles VALUES (?, ?, ?)", [(i, str(i), str(i)) for i in range(9)])
        db.commit()
    return path


class Model:
    def __init__(self):
        self.active = 0
        self.maximum = 0
        self.threads = set()

    def prepare(self, texts):
        return texts

    def predict_prepared(self, prepared, batch_size):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.threads.add(threading.get_ident())
        time.sleep(0.005)
        self.active -= 1
        return Prediction([[] for _ in prepared])


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_staging_preserves_order_and_single_inference_owner(tmp_path, depth):
    model = Model()
    batches = list(prediction_batches(model, sample(tmp_path), 9, 2, 16, depth))
    assert [row["article_id"] for batch in batches for row in batch.rows] == list(map(str, range(9)))
    assert model.maximum == 1
    assert len(model.threads) == 1
    assert [len(batch.prediction.entities) for batch in batches] == [2, 2, 2, 2, 1]


def test_staging_propagates_failure_and_cleans_up(tmp_path):
    class Broken(Model):
        def prepare(self, texts):
            raise RuntimeError("tokenization failed")

    with pytest.raises(RuntimeError, match="tokenization failed"):
        list(prediction_batches(Broken(), sample(tmp_path), 9, 2, 16, 3))
    assert not any(t.name.startswith(("ner-prepare", "ner-infer")) for t in threading.enumerate())


def test_early_close_cleans_up(tmp_path):
    iterator = prediction_batches(Model(), sample(tmp_path), 9, 2, 16, 3)
    next(iterator)
    iterator.close()
    assert not any(t.name.startswith(("ner-prepare", "ner-infer")) for t in threading.enumerate())
