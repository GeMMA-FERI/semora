"""Adapt a fixed benchmark sample to the shared bounded pipeline."""

from contextlib import closing

from semora.ner.staging import BatchResult, iter_predictions

from .sample import readonly

__all__ = ["BatchResult", "prediction_batches"]


def prediction_batches(model, sample, total, batch_articles, batch_size, depth=1):
    offsets = iter(range(0, total, batch_articles))

    def read_batch():
        offset = next(offsets, None)
        if offset is None:
            return []
        with closing(readonly(sample)) as source:
            return [dict(row) for row in source.execute(
                "SELECT article_id, content FROM articles WHERE position >= ? AND position < ? ORDER BY position",
                (offset, min(offset + batch_articles, total)),
            )]

    yield from iter_predictions(model, read_batch, batch_size, depth)
