"""Bounded CPU preparation, single-model inference and caller-owned writing."""

import time
from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .base import Entity, Prediction


@dataclass
class BatchResult:
    rows: list[dict[str, Any]]
    prediction: Prediction
    read_seconds: float
    predict_seconds: float


def iter_predictions(
    model: Any, read_batch: Callable[[], list[dict[str, Any]]], batch_size: int, depth: int = 1
) -> Generator[BatchResult, None, None]:
    """One preparation thread and one GPU owner; at most depth batches in flight.

    Each reader owns its SQLite connection. The caller writes results while the
    next prediction runs. Exceptions propagate through futures; pending jobs are
    cancelled on early exit. Depth 1 preserves the sequential baseline.
    """
    if depth < 1:
        raise ValueError("Pipeline depth must be positive")
    staged = hasattr(model, "prepare") and hasattr(model, "predict_prepared")
    if depth > 1 and not staged:
        raise ValueError("This adapter does not support staged inference")

    def prepare():
        tick = time.perf_counter()
        rows = read_batch()
        if not rows:
            return None
        read_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        texts = [row.get("inference_content", row["content"]) for row in rows]
        prepared = model.prepare(texts) if staged else texts
        return rows, prepared, read_seconds, time.perf_counter() - tick

    def infer(prepared):
        if prepared is None:
            return None
        rows, value, read_seconds, preparation_seconds = prepared
        tick = time.perf_counter()
        result = model.predict_prepared(value, batch_size) if staged else model.predict(value, batch_size)
        return BatchResult(rows, result, read_seconds, preparation_seconds + time.perf_counter() - tick)

    if depth == 1:
        while (result := infer(prepare())) is not None:
            yield result
        return

    cpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ner-prepare")
    gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ner-infer")
    pending: deque[Future] = deque()

    def infer_after(future):
        return infer(future.result())

    try:
        for _ in range(depth):
            pending.append(gpu.submit(infer_after, cpu.submit(prepare)))
        while pending:
            result = pending.popleft().result()
            if result is None:
                break
            yield result
            pending.append(gpu.submit(infer_after, cpu.submit(prepare)))
    finally:
        for future in pending:
            future.cancel()
        gpu.shutdown(wait=True, cancel_futures=True)
        cpu.shutdown(wait=True, cancel_futures=True)


def coalesce_batches(
    batches: Iterable[BatchResult], max_articles: int = 0,
) -> Generator[BatchResult, None, None]:
    """Group transport results into bounded writes without changing predictions.

    Zero preserves transport commit boundaries. The caller retains ownership of
    the input iterator and must close it on early exit. Partial final batches
    are emitted only on successful exhaustion, never after a producer failure.
    """
    if type(max_articles) is not int or max_articles < 0:
        raise ValueError("NER write batch size must be a nonnegative integer.")
    if not max_articles:
        yield from batches
        return
    rows: list[dict[str, Any]] = []
    entities: list[list[Entity]] = []
    read_seconds = predict_seconds = 0.0
    for batch in batches:
        if len(batch.rows) != len(batch.prediction.entities):
            raise ValueError("NER batch rows and predictions differ in length.")
        read_seconds += batch.read_seconds
        predict_seconds += batch.predict_seconds
        offset = 0
        while offset < len(batch.rows):
            count = min(max_articles - len(rows), len(batch.rows) - offset)
            rows.extend(batch.rows[offset:offset + count])
            entities.extend(batch.prediction.entities[offset:offset + count])
            offset += count
            if len(rows) == max_articles:
                yield BatchResult(rows, Prediction(entities), read_seconds, predict_seconds)
                rows, entities = [], []
                read_seconds = predict_seconds = 0.0
    if rows:
        yield BatchResult(rows, Prediction(entities), read_seconds, predict_seconds)
