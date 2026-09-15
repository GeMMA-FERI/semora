"""Bounded CPU preparation, single-model inference and caller-owned writing."""

import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .base import Prediction


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
        texts = [row["content"] for row in rows]
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
