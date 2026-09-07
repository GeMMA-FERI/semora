"""Slovene lemmatization through the optional CLASSLA pipeline."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Protocol

# Obeliks keeps this alphanumeric sentinel as one token. Punctuation-based
# markers such as @@EOD@@ are split into several tokens.
DOCUMENT_BOUNDARY = "SEMORAEODBOUNDARYZXQ"
@dataclass(frozen=True)
class LemmatizationProfile:
    documents: int
    characters: int
    tokens: int
    tokenize_seconds: float
    pos_seconds: float
    lemma_seconds: float
    peak_cuda_bytes: int
    wall_seconds: float | None = None

    @property
    def total_seconds(self) -> float:
        return self.tokenize_seconds + self.pos_seconds + self.lemma_seconds

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.wall_seconds if self.wall_seconds is not None else self.total_seconds
        return self.tokens / elapsed if elapsed else 0.0


@dataclass
class _PipelineBatch:
    texts: tuple[str, ...]
    boundary: str
    combined: str
    document: Any | None = None
    tokenize_seconds: float = 0.0
    pos_seconds: float = 0.0
    lemma_seconds: float = 0.0


@dataclass(frozen=True)
class _LemmaCollection:
    documents: list[str]
    tokens: int


class Lemmatizer(Protocol):
    def lemmatize(self, text: str) -> str: ...

    def lemmatize_many(self, texts: Sequence[str]) -> list[str]: ...


class ClasslaLemmatizer:
    """Minimal tokenize/POS/lemma CLASSLA pipeline for standard Slovene."""

    def __init__(
        self,
        *,
        pipeline_type: str = "default",
        device: str = "auto",
        resources_dir: str | Path | None = None,
        pos_batch_size: int | None = None,
        lemma_batch_size: int | None = None,
        tokenizer_workers: int = 0,
    ) -> None:
        try:
            import classla
            import torch
        except ImportError as exc:
            raise RuntimeError("Slovene lemmatization requires the 'classla' extra.") from exc
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("CLASSLA device must be auto, cpu, or cuda.")
        use_gpu = torch.cuda.is_available() if device == "auto" else device == "cuda"
        if device == "cuda" and not use_gpu:
            raise RuntimeError("CLASSLA was asked to use CUDA, but CUDA is unavailable.")
        if pos_batch_size is not None and pos_batch_size <= 0:
            raise ValueError("CLASSLA POS batch size must be positive.")
        if lemma_batch_size is not None and lemma_batch_size <= 0:
            raise ValueError("CLASSLA lemma batch size must be positive.")
        if tokenizer_workers < 0:
            raise ValueError("CLASSLA tokenizer worker count cannot be negative.")
        options: dict[str, Any] = {
            "type": pipeline_type,
            "processors": "tokenize,pos,lemma",
            "use_gpu": use_gpu,
            "verbose": False,
        }
        if resources_dir is not None:
            options["dir"] = str(Path(resources_dir).resolve())
        if pos_batch_size is not None:
            options["pos_batch_size"] = pos_batch_size
        if lemma_batch_size is not None:
            options["lemma_batch_size"] = lemma_batch_size
        self._torch = torch
        self._use_gpu = use_gpu
        self._pos_stream = torch.cuda.Stream() if use_gpu else None
        self._lemma_stream = torch.cuda.Stream() if use_gpu else None
        self._tokenizer_workers = tokenizer_workers
        self._tokenizer_pool: ProcessPoolExecutor | None = None
        self.last_profile: LemmatizationProfile | None = None
        self.pipeline = classla.Pipeline("sl", **options)

    def lemmatize_many(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            self.last_profile = LemmatizationProfile(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0)
            return []
        self._begin_profile()
        wall_started = time.perf_counter()
        batch = self._tokenize_batch(tuple(texts))
        batch = self._pos_batch(batch)
        batch = self._lemma_batch(batch)
        collection = self._collect_batch(batch)
        wall_seconds = time.perf_counter() - wall_started
        peak_cuda_bytes = self._end_profile()
        self.last_profile = _profile_batches(
            [batch], collection.tokens, peak_cuda_bytes, wall_seconds
        )
        return collection.documents

    def lemmatize_batches(
        self,
        text_batches: Sequence[Sequence[str]],
        *,
        pipeline_depth: int = 3,
    ) -> list[list[str]]:
        """Lemmatize bounded batches through independent tokenizer, POS, and lemma stages."""
        if pipeline_depth <= 0:
            raise ValueError("pipeline_depth must be positive.")
        batches = [tuple(texts) for texts in text_batches]
        if not batches:
            self.last_profile = LemmatizationProfile(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0)
            return []

        self._begin_profile()
        wall_started = time.perf_counter()
        completed: list[tuple[_PipelineBatch, _LemmaCollection]] = []
        pending: deque[Future[_PipelineBatch]] = deque()
        with ExitStack() as stack:
            if self._tokenizer_workers:
                tokenize_pool = self._get_tokenizer_pool()
            else:
                tokenize_pool = stack.enter_context(
                    ThreadPoolExecutor(max_workers=1, thread_name_prefix="classla-tokenize")
                )
            annotation_pool = stack.enter_context(
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="classla-annotate")
            )
            for texts in batches:
                if self._tokenizer_workers:
                    tokenized = tokenize_pool.submit(_tokenize_external, texts)
                    annotated = annotation_pool.submit(self._annotate_external_after, tokenized)
                else:
                    tokenized = tokenize_pool.submit(self._tokenize_batch, texts)
                    annotated = annotation_pool.submit(self._annotate_after, tokenized)
                pending.append(annotated)
                if len(pending) >= pipeline_depth:
                    batch = pending.popleft().result()
                    completed.append((batch, self._collect_batch(batch)))
            while pending:
                batch = pending.popleft().result()
                completed.append((batch, self._collect_batch(batch)))

        wall_seconds = time.perf_counter() - wall_started
        peak_cuda_bytes = self._end_profile()
        stage_batches = [batch for batch, _ in completed]
        results = [collection.documents for _, collection in completed]
        token_count = sum(collection.tokens for _, collection in completed)
        self.last_profile = _profile_batches(
            stage_batches, token_count, peak_cuda_bytes, wall_seconds
        )
        return results

    def close(self) -> None:
        if self._tokenizer_pool is not None:
            self._tokenizer_pool.shutdown(wait=True, cancel_futures=True)
            self._tokenizer_pool = None

    def release_cuda_cache(self) -> None:
        """Return unused cached CUDA blocks without unloading the pipeline models."""
        if self._use_gpu:
            self._torch.cuda.empty_cache()

    def start_tokenizer_workers(self) -> None:
        """Start optional tokenizer processes before timed batch processing."""
        if self._tokenizer_workers:
            self._get_tokenizer_pool().submit(_tokenizer_ready).result()

    def _get_tokenizer_pool(self) -> ProcessPoolExecutor:
        if self._tokenizer_pool is None:
            self._tokenizer_pool = ProcessPoolExecutor(
                max_workers=self._tokenizer_workers,
                mp_context=get_context("spawn"),
            )
        return self._tokenizer_pool

    def _annotate_after(self, future: Future[_PipelineBatch]) -> _PipelineBatch:
        batch = future.result()
        return self._lemma_batch(self._pos_batch(batch))

    def _annotate_external_after(self, future: Future[tuple[Any, ...]]) -> _PipelineBatch:
        texts, boundary, combined, raw_text, document, metadocument, tokenize_seconds = future.result()
        batch = _PipelineBatch(
            texts=texts,
            boundary=boundary,
            combined=combined,
            tokenize_seconds=tokenize_seconds,
        )
        if combined.strip():
            batch.document = self.pipeline.processors["tokenize"].process_tokenized(
                raw_text,
                document,
                metadocument,
            )
        return self._lemma_batch(self._pos_batch(batch))

    def _tokenize_batch(self, texts: tuple[str, ...]) -> _PipelineBatch:
        boundary = _unused_boundary(texts)
        separator = f"\n\n{boundary}\n\n"
        combined = separator.join(texts)
        batch = _PipelineBatch(texts=texts, boundary=boundary, combined=combined)
        if not combined.strip():
            return batch
        started = time.perf_counter()
        with self._torch.inference_mode():
            batch.document = self.pipeline.processors["tokenize"].process(combined)
        batch.tokenize_seconds = time.perf_counter() - started
        return batch

    def _pos_batch(self, batch: _PipelineBatch) -> _PipelineBatch:
        if batch.document is None:
            return batch
        started = time.perf_counter()
        batch.document = self._run_gpu_stage("pos", batch.document, self._pos_stream)
        batch.pos_seconds = time.perf_counter() - started
        return batch

    def _lemma_batch(self, batch: _PipelineBatch) -> _PipelineBatch:
        if batch.document is None:
            return batch
        started = time.perf_counter()
        batch.document = self._run_gpu_stage("lemma", batch.document, self._lemma_stream)
        batch.lemma_seconds = time.perf_counter() - started
        return batch

    def _run_gpu_stage(self, name: str, document: Any, stream: Any | None) -> Any:
        with self._torch.inference_mode():
            if stream is None:
                return self.pipeline.processors[name].process(document)
            with self._torch.cuda.stream(stream):
                result = self.pipeline.processors[name].process(document)
            stream.synchronize()
            return result

    def _collect_batch(self, batch: _PipelineBatch) -> _LemmaCollection:
        if batch.document is None:
            return _LemmaCollection(["" for _ in batch.texts], 0)
        return _lemmas_from_document(batch.texts, batch.boundary, batch.document)

    def _begin_profile(self) -> None:
        if self._use_gpu:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()

    def _end_profile(self) -> int:
        if not self._use_gpu:
            return 0
        self._torch.cuda.synchronize()
        return int(self._torch.cuda.max_memory_allocated())

    def _process_profiled(self, text: str) -> tuple[Any, dict[str, float], int]:
        """Compatibility helper used by the operator profiler."""
        self._begin_profile()
        document: Any = text
        stage_seconds: dict[str, float] = {}
        with self._torch.inference_mode():
            for processor_name in ("tokenize", "pos", "lemma"):
                started = time.perf_counter()
                document = self.pipeline.processors[processor_name].process(document)
                if self._use_gpu:
                    self._torch.cuda.synchronize()
                stage_seconds[processor_name] = time.perf_counter() - started
        return document, stage_seconds, self._end_profile()

    def lemmatize(self, text: str) -> str:
        return self.lemmatize_many([text])[0]


def _lemmas_from_document(
    texts: Sequence[str],
    boundary: str,
    document: Any,
) -> _LemmaCollection:
    results: list[list[str]] = [[] for _ in texts]
    document_index = 0
    boundaries_seen = 0
    token_count = 0
    for sentence in document.sentences:
        for token in sentence.tokens:
            token_text = str(token.text)
            if token_text == boundary:
                boundaries_seen += 1
                document_index += 1
                continue
            if document_index >= len(texts):
                raise ValueError("CLASSLA returned tokens after the final EOD document boundary.")
            lemmas = [
                str(word.lemma or word.text).strip()
                for word in token.words
                if str(word.lemma or word.text).strip() not in {"", "_"}
            ]
            if not lemmas and token_text.strip():
                lemmas = [token_text.strip()]
            if lemmas:
                results[document_index].extend(lemmas)
                token_count += 1
    expected_boundaries = len(texts) - 1
    if boundaries_seen != expected_boundaries:
        raise ValueError(
            f"CLASSLA returned {boundaries_seen} EOD boundaries; expected {expected_boundaries}."
        )
    return _LemmaCollection([" ".join(lemmas) for lemmas in results], token_count)


def _profile_batches(
    batches: Sequence[_PipelineBatch],
    token_count: int,
    peak_cuda_bytes: int,
    wall_seconds: float,
) -> LemmatizationProfile:
    return LemmatizationProfile(
        documents=sum(len(batch.texts) for batch in batches),
        characters=sum(len(text) for batch in batches for text in batch.texts),
        tokens=token_count,
        tokenize_seconds=sum(batch.tokenize_seconds for batch in batches),
        pos_seconds=sum(batch.pos_seconds for batch in batches),
        lemma_seconds=sum(batch.lemma_seconds for batch in batches),
        peak_cuda_bytes=peak_cuda_bytes,
        wall_seconds=wall_seconds,
    )


def _tokenize_external(texts: tuple[str, ...]) -> tuple[Any, ...]:
    """Tokenize without loading POS, lemma, or CUDA state in the worker."""
    from classla.utils.obeliks import ObeliksTrainer

    boundary = _unused_boundary(texts)
    combined = f"\n\n{boundary}\n\n".join(texts)
    if not combined.strip():
        return texts, boundary, combined, combined, [], [], 0.0
    started = time.perf_counter()
    raw_text, document, metadocument = ObeliksTrainer.tokenize(combined)
    return (
        texts,
        boundary,
        combined,
        raw_text,
        document,
        metadocument,
        time.perf_counter() - started,
    )


def _tokenizer_ready() -> None:
    """Import tokenizer resources eagerly in a persistent worker."""
    from classla.utils.obeliks import ObeliksTrainer  # noqa: F401


def download_classla_models(
    *,
    pipeline_type: str = "default",
    resources_dir: str | Path | None = None,
) -> None:
    try:
        import classla
    except ImportError as exc:
        raise RuntimeError("CLASSLA model download requires the 'classla' extra.") from exc
    options: dict[str, Any] = {"type": pipeline_type, "processors": "tokenize,pos,lemma"}
    if resources_dir is not None:
        options["dir"] = str(Path(resources_dir).resolve())
    classla.download("sl", **options)


def _unused_boundary(texts: Sequence[str]) -> str:
    boundary = DOCUMENT_BOUNDARY
    suffix = 1
    while any(boundary in text for text in texts):
        boundary = f"{DOCUMENT_BOUNDARY}{suffix}"
        suffix += 1
    return boundary
