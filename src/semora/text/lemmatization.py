"""Slovene lemmatization through the optional CLASSLA pipeline."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Obeliks keeps this alphanumeric sentinel as one token. Punctuation-based
# markers such as @@EOD@@ are split into several tokens.
DOCUMENT_BOUNDARY = "SEMORAEODBOUNDARYZXQ"


@dataclass(frozen=True)
class LemmaToken:
    start: int
    end: int
    lemmas: tuple[str, ...]


@dataclass(frozen=True)
class LemmatizationProfile:
    documents: int
    characters: int
    tokens: int
    tokenize_seconds: float
    pos_seconds: float
    lemma_seconds: float
    peak_cuda_bytes: int

    @property
    def total_seconds(self) -> float:
        return self.tokenize_seconds + self.pos_seconds + self.lemma_seconds

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.total_seconds if self.total_seconds else 0.0


class Lemmatizer(Protocol):
    def annotate(self, text: str) -> list[LemmaToken]: ...

    def annotate_many(self, texts: Sequence[str]) -> list[list[LemmaToken]]: ...

    def lemmatize(self, text: str) -> str: ...


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
        self.last_profile: LemmatizationProfile | None = None
        self.pipeline = classla.Pipeline("sl", **options)

    def annotate(self, text: str) -> list[LemmaToken]:
        return self.annotate_many([text])[0]

    def annotate_many(self, texts: Sequence[str]) -> list[list[LemmaToken]]:
        if not texts:
            self.last_profile = LemmatizationProfile(0, 0, 0, 0.0, 0.0, 0.0, 0)
            return []
        boundary = _unused_boundary(texts)
        separator = f"\n\n{boundary}\n\n"
        parts: list[str] = []
        for index, text in enumerate(texts):
            if index:
                parts.append(separator)
            parts.append(text)
        combined = "".join(parts)
        if not combined.strip():
            self.last_profile = LemmatizationProfile(len(texts), len(combined), 0, 0.0, 0.0, 0.0, 0)
            return [[] for _ in texts]
        document, stage_seconds, peak_cuda_bytes = self._process_profiled(combined)
        results: list[list[LemmaToken]] = [[] for _ in texts]
        document_index = 0
        search_start = 0
        boundaries_seen = 0
        for sentence in document.sentences:
            for token in sentence.tokens:
                token_text = str(token.text)
                if token_text == boundary:
                    boundaries_seen += 1
                    document_index += 1
                    search_start = 0
                    continue
                if document_index >= len(texts):
                    raise ValueError("CLASSLA returned tokens after the final EOD document boundary.")
                token_start = texts[document_index].find(token_text, search_start)
                if token_start < 0:
                    raise ValueError(
                        "Could not map a CLASSLA token back to its source document after the EOD boundary. "
                        f"Token: {token_text!r}."
                    )
                token_end = token_start + len(token_text)
                search_start = token_end
                lemmas = tuple(
                    str(word.lemma or word.text).strip()
                    for word in token.words
                    if str(word.lemma or word.text).strip() not in {"", "_"}
                )
                if not lemmas and token_text.strip():
                    lemmas = (token_text.strip(),)
                if lemmas:
                    results[document_index].append(
                        LemmaToken(
                            start=token_start,
                            end=token_end,
                            lemmas=lemmas,
                        )
                    )
        expected_boundaries = len(texts) - 1
        if boundaries_seen != expected_boundaries:
            raise ValueError(
                f"CLASSLA returned {boundaries_seen} EOD boundaries; expected {expected_boundaries}."
            )
        token_count = sum(len(tokens) for tokens in results)
        self.last_profile = LemmatizationProfile(
            documents=len(texts),
            characters=sum(len(text) for text in texts),
            tokens=token_count,
            tokenize_seconds=stage_seconds.get("tokenize", 0.0),
            pos_seconds=stage_seconds.get("pos", 0.0),
            lemma_seconds=stage_seconds.get("lemma", 0.0),
            peak_cuda_bytes=peak_cuda_bytes,
        )
        return results

    def _process_profiled(self, text: str) -> tuple[Any, dict[str, float], int]:
        if self._use_gpu:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()
        document: Any = text
        stage_seconds: dict[str, float] = {}
        with self._torch.inference_mode():
            for processor_name in ("tokenize", "pos", "lemma"):
                started = time.perf_counter()
                document = self.pipeline.processors[processor_name].process(document)
                if self._use_gpu:
                    self._torch.cuda.synchronize()
                stage_seconds[processor_name] = time.perf_counter() - started
        peak_cuda_bytes = int(self._torch.cuda.max_memory_allocated()) if self._use_gpu else 0
        return document, stage_seconds, peak_cuda_bytes

    def lemmatize(self, text: str) -> str:
        return " ".join(lemma for token in self.annotate(text) for lemma in token.lemmas)


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
        boundary = f"@@EOD_{suffix}@@"
        suffix += 1
    return boundary
