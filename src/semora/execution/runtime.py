"""Single-owner NER inference, independent of the optional HTTP server."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import platform
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from semora.text.diagnostics import TokenAlignmentWarnings

from .contracts import DocumentResult, Request, Response, semantic_fingerprint
from .identity import build_classla_ner_identity

logger = logging.getLogger(__name__)


class WorkerError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class WorkerSettings:
    resources: Path
    device: str = "cpu"
    package: str = "default"
    batch_size: int = 16
    tokenizer_workers: int = 0
    ner_pipeline_depth: int = 1
    decoder: str = "numpy"
    max_documents: int = 32
    max_input_bytes: int = 1_048_576
    max_request_bytes: int = 2_097_152
    max_response_bytes: int = 4_194_304
    body_timeout: float = 30.0
    expected_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda") or self.decoder not in ("numpy", "numba"):
            raise ValueError("Worker device must be cpu/cuda and decoder must be numpy/numba.")
        if not isinstance(self.package, str) or not self.package.strip():
            raise ValueError("Worker model package must be nonempty.")
        for value in (
            self.batch_size, self.ner_pipeline_depth, self.max_documents,
            self.max_input_bytes, self.max_request_bytes, self.max_response_bytes,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Worker batch sizes, depths and byte/document limits must be positive integers.")
        if type(self.tokenizer_workers) is not int or self.tokenizer_workers < 0:
            raise ValueError("Worker tokenizer count must be a nonnegative integer.")
        if type(self.body_timeout) not in (int, float) or not isfinite(self.body_timeout) or self.body_timeout <= 0:
            raise ValueError("Worker body timeout must be finite and positive.")
        if self.expected_fingerprint is not None and (
            not isinstance(self.expected_fingerprint, str) or len(self.expected_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.expected_fingerprint)
        ):
            raise ValueError("Expected fingerprint must be a lowercase SHA-256 digest.")


def _load_adapter(settings: WorkerSettings):
    from semora.ner.adapters import ClasslaAdapter

    if not Path(settings.resources).is_dir():
        raise ValueError("Worker resources directory must already exist; serving never downloads models.")
    if settings.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Worker requested CUDA, but CUDA is unavailable.")
    return ClasslaAdapter(
        device=settings.device, model=settings.package, resources=str(settings.resources),
        download=False, tokenizer_workers=settings.tokenizer_workers,
        sentence_order="input", ner_pipeline_depth=settings.ner_pipeline_depth, ner_decoder=settings.decoder,
    )


class NERWorker:
    """Load, warm, infer and close the model on the same dedicated thread.

    There is no inference queue. Cancellation of an HTTP caller cannot release
    model capacity while the underlying inference is still running.
    Control methods belong to one event loop, with sequential lifespan startup
    and shutdown. Admission has no suspension point before executor submission.
    """

    def __init__(
        self, settings: WorkerSettings, *, adapter_factory: Callable = _load_adapter,
        identity_factory: Callable = build_classla_ner_identity,
    ):
        self.settings = settings
        self._adapter_factory = adapter_factory
        self._identity_factory = identity_factory
        self._executor: ThreadPoolExecutor | None = None
        self._model: Any = None
        self._busy = threading.Lock()
        self.ready = False
        self.identity: dict = {}
        self.fingerprint = ""
        self.completed_requests = self.failed_requests = self.inferred_documents = 0
        self.token_alignment_warnings = 0
        self.versions: dict[str, str | None] = {"python": platform.python_version()}

    def _initialize(self) -> None:
        self._model = self._adapter_factory(self.settings)
        try:
            self.identity = self._identity_factory(self._model)
            self.fingerprint = semantic_fingerprint(self.identity)
            for name in ("classla", "obeliks", "torch", "numpy", "numba"):
                try:
                    self.versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    self.versions[name] = None
            if (
                self.settings.expected_fingerprint is not None
                and self.fingerprint != self.settings.expected_fingerprint
            ):
                raise ValueError("Worker semantic fingerprint differs from the expected fingerprint.")
            # Exercise tokenization/model/decoder initialization before readiness.
            with redirect_stdout(TokenAlignmentWarnings()):
                self._model.predict(["Ana Novak je v Mariboru."], self.settings.batch_size)
        except BaseException:
            self._close_model()
            raise

    def _close_model(self) -> None:
        model, self._model = self._model, None
        if model is not None and hasattr(model, "close"):
            model.close()

    async def start(self) -> None:
        if self._executor is not None:
            raise RuntimeError("Worker has already been started.")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="classla-owner")
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, self._initialize)
            self.ready = True
        except BaseException:
            # Startup cancellation does not stop model loading in its thread.
            # Queue teardown behind it, even if no readiness was published.
            try:
                await asyncio.shield(asyncio.get_running_loop().run_in_executor(self._executor, self._close_model))
            finally:
                self._executor.shutdown(wait=True)
                self._executor = None
            raise

    async def close(self) -> None:
        self.ready = False
        executor = self._executor
        if executor is not None:
            # FIFO submission waits behind an inference abandoned by its caller.
            try:
                await asyncio.shield(asyncio.get_running_loop().run_in_executor(executor, self._close_model))
            finally:
                executor.shutdown(wait=True)
                self._executor = None

    def capabilities(self) -> dict:
        return {
            "api_version": "v1", "ready": self.ready,
            "tasks": {
                "ner": {"enabled": True, "ready": self.ready, "fingerprint": self.fingerprint,
                        "identity": self.identity},
                "lemma": {"enabled": False, "ready": False},
            },
            "limits": {
                "max_documents": self.settings.max_documents, "max_input_bytes": self.settings.max_input_bytes,
                "max_request_bytes": self.settings.max_request_bytes,
                "max_response_bytes": self.settings.max_response_bytes,
            },
            "capacity": {"max_in_flight": 1, "in_flight": int(self._busy.locked()), "queue_depth": 0},
            "runtime": {"device": self.settings.device, "batch_size": self.settings.batch_size,
                        "decoder": self.settings.decoder, "sentence_order": "input",
                        "tokenizer_workers": self.settings.tokenizer_workers,
                        "ner_pipeline_depth": self.settings.ner_pipeline_depth, "versions": self.versions},
            "counters": {"completed_requests": self.completed_requests, "failed_requests": self.failed_requests,
                         "inferred_documents": self.inferred_documents,
                         "token_alignment_warnings": self.token_alignment_warnings},
        }

    def _predict(self, request: Request) -> bytes:
        try:
            texts = [document.text for document in request.documents]
            warnings = TokenAlignmentWarnings()
            try:
                # Obeliks/CLASSLA diagnostics can print whole source paragraphs.
                # The dedicated service has one model owner; retain counts only.
                with redirect_stdout(warnings):
                    prediction = self._model.predict(texts, self.settings.batch_size)
            finally:
                self.token_alignment_warnings += warnings.count
            results = tuple(
                DocumentResult(document.document_id, document.content_sha256, entities=tuple(entities))
                for document, entities in zip(request.documents, prediction.entities, strict=True)
            )
            response = Response(request.request_id, "ner", self.fingerprint, results)
            response.validate_against(request)
            encoded = json.dumps(response.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(encoded) > self.settings.max_response_bytes:
                raise WorkerError(413, "response_too_large", "Inference output exceeds the worker response limit.")
            self.completed_requests += 1
            self.inferred_documents += len(texts)
            return encoded
        except Exception as error:
            self.failed_requests += 1
            # Exceptions from token alignment can contain source text. Log only type.
            logger.error("NER inference failed (%s)", type(error).__name__)
            if isinstance(error, WorkerError):
                raise
            raise WorkerError(500, "inference_failed", "NER inference failed; no results were committed.") from None

    async def infer(self, request: Request) -> bytes:
        if not self.ready or self._executor is None:
            raise WorkerError(503, "not_ready", "Worker is not ready.")
        if request.task != "ner":
            raise WorkerError(501, "unsupported_task", "This worker supports NER only.")
        if request.fingerprint != self.fingerprint:
            raise WorkerError(409, "incompatible_fingerprint", "Request fingerprint differs from the loaded task.")
        if len(request.documents) > self.settings.max_documents or sum(
            len(document.text.encode("utf-8")) for document in request.documents
        ) > self.settings.max_input_bytes:
            raise WorkerError(413, "input_too_large", "Request exceeds the document or input-byte limit.")
        if not self._busy.acquire(blocking=False):
            raise WorkerError(503, "busy", "Worker is busy; retry later.")
        try:
            future = self._executor.submit(self._predict, request)
        except BaseException:
            self._busy.release()
            raise
        future.add_done_callback(lambda _: self._busy.release())
        wrapped = asyncio.wrap_future(future)
        # Retrieve failures even if a disconnected HTTP caller no longer awaits them.
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(wrapped)
