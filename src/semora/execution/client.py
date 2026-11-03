"""Bounded, authenticated HTTP execution for remote NER workers."""

from __future__ import annotations

import copy
import http.client
import json
import socket
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any
from urllib.parse import urlsplit

from semora.ner.base import Prediction
from semora.ner.staging import BatchResult

from .config import ExecutionConfig
from .contracts import Document, Request, Response, semantic_fingerprint

# Independent client ceilings prevent a worker from advertising unbounded allocations.
_HARD_LIMITS = {
    "max_documents": 1024,
    "max_input_bytes": 16 * 1024 * 1024,
    "max_request_bytes": 32 * 1024 * 1024,
    "max_response_bytes": 64 * 1024 * 1024,
}
_RETRY = {408, 429, 502, 503, 504}


class RemoteExecutionError(RuntimeError):
    """A remote worker could not safely execute a request."""


class _Retryable(RemoteExecutionError):
    pass


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("non-finite number")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


class RemoteNERPool:
    """Remote NER transport; DNS resolution is subject to the OS resolver deadline."""

    def __init__(self, config: ExecutionConfig):
        config.validate_for("ner")
        if config.backend != "remote":
            raise ValueError("Remote NER pool requires the remote execution backend.")
        self.config = config
        try:
            token = config.token_file.read_text(encoding="ascii").strip() if config.token_file else ""
        except (OSError, UnicodeError):
            raise RemoteExecutionError("Worker token file could not be read.") from None
        if len(token) < 32 or any(not 33 <= ord(char) <= 126 for char in token):
            raise RemoteExecutionError("Worker token must contain at least 32 visible ASCII characters.")
        self._token = token
        self._urls = config.worker_urls
        self._locks = [threading.Lock() for _ in self._urls]
        self._diagnostics_lock = threading.Lock()
        self._diagnostics: dict[str, Any] = {
            "worker_count": len(self._urls),
            "requests": 0,
            "attempts": 0,
            "retries": 0,
            "failures": 0,
            "last_failure": None,
            "workers_attempted": [],
        }
        self._limits = dict(_HARD_LIMITS)
        workers = []
        identity = fingerprint = None
        for url in self._urls:
            capability = self._request(url, "GET", "/v1/capabilities", None, _HARD_LIMITS["max_response_bytes"])
            try:
                ner = capability["tasks"]["ner"]
                current = ner["identity"]
                reported = ner["fingerprint"]
                if (
                    capability["api_version"] != "v1"
                    or capability["ready"] is not True
                    or ner["enabled"] is not True
                    or ner["ready"] is not True
                    or current["format"] != "semora.classla-ner.identity.v1"
                    or semantic_fingerprint(current) != reported
                ):
                    raise ValueError()
                if identity is not None and (identity != current or fingerprint != reported):
                    raise ValueError()
                identity, fingerprint = current, reported
                for name, ceiling in _HARD_LIMITS.items():
                    value = capability["limits"][name]
                    if type(value) is not int or value <= 0:
                        raise ValueError()
                    self._limits[name] = min(self._limits[name], value, ceiling)
                runtime = capability.get("runtime", {})
                # Retain JSON provenance without permitting non-finite numbers
                # in the run record. Runtime is deliberately outside identity.
                _encode(runtime)
                workers.append({"url": url, "runtime": runtime, "limits": capability["limits"]})
            except (KeyError, TypeError, ValueError, RecursionError):
                raise RemoteExecutionError("Worker capabilities are invalid or incompatible.") from None
        self.metadata = {
            "adapter": "classla", "worker_identity": identity, "worker_fingerprint": fingerprint,
            "execution": {"backend": "remote", "workers": workers},
        }
        self._diagnostics["workers"] = workers
        if not isinstance(fingerprint, str):
            raise RemoteExecutionError("Remote NER requires at least one compatible worker.")
        self._fingerprint = fingerprint

    @property
    def diagnostics(self) -> dict[str, Any]:
        with self._diagnostics_lock:
            return copy.deepcopy(self._diagnostics)

    def _request(self, url, method, path, body, limit):
        parts = urlsplit(url)
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = cls(parts.hostname, parts.port, timeout=self.config.connect_timeout)
        # A timer interrupts even a peer that continuously dribbles bytes.
        expired = threading.Event()
        active_socket: list[Any] = [None]

        def expire():
            expired.set()
            sock = active_socket[0] or connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        timer = threading.Timer(self.config.connect_timeout + self.config.read_timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            connection.connect()
            active_socket[0] = connection.sock
            if expired.is_set():
                raise _Retryable("Worker request deadline exceeded.")
            connection.sock.settimeout(self.config.read_timeout)
            headers = {"Accept": "application/json"}
            if self._token is not None:
                headers["Authorization"] = "Bearer " + self._token
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                error = _Retryable if response.status in _RETRY else RemoteExecutionError
                raise error(f"Worker returned HTTP {response.status}.")
            raw = response.read(limit + 1)
            if expired.is_set():
                raise _Retryable("Worker request deadline exceeded.")
            if len(raw) > limit:
                raise RemoteExecutionError("Worker response exceeds the byte limit.")
            try:
                return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_invalid_constant)
            except (ValueError, UnicodeError, RecursionError):
                raise RemoteExecutionError("Worker returned invalid JSON.") from None
        except (OSError, http.client.HTTPException):
            raise _Retryable("Worker connection failed or timed out.") from None
        finally:
            timer.cancel()
            connection.close()

    def _subbatches(self, rows):
        current: list[Any] = []
        size = 0
        # UUIDs have fixed ASCII length. Compact JSON list size is the empty
        # request plus each encoded document and commas; encode each text once.
        header = len(_encode(Request("0" * 36, "ner", self._fingerprint, ()).to_dict()))
        raw_size = header

        def fits(count, text_bytes, request_bytes):
            return (
                count <= self._limits["max_documents"]
                and text_bytes <= self._limits["max_input_bytes"]
                and request_bytes <= self._limits["max_request_bytes"]
            )

        for row in rows:
            text = row.get("inference_content", row["content"])
            document = Document.create(str(row["article_id"]), text)
            nbytes = len(text.encode("utf-8"))
            encoded_size = len(_encode(document.to_dict()))
            proposed = raw_size + encoded_size + int(bool(current))
            if current and not fits(len(current) + 1, size + nbytes, proposed):
                yield current
                current, size, raw_size = [], 0, header
                proposed = header + encoded_size
            if not fits(len(current) + 1, size + nbytes, proposed):
                raise RemoteExecutionError("A single document exceeds the worker input limits.")
            current.append((row, document))
            size += nbytes
            raw_size = proposed
        if current:
            yield current

    def _infer(self, items, start, read_seconds):
        request = Request(str(uuid.uuid4()), "ner", self._fingerprint, tuple(doc for _, doc in items))
        body = _encode(request.to_dict())
        tick = time.monotonic()
        with self._diagnostics_lock:
            self._diagnostics["requests"] += 1
        for attempt in range(self.config.max_attempts):
            index = (start + attempt) % len(self._urls)
            with self._diagnostics_lock:
                self._diagnostics["attempts"] += 1
                if attempt:
                    self._diagnostics["retries"] += 1
                if self._urls[index] not in self._diagnostics["workers_attempted"]:
                    self._diagnostics["workers_attempted"].append(self._urls[index])
            try:
                with self._locks[index]:
                    value = self._request(
                        self._urls[index], "POST", "/v1/ner", body, self._limits["max_response_bytes"]
                    )
                try:
                    results = Response.from_dict(value).validate_against(request)
                except (ValueError, TypeError, KeyError, RecursionError):
                    raise RemoteExecutionError("Worker returned an invalid inference result.") from None
                entities = []
                for item in results:
                    if item.entities is None:
                        raise RemoteExecutionError("Worker returned results for another task.")
                    entities.append(list(item.entities))
                return BatchResult(
                    [row for row, _ in items],
                    Prediction(entities),
                    read_seconds,
                    time.monotonic() - tick,
                )
            except _Retryable as error:
                with self._diagnostics_lock:
                    self._diagnostics["last_failure"] = str(error)
                    if attempt + 1 == self.config.max_attempts:
                        self._diagnostics["failures"] += 1
                if attempt + 1 == self.config.max_attempts:
                    raise RemoteExecutionError("Worker request failed after retries.") from None
                time.sleep(self.config.retry_delay)
            except RemoteExecutionError as error:
                with self._diagnostics_lock:
                    self._diagnostics["failures"] += 1
                    self._diagnostics["last_failure"] = str(error)
                raise

    def iter_predictions(self, read_batch):
        executor = ThreadPoolExecutor(max_workers=len(self._urls), thread_name_prefix="remote-ner")
        pending: dict[Any, int] = {}
        buffered: deque[Any] = deque()
        exhausted = False
        available = deque(range(len(self._urls)))
        try:
            while pending or not exhausted or buffered:
                while available and (buffered or not exhausted):
                    if not buffered:
                        tick = time.monotonic()
                        rows = read_batch()
                        read_seconds = time.monotonic() - tick
                        if not rows:
                            exhausted = True
                            break
                        buffered.extend(
                            (batch, read_seconds if i == 0 else 0.0) for i, batch in enumerate(self._subbatches(rows))
                        )
                    if buffered:
                        items, elapsed = buffered.popleft()
                        endpoint = available.popleft()
                        pending[executor.submit(self._infer, items, endpoint, elapsed)] = endpoint
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        # Reuse the worker that finished, rather than queueing behind
                        # a slower endpoint while a faster model sits idle.
                        available.append(pending.pop(future))
                        yield future.result()
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
