"""Task-independent execution selection, kept separate from model identity."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ExecutionConfig:
    """Select transport independently for each NER or lemmatization invocation.

    Validation does not discover endpoints, import a client or start a model.
    """

    backend: Literal["local", "remote"] = "local"
    worker_urls: tuple[str, ...] = ()
    token_file: Path | None = None
    connect_timeout: float = 5.0
    read_timeout: float = 300.0
    max_attempts: int = 3
    retry_delay: float = 0.25

    def __post_init__(self) -> None:
        if self.backend not in ("local", "remote"):
            raise ValueError("Execution backend must be local or remote.")
        if not isinstance(self.worker_urls, tuple):
            raise ValueError("Worker URLs must be a tuple.")
        if self.backend == "local" and self.worker_urls:
            raise ValueError("Worker URLs require the remote execution backend.")
        if self.backend == "local" and self.token_file is not None:
            raise ValueError("Worker token file requires the remote execution backend.")
        if self.token_file is not None and not isinstance(self.token_file, Path):
            raise ValueError("Worker token file must be a Path.")
        for value in (self.connect_timeout, self.read_timeout, self.retry_delay):
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError("Worker timeouts and retry delay must be finite and positive.")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("Worker attempts must be an integer between 1 and 10.")
        if self.backend == "remote" and not self.worker_urls:
            raise ValueError("Remote execution requires at least one worker URL.")
        origins = set()
        for url in self.worker_urls:
            if not isinstance(url, str) or not url or "\\" in url or any(
                char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url
            ):
                raise ValueError("Worker URLs must be nonempty HTTP(S) URLs without whitespace.")
            try:
                parts = urlsplit(url)
                port = parts.port
            except ValueError as exc:
                raise ValueError("Invalid worker URL.") from exc
            if (
                parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or parts.path not in ("", "/")
                or (port is not None and port == 0)
            ):
                raise ValueError(
                    "Worker URLs must be HTTP(S) origins without credentials, paths, queries or fragments."
                )
            origins.add((parts.scheme, parts.hostname.lower(), port or (443 if parts.scheme == "https" else 80)))
        if len(origins) != len(self.worker_urls):
            raise ValueError("Worker URLs must be unique.")

    def validate_for(self, task: Literal["ner", "lemma"], *, injected: bool = False, local_workers: int = 1) -> None:
        """Reject unsupported or conflicting execution before any pipeline writes."""
        if task not in ("ner", "lemma"):
            raise ValueError("Execution task must be ner or lemma.")
        if self.backend == "remote":
            if injected:
                raise ValueError("Remote execution cannot be combined with an injected local implementation.")
            if local_workers != 1:
                raise ValueError("Remote execution cannot be combined with local CLASSLA worker processes.")
            if task == "lemma":
                raise NotImplementedError("Remote lemma execution is not available yet; use the local backend.")
            if self.token_file is None:
                raise ValueError("Remote NER requires a worker token file.")
