from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from semora.embeddings.serialization import build_embedding
from semora.storage import Database

OPENAI_BATCH_ENDPOINT = "/v1/embeddings"
MAX_BATCH_INPUTS = 50_000
MAX_BATCH_FILE_BYTES = 200_000_000
MAX_REQUEST_INPUTS = 2_048
MAX_REQUEST_TOKENS = 300_000
MAX_INPUT_TOKENS = 8_192
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


class BatchClient(Protocol):
    def upload_file(self, path: Path) -> str: ...

    def create_batch(self, *, input_file_id: str, metadata: dict[str, str]) -> dict: ...

    def retrieve_batch(self, batch_id: str) -> dict: ...

    def download_file(self, file_id: str) -> str: ...


class BatchSubmissionError(RuntimeError):
    def __init__(self, *, submitted: int, failures: list[tuple[str, str]]) -> None:
        self.submitted = submitted
        self.failures = failures
        jobs = ", ".join(batch_job_id for batch_job_id, _ in failures)
        super().__init__(f"Failed to submit {len(failures)} batches: {jobs}")


class OpenAIBatchClient:
    """Small REST client for the OpenAI files and batches endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        timeout: int = 120,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("Set OPENAI_API_KEY or pass --api-key")
        self.api_key = resolved_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.organization = organization
        self.project = project

    def upload_file(self, path: Path) -> str:
        requests = _requests()
        with path.open("rb") as handle:
            response = requests.post(
                f"{self.api_base}/files",
                headers=self._headers(content_type=False),
                data={"purpose": "batch"},
                files={"file": (path.name, handle, "application/jsonl")},
                timeout=self.timeout,
            )
        payload = _response_json(response)
        return _required_string(payload, "id")

    def create_batch(self, *, input_file_id: str, metadata: dict[str, str]) -> dict:
        requests = _requests()
        response = requests.post(
            f"{self.api_base}/batches",
            headers=self._headers(),
            json={
                "input_file_id": input_file_id,
                "endpoint": OPENAI_BATCH_ENDPOINT,
                "completion_window": "24h",
                "metadata": metadata,
            },
            timeout=self.timeout,
        )
        return _response_json(response)

    def retrieve_batch(self, batch_id: str) -> dict:
        requests = _requests()
        response = requests.get(
            f"{self.api_base}/batches/{batch_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        return _response_json(response)

    def download_file(self, file_id: str) -> str:
        requests = _requests()
        response = requests.get(
            f"{self.api_base}/files/{file_id}/content",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def _headers(self, *, content_type: bool = True) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            headers["Content-Type"] = "application/json"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers


@dataclass(frozen=True)
class PreparedBatch:
    batch_job_id: str
    batch_index: int
    input_file_path: Path
    request_count: int
    input_count: int


@dataclass(frozen=True)
class CollectionCounts:
    batches: int = 0
    imported: int = 0
    failed: int = 0


def make_token_counter(model_id: str) -> Callable[[str], int]:
    try:
        import tiktoken  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required to validate OpenAI embedding input limits"
        ) from exc

    try:
        encoding = tiktoken.encoding_for_model(model_id)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text))


def prepare_batches(
    db: Database,
    *,
    embedding_run_id: str,
    chunking_run_id: str,
    model_id: str,
    artifact_dir: str | Path,
    request_size: int = 128,
    max_batch_inputs: int = MAX_BATCH_INPUTS,
    max_batch_bytes: int = 190_000_000,
    max_characters: int | None = None,
    limit: int | None = None,
    dimensions: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> list[PreparedBatch]:
    _validate_limits(
        request_size=request_size,
        max_batch_inputs=max_batch_inputs,
        max_batch_bytes=max_batch_bytes,
        dimensions=dimensions,
    )
    counter = token_counter or make_token_counter(model_id)
    chunks = db.get_unembedded_chunks(
        embedding_run_id=embedding_run_id,
        chunking_run_id=chunking_run_id,
        max_characters=max_characters,
    )
    active_chunk_ids = {
        row["chunk_id"]
        for row in db.conn.execute(
            """
            SELECT DISTINCT chunk_id
            FROM openai_embedding_batch_items
            WHERE embedding_run_id = ? AND status IN ('pending', 'imported')
            """,
            (embedding_run_id,),
        )
    }
    chunks = [row for row in chunks if row["chunk_id"] not in active_chunk_ids]
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        return []

    request_lines = _build_request_lines(
        chunks,
        model_id=model_id,
        request_size=request_size,
        dimensions=dimensions,
        token_counter=counter,
    )
    packed = _pack_request_lines(
        request_lines,
        max_batch_inputs=max_batch_inputs,
        max_batch_bytes=max_batch_bytes,
    )
    first_index_row = db.conn.execute(
        """
        SELECT COALESCE(MAX(batch_index), -1) + 1 AS next_index
        FROM openai_embedding_batches
        WHERE embedding_run_id = ?
        """,
        (embedding_run_id,),
    ).fetchone()
    first_index = int(first_index_row["next_index"])
    run_dir = Path(artifact_dir) / embedding_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[PreparedBatch] = []
    for offset, batch_lines in enumerate(packed):
        batch_index = first_index + offset
        batch_job_id = f"{embedding_run_id}_openai_{batch_index:05d}"
        input_path = run_dir / f"batch_{batch_index:05d}.input.jsonl"
        rendered = "".join(line.rendered for line in batch_lines)
        input_path.write_text(rendered, encoding="utf-8", newline="\n")
        input_count = sum(len(line.chunk_ids) for line in batch_lines)

        with db.conn:
            db.conn.execute(
                """
                INSERT INTO openai_embedding_batches (
                    batch_job_id, embedding_run_id, batch_index, status,
                    input_file_path, request_count, input_count
                ) VALUES (?, ?, ?, 'prepared', ?, ?, ?)
                """,
                (
                    batch_job_id,
                    embedding_run_id,
                    batch_index,
                    str(input_path),
                    len(batch_lines),
                    input_count,
                ),
            )
            db.conn.executemany(
                """
                INSERT INTO openai_embedding_batch_items (
                    batch_job_id, embedding_run_id, custom_id,
                    input_index, chunk_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        batch_job_id,
                        embedding_run_id,
                        line.custom_id,
                        input_index,
                        chunk_id,
                    )
                    for line in batch_lines
                    for input_index, chunk_id in enumerate(line.chunk_ids)
                ],
            )
        prepared.append(
            PreparedBatch(
                batch_job_id=batch_job_id,
                batch_index=batch_index,
                input_file_path=input_path,
                request_count=len(batch_lines),
                input_count=input_count,
            )
        )
    return prepared


def submit_prepared_batches(
    db: Database,
    client: BatchClient,
    *,
    embedding_run_id: str,
    on_batch: Callable[[str], None] | None = None,
) -> int:
    rows = db.conn.execute(
        """
        SELECT *
        FROM openai_embedding_batches
        WHERE embedding_run_id = ? AND status IN ('prepared', 'uploaded')
        ORDER BY batch_index
        """,
        (embedding_run_id,),
    ).fetchall()
    submitted = 0
    failures: list[tuple[str, str]] = []
    for row in rows:
        try:
            input_file_id = row["input_file_id"]
            if not input_file_id:
                input_file_id = client.upload_file(Path(row["input_file_path"]))
                with db.conn:
                    db.conn.execute(
                        """
                        UPDATE openai_embedding_batches
                        SET input_file_id = ?, status = 'uploaded', error_json = NULL
                        WHERE batch_job_id = ?
                        """,
                        (input_file_id, row["batch_job_id"]),
                    )

            payload = client.create_batch(
                input_file_id=input_file_id,
                metadata={
                    "embedding_run_id": embedding_run_id,
                    "batch_job_id": row["batch_job_id"],
                },
            )
            provider_batch_id = _required_string(payload, "id")
            status = str(payload.get("status") or "validating")
            with db.conn:
                db.conn.execute(
                    """
                    UPDATE openai_embedding_batches
                    SET provider_batch_id = ?, status = ?, submitted_at = ?,
                        error_json = NULL
                    WHERE batch_job_id = ?
                    """,
                    (provider_batch_id, status, _utc_now(), row["batch_job_id"]),
                )
            submitted += 1
        except Exception as exc:
            message = str(exc)
            failures.append((row["batch_job_id"], message))
            with db.conn:
                db.conn.execute(
                    """
                    UPDATE openai_embedding_batches
                    SET error_json = ?
                    WHERE batch_job_id = ?
                    """,
                    (
                        json.dumps({"submission_error": message}, ensure_ascii=False),
                        row["batch_job_id"],
                    ),
                )
        finally:
            if on_batch:
                on_batch(row["batch_job_id"])
    if failures:
        raise BatchSubmissionError(submitted=submitted, failures=failures)
    return submitted


def refresh_batches(
    db: Database,
    client: BatchClient,
    *,
    embedding_run_id: str,
    on_batch: Callable[[str], None] | None = None,
) -> dict[str, int]:
    rows = db.conn.execute(
        """
        SELECT *
        FROM openai_embedding_batches
        WHERE embedding_run_id = ?
          AND provider_batch_id IS NOT NULL
          AND (
              status NOT IN ('completed', 'failed', 'expired', 'cancelled')
              OR (
                  status = 'completed'
                  AND output_file_id IS NULL
                  AND error_file_id IS NULL
              )
          )
        ORDER BY batch_index
        """,
        (embedding_run_id,),
    ).fetchall()
    for row in rows:
        payload = client.retrieve_batch(row["provider_batch_id"])
        status = str(payload.get("status") or row["status"])
        request_counts = payload.get("request_counts") or {}
        completed_at = _unix_timestamp(payload.get("completed_at"))
        error_payload = payload.get("errors")
        with db.conn:
            db.conn.execute(
                """
                UPDATE openai_embedding_batches
                SET status = ?, output_file_id = ?, error_file_id = ?,
                    completed_request_count = ?, failed_request_count = ?,
                    error_json = ?, last_polled_at = ?,
                    completed_at = COALESCE(?, completed_at)
                WHERE batch_job_id = ?
                """,
                (
                    status,
                    payload.get("output_file_id"),
                    payload.get("error_file_id"),
                    int(request_counts.get("completed") or 0),
                    int(request_counts.get("failed") or 0),
                    json.dumps(error_payload, ensure_ascii=False) if error_payload else None,
                    _utc_now(),
                    completed_at,
                    row["batch_job_id"],
                ),
            )
        if on_batch:
            on_batch(row["batch_job_id"])
    return batch_status_counts(db, embedding_run_id=embedding_run_id)


def collect_completed_batches(
    db: Database,
    client: BatchClient,
    *,
    embedding_run_id: str,
    on_batch: Callable[[str], None] | None = None,
) -> CollectionCounts:
    rows = db.conn.execute(
        """
        SELECT *
        FROM openai_embedding_batches
        WHERE embedding_run_id = ?
          AND status IN ('completed', 'failed', 'expired', 'cancelled')
          AND imported_at IS NULL
        ORDER BY batch_index
        """,
        (embedding_run_id,),
    ).fetchall()
    imported_total = 0
    failed_total = 0
    collected_batches = 0
    for row in rows:
        input_path = Path(row["input_file_path"])
        output_text = ""
        error_text = ""
        if row["output_file_id"]:
            output_text = client.download_file(row["output_file_id"])
            input_path.with_suffix(".output.jsonl").write_text(
                output_text, encoding="utf-8", newline="\n"
            )
        if row["error_file_id"]:
            error_text = client.download_file(row["error_file_id"])
            input_path.with_suffix(".errors.jsonl").write_text(
                error_text, encoding="utf-8", newline="\n"
            )

        item_rows = db.conn.execute(
            """
            SELECT custom_id, input_index, chunk_id
            FROM openai_embedding_batch_items
            WHERE batch_job_id = ? AND status = 'pending'
            """,
            (row["batch_job_id"],),
        ).fetchall()
        item_map = {
            (item["custom_id"], int(item["input_index"])): item["chunk_id"]
            for item in item_rows
        }
        embeddings = []
        imported_keys: list[tuple[str, int]] = []
        errors: dict[tuple[str, int], object] = {}

        for result in _parse_jsonl(output_text):
            custom_id = _required_string(result, "custom_id")
            response = result.get("response")
            if not isinstance(response, dict) or int(response.get("status_code") or 0) != 200:
                _mark_request_error(errors, item_map, custom_id, result)
                continue
            body = response.get("body")
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list):
                _mark_request_error(errors, item_map, custom_id, result)
                continue
            for value in data:
                if not isinstance(value, dict):
                    continue
                input_index = int(value.get("index", -1))
                key = (custom_id, input_index)
                chunk_id = item_map.get(key)
                vector = value.get("embedding")
                if chunk_id is None or not isinstance(vector, list):
                    raise ValueError(
                        f"Unmappable embedding result {custom_id!r} index {input_index}"
                    )
                embeddings.append(
                    build_embedding(
                        embedding_run_id=embedding_run_id,
                        chunk_id=chunk_id,
                        vector=vector,
                    )
                )
                imported_keys.append(key)

        for result in _parse_jsonl(error_text):
            custom_id = _required_string(result, "custom_id")
            _mark_request_error(errors, item_map, custom_id, result)

        for key in imported_keys:
            errors.pop(key, None)

        handled = set(imported_keys) | set(errors)
        for key in item_map.keys() - handled:
            errors[key] = {"message": "No result was returned for this input"}

        with db.conn:
            db.conn.executemany(
                """
                INSERT INTO embeddings (
                    embedding_id, embedding_run_id, chunk_id, tensor_blob, is_valid
                ) VALUES (
                    ?, ?, ?, ?,
                    COALESCE(
                        (
                            SELECT articles.is_valid
                            FROM chunks
                            JOIN articles ON articles.article_id = chunks.article_id
                            WHERE chunks.chunk_id = ?
                        ),
                        0
                    )
                )
                ON CONFLICT(embedding_id) DO NOTHING
                """,
                [
                    (
                        embedding.embedding_id,
                        embedding.embedding_run_id,
                        embedding.chunk_id,
                        embedding.tensor_blob,
                        embedding.chunk_id,
                    )
                    for embedding in embeddings
                ],
            )
            db.conn.executemany(
                """
                UPDATE openai_embedding_batch_items
                SET status = 'imported', imported_at = ?, error_json = NULL
                WHERE batch_job_id = ? AND custom_id = ? AND input_index = ?
                """,
                [
                    (_utc_now(), row["batch_job_id"], custom_id, input_index)
                    for custom_id, input_index in imported_keys
                ],
            )
            db.conn.executemany(
                """
                UPDATE openai_embedding_batch_items
                SET status = 'error', error_json = ?
                WHERE batch_job_id = ? AND custom_id = ? AND input_index = ?
                """,
                [
                    (
                        json.dumps(error, ensure_ascii=False),
                        row["batch_job_id"],
                        custom_id,
                        input_index,
                    )
                    for (custom_id, input_index), error in errors.items()
                ],
            )
            db.conn.execute(
                """
                UPDATE openai_embedding_batches
                SET imported_at = ?
                WHERE batch_job_id = ?
                """,
                (_utc_now(), row["batch_job_id"]),
            )
        imported_total += len(imported_keys)
        failed_total += len(errors)
        collected_batches += 1
        if on_batch:
            on_batch(row["batch_job_id"])
    return CollectionCounts(
        batches=collected_batches,
        imported=imported_total,
        failed=failed_total,
    )


def batch_status_counts(db: Database, *, embedding_run_id: str) -> dict[str, int]:
    rows = db.conn.execute(
        """
        SELECT status, COUNT(*) AS batch_count
        FROM openai_embedding_batches
        WHERE embedding_run_id = ?
        GROUP BY status
        """,
        (embedding_run_id,),
    ).fetchall()
    return {row["status"]: int(row["batch_count"]) for row in rows}


def all_batches_terminal(db: Database, *, embedding_run_id: str) -> bool:
    row = db.conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status IN ('completed', 'failed', 'expired', 'cancelled')
                     THEN 1 ELSE 0 END) AS terminal
        FROM openai_embedding_batches
        WHERE embedding_run_id = ?
        """,
        (embedding_run_id,),
    ).fetchone()
    return int(row["total"] or 0) > 0 and int(row["terminal"] or 0) == int(row["total"])


@dataclass(frozen=True)
class _RequestLine:
    custom_id: str
    chunk_ids: tuple[str, ...]
    rendered: str

    @property
    def byte_count(self) -> int:
        return len(self.rendered.encode("utf-8"))


def _build_request_lines(
    chunks: Iterable,
    *,
    model_id: str,
    request_size: int,
    dimensions: int | None,
    token_counter: Callable[[str], int],
) -> list[_RequestLine]:
    groups: list[list] = []
    current: list = []
    current_tokens = 0
    for chunk in chunks:
        token_count = token_counter(chunk["text"])
        if token_count < 1:
            raise ValueError(f"Chunk {chunk['chunk_id']} has empty embedding input")
        if token_count > MAX_INPUT_TOKENS:
            raise ValueError(
                f"Chunk {chunk['chunk_id']} has {token_count} tokens; maximum is {MAX_INPUT_TOKENS}"
            )
        if current and (
            len(current) >= request_size
            or current_tokens + token_count > MAX_REQUEST_TOKENS
        ):
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += token_count
    if current:
        groups.append(current)

    lines: list[_RequestLine] = []
    for ordinal, group in enumerate(groups):
        custom_id = f"embedding-request-{ordinal:08d}"
        body: dict[str, object] = {
            "model": model_id,
            "input": [chunk["text"] for chunk in group],
            "encoding_format": "float",
        }
        if dimensions is not None:
            body["dimensions"] = dimensions
        payload = {
            "custom_id": custom_id,
            "method": "POST",
            "url": OPENAI_BATCH_ENDPOINT,
            "body": body,
        }
        lines.append(
            _RequestLine(
                custom_id=custom_id,
                chunk_ids=tuple(chunk["chunk_id"] for chunk in group),
                rendered=json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            )
        )
    return lines


def _pack_request_lines(
    lines: Iterable[_RequestLine],
    *,
    max_batch_inputs: int,
    max_batch_bytes: int,
) -> list[list[_RequestLine]]:
    batches: list[list[_RequestLine]] = []
    current: list[_RequestLine] = []
    current_inputs = 0
    current_bytes = 0
    for line in lines:
        line_inputs = len(line.chunk_ids)
        if line_inputs > max_batch_inputs or line.byte_count > max_batch_bytes:
            raise ValueError("A single embeddings request exceeds the configured batch limits")
        if current and (
            current_inputs + line_inputs > max_batch_inputs
            or current_bytes + line.byte_count > max_batch_bytes
        ):
            batches.append(current)
            current = []
            current_inputs = 0
            current_bytes = 0
        current.append(line)
        current_inputs += line_inputs
        current_bytes += line.byte_count
    if current:
        batches.append(current)
    return batches


def _validate_limits(
    *,
    request_size: int,
    max_batch_inputs: int,
    max_batch_bytes: int,
    dimensions: int | None,
) -> None:
    if not 1 <= request_size <= MAX_REQUEST_INPUTS:
        raise ValueError(f"request-size must be between 1 and {MAX_REQUEST_INPUTS}")
    if not 1 <= max_batch_inputs <= MAX_BATCH_INPUTS:
        raise ValueError(f"max-batch-inputs must be between 1 and {MAX_BATCH_INPUTS}")
    if not 1 <= max_batch_bytes <= MAX_BATCH_FILE_BYTES:
        raise ValueError(f"max-batch-bytes must be between 1 and {MAX_BATCH_FILE_BYTES}")
    if dimensions is not None and dimensions < 1:
        raise ValueError("dimensions must be positive")


def _mark_request_error(
    errors: dict[tuple[str, int], object],
    item_map: dict[tuple[str, int], str],
    custom_id: str,
    error: object,
) -> None:
    matching_keys = [key for key in item_map if key[0] == custom_id]
    if not matching_keys:
        raise ValueError(f"Unknown custom_id in batch result: {custom_id}")
    for key in matching_keys:
        errors[key] = error


def _parse_jsonl(text: str) -> list[dict]:
    parsed: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object on line {line_number}")
        parsed.append(value)
    return parsed


def _requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError("requests is required for OpenAI batch operations") from exc
    return requests


def _response_json(response) -> dict:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OpenAI returned a non-object JSON response")
    return payload


def _required_string(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string field {key!r}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unix_timestamp(value) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
