from __future__ import annotations

import json
import pathlib
import struct

from semora.embeddings.openai_batch import (
    BatchSubmissionError,
    collect_completed_batches,
    prepare_batches,
    refresh_batches,
    submit_prepared_batches,
)
from semora.storage import (
    Article,
    Chunk,
    ChunkingRun,
    Database,
    EmbeddingRun,
    Newspaper,
    Run,
)


class FakeBatchClient:
    def __init__(self) -> None:
        self.uploaded: list[pathlib.Path] = []
        self.created: list[tuple[str, dict[str, str]]] = []
        self.batches: dict[str, dict] = {}
        self.files: dict[str, str] = {}

    def upload_file(self, path: pathlib.Path) -> str:
        self.uploaded.append(path)
        return f"file-input-{len(self.uploaded)}"

    def create_batch(self, *, input_file_id: str, metadata: dict[str, str]) -> dict:
        self.created.append((input_file_id, metadata))
        batch_id = f"batch-{len(self.created)}"
        payload = {"id": batch_id, "status": "validating"}
        self.batches[batch_id] = payload
        return payload

    def retrieve_batch(self, batch_id: str) -> dict:
        return self.batches[batch_id]

    def download_file(self, file_id: str) -> str:
        return self.files[file_id]


def test_migration_adds_openai_batch_tracking_tables(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "batches.sqlite")
    try:
        db.initialize()
        tables = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "openai_embedding_batches" in tables
        assert "openai_embedding_batch_items" in tables

        indexes = {
            row["name"]
            for row in db.conn.execute(
                "PRAGMA index_list(openai_embedding_batch_items)"
            )
        }
        assert "idx_openai_embedding_batch_items_batch_status" in indexes
        assert "idx_openai_embedding_batch_items_run_chunk_status" in indexes
    finally:
        db.close()


def test_multiple_batches_can_be_submitted_polled_and_collected_later(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "batches.sqlite")
    client = FakeBatchClient()
    try:
        _seed_chunks(db, count=3)
        prepared = prepare_batches(
            db,
            embedding_run_id="embedding-run",
            chunking_run_id="chunking-run",
            model_id="text-embedding-3-small",
            artifact_dir=tmp_path / "artifacts",
            request_size=2,
            max_batch_inputs=2,
            max_batch_bytes=10_000,
            token_counter=len,
        )

        assert len(prepared) == 2
        assert [batch.input_count for batch in prepared] == [2, 1]
        first_lines = _read_jsonl(prepared[0].input_file_path)
        assert first_lines[0]["url"] == "/v1/embeddings"
        assert first_lines[0]["body"]["model"] == "text-embedding-3-small"
        assert first_lines[0]["body"]["input"] == ["text 0", "text 1"]

        assert submit_prepared_batches(
            db, client, embedding_run_id="embedding-run"
        ) == 2
        assert len(client.uploaded) == 2
        assert len(client.created) == 2

        expected_vectors = {
            "chunk-0": [0.0, 0.5],
            "chunk-1": [1.0, 1.5],
            "chunk-2": [2.0, 2.5],
        }
        for batch_row in db.conn.execute(
            "SELECT * FROM openai_embedding_batches ORDER BY batch_index"
        ).fetchall():
            output_file_id = f"file-output-{batch_row['batch_index']}"
            client.batches[batch_row["provider_batch_id"]] = {
                "id": batch_row["provider_batch_id"],
                "status": "completed",
                "output_file_id": output_file_id,
                "error_file_id": None,
                "completed_at": 1_700_000_000,
                "request_counts": {"completed": batch_row["request_count"], "failed": 0},
            }
            client.files[output_file_id] = _successful_output(
                db, batch_row["batch_job_id"], expected_vectors
            )

        statuses = refresh_batches(db, client, embedding_run_id="embedding-run")
        assert statuses == {"completed": 2}

        counts = collect_completed_batches(
            db, client, embedding_run_id="embedding-run"
        )
        assert (counts.batches, counts.imported, counts.failed) == (2, 3, 0)
        assert collect_completed_batches(
            db, client, embedding_run_id="embedding-run"
        ).batches == 0

        stored = db.conn.execute(
            "SELECT chunk_id, tensor_blob FROM embeddings ORDER BY chunk_id"
        ).fetchall()
        assert len(stored) == 3
        assert {
            row["chunk_id"]: list(struct.unpack("<2f", row["tensor_blob"]))
            for row in stored
        } == expected_vectors
        assert {
            row["status"]
            for row in db.conn.execute(
                "SELECT status FROM openai_embedding_batch_items"
            )
        } == {"imported"}
    finally:
        db.close()


def test_failed_terminal_batch_is_recorded_and_can_be_prepared_again(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "batches.sqlite")
    client = FakeBatchClient()
    try:
        _seed_chunks(db, count=1)
        prepared = prepare_batches(
            db,
            embedding_run_id="embedding-run",
            chunking_run_id="chunking-run",
            model_id="text-embedding-3-large",
            artifact_dir=tmp_path / "artifacts",
            token_counter=len,
        )
        assert len(prepared) == 1
        submit_prepared_batches(db, client, embedding_run_id="embedding-run")
        client.batches["batch-1"] = {
            "id": "batch-1",
            "status": "failed",
            "errors": {"data": [{"message": "validation failed"}]},
            "request_counts": {"completed": 0, "failed": 1},
        }

        assert refresh_batches(db, client, embedding_run_id="embedding-run") == {
            "failed": 1
        }
        counts = collect_completed_batches(
            db, client, embedding_run_id="embedding-run"
        )
        assert (counts.imported, counts.failed) == (0, 1)
        assert db.conn.execute(
            "SELECT status FROM openai_embedding_batch_items"
        ).fetchone()["status"] == "error"

        retried = prepare_batches(
            db,
            embedding_run_id="embedding-run",
            chunking_run_id="chunking-run",
            model_id="text-embedding-3-large",
            artifact_dir=tmp_path / "artifacts",
            token_counter=len,
        )
        assert len(retried) == 1
        assert retried[0].batch_index == 1
    finally:
        db.close()


def test_submission_attempts_later_batches_after_one_failure(
    tmp_path: pathlib.Path,
) -> None:
    class PartiallyFailingClient(FakeBatchClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def create_batch(self, *, input_file_id: str, metadata: dict[str, str]) -> dict:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary submission failure")
            return super().create_batch(input_file_id=input_file_id, metadata=metadata)

    db = Database(tmp_path / "batches.sqlite")
    client = PartiallyFailingClient()
    try:
        _seed_chunks(db, count=2)
        prepare_batches(
            db,
            embedding_run_id="embedding-run",
            chunking_run_id="chunking-run",
            model_id="text-embedding-3-small",
            artifact_dir=tmp_path / "artifacts",
            request_size=1,
            max_batch_inputs=1,
            token_counter=len,
        )
        try:
            submit_prepared_batches(db, client, embedding_run_id="embedding-run")
        except BatchSubmissionError as error:
            assert error.submitted == 1
            assert len(error.failures) == 1
        else:
            raise AssertionError("Expected the partial submission failure to be reported")

        rows = db.conn.execute(
            """
            SELECT status, provider_batch_id, error_json
            FROM openai_embedding_batches
            ORDER BY batch_index
            """
        ).fetchall()
        assert rows[0]["status"] == "uploaded"
        assert "temporary submission failure" in rows[0]["error_json"]
        assert rows[1]["status"] == "validating"
        assert rows[1]["provider_batch_id"] == "batch-1"
    finally:
        db.close()


def test_prepare_rejects_an_oversized_input(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "batches.sqlite")
    try:
        _seed_chunks(db, count=1)
        try:
            prepare_batches(
                db,
                embedding_run_id="embedding-run",
                chunking_run_id="chunking-run",
                model_id="text-embedding-3-small",
                artifact_dir=tmp_path / "artifacts",
                token_counter=lambda _: 8_193,
            )
        except ValueError as error:
            assert "maximum is 8192" in str(error)
        else:
            raise AssertionError("Expected the oversized input to be rejected")
    finally:
        db.close()


def _seed_chunks(db: Database, *, count: int) -> None:
    db.initialize()
    db.insert_run(Run(run_id="run", run_type="test"))
    db.insert_newspaper(
        Newspaper(newspaper_id="newspaper", run_id="run", content="newspaper")
    )
    db.insert_article(
        Article(
            article_id="article",
            run_id="run",
            newspaper_id="newspaper",
            title="article",
            content="article",
        )
    )
    db.update_article_validation("article", is_valid=True)
    db.insert_chunking_run(
        ChunkingRun(chunking_run_id="chunking-run", run_id="run", method="test")
    )
    db.insert_chunks(
        [
            Chunk(
                chunk_id=f"chunk-{index}",
                run_id="run",
                article_id="article",
                chunking_run_id="chunking-run",
                chunk_index=index,
                method="test",
                text=f"text {index}",
            )
            for index in range(count)
        ]
    )
    db.insert_embedding_run(
        EmbeddingRun(
            embedding_run_id="embedding-run",
            run_id="run",
            model_id="text-embedding-3-small",
            config={"chunking_run_id": "chunking-run"},
        )
    )


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _successful_output(
    db: Database,
    batch_job_id: str,
    expected_vectors: dict[str, list[float]],
) -> str:
    rows = db.conn.execute(
        """
        SELECT custom_id, input_index, chunk_id
        FROM openai_embedding_batch_items
        WHERE batch_job_id = ?
        ORDER BY custom_id, input_index DESC
        """,
        (batch_job_id,),
    ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["custom_id"], []).append(
            {
                "object": "embedding",
                "index": row["input_index"],
                "embedding": expected_vectors[row["chunk_id"]],
            }
        )
    results = [
        {
            "id": f"result-{custom_id}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": f"request-{custom_id}",
                "body": {"object": "list", "data": values},
            },
            "error": None,
        }
        for custom_id, values in reversed(grouped.items())
    ]
    return "\n".join(json.dumps(result) for result in results) + "\n"
