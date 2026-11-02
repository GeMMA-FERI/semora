"""Optional authenticated HTTP worker: python -m semora.execution.worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse
from fastapi.responses import Response as HTTPResponse
from starlette.requests import ClientDisconnect

from .contracts import Request
from .runtime import NERWorker, WorkerError, WorkerSettings


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object fields are not supported.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON values are not supported.")


def create_app(settings: WorkerSettings, *, token: str, worker_factory: Callable = NERWorker) -> FastAPI:
    """Create a worker without loading its model until the application lifespan."""
    if not isinstance(token, str) or len(token) < 32 or any(char.isspace() for char in token):
        raise ValueError("Worker token must have at least 32 characters and contain no whitespace.")
    worker = worker_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await worker.start()
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(title="Semora CLASSLA worker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.worker = worker

    def authenticate(request: HTTPRequest) -> None:
        expected = f"Bearer {token}".encode()
        received = request.headers.get("authorization", "").encode("utf-8")
        if not secrets.compare_digest(received, expected):
            raise WorkerError(401, "unauthorized", "A valid bearer token is required.")

    @app.exception_handler(WorkerError)
    async def worker_error(request: HTTPRequest, error: WorkerError):
        headers = {"Retry-After": "1"} if error.status == 503 else None
        if error.status == 401:
            headers = {"WWW-Authenticate": "Bearer"}
        return JSONResponse({"error": {"code": error.code, "message": str(error)}}, error.status, headers=headers)

    @app.get("/health/live")
    async def live():
        return {"status": "alive", "api_version": "v1"}

    @app.get("/v1/capabilities")
    async def capabilities(request: HTTPRequest):
        authenticate(request)
        return worker.capabilities()

    async def read_request(request: HTTPRequest) -> Request:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise WorkerError(415, "unsupported_media_type", "Use application/json.")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
            except ValueError:
                raise WorkerError(400, "invalid_request", "Invalid Content-Length.") from None
            if size < 0:
                raise WorkerError(400, "invalid_request", "Invalid Content-Length.")
            if size > settings.max_request_bytes:
                raise WorkerError(413, "request_too_large", "Request body exceeds the byte limit.")

        async def read_limited() -> bytes:
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > settings.max_request_bytes:
                    raise WorkerError(413, "request_too_large", "Request body exceeds the byte limit.")
                body.extend(chunk)
            return bytes(body)

        try:
            body = await asyncio.wait_for(read_limited(), timeout=settings.body_timeout)
            value = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            return Request.from_dict(value)
        except asyncio.TimeoutError:
            raise WorkerError(408, "body_timeout", "Request body timed out.") from None
        except (ValueError, UnicodeError, RecursionError, ClientDisconnect):
            raise WorkerError(400, "invalid_request", "Request does not match the v1 batch contract.") from None

    @app.post("/v1/ner")
    async def ner(request: HTTPRequest):
        authenticate(request)
        batch = await read_request(request)
        encoded = await worker.infer(batch)
        return HTTPResponse(encoded, media_type="application/json")

    @app.post("/v1/lemma")
    async def lemma(request: HTTPRequest):
        authenticate(request)
        raise WorkerError(501, "unsupported_task", "Lemmatization is not enabled on this worker.")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve optional CLASSLA NER inference without corpus/database access.")
    parser.add_argument("--resources", type=Path, required=True, help="Existing CLASSLA resources directory.")
    parser.add_argument(
        "--token-file", type=Path, required=True, help="File containing a bearer token (32+ characters)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--package", default="default")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decoder", choices=("numpy", "numba"), default="numpy")
    parser.add_argument("--tokenizer-workers", type=int, default=0)
    parser.add_argument("--ner-pipeline-depth", type=int, default=1)
    parser.add_argument("--max-documents", type=int, default=32)
    parser.add_argument("--max-input-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-request-bytes", type=int, default=2_097_152)
    parser.add_argument("--max-response-bytes", type=int, default=4_194_304)
    parser.add_argument("--body-timeout", type=float, default=30.0)
    parser.add_argument(
        "--expected-fingerprint", help="Fail startup unless resource/code identity matches this SHA-256."
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    settings = WorkerSettings(
        resources=args.resources.resolve(strict=True), device=args.device, package=args.package,
        batch_size=args.batch_size, decoder=args.decoder, tokenizer_workers=args.tokenizer_workers,
        ner_pipeline_depth=args.ner_pipeline_depth, max_documents=args.max_documents,
        max_input_bytes=args.max_input_bytes, max_request_bytes=args.max_request_bytes,
        max_response_bytes=args.max_response_bytes, body_timeout=args.body_timeout,
        expected_fingerprint=args.expected_fingerprint,
    )
    token = args.token_file.read_text(encoding="utf-8").strip()
    app = create_app(settings, token=token)
    import uvicorn

    # One process owns one model. Bound concurrent body readers and disable raw URL access logs.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, limit_concurrency=32, access_log=False)


if __name__ == "__main__":
    main()
