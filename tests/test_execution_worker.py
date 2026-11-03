from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from semora.execution.contracts import Document, Request, Response  # noqa: E402
from semora.execution.runtime import NERWorker, WorkerError, WorkerSettings  # noqa: E402
from semora.execution.worker import create_app  # noqa: E402
from semora.ner.base import Entity, Prediction  # noqa: E402

TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeModel:
    def __init__(self, *, fail_warmup=False):
        self.events = [("create", threading.get_ident())]
        self.fail_warmup = fail_warmup
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False

    def predict(self, texts, batch_size):
        self.events.append(("predict", threading.get_ident()))
        if self.fail_warmup or self.fail:
            raise RuntimeError("private OCR content")
        if self.block:
            self.entered.set()
            assert self.release.wait(5)
        return Prediction([[Entity(0, 3, "PER", text[:3])] if text.startswith("Ana") else [] for text in texts])

    def close(self):
        self.events.append(("close", threading.get_ident()))


def make_app(tmp_path: Path, **options):
    settings = WorkerSettings(resources=tmp_path, **options)
    models = []

    def factory(_settings):
        model = FakeModel()
        models.append(model)
        return model

    app = create_app(
        settings, token=TOKEN,
        worker_factory=lambda config: NERWorker(
            config, adapter_factory=factory, identity_factory=lambda model: {"format": "fake.ner.v1"},
        ),
    )
    return app, models


def batch(client, texts=("Ana č😀", "", "nobody"), *, task="ner", fingerprint=None):
    capability = client.get("/v1/capabilities", headers=AUTH).json()
    return Request(
        "request-1", task, fingerprint or capability["tasks"]["ner"]["fingerprint"],
        tuple(Document.create(str(index), text) for index, text in enumerate(texts)),
    )


def test_lifecycle_auth_and_unicode_results(tmp_path: Path) -> None:
    app, models = make_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health/live").json()["status"] == "alive"
        assert client.get("/v1/capabilities").status_code == 401
        capabilities = client.get("/v1/capabilities", headers=AUTH).json()
        assert capabilities["ready"] and not capabilities["tasks"]["lemma"]["enabled"]
        assert client.post("/v1/lemma", headers=AUTH).status_code == 501
        request = batch(client)
        response = client.post("/v1/ner", headers=AUTH, json=request.to_dict())
        assert response.status_code == 200
        ordered = Response.from_dict(response.json()).validate_against(request)
        assert [list(item.entities) for item in ordered] == [[Entity(0, 3, "PER", "Ana")], [], []]
        empty = batch(client, ())
        assert Response.from_dict(client.post("/v1/ner", headers=AUTH, json=empty.to_dict()).json()).validate_against(
            empty
        ) == ()
    assert [event[0] for event in models[0].events] == ["create", "predict", "predict", "predict", "close"]
    assert len({event[1] for event in models[0].events}) == 1


def test_validation_limits_and_sanitized_errors(tmp_path: Path) -> None:
    app, models = make_app(tmp_path, max_documents=2, max_input_bytes=5, max_request_bytes=1024)
    with TestClient(app) as client:
        request = batch(client, ("Ana",))
        assert client.post("/v1/ner", headers=AUTH, json=batch(client, ("a", "b", "c")).to_dict()).status_code == 413
        assert client.post("/v1/ner", headers=AUTH, json=batch(client, ("😀😀",)).to_dict()).status_code == 413
        incompatible = batch(client, ("a",), fingerprint="a" * 64).to_dict()
        assert client.post("/v1/ner", headers=AUTH, json=incompatible).status_code == 409
        unsupported = batch(client, ("a",), task="lemma").to_dict()
        assert client.post("/v1/ner", headers=AUTH, json=unsupported).status_code == 501
        assert client.post("/v1/ner", headers=AUTH, content=b"x" * 1025).status_code == 415
        json_headers = {**AUTH, "Content-Type": "application/json"}
        assert client.post("/v1/ner", headers=json_headers, content=b"x" * 1025).status_code == 413
        assert client.post("/v1/ner", headers=AUTH, content="{}",).status_code == 415
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":'):
            assert client.post("/v1/ner", headers=json_headers, content=raw).status_code == 400
        invalid = request.to_dict()
        invalid["documents"][0]["content_sha256"] = "a" * 64
        assert client.post("/v1/ner", headers=AUTH, json=invalid).status_code == 400
        duplicate = request.to_dict()
        duplicate["documents"].append(duplicate["documents"][0])
        assert client.post("/v1/ner", headers=AUTH, json=duplicate).status_code == 400
        models[0].fail = True
        response = client.post("/v1/ner", headers=AUTH, json=request.to_dict())
        assert response.status_code == 500 and "private OCR content" not in response.text


def test_response_limit_and_startup_failure_close(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path, max_response_bytes=10)
    with TestClient(app) as client:
        response = client.post("/v1/ner", headers=AUTH, json=batch(client, ("Ana",)).to_dict())
        assert response.status_code == 413 and response.json()["error"]["code"] == "response_too_large"

    models = []

    def failing(_settings):
        model = FakeModel(fail_warmup=True)
        models.append(model)
        return model

    app = create_app(
        WorkerSettings(resources=tmp_path), token=TOKEN,
        worker_factory=lambda settings: NERWorker(
            settings, adapter_factory=failing, identity_factory=lambda model: {"format": "fake.ner.v1"},
        ),
    )
    with pytest.raises(RuntimeError, match="private OCR"):
        with TestClient(app):
            pass
    assert [name for name, _ in models[0].events] == ["create", "predict", "close"]
    assert len({thread for _, thread in models[0].events}) == 1


def test_busy_service_keeps_health_responsive(tmp_path: Path) -> None:
    app, models = make_app(tmp_path)
    with TestClient(app) as client:
        request = batch(client, ("Ana",)).to_dict()
        model = models[0]
        model.block = True
        responses = []
        thread = threading.Thread(target=lambda: responses.append(client.post("/v1/ner", headers=AUTH, json=request)))
        thread.start()
        try:
            assert model.entered.wait(5)
            assert client.get("/health/live").status_code == 200
            assert client.get("/v1/capabilities", headers=AUTH).json()["capacity"]["in_flight"] == 1
            busy = client.post("/v1/ner", headers=AUTH, json=request)
            assert busy.status_code == 503 and busy.json()["error"]["code"] == "busy"
        finally:
            model.release.set()
            thread.join(5)
        assert not thread.is_alive() and responses[0].status_code == 200


def test_cancellation_keeps_worker_busy_until_prediction_finishes(tmp_path: Path) -> None:
    async def run():
        models = []

        def factory(_settings):
            model = FakeModel()
            models.append(model)
            return model

        worker = NERWorker(
            WorkerSettings(resources=tmp_path), adapter_factory=factory,
            identity_factory=lambda model: {"format": "fake.ner.v1"},
        )
        await worker.start()
        model = models[0]
        model.block = True
        request = Request("id", "ner", worker.fingerprint, (Document.create("1", "Ana"),))
        task = asyncio.create_task(worker.infer(request))
        try:
            assert await asyncio.to_thread(model.entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(WorkerError, match="busy") as error:
                await worker.infer(request)
            assert error.value.status == 503
        finally:
            model.release.set()
            await worker.close()

    asyncio.run(run())


def test_streaming_body_limit_and_timeout(tmp_path: Path) -> None:
    async def run():
        app, _ = make_app(tmp_path, max_request_bytes=32, body_timeout=0.02)

        async def chunks(*, slow=False):
            yield b"{" * 20
            if slow:
                await asyncio.sleep(0.1)
            yield b"}" * 20

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                headers = {**AUTH, "Content-Type": "application/json"}
                assert (await client.post("/v1/ner", headers=headers, content=chunks())).status_code == 413
                assert (await client.post("/v1/ner", headers=headers, content=chunks(slow=True))).status_code == 408

    asyncio.run(run())


def test_expected_identity_rejects_startup_and_closes_model(tmp_path: Path) -> None:
    app, models = make_app(tmp_path, expected_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        with TestClient(app):
            pass
    assert [name for name, _ in models[0].events] == ["create", "close"]


def test_cuda_request_does_not_fall_back_to_cpu(tmp_path: Path, monkeypatch) -> None:
    import sys

    from semora.execution.runtime import _load_adapter

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _load_adapter(WorkerSettings(resources=tmp_path, device="cuda"))


def test_tokenizer_stdout_is_counted_without_retaining_ocr(tmp_path: Path, capsys) -> None:
    class NoisyModel(FakeModel):
        def predict(self, texts, batch_size):
            print("Warning: Cannot compute token index. private OCR paragraph")
            return super().predict(texts, batch_size)

    app = create_app(
        WorkerSettings(resources=tmp_path), token=TOKEN,
        worker_factory=lambda settings: NERWorker(
            settings, adapter_factory=lambda _: NoisyModel(),
            identity_factory=lambda _: {"format": "fake.ner.v1"},
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/ner", headers=AUTH, json=batch(client, ("Ana",)).to_dict())
        assert response.status_code == 200
        assert client.get("/v1/capabilities", headers=AUTH).json()["counters"]["token_alignment_warnings"] == 1
    assert "private OCR paragraph" not in capsys.readouterr().out


def test_startup_cancellation_still_closes_model_on_owner_thread(tmp_path: Path) -> None:
    async def run():
        models = []

        def factory(_settings):
            model = FakeModel()
            model.block = True
            models.append(model)
            return model

        worker = NERWorker(
            WorkerSettings(resources=tmp_path), adapter_factory=factory,
            identity_factory=lambda _: {"format": "fake.ner.v1"},
        )
        startup = asyncio.create_task(worker.start())
        while not models:
            await asyncio.sleep(0)
        model = models[0]
        try:
            assert await asyncio.to_thread(model.entered.wait, 5)
            startup.cancel()
        finally:
            model.release.set()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert not worker.ready
        assert [name for name, _ in model.events] == ["create", "predict", "close"]
        assert len({thread for _, thread in model.events}) == 1

    asyncio.run(run())


def test_worker_counts_parallel_tokenizer_warnings(tmp_path):
    app, models = make_app(tmp_path)
    with TestClient(app) as client:
        model = models[0]
        original = model.predict
        model.token_alignment_warnings = 0

        def predict(texts, batch_size):
            model.token_alignment_warnings += 3
            return original(texts, batch_size)

        model.predict = predict
        request = batch(client, ("Ana",))
        assert client.post("/v1/ner", headers=AUTH, json=request.to_dict()).status_code == 200
        capability = client.get("/v1/capabilities", headers=AUTH).json()
        assert capability["counters"]["token_alignment_warnings"] == 3


class StagedModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.preparation_threads = []
        self.next_prepared = threading.Event()
        self.preparation_release = threading.Event()
        self.block_preparation = False
        self.check_overlap = False
        self.fail_prediction = False

    def prepare(self, texts):
        self.preparation_threads.append(threading.get_ident())
        if texts == ["Ana2"]:
            self.next_prepared.set()
            if self.block_preparation:
                assert self.preparation_release.wait(5)
        return texts

    def predict_prepared(self, texts, batch_size):
        if texts == ["Ana1"] and self.check_overlap:
            assert self.next_prepared.wait(3)
        if texts == ["Ana1"] and self.fail_prediction:
            raise RuntimeError("private OCR content")
        return super().predict(texts, batch_size)


def staged_worker(tmp_path):
    model = StagedModel()
    def factory(_settings):
        # Record creation on the actual model-owner thread, not the test caller.
        model.events = [("create", threading.get_ident())]
        return model
    worker = NERWorker(WorkerSettings(resources=tmp_path, max_documents=3, prepare_batch_documents=1),
                       adapter_factory=factory, identity_factory=lambda _model: {"format": "fake.ner.v1"})
    return worker, model


def staged_request(worker):
    return Request("staged", "ner", worker.fingerprint,
                   tuple(Document.create(str(i), f"Ana{i + 1}") for i in range(3)))


def test_document_preparation_overlaps_inference_and_preserves_owner_and_order(tmp_path):
    async def scenario():
        worker, model = staged_worker(tmp_path)
        await worker.start()
        model.check_overlap = True
        try:
            request = staged_request(worker)
            response = Response.from_dict(json.loads(await worker.infer(request)))
            results = response.validate_against(request)
            assert [result.document_id for result in results] == ["0", "1", "2"]
        finally:
            await worker.close()
        owner = model.events[0][1]
        assert all(thread == owner for _event, thread in model.events)
        assert model.preparation_threads and all(thread != owner for thread in model.preparation_threads)
    asyncio.run(scenario())


def test_staged_failure_waits_for_prefetch_before_releasing_capacity(tmp_path):
    async def scenario():
        worker, model = staged_worker(tmp_path)
        await worker.start()
        model.check_overlap = model.block_preparation = model.fail_prediction = True
        request = staged_request(worker)
        active = asyncio.create_task(worker.infer(request))
        try:
            assert await asyncio.to_thread(model.next_prepared.wait, 3)
            assert not active.done()
            with pytest.raises(WorkerError) as busy:
                await worker.infer(request)
            assert busy.value.status == 503
            model.preparation_release.set()
            with pytest.raises(WorkerError) as failed:
                await active
            assert failed.value.status == 500
            assert "private" not in str(failed.value)
            model.check_overlap = model.block_preparation = model.fail_prediction = False
            await worker.infer(request)
        finally:
            model.preparation_release.set()
            await worker.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("chunk", [-1, True, 33])
def test_preparation_chunk_bounds(tmp_path, chunk):
    with pytest.raises(ValueError, match="Preparation chunk"):
        WorkerSettings(resources=tmp_path, prepare_batch_documents=chunk)


def test_cancellation_keeps_capacity_during_document_preparation(tmp_path):
    async def scenario():
        worker, model = staged_worker(tmp_path)
        await worker.start()
        model.check_overlap = model.block_preparation = True
        request = staged_request(worker)
        active = asyncio.create_task(worker.infer(request))
        try:
            assert await asyncio.to_thread(model.next_prepared.wait, 3)
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            with pytest.raises(WorkerError) as busy:
                await worker.infer(request)
            assert busy.value.status == 503
        finally:
            model.preparation_release.set()
            await worker.close()
    asyncio.run(scenario())
