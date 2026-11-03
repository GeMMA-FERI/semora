import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from semora.execution.client import RemoteExecutionError, RemoteNERPool
from semora.execution.config import ExecutionConfig
from semora.execution.contracts import DocumentResult, Request, Response, semantic_fingerprint


@pytest.fixture
def worker(tmp_path):
    identity = {"format": "semora.classla-ner.identity.v1", "model": "test"}
    fingerprint = semantic_fingerprint(identity)
    state = {"calls": [], "fail": 0, "identity": identity, "response_mode": "normal", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state["calls"].append(self.headers.get("Authorization"))
            self.send_json(
                200,
                {
                    "api_version": "v1",
                    "ready": True,
                    "tasks": {
                        "ner": {
                            "enabled": True,
                            "ready": True,
                            "identity": state["identity"],
                            "fingerprint": fingerprint,
                        }
                    },
                    "limits": {
                        "max_documents": 1,
                        "max_input_bytes": 1000,
                        "max_request_bytes": 2000,
                        "max_response_bytes": 2000,
                    },
                },
            )

        def do_POST(self):
            state["calls"].append(self.headers.get("Authorization"))
            request = Request.from_dict(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            state["requests"].append(request.to_dict())
            if state.get("before_post"):
                state["before_post"]()
            mode = state["response_mode"]
            if isinstance(mode, int):
                self.send_json(mode, {"secret": "do not expose"})
                return
            if mode in ("duplicate", "nonfinite", "oversize", "dribble"):
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                payload = {
                    "duplicate": b'{"x":1,"x":2}',
                    "nonfinite": b'{"x":NaN}',
                    "oversize": b"x" * 2001,
                    "dribble": b"x" * 100,
                }[mode]
                try:
                    for byte in (bytes([b]) for b in payload) if mode == "dribble" else (payload,):
                        self.wfile.write(f"{len(byte):x}\r\n".encode() + byte + b"\r\n")
                        self.wfile.flush()
                        if mode == "dribble":
                            time.sleep(0.02)
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass
                return
            if state["fail"]:
                state["fail"] -= 1
                self.send_json(503, {})
                return
            response = Response(
                request.request_id,
                "ner",
                fingerprint,
                tuple(DocumentResult(doc.document_id, doc.content_sha256, entities=()) for doc in request.documents),
            )
            value = response.to_dict()
            if state["response_mode"] == "fingerprint":
                value["fingerprint"] = "0" * 64
            elif state["response_mode"] == "drop":
                value["results"] = []
            self.send_json(200, value)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("x" * 32)
    config = ExecutionConfig(
        backend="remote",
        worker_urls=(f"http://127.0.0.1:{server.server_port}",),
        token_file=token,
        connect_timeout=1,
        read_timeout=2,
        max_attempts=2,
        retry_delay=0.001,
    )
    try:
        yield config, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_auth_splitting_and_retry(worker):
    config, state = worker
    pool = RemoteNERPool(config)
    state["fail"] = 1
    batches = iter(
        [
            [
                {"article_id": "1", "content": "original", "inference_content": "masked"},
                {"article_id": "2", "content": "second"},
            ],
            [],
        ]
    )
    result = list(pool.iter_predictions(lambda: next(batches)))
    assert [item.rows[0]["article_id"] for item in result] == ["1", "2"]
    assert all(item.prediction.entities == [[]] for item in result)
    assert state["calls"] == ["Bearer " + "x" * 32] * 4


def test_rejects_bad_capabilities(worker):
    config, state = worker
    state["identity"] = {"format": "wrong"}
    with pytest.raises(RemoteExecutionError, match="capabilities"):
        RemoteNERPool(config)


def test_rejects_oversized_article(worker):
    config, _ = worker
    pool = RemoteNERPool(config)
    with pytest.raises(RemoteExecutionError, match="single document"):
        list(pool.iter_predictions(lambda: [{"article_id": "1", "content": "a" * 1001}]))


@pytest.mark.parametrize("mode", ["fingerprint", "drop"])
def test_invalid_results_are_not_retried(worker, mode):
    config, state = worker
    pool = RemoteNERPool(config)
    state["response_mode"] = mode
    with pytest.raises(RemoteExecutionError, match="invalid inference"):
        list(pool.iter_predictions(lambda: [{"article_id": "1", "content": "test"}]))
    assert len(state["calls"]) == 2
    assert pool.diagnostics["failures"] == 1


@pytest.mark.parametrize("token", [b"short", b"x" * 31 + b" ", b"x" * 32 + b"\xff", b"x" * 16 + b" " + b"x" * 16])
def test_bad_token_rejected_without_leaking(worker, token):
    config, state = worker
    config.token_file.write_bytes(token)
    with pytest.raises(RemoteExecutionError, match="token") as error:
        RemoteNERPool(config)
    assert "xxxx" not in str(error.value)
    assert not state["calls"]


@pytest.mark.parametrize("status", [301, 401, 500])
def test_http_errors_and_redirects_are_not_retried(worker, status):
    config, state = worker
    pool = RemoteNERPool(config)
    state["response_mode"] = status
    with pytest.raises(RemoteExecutionError, match=f"HTTP {status}") as error:
        list(pool.iter_predictions(lambda: [{"article_id": "1", "content": "secret"}]))
    assert "do not expose" not in str(error.value)
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("mode", ["duplicate", "nonfinite", "oversize"])
def test_malformed_or_large_chunked_response(worker, mode):
    config, state = worker
    pool = RemoteNERPool(config)
    state["response_mode"] = mode
    with pytest.raises(RemoteExecutionError):
        list(pool.iter_predictions(lambda: [{"article_id": "1", "content": "test"}]))
    assert len(state["requests"]) == 1


def test_dribble_has_absolute_deadline_and_retries_same_request(worker):
    config, state = worker
    config = replace(config, connect_timeout=0.1, read_timeout=0.15)
    pool = RemoteNERPool(config)
    state["response_mode"] = "dribble"
    tick = time.monotonic()
    with pytest.raises(RemoteExecutionError, match="retries"):
        list(pool.iter_predictions(lambda: [{"article_id": "1", "content": "test"}]))
    assert time.monotonic() - tick < 2
    assert len(state["requests"]) == 2
    assert state["requests"][0] == state["requests"][1]


def test_retry_moves_to_next_worker_with_same_request(worker, tmp_path):
    config, first = worker
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second_fixture = worker_fixture(second_dir)
    second_config, second = next(second_fixture)
    try:
        config = replace(config, worker_urls=config.worker_urls + second_config.worker_urls)
        pool = RemoteNERPool(config)
        first["fail"] = 1
        batches = iter([[{"article_id": "1", "content": "test"}], []])
        assert len(list(pool.iter_predictions(lambda: next(batches)))) == 1
        assert first["requests"] == second["requests"]
    finally:
        next(second_fixture, None)


def test_endpoint_lock_serializes_concurrent_requests(worker, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    config, _ = worker
    pool = RemoteNERPool(config)
    original = pool._request
    guard = threading.Lock()
    counts = {"active": 0, "maximum": 0}

    def monitored(*args):
        with guard:
            counts["active"] += 1
            counts["maximum"] = max(counts["maximum"], counts["active"])
        try:
            time.sleep(0.03)
            return original(*args)
        finally:
            with guard:
                counts["active"] -= 1

    monkeypatch.setattr(pool, "_request", monitored)

    def run(index):
        rows = [{"article_id": str(index), "content": "test"}]
        items = list(pool._subbatches(rows))[0]
        return pool._infer(items, 0, 0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, range(4)))
    assert counts["maximum"] == 1


worker_fixture = worker.__wrapped__


def test_free_worker_continues_while_other_worker_is_busy(worker, tmp_path):
    config, fast = worker
    directory = tmp_path / "slow"
    directory.mkdir()
    fixture = worker_fixture(directory)
    slow_config, slow = next(fixture)
    release = threading.Event()
    entered = threading.Event()
    timed_out = []

    def hold_slow():
        entered.set()
        if not release.wait(3):
            timed_out.append(True)

    def fast_request():
        if len(fast["requests"]) >= 3:
            assert entered.wait(3)
            release.set()

    slow["before_post"] = hold_slow
    fast["before_post"] = fast_request
    config = replace(config, worker_urls=config.worker_urls + slow_config.worker_urls, read_timeout=5)
    pages = iter([[{"article_id": str(i), "content": "test"}] for i in range(7)] + [[]])
    try:
        results = list(RemoteNERPool(config).iter_predictions(lambda: next(pages)))
        assert len(results) == 7
        assert len(fast["requests"]) >= 3
        assert not timed_out
    finally:
        release.set()
        next(fixture, None)


def test_batch_sizing_is_linear_and_exact_for_escaped_unicode(worker, monkeypatch):
    from semora.execution import client
    from semora.execution.contracts import Request

    config, _ = worker
    pool = RemoteNERPool(config)
    rows = [{"article_id": str(i), "content": 'č😀\\"\n' * 100} for i in range(128)]
    pool._limits.update(max_documents=128, max_input_bytes=1048576, max_request_bytes=2097152)
    original = client._encode
    encoded_total = []

    def counted(value):
        encoded = original(value)
        encoded_total.append(len(encoded))
        return encoded

    monkeypatch.setattr(client, "_encode", counted)
    batches = list(pool._subbatches(rows))
    assert len(batches) == 1
    request = Request("0" * 36, "ner", pool._fingerprint, tuple(doc for _row, doc in batches[0]))
    exact_size = len(original(request.to_dict()))
    assert sum(encoded_total) <= exact_size
    pool._limits["max_request_bytes"] = exact_size - 1
    split = list(pool._subbatches(rows))
    assert len(split) == 2
    for batch in split:
        value = Request("0" * 36, "ner", pool._fingerprint, tuple(doc for _row, doc in batch))
        assert len(original(value.to_dict())) <= exact_size - 1


def test_dispatch_continues_during_slow_consumer_and_is_bounded(worker):
    config, state = worker
    second = threading.Event()
    def posted():
        if len(state["requests"]) >= 2:
            second.set()
    state["before_post"] = posted
    pages = iter([[{"article_id": str(i), "content": "test"}] for i in range(30)] + [[]])
    stream = RemoteNERPool(config).iter_predictions(lambda: next(pages))
    next(stream)
    try:
        assert second.wait(2), "Writer must not block dispatch of the next request"
        time.sleep(0.1)
        assert 2 <= len(state["requests"]) <= 3
    finally:
        stream.close()
    count = len(state["requests"])
    time.sleep(0.05)
    assert len(state["requests"]) == count
    assert not any(t.name == "remote-ner-dispatch" for t in threading.enumerate())


def test_reader_exception_propagates_and_dispatcher_joins(worker):
    config, _ = worker
    def read():
        raise RuntimeError("reader failed")
    with pytest.raises(RuntimeError, match="reader failed"):
        list(RemoteNERPool(config).iter_predictions(read))
    assert not any(t.name == "remote-ner-dispatch" for t in threading.enumerate())
