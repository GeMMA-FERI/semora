"""Opt-in CPU/CUDA model parity; never downloads resources or touches a corpus."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("SEMORA_TEST_CLASSLA_WORKER") != "1", reason="Requires explicitly selected cached CLASSLA resources"
)
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        os.environ.get("SEMORA_TEST_CLASSLA_CUDA") != "1", reason="Requires explicitly enabled CUDA qualification"
    )),
])
def test_real_classla_local_and_api_predictions_match(device: str) -> None:
    # Explicit qualification must fail if its test environment is incomplete.
    for dependency in ("classla", "fastapi", "httpx"):
        importlib.import_module(dependency)
    from fastapi.testclient import TestClient

    from semora.corpus.spans import mask_html_tables
    from semora.execution.contracts import Document, Request, Response
    from semora.execution.runtime import WorkerSettings, _load_adapter
    from semora.execution.worker import create_app

    settings = WorkerSettings(resources=Path(os.environ["SEMORA_CLASSLA_RESOURCES"]), device=device)
    original = [
        "Ana Novak živi v Mariboru.", "", "Danes je lepo vreme.",
        "J\x03. Novak je obiskal Ljubljano. Mehki\xadvezaj ostane.",
        "To je preprost stavek.\n\n" * 30 + "Janez Novak živi v Ljubljani.",
        "Ana Novak. <table><tr><td>Jože Čop</td></tr></table> Maribor.",
    ]
    texts = [mask_html_tables(text) for text in original]
    reference = _load_adapter(settings)
    try:
        expected = reference.predict(texts, settings.batch_size).entities
    finally:
        reference.close()
    token = "test-only-token-" * 3
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(create_app(settings, token=token)) as client:
        fingerprint = client.get("/v1/capabilities", headers=headers).json()["tasks"]["ner"]["fingerprint"]
        request = Request(
            "fixed-model-parity", "ner", fingerprint,
            tuple(Document.create(str(index), text) for index, text in enumerate(texts)),
        )
        result = client.post("/v1/ner", headers=headers, json=request.to_dict())
        assert result.status_code == 200, result.text
        ordered = Response.from_dict(result.json()).validate_against(request)
        assert [list(item.entities or ()) for item in ordered] == expected
        for source, entities in zip(original, expected, strict=True):
            assert all(source[entity.start:entity.end] == entity.text for entity in entities)
