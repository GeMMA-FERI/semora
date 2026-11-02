import hashlib

import pytest

from semora.execution.contracts import (
    Document,
    DocumentResult,
    Request,
    Response,
    semantic_fingerprint,
)
from semora.ner.base import Entity

FP = "a" * 64


def test_non_json_keys_and_out_of_range_score_fail_with_validation_error():
    with pytest.raises(ValueError, match="keys must be strings"):
        Request.from_dict({1: "value", "extra": "value"})
    with pytest.raises(ValueError, match="finite"):
        DocumentResult("d", FP, entities=(Entity(0, 1, "PER", "a", 10**1000),))


def request(task="ner", docs=None):
    if docs is None:
        docs = [Document.create("d1", "")]
    return Request("req-1", task, FP, tuple(docs))


def test_fingerprint_is_canonical_and_rejects_non_json_values():
    assert semantic_fingerprint({"format": "v1", "model": "x"}) == semantic_fingerprint({"model": "x", "format": "v1"})
    with pytest.raises(ValueError):
        semantic_fingerprint({"format": "v1", "version": float("nan")})
    with pytest.raises(ValueError):
        semantic_fingerprint({"format": "v1", 1: "coerced key"})
    with pytest.raises(ValueError):
        semantic_fingerprint({"format": "v1", "values": (1, 2)})
    with pytest.raises(ValueError):
        semantic_fingerprint({"format": "  "})


def test_request_roundtrip_unicode_empty_text_and_hash_validation():
    docs = (Document.create("empty", ""), Document.create("unicode", "A😀café"))
    req = request(docs=docs)
    assert Request.from_dict(req.to_dict()) == req
    bad = req.to_dict()
    bad["documents"][1]["content_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValueError, match="does not match"):
        Request.from_dict(bad)


def test_ner_response_validates_unicode_character_offsets_and_reorders():
    docs = (Document.create("one", "😀 café"), Document.create("two", ""))
    req = request(docs=docs)
    resp = Response(
        "req-1",
        "ner",
        FP,
        (
            DocumentResult("two", docs[1].content_sha256, entities=()),
            DocumentResult("one", docs[0].content_sha256, entities=(Entity(2, 6, "LOC", "café", 0.8),)),
        ),
    )
    ordered = Response.from_dict(resp.to_dict()).validate_against(req)
    assert [r.document_id for r in ordered] == ["one", "two"]


@pytest.mark.parametrize(
    "bad",
    [
        {"start": True, "end": 1, "label": "X", "text": "a", "score": None},
        {"start": 0, "end": 1, "label": "X", "text": "a", "score": float("nan")},
        {"start": 0, "end": 1, "label": "X", "text": "a", "score": None, "extra": 0},
    ],
)
def test_ner_response_rejects_malformed_entities(bad):
    doc = Document.create("d", "a")
    raw = {
        "api_version": "v1",
        "request_id": "r",
        "task": "ner",
        "fingerprint": FP,
        "results": [{"document_id": "d", "content_sha256": doc.content_sha256, "entities": [bad]}],
    }
    with pytest.raises(ValueError):
        Response.from_dict(raw)


def test_lemma_response_preserves_empty_lemma_and_rejects_mismatch():
    doc = Document.create("d", "")
    req = request("lemma", [doc])
    resp = Response("req-1", "lemma", FP, (DocumentResult("d", doc.content_sha256, lemma=""),))
    assert Response.from_dict(resp.to_dict()).validate_against(req)[0].lemma == ""
    mismatched = Response(
        "req-1",
        "ner",
        FP,
        (DocumentResult("d", doc.content_sha256, entities=()),),
    )
    with pytest.raises(ValueError, match="identity"):
        mismatched.validate_against(req)


def test_response_rejects_duplicate_missing_and_hash_mismatch():
    docs = (Document.create("a", "a"), Document.create("b", "b"))
    req = request(docs=docs)
    duplicate = Response(
        "req-1", "ner", FP, tuple(DocumentResult("a", docs[0].content_sha256, entities=()) for _ in range(2))
    )
    with pytest.raises(ValueError, match="IDs/cardinality"):
        duplicate.validate_against(req)
    missing = Response("req-1", "ner", FP, (DocumentResult("a", docs[0].content_sha256, entities=()),))
    with pytest.raises(ValueError, match="IDs/cardinality"):
        missing.validate_against(req)
    wrong_hash = Response(
        "req-1",
        "ner",
        FP,
        (
            DocumentResult("a", "b" * 64, entities=()),
            DocumentResult("b", docs[1].content_sha256, entities=()),
        ),
    )
    with pytest.raises(ValueError, match="hash"):
        wrong_hash.validate_against(req)


def test_request_rejects_unknown_fields_and_duplicate_ids():
    doc = Document.create("d", "")
    with pytest.raises(ValueError, match="unknown"):
        Request.from_dict({**request().to_dict(), "extra": 1})
    with pytest.raises(ValueError, match="unique"):
        request(docs=[doc, doc])


@pytest.mark.parametrize("factory", [Request.from_dict, Response.from_dict])
def test_wire_objects_require_supported_api_version(factory):
    doc = Document.create("d", "")
    if factory is Request.from_dict:
        raw = request().to_dict()
        del raw["api_version"]
    else:
        raw = Response("r", "ner", FP, (DocumentResult("d", doc.content_sha256, entities=()),)).to_dict()
        raw["api_version"] = "v2"
    with pytest.raises(ValueError):
        factory(raw)


def test_response_rejects_wrong_task_results_and_invalid_source_offsets():
    doc = Document.create("d", "abc")
    with pytest.raises(ValueError, match="another task"):
        Response("r", "lemma", FP, (DocumentResult("d", doc.content_sha256, entities=()),))
    req = request(docs=[doc])
    bad_span = Response(
        "req-1", "ner", FP, (DocumentResult("d", doc.content_sha256, entities=(Entity(0, 1, "LOC", "z"),)),)
    )
    with pytest.raises(ValueError, match="does not match source"):
        bad_span.validate_against(req)


def test_empty_batch_and_tampered_identity_validation():
    req = request(docs=[])
    assert Response("req-1", "ner", FP, ()).validate_against(req) == ()
    for response in (
        Response("other", "ner", FP, ()),
        Response("req-1", "ner", "b" * 64, ()),
    ):
        with pytest.raises(ValueError, match="identity"):
            response.validate_against(req)


@pytest.mark.parametrize(
    "entity",
    [
        Entity(True, 1, "X", "a"),
        Entity(0, 1, "  ", "a"),
        Entity(0, 1, "X", "a", float("nan")),
        Entity(0, 1, "X", "a", True),
    ],
)
def test_direct_result_construction_validates_entities(entity):
    with pytest.raises(ValueError):
        DocumentResult("d", Document.create("d", "a").content_sha256, entities=(entity,))
