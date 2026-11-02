"""Small, versioned JSON contracts for CLASSLA execution batches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from semora.ner.base import Entity, validate

Task = Literal["ner", "lemma"]
API_VERSION = "v1"


def _json_value(value: Any, path: str = "identity") -> None:
    """Reject Python conveniences that JSON silently coerces or cannot represent."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} object keys must be strings")
            _json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value")


def semantic_fingerprint(operation_identity: dict[str, Any]) -> str:
    """Hash a versioned semantic identity with a non-empty ``format`` field."""
    if type(operation_identity) is not dict:
        raise ValueError("operation_identity must be a JSON object")
    format_name = operation_identity.get("format")
    if not isinstance(format_name, str) or not format_name.strip():
        raise ValueError("operation_identity.format must be a non-empty version string")
    _json_value(operation_identity)
    payload = json.dumps(
        operation_identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{where} object keys must be strings")
    unknown = value.keys() - expected
    missing = expected - value.keys()
    if unknown or missing:
        raise ValueError(f"{where} fields invalid (missing={sorted(missing)}, unknown={sorted(unknown)})")
    return value


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        qualifier = "a non-empty" if nonempty else "a"
        raise ValueError(f"{where} must be {qualifier} string")
    return value


def _digest(value: Any, where: str) -> str:
    value = _string(value, where)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{where} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    content_sha256: str

    @classmethod
    def create(cls, document_id: str, text: str) -> Document:
        text = _string(text, "text", nonempty=False)
        return cls(
            _string(document_id, "document_id"),
            text,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        _string(self.document_id, "document_id")
        _string(self.text, "text", nonempty=False)
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if _digest(self.content_sha256, "content_sha256") != actual:
            raise ValueError("content_sha256 does not match text")

    def to_dict(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Document:
        data = _keys(value, {"document_id", "text", "content_sha256"}, "document")
        return cls(data["document_id"], data["text"], data["content_sha256"])


@dataclass(frozen=True)
class Request:
    request_id: str
    task: Task
    fingerprint: str
    documents: tuple[Document, ...]
    api_version: str = API_VERSION

    def __post_init__(self) -> None:
        if self.api_version != API_VERSION:
            raise ValueError("unsupported api_version")
        _string(self.request_id, "request_id")
        if self.task not in ("ner", "lemma"):
            raise ValueError("task must be ner or lemma")
        _digest(self.fingerprint, "fingerprint")
        if not isinstance(self.documents, tuple) or any(
            not isinstance(document, Document) for document in self.documents
        ):
            raise ValueError("documents must be a tuple of Document")
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "request_id": self.request_id,
            "task": self.task,
            "fingerprint": self.fingerprint,
            "documents": [document.to_dict() for document in self.documents],
        }

    @classmethod
    def from_dict(cls, value: Any) -> Request:
        data = _keys(
            value,
            {"api_version", "request_id", "task", "fingerprint", "documents"},
            "request",
        )
        if not isinstance(data["documents"], list):
            raise ValueError("documents must be a list")
        return cls(
            data["request_id"],
            data["task"],
            data["fingerprint"],
            tuple(Document.from_dict(document) for document in data["documents"]),
            data["api_version"],
        )


@dataclass(frozen=True)
class DocumentResult:
    document_id: str
    content_sha256: str
    entities: tuple[Entity, ...] | None = None
    lemma: str | None = None

    def __post_init__(self) -> None:
        _string(self.document_id, "document_id")
        _digest(self.content_sha256, "content_sha256")
        if (self.entities is None) == (self.lemma is None):
            raise ValueError("result must contain exactly one of entities or lemma")
        if self.entities is not None:
            if not isinstance(self.entities, tuple) or any(not isinstance(entity, Entity) for entity in self.entities):
                raise ValueError("entities must be a tuple of Entity")
            for entity in self.entities:
                if type(entity.start) is not int or type(entity.end) is not int:
                    raise ValueError("entity offsets must be integers")
                _string(entity.label, "label")
                _string(entity.text, "entity text", nonempty=False)
                if entity.score is not None:
                    try:
                        finite = type(entity.score) in (int, float) and isfinite(entity.score)
                    except OverflowError:
                        finite = False
                    if not finite:
                        raise ValueError("entity score must be finite or null")
        if self.lemma is not None:
            _string(self.lemma, "lemma", nonempty=False)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "document_id": self.document_id,
            "content_sha256": self.content_sha256,
        }
        if self.entities is not None:
            result["entities"] = [
                {
                    "start": entity.start,
                    "end": entity.end,
                    "label": entity.label,
                    "text": entity.text,
                    "score": entity.score,
                }
                for entity in self.entities
            ]
        else:
            result["lemma"] = self.lemma
        return result

    @classmethod
    def from_dict(cls, value: Any) -> DocumentResult:
        if not isinstance(value, dict):
            raise ValueError("document result must be an object")
        common = {"document_id", "content_sha256"}
        if "entities" in value:
            data = _keys(value, common | {"entities"}, "NER result")
            if not isinstance(data["entities"], list):
                raise ValueError("entities must be a list")
            entities = []
            for raw in data["entities"]:
                item = _keys(raw, {"start", "end", "label", "text", "score"}, "entity")
                entities.append(
                    Entity(
                        item["start"],
                        item["end"],
                        item["label"],
                        item["text"],
                        item["score"],
                    )
                )
            return cls(data["document_id"], data["content_sha256"], tuple(entities), None)
        data = _keys(value, common | {"lemma"}, "lemma result")
        return cls(
            data["document_id"],
            data["content_sha256"],
            None,
            _string(data["lemma"], "lemma", nonempty=False),
        )


@dataclass(frozen=True)
class Response:
    request_id: str
    task: Task
    fingerprint: str
    results: tuple[DocumentResult, ...]
    api_version: str = API_VERSION

    def __post_init__(self) -> None:
        if self.api_version != API_VERSION:
            raise ValueError("unsupported api_version")
        _string(self.request_id, "request_id")
        if self.task not in ("ner", "lemma"):
            raise ValueError("task must be ner or lemma")
        _digest(self.fingerprint, "fingerprint")
        if not isinstance(self.results, tuple) or any(
            not isinstance(result, DocumentResult) for result in self.results
        ):
            raise ValueError("results must be a tuple of DocumentResult")
        expected_entities = self.task == "ner"
        if any((result.entities is not None) != expected_entities for result in self.results):
            raise ValueError(f"{self.task} response contains results for another task")

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "request_id": self.request_id,
            "task": self.task,
            "fingerprint": self.fingerprint,
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, value: Any) -> Response:
        data = _keys(
            value,
            {"api_version", "request_id", "task", "fingerprint", "results"},
            "response",
        )
        if not isinstance(data["results"], list):
            raise ValueError("results must be a list")
        return cls(
            data["request_id"],
            data["task"],
            data["fingerprint"],
            tuple(DocumentResult.from_dict(result) for result in data["results"]),
            data["api_version"],
        )

    def validate_against(self, request: Request) -> tuple[DocumentResult, ...]:
        if (self.request_id, self.task, self.fingerprint) != (
            request.request_id,
            request.task,
            request.fingerprint,
        ):
            raise ValueError("response request identity does not match")
        expected = {document.document_id: document for document in request.documents}
        actual = {result.document_id: result for result in self.results}
        if len(actual) != len(self.results) or actual.keys() != expected.keys():
            raise ValueError("response document IDs/cardinality do not match")
        ordered = []
        for document in request.documents:
            result = actual[document.document_id]
            if result.content_sha256 != document.content_sha256:
                raise ValueError("response document hash does not match")
            if request.task == "ner":
                if result.entities is None:
                    raise ValueError("NER response requires entities")
                validate(document.text, list(result.entities))
            elif result.lemma is None:
                raise ValueError("lemma response requires lemma")
            ordered.append(result)
        return tuple(ordered)
