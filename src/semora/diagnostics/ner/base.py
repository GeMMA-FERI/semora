"""Shared adapter contract and source-relative entity spans."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    text: str
    score: float | None = None


@dataclass
class Prediction:
    entities: list[list[Entity]]
    timings: dict[str, float] = field(default_factory=dict)


class Adapter(Protocol):
    metadata: dict[str, Any]

    def predict(self, texts: list[str], batch_size: int) -> Prediction: ...


def validate(text: str, entities: list[Entity]) -> None:
    """Never silently save misaligned spans or unsupported labels."""
    for entity in entities:
        if not 0 <= entity.start < entity.end <= len(text):
            raise ValueError(f"Invalid entity offsets: {entity}")
        if text[entity.start : entity.end] != entity.text:
            raise ValueError(f"Entity does not match source text: {entity}")
        if not entity.label:
            raise ValueError("Entity label cannot be empty")
