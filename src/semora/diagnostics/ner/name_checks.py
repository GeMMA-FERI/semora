"""Literal-name spot checks, not an annotated precision/recall evaluation."""

import re


def check_names(text: str, predictions: dict[str, list[dict]], queries: list[str]) -> list[dict]:
    checks = []
    for query in queries:
        if not query.strip():
            raise ValueError("Name queries cannot be blank")
        for match in re.finditer(r"(?<!\w)" + re.escape(query) + r"(?!\w)", text, flags=re.IGNORECASE):
            checks.append(
                {
                    "query": query,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "context": text[max(0, match.start() - 80) : match.end() + 80],
                    "models": {
                        name: {
                            "covering_labels": sorted(
                                {
                                    e["label"]
                                    for e in entities
                                    if e["start"] <= match.start() and e["end"] >= match.end()
                                }
                            ),
                            "overlapping_entities": [
                                e for e in entities if e["start"] < match.end() and e["end"] > match.start()
                            ],
                        }
                        for name, entities in predictions.items()
                    },
                }
            )
    return checks
