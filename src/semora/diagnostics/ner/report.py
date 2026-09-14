"""Offline comparison of cached predictions, with escaped source highlights."""

import csv
import html
import json
import random
from contextlib import ExitStack, closing
from pathlib import Path

from .name_checks import check_names
from .sample import readonly


def signature(entities: list[dict], labels: set[str]) -> set[tuple]:
    return {(e["start"], e["end"], e["label"]) for e in entities if e["label"] in labels}


def highlight(text: str, entities: list[dict], start: int, end: int) -> str:
    relevant = [e for e in entities if e["start"] < end and e["end"] > start]
    boundaries = sorted(
        {start, end} | {max(start, e["start"]) for e in relevant} | {min(end, e["end"]) for e in relevant}
    )
    parts = []
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        labels = sorted({e["label"] for e in relevant if e["start"] <= left and e["end"] >= right})
        fragment = html.escape(text[left:right])
        parts.append(
            f'<mark title="{html.escape(", ".join(labels), quote=True)}">{fragment}</mark>' if labels else fragment
        )
    return "".join(parts)


def report(
    sample: Path,
    runs: list[Path],
    output: Path,
    *,
    limit: int = 40,
    names: list[str] | None = None,
    seed: int = 42,
    labels: list[str] | None = None,
    queries: list[str] | None = None,
) -> None:
    if len(runs) < 2 or limit <= 0:
        raise ValueError("Provide at least two runs and a positive report limit")
    names = names or [path.name for path in runs]
    if len(names) != len(runs) or len(set(names)) != len(names):
        raise ValueError("Run names must be unique and match the number of runs")
    labels_set = set(labels or ["PER", "LOC", "ORG"])
    name_checks: list[dict] = []
    metrics = [json.loads((path / "metrics.json").read_text(encoding="utf-8")) for path in runs]
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        source = stack.enter_context(closing(readonly(sample)))
        metadata = json.loads(source.execute("SELECT value FROM metadata").fetchone()[0])
        for item in metrics:
            if item["status"] != "complete" or item["sample"]["sample_sha256"] != metadata["sample_sha256"]:
                raise ValueError("Only completed runs on the identical sample can be compared")
        if len({item["completed_articles"] for item in metrics}) != 1:
            raise ValueError("Runs processed different numbers of articles")
        databases = [stack.enter_context(closing(readonly(path / "predictions.sqlite"))) for path in runs]
        rng = random.Random(seed)
        choices: list[dict] = []
        disagreement_count = 0
        total = metrics[0]["completed_articles"]
        for row in source.execute("SELECT * FROM articles ORDER BY position LIMIT ?", (total,)):
            predictions = []
            for db in databases:
                found = db.execute(
                    "SELECT entities_json FROM predictions WHERE article_id=?", (row["article_id"],)
                ).fetchone()
                if found is None:
                    raise ValueError(f"Missing predictions for {row['article_id']}")
                predictions.append(json.loads(found[0]))
            if queries:
                checks = check_names(row["content"], dict(zip(names, predictions, strict=True)), queries)
                name_checks.extend({**check, "article_id": row["article_id"], "urn": row["urn"]} for check in checks)
            signatures = [signature(entities, labels_set) for entities in predictions]
            if all(value == signatures[0] for value in signatures):
                continue
            disagreement_count += 1
            differences = signatures[0].union(*signatures[1:]) - signatures[0].intersection(*signatures[1:])
            # Prefer a person disagreement when available; show a short local passage.
            anchor = min(differences, key=lambda s: (s[2] != "PER", s[0]))
            start, end = max(0, anchor[0] - 130), min(len(row["content"]), anchor[1] + 180)
            item = {
                "article_id": row["article_id"],
                "newspaper": row["newspaper"],
                "date": row["date"],
                "urn": row["urn"],
                "start": start,
                "end": end,
                "text": row["content"][start:end],
                "predictions": {
                    name: [e for e in entities if e["start"] < end and e["end"] > start]
                    for name, entities in zip(names, predictions, strict=True)
                },
                "rendered": [highlight(row["content"], entities, start, end) for entities in predictions],
            }
            if len(choices) < limit:
                choices.append(item)
            else:
                replacement = rng.randrange(disagreement_count)
                if replacement < limit:
                    choices[replacement] = item
    body = []
    for item in choices:
        columns = ""
        for name, rendered in zip(names, item.pop("rendered"), strict=True):
            mentions = (
                " · ".join(
                    html.escape(f"{e['label']}: {e['text']} [{e['start']}:{e['end']}]")
                    for e in item["predictions"][name]
                )
                or "No entities"
            )
            columns += (
                f'<td><strong>{html.escape(name)}</strong><p>{rendered}</p><p class="entities">{mentions}</p></td>'
            )
        caption = html.escape(f"{item['newspaper']} | {item['date']} | {item['urn']} | {item['article_id']}")
        body.append(f"<section><p>{caption}</p><table><tr>{columns}</tr></table></section>")
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
    <title>NER comparison</title><style>
    body{{font:16px system-ui;max-width:1400px;margin:2rem auto;padding:1rem;background:#fafafa}}
    table{{width:100%;table-layout:fixed;border-collapse:collapse}}
    td{{vertical-align:top;padding:1rem;border:1px solid #ddd}}
    p{{white-space:pre-wrap;overflow-wrap:anywhere}}mark{{background:#ffe7a3}}section{{margin:2rem 0}}
    .entities{{font-size:0.85rem;color:#444;border-top:1px solid #ddd;padding-top:0.6rem}}
    </style><h1>NER disagreements</h1>
    <p>{disagreement_count} of {total} articles disagree on {html.escape(", ".join(sorted(labels_set)))}.
    Showing {len(choices)} sampled articles, one passage each. Hover highlights for labels.
    Disagreement is not accuracy; shared omissions are invisible. No precision/recall/F1 is inferred.</p>
    {"".join(body)}</html>"""
    (output / "report.html").write_text(document, encoding="utf-8")
    (output / "review.json").write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")
    if queries:
        name_results = {
            "note": "Literal whole-name matches only; absent names and inflections are not NER misses. Not recall.",
            "occurrences": {query: sum(check["query"] == query for check in name_checks) for query in queries},
            "checks": name_checks,
        }
        (output / "name_checks.json").write_text(
            json.dumps(name_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    fields = [
        "name",
        "adapter",
        "batch_size",
        "completed_articles",
        "articles_per_second",
        "characters_per_second",
        "load_seconds",
        "warmup_seconds",
        "read_seconds",
        "predict_seconds",
        "write_seconds",
        "peak_process_rss_bytes",
        "peak_cuda_allocated_bytes",
        "estimated_corpus_hours",
        "entities",
    ]
    with (output / "timings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, item in zip(names, metrics, strict=True):
            writer.writerow({**item, "name": name})
    print(f"Report: {output / 'report.html'} ({disagreement_count}/{total} articles disagree)")
