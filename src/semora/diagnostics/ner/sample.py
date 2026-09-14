"""Reproducible uniform article sampling without scanning corpus text."""

import hashlib
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path


def readonly(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def create_sample(database: Path, output: Path, count: int = 10_000, seed: int = 42) -> None:
    if count <= 0:
        raise ValueError("Sample size must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing artifacts and the production database.
    with output.open("xb"):
        pass
    rng = random.Random(seed)
    with closing(readonly(database)) as source, closing(sqlite3.connect(output)) as target, target:
        target.executescript("""
            CREATE TABLE articles (
                position INTEGER PRIMARY KEY, article_id TEXT UNIQUE, newspaper_id TEXT,
                title TEXT, content TEXT, newspaper TEXT, date TEXT, urn TEXT
            );
            CREATE TABLE metadata (value TEXT NOT NULL);
        """)
        bounds = source.execute(
            "SELECT (SELECT MIN(rowid) FROM articles), (SELECT MAX(rowid) FROM articles)"
        ).fetchone()
        if bounds[0] is None:
            raise ValueError("The source has no articles")
        low, high = bounds
        seen: set[int] = set()
        selected: list[sqlite3.Row] = []
        # Rejection sampling over rowids: each eligible article has equal probability,
        # including when rowids have gaps. Do not use random rowid seek (gap-biased).
        attempts = 0
        while len(selected) < count and len(seen) < high - low + 1:
            ids = []
            for _ in range(min(500, high - low + 1)):
                value = rng.randint(low, high)
                if value not in seen:
                    ids.append(value)
                    seen.add(value)
            attempts += 1
            if attempts > max(1000, count * 10):
                raise ValueError("Source rowids are too sparse; could not obtain the requested sample")
            if not ids:
                continue
            placeholders = ",".join("?" for _ in ids)
            rows = source.execute(
                f"""
                SELECT a.rowid AS source_rowid, a.article_id, a.newspaper_id,
                       a.title, a.content, n.title AS newspaper, n.date, n.urn
                FROM articles a LEFT JOIN newspapers n ON n.newspaper_id = a.newspaper_id
                WHERE a.rowid IN ({placeholders}) AND a.is_valid = 1
                  AND LENGTH(a.content) > 0
            """,
                ids,
            ).fetchall()
            found = {row["source_rowid"]: row for row in rows}
            selected.extend(found[value] for value in ids if value in found)
        if len(selected) < count:
            raise ValueError(f"Only {len(selected)} eligible articles; requested {count}")
        selected = selected[:count]
        digest = hashlib.sha256()
        for position, row in enumerate(selected):
            digest.update(json.dumps([row["article_id"], row["content"]], ensure_ascii=False).encode())
            target.execute(
                "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    position,
                    row["article_id"],
                    row["newspaper_id"],
                    row["title"],
                    row["content"],
                    row["newspaper"],
                    row["date"],
                    row["urn"],
                ),
            )
        metadata = {
            "sample_sha256": digest.hexdigest(),
            "seed": seed,
            "articles": count,
            "source": str(database.resolve()),
            "characters": sum(len(r["content"]) for r in selected),
            "sampling": "uniform valid nonempty articles via rowid rejection; not balanced by decade",
            "decades": dict(
                Counter(r["date"][:3] + "0s" if r["date"] and r["date"][:4].isdigit() else "unknown" for r in selected)
            ),
            "newspapers": dict(Counter(r["newspaper"] or "unknown" for r in selected)),
            "length_buckets": dict(
                Counter(
                    "<1k" if len(r["content"]) < 1000 else "1k-5k" if len(r["content"]) < 5000 else ">=5k"
                    for r in selected
                )
            ),
        }
        target.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata, ensure_ascii=False),))
    print(json.dumps({key: value for key, value in metadata.items() if key != "newspapers"}, indent=2))
