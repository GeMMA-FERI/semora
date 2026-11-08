"""Bounded, explainable candidate discovery and source evidence."""

from __future__ import annotations

import difflib
import re
import time
from functools import wraps
from typing import Any

from semora.ner.names import normalize_name

RULE_VERSION = "ocr-candidates-v1"


def comparison_key(name):
    return re.sub(r"[\W_]+", "", normalize_name(name))


def ocr_key(name):
    return comparison_key(name).replace("rn", "m").translate(str.maketrans({"l": "i", "0": "o", "1": "i"}))


def review_read(method):
    @wraps(method)
    def bounded(self, *args, **kwargs):
        deadline = time.monotonic() + (
            900
            if method.__name__ == "count_articles"
            else 120
            if method.__name__ == "suggest_variants"
            else 30
            if method.__name__ == "discovery"
            else 5
        )
        self.connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            with self.store.snapshot() as revision:
                result = method(self, *args, **kwargs)
                result["revision"] = revision
                return result
        finally:
            self.connection.set_progress_handler(None, 0)

    return bounded


class CurationCatalogue:
    def __init__(self, entities, store) -> None:
        self.entities = entities
        self.store = store
        self.connection = entities.connection

    @review_read
    def find_names(self, query, *, limit=100):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise ValueError("Name queries require 1-200 characters.")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Use a name page size of 1-100.")
        query = normalize_name(query)
        rows = self.connection.execute(
            "SELECT normalized_name,name,article_count FROM names WHERE normalized_name>=? AND normalized_name<? "
            "ORDER BY normalized_name LIMIT ?",
            (query, query + chr(0x10FFFF), limit + 1),
        ).fetchall()
        return {
            "names": [{**dict(row), "group_id": self.store.membership(row["normalized_name"])} for row in rows[:limit]],
            "limited": len(rows) > limit,
            "generation": self.entities.generation,
        }

    @review_read
    def suggest_variants(self, name, *, group_id=None, limit=100):
        name = normalize_name(name) if isinstance(name, str) else ""
        if not 3 <= len(name) <= 200 or any(c in name for c in "*?[]"):
            raise ValueError("Suggestion anchors require 3-200 literal characters.")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Use a candidate limit of 1-100.")
        anchors = [name]
        members = []
        if group_id is not None:
            members = self.store.group(group_id)["members"]
            anchors += [m for m in members if m != name][:4]
        anchors = [a for a in anchors if 3 <= len(a) <= 200 and not any(c in a for c in "*?[]")]
        candidates: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            discovered = self.entities.search_entities(anchor)
            for item in discovered["entities"]:
                key = normalize_name(item["name"])
                if key in set(anchors + members) or any(self.store.rejected(a, key) for a in set(anchors + members)):
                    continue
                reasons = ["spellfix candidate"]
                if comparison_key(anchor) == comparison_key(key):
                    reasons.append("spacing/punctuation variant")
                if ocr_key(anchor) == ocr_key(key):
                    reasons.append("OCR-confusion comparison")
                similarity = round(difflib.SequenceMatcher(None, anchor, key, autojunk=False).ratio(), 4)
                proposal = {
                    **item,
                    "normalized_name": key,
                    "anchor": anchor,
                    "reasons": reasons,
                    "similarity": similarity,
                    "group_id": self.store.membership(key),
                }
                current = candidates.get(key)
                if current is None or (proposal["distance"], -similarity) < (
                    current["distance"],
                    -current["similarity"],
                ):
                    candidates[key] = proposal
        ranked = sorted(
            candidates.values(),
            key=lambda r: (-len(r["reasons"]), r["distance"], -r["article_count"], r["normalized_name"]),
        )
        return {
            "candidates": ranked[:limit],
            "limited": len(ranked) > limit,
            "coverage": "At most 100 spellfix candidates per anchor, at most five anchors; not exhaustive.",
            "rule_version": RULE_VERSION,
            "generation": self.entities.generation,
            "revision": self.store.metadata()["revision"],
        }

    @review_read
    def evidence(self, name, *, limit=3):
        if not isinstance(name, str) or not name.strip() or len(name) > 2000:
            raise ValueError("Supply a name of at most 2000 characters.")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("Use 1-5 evidence passages.")
        # Bounded membership sampling; never sort the whole archive for evidence.
        rows = self.connection.execute(
            "WITH sample AS MATERIALIZED (SELECT m.* FROM names n JOIN entity_articles m "
            "ON m.entity_id=n.entity_id WHERE n.normalized_name=? ORDER BY m.article_id LIMIT 40) "
            "SELECT m.article_id,m.char_start,s.content_sha256,p.source,p.date,a.title,"
            "substr(a.content,max(1,m.char_start-100),320) AS passage "
            "FROM sample m "
            "JOIN corpus.article_ner_state s ON s.article_id=m.article_id AND s.ner_run_id=? "
            "AND s.content_sha256=m.content_sha256 "
            "JOIN corpus.articles a ON a.article_id=m.article_id "
            "JOIN corpus.newspapers p ON p.newspaper_id=a.newspaper_id "
            "WHERE a.is_valid=1 ORDER BY m.article_id",
            (normalize_name(name), self.entities.run_id),
        ).fetchall()
        selected = []
        seen = set()
        # Spread within the bounded sample, making its limited coverage explicit.
        for row in sorted(rows, key=lambda r: (r["date"] or "", r["source"] or "", r["article_id"])):
            bucket = (row["source"], (row["date"] or "")[:4])
            if bucket in seen:
                continue
            seen.add(bucket)
            selected.append(row)
            if len(selected) == limit:
                break
        for row in rows:
            if len(selected) == limit:
                break
            if row not in selected:
                selected.append(row)
        result = []
        for row in selected:
            mentions = self.connection.execute(
                "SELECT text,label FROM corpus.article_entities WHERE ner_run_id=? AND article_id=? AND char_start=?",
                (self.entities.run_id, row["article_id"], row["char_start"]),
            ).fetchall()
            result.append({**dict(row), "mentions": [dict(r) for r in mentions]})
        return {
            "passages": result,
            "sampling": (
                "First 40 dictionary memberships in article-ID order, "
                "filtered for valid source hashes; not representative."
            ),
            "generation": self.entities.generation,
        }

    @review_read
    def preview_group(self, group_id):
        group = self.store.group(group_id)
        members = []
        for name in group["members"]:
            row = self.connection.execute(
                "SELECT name,article_count FROM names WHERE normalized_name=?", (name,)
            ).fetchone()
            members.append(
                {
                    "normalized_name": name,
                    "resolved": row is not None,
                    "name": row["name"] if row else None,
                    "article_count": row["article_count"] if row else None,
                }
            )
        revision = self.store.metadata()["revision"]
        count = self.store.connection.execute(
            "SELECT article_count FROM group_counts WHERE group_id=? AND publication=? AND revision=?",
            (group_id, self.entities.generation, group["version"]),
        ).fetchone()
        return {
            **group,
            "members": members,
            "revision": revision,
            "generation": self.entities.generation,
            "article_count": count[0] if count else None,
            "count_status": "ready" if count else "pending",
        }
