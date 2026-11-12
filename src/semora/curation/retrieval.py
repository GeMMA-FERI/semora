"""Read-only curated discovery and bounded merged article paging."""

from __future__ import annotations

import heapq
from datetime import date

from semora.ner.names import normalize_name

from .catalogue import CurationCatalogue, review_read
from .store import membership_key


class CuratedRetrieval(CurationCatalogue):
    @review_read
    def discovery(self, query):
        result = []
        seen = set()
        revision = self.store.metadata()["revision"]
        key = normalize_name(query)
        literal = key[:-1] if key.endswith("*") else key
        if not 3 <= len(literal) <= 200 or any(c in literal for c in "*?[]"):
            raise ValueError("Use 3-200 characters and at most one trailing * wildcard.")
        is_prefix = key.endswith("*")
        if is_prefix:
            key = key[:-1]
            preferred = self.store.connection.execute(
                "SELECT group_id FROM name_groups WHERE preferred_key>=? AND preferred_key<? "
                "ORDER BY preferred_key,group_id LIMIT 101",
                (key, key + chr(0x10FFFF)),
            ).fetchall()
        else:
            preferred = self.store.connection.execute(
                "SELECT group_id FROM name_groups WHERE preferred_key=? ORDER BY group_id LIMIT 101", (key,)
            ).fetchall()
        if preferred:
            # Preserve exact/raw-prefix matches even when a preferred name collides with an
            # unrelated extraction. Indexed matches keep this fast without invoking spellfix.
            if is_prefix:
                rows = self.connection.execute(
                    "SELECT name,article_count FROM names WHERE normalized_name>=? AND normalized_name<? "
                    "ORDER BY normalized_name LIMIT 101",
                    (literal, literal + chr(0x10FFFF)),
                ).fetchall()
            else:
                rows = self.connection.execute(
                    "SELECT name,article_count FROM names WHERE normalized_name=?", (literal,)
                ).fetchall()
            raw = {
                "entities": [{**dict(r), "distance": 0} for r in rows[:100]],
                "limited": len(rows) > 100,
                "candidate_limit": 100,
                "generation": self.entities.generation,
            }
        else:
            raw = self.entities.search_entities(query)
        extra = [
            {"name": self.store.group(row[0])["preferred_name"], "distance": 0, "preferred_group": row[0]}
            for row in preferred[:100]
        ]
        for candidate in extra + raw["entities"]:
            group_id = candidate.get("preferred_group") or self.store.membership(candidate["name"])
            if group_id is None:
                result.append(candidate)
                continue
            if group_id in seen:
                continue
            seen.add(group_id)
            group = self.store.group(group_id)
            count = self.store.connection.execute(
                "SELECT article_count FROM group_counts WHERE group_id=? AND publication=? AND revision=?",
                (group_id, self.entities.generation, group["version"]),
            ).fetchone()
            result.append(
                {
                    "name": group["preferred_name"],
                    "group_id": group_id,
                    "origin": group["origin"],
                    "human_reviewed": group["human_reviewed"],
                    "matched_name": None if candidate.get("preferred_group") else candidate["name"],
                    "article_count": count[0] if count else None,
                    "count_status": "ready" if count else "pending",
                    "distance": candidate["distance"],
                }
            )
        return {
            **raw,
            "entities": result[:100],
            "limited": raw["limited"] or len(preferred) > 100 or len(result) > 100,
            "curation_revision": revision,
        }

    @review_read
    def articles(
        self,
        group_id,
        *,
        limit=20,
        after=None,
        generation=None,
        curation_revision=None,
        newspaper=None,
        date_from=None,
        date_to=None,
        max_snippet_chars=320,
        source_wrap_chars=240,
    ):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Use an article page size of 1-20.")
        revision = self.store.metadata()["revision"]
        if (
            generation is not None
            and generation != self.entities.generation
            or curation_revision is not None
            and curation_revision != revision
        ):
            raise ValueError("Entity publication or curation revision changed; restart search.")
        if after is not None and (generation is None or curation_revision is None):
            raise ValueError("Group pagination requires publication and curation revision.")
        start = date.fromisoformat(date_from) if date_from else None
        end = date.fromisoformat(date_to) if date_to else None
        if start and end and start > end:
            raise ValueError("date_to must not precede date_from")
        group = self.store.group(group_id)
        streams = []
        unresolved = []
        for member in group["members"]:
            if self.connection.execute("SELECT 1 FROM names WHERE normalized_name=?", (member,)).fetchone() is None:
                unresolved.append(member)
                continue
            rows = self.entities._candidates(
                member,
                rows=limit + 1,
                after=after,
                newspaper=newspaper,
                date_from=date_from,
                date_to=date_to,
                full=False,
            )
            streams.append([(r["article_id"], r["mention_start"], member) for r in rows])
        chosen = []
        previous = None
        for item in heapq.merge(*streams):
            if item[0] == previous:
                continue
            previous = item[0]
            chosen.append(item)
            if len(chosen) == limit + 1:
                break
        hits = []
        for article_id, _, member in chosen[:limit]:
            # Materialize full source only for the final page, retaining normal source-hash checks.
            rows = self.entities._candidates(member, rows=1, after=article_id, full=True, inclusive=True)
            if not rows or rows[0]["article_id"] != article_id:
                raise ValueError("Entity source changed during paging; restart search.")
            hit = self.entities._hits(rows, max_snippet_chars, source_wrap_chars)[0]
            originals = self.connection.execute(
                "SELECT text FROM corpus.article_entities WHERE ner_run_id=? AND article_id=? AND char_start=?",
                (self.entities.run_id, article_id, rows[0]["mention_start"]),
            ).fetchall()
            original = next((r[0] for r in originals if normalize_name(r[0]) == member), None)
            if original is None:
                raise ValueError("Original entity mention is missing; rebuild or repair the entity index.")
            hit["matched_name"] = original
            hits.append(hit)
        return {
            "hits": hits,
            "generation": self.entities.generation,
            "curation_revision": revision,
            "group_id": group_id,
            "unresolved_members": unresolved,
            "next_after": chosen[limit - 1][0] if len(chosen) > limit else None,
        }

    @review_read
    def count_articles(self, group_id):
        group = self.store.group(group_id)
        cached = self.store.connection.execute(
            "SELECT article_count FROM group_counts WHERE group_id=? AND publication=? AND revision=?",
            (group_id, self.entities.generation, group["version"]),
        ).fetchone()
        if cached is not None:
            return {
                "group_id": group_id,
                "article_count": cached[0],
                "generation": self.entities.generation,
                "curation_revision": self.store.metadata()["revision"],
                "group_version": group["version"],
                "membership_key": membership_key(group["members"]),
            }
        # Exact, potentially expensive work is a separate bounded admin job, never discovery work.
        parts = []
        parameters: list[str] = []
        unresolved = []
        for member in group["members"]:
            if self.connection.execute("SELECT 1 FROM names WHERE normalized_name=?", (member,)).fetchone() is None:
                unresolved.append(member)
            parts.append(
                "SELECT m.article_id FROM names n JOIN entity_articles m ON m.entity_id=n.entity_id "
                "JOIN corpus.article_ner_state s ON s.article_id=m.article_id AND s.ner_run_id=? "
                "AND s.content_sha256=m.content_sha256 JOIN corpus.articles a ON a.article_id=m.article_id "
                "WHERE n.normalized_name=? AND a.is_valid=1 AND a.char_start IS NOT NULL "
                "AND a.char_end IS NOT NULL AND a.line_start IS NOT NULL AND a.line_end IS NOT NULL"
            )
            parameters.extend((self.entities.run_id, member))
        if unresolved:
            raise ValueError("Resolve missing group members before computing a complete group count.")
        total = self.connection.execute("SELECT COUNT(*) FROM (" + " UNION ".join(parts) + ")", parameters).fetchone()[
            0
        ]
        return {
            "group_id": group_id,
            "article_count": total,
            "generation": self.entities.generation,
            "curation_revision": self.store.metadata()["revision"],
            "group_version": group["version"],
            "membership_key": membership_key(group["members"]),
        }
