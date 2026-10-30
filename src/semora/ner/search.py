"""Bounded spellfix discovery and keyset-paged source lookup."""

import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path

from .names import normalize_name
from .spellfix import extension_path, load_spellfix


class EntitySearch:
    def __init__(self, engine, root):
        self.engine = engine
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Entity index is not built. Run semora index entities after NER extraction.")
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        legacy_generation = manifest["generation"]
        if len(legacy_generation) != 32 or any(c not in "0123456789abcdef" for c in legacy_generation):
            raise ValueError("Invalid entity index generation")
        publication = engine.publication
        if publication is not None and (
            legacy_generation != publication["entities"]["generation"]
            or hashlib.sha256(raw).hexdigest() != publication["entities"]["manifest_sha256"]
            or manifest["ner_run_id"] != publication["entities"]["ner_run_id"]
        ):
            raise ValueError("Entity dictionary differs from the serving publication.")
        self.generation = publication["publication_id"] if publication is not None else legacy_generation
        self.run_id = manifest["ner_run_id"]
        path = manifest_path.parent / f"{legacy_generation}.sqlite"
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Entity dictionary file escapes its artifact root.")
        self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        try:
            if publication is not None:
                binding = engine.database.conn.execute(
                    "SELECT advancement_token,pending_token FROM pipeline_entity_bindings "
                    "WHERE ner_run_id=? AND corpus_id=? AND generation=?",
                    (self.run_id, publication["corpus_id"], legacy_generation),
                ).fetchone()
                profile = self.connection.execute(
                    "SELECT advancement_token FROM pipeline_entity_profile WHERE profile_id=1"
                ).fetchone()
                if (
                    binding is None or profile is None or binding["pending_token"] is not None
                    or binding["advancement_token"] != profile["advancement_token"]
                ):
                    raise ValueError("Entity dictionary advancement differs from the corpus.")
            load_spellfix(self.connection, extension_path(Path(engine.database.path).resolve().parent.parent))
            self.connection.execute("ATTACH DATABASE ? AS corpus", (
                Path(engine.database.path).resolve().as_uri() + "?mode=ro",))
            self.connection.execute("PRAGMA query_only=ON")
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def search_entities(self, query):
        query = normalize_name(query)
        literal = query[:-1] if query.endswith("*") else query
        if not 3 <= len(literal) <= 200 or any(c in literal for c in "*?[]"):
            raise ValueError("Use 3–200 characters and at most one trailing * wildcard.")
        rows = self.connection.execute(
            "SELECT n.name,n.article_count,s.distance FROM name_spellfix s "
            "JOIN names n ON n.entity_id=s.rowid "
            "WHERE s.word MATCH ? AND s.top=100 "
            "ORDER BY s.distance,n.article_count DESC,n.normalized_name", (query,),
        ).fetchall()
        return {"entities": [dict(row) for row in rows], "candidate_limit": 100,
                "generation": self.generation, "limited": len(rows) == 100}

    def search_entity_articles(self, entity, *, limit=20, after=None, generation=None,
                               newspaper=None, date_from=None, date_to=None,
                               max_snippet_chars=320, source_wrap_chars=240):
        if not 1 <= limit <= 20:
            raise ValueError("Entity article page size must be between 1 and 20")
        if generation is not None and generation != self.generation:
            raise ValueError("Entity index changed; restart this entity search.")
        if not normalize_name(entity) or len(entity) > 2000:
            raise ValueError("Supply an exact entity name returned by search_entities.")
        start = date.fromisoformat(date_from) if date_from else None
        end = date.fromisoformat(date_to) if date_to else None
        if start and end and start > end:
            raise ValueError("date_to must not precede date_from")
        # Membership is indexed by entity and article. Check live completion hashes
        # so updated/deleted articles never use stale offsets from the snapshot.
        candidates = self.connection.execute(
            "SELECT a.*,m.char_start AS mention_start,a.title AS article_title, "
            "p.source,p.title AS newspaper_title,p.date,p.urn,p.newspaper_id AS document_id, "
            "p.relative_path,p.content AS newspaper_content FROM names n "
            "JOIN entity_articles m ON m.entity_id=n.entity_id "
            "JOIN corpus.article_ner_state s ON s.article_id=m.article_id AND s.ner_run_id=? "
            "AND s.content_sha256=m.content_sha256 "
            "JOIN corpus.articles a ON a.article_id=m.article_id "
            "JOIN corpus.newspapers p ON p.newspaper_id=a.newspaper_id "
            "WHERE n.normalized_name=? AND m.article_id>? AND a.is_valid=1 "
            "AND a.char_start IS NOT NULL AND a.char_end IS NOT NULL "
            "AND a.line_start IS NOT NULL AND a.line_end IS NOT NULL "
            "AND (? IS NULL OR p.source=? OR p.title=?) "
            "AND (? IS NULL OR p.date>=?) AND (? IS NULL OR p.date<=?) "
            "ORDER BY m.article_id LIMIT ?",
            (self.run_id, normalize_name(entity), after or "", newspaper, newspaper, newspaper,
             date_from, date_from, date_to, date_to, limit + 1),
        ).fetchall()
        hits = []
        for candidate in candidates[:limit]:
            row = dict(candidate)
            row["_focus_char"] = int(row["char_end"]) - len(row["content"]) + row["mention_start"]
            hits.append(self.engine._make_hit(
                row, 1.0, before=0, after=0, context_lines=0,
                max_snippet_chars=max_snippet_chars, source_wrap_chars=source_wrap_chars,
            ).as_dict())
        return {"hits": hits, "generation": self.generation,
                "next_after": candidates[limit - 1]["article_id"] if len(candidates) > limit else None}
