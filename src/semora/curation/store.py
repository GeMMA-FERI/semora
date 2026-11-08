"""Persistent, transactional name groups. Only this store is writable."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from semora.ner.names import normalize_name

NORMALIZATION = "nfc-casefold-whitespace-v1"
MAX_MEMBERS = 100


class Conflict(ValueError):
    """The caller must refresh state before retrying a different operation."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text(value, field, maximum=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{field} must be nonblank text of at most {maximum} characters.")
    return value.strip()


def _identifier(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a UUID string.")
    try:
        return uuid.UUID(value).hex
    except (ValueError, AttributeError) as error:
        raise ValueError(f"{field} must be a UUID string.") from error


class CurationStore:
    def __init__(self, path: str | Path, *, corpus_id: str, initialize: bool = False, read_only: bool = False) -> None:
        self.path = Path(path).resolve()
        if initialize and read_only:
            raise ValueError("Cannot initialize a read-only store.")
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ro" if read_only else ("rwc" if initialize else "rw")
        self.connection = sqlite3.connect(self.path.as_uri() + f"?mode={mode}", uri=True, timeout=5)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        try:
            if initialize:
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=FULL")
                self.connection.executescript("""
                    CREATE TABLE IF NOT EXISTS curation_metadata (
                        id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL,
                        corpus_id TEXT NOT NULL, normalization TEXT NOT NULL, revision INTEGER NOT NULL);
                    CREATE TABLE IF NOT EXISTS name_groups (
                        group_id TEXT PRIMARY KEY, preferred_name TEXT NOT NULL,
                        description TEXT NOT NULL, version INTEGER NOT NULL, preferred_key TEXT NOT NULL);
                    CREATE INDEX IF NOT EXISTS curation_preferred ON name_groups(preferred_key,group_id);
                    CREATE TABLE IF NOT EXISTS group_members (
                        normalized_name TEXT PRIMARY KEY, group_id TEXT NOT NULL
                        REFERENCES name_groups(group_id) ON DELETE CASCADE);
                    CREATE INDEX IF NOT EXISTS curation_group_members ON group_members(group_id,normalized_name);
                    CREATE TABLE IF NOT EXISTS rejected_pairs (
                        first_name TEXT NOT NULL, second_name TEXT NOT NULL, reason TEXT NOT NULL,
                        PRIMARY KEY(first_name,second_name), CHECK(first_name<second_name));
                    CREATE TABLE IF NOT EXISTS curation_changes (
                        operation_id TEXT PRIMARY KEY, actor TEXT NOT NULL, request_hash TEXT NOT NULL,
                        action TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        before_json TEXT NOT NULL, after_json TEXT NOT NULL,
                        revision INTEGER NOT NULL UNIQUE, result_json TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS group_counts (
                        group_id TEXT NOT NULL, publication TEXT NOT NULL, revision INTEGER NOT NULL,
                        article_count INTEGER NOT NULL CHECK(article_count>=0),
                        PRIMARY KEY(group_id,publication,revision));
                """)
                with self.connection:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO curation_metadata VALUES(1,1,?,?,0)",
                        (_text(corpus_id, "corpus_id", 128), NORMALIZATION),
                    )
            metadata = self.metadata()
            if (
                metadata["schema_version"] != 1
                or metadata["corpus_id"] != corpus_id
                or metadata["normalization"] != NORMALIZATION
            ):
                raise ValueError("Curation store schema, corpus or normalization differs.")
            if read_only:
                self.connection.execute("PRAGMA query_only=ON")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def metadata(self):
        return dict(self.connection.execute("SELECT * FROM curation_metadata WHERE id=1").fetchone())

    @contextmanager
    def snapshot(self):
        self.connection.execute("BEGIN")
        try:
            yield self.metadata()["revision"]
        finally:
            self.connection.rollback()

    def group(self, group_id):
        row = self.connection.execute("SELECT * FROM name_groups WHERE group_id=?", (group_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown name group.")
        return {
            **dict(row),
            "members": [
                r[0]
                for r in self.connection.execute(
                    "SELECT normalized_name FROM group_members WHERE group_id=? ORDER BY normalized_name", (group_id,)
                )
            ],
        }

    def groups(self, *, after="", limit=50):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Use a group page size of 1-100.")
        return [
            self.group(r[0])
            for r in self.connection.execute(
                "SELECT group_id FROM name_groups WHERE group_id>? ORDER BY group_id LIMIT ?", (after, limit)
            )
        ]

    def membership(self, name):
        row = self.connection.execute(
            "SELECT group_id FROM group_members WHERE normalized_name=?", (normalize_name(name),)
        ).fetchone()
        return row[0] if row else None

    def rejected(self, first, second):
        pair = sorted((normalize_name(first), normalize_name(second)))
        return (
            self.connection.execute(
                "SELECT 1 FROM rejected_pairs WHERE first_name=? AND second_name=?", pair
            ).fetchone()
            is not None
        )

    def save_count(self, result, *, authorize=None):
        """Persist only a complete count for the still-current curation revision."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.group(result["group_id"])["version"] != result["group_version"]:
                raise Conflict("Curation changed while counting; retry the count.")
            self.group(result["group_id"])
            count = result["article_count"]
            if type(count) is not int or count < 0:
                raise ValueError("Expected an exact nonnegative count.")
            self.connection.execute(
                "INSERT OR REPLACE INTO group_counts VALUES(?,?,?,?)",
                (result["group_id"], result["generation"], self.group(result["group_id"])["version"], count),
            )
            if authorize is not None:
                authorize()
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _capture(self, group_id=None, pair=None):
        group = None
        if group_id is not None:
            row = self.connection.execute("SELECT 1 FROM name_groups WHERE group_id=?", (group_id,)).fetchone()
            group = self.group(group_id) if row else None
        rejection = None
        if pair is not None:
            row = self.connection.execute(
                "SELECT * FROM rejected_pairs WHERE first_name=? AND second_name=?", pair
            ).fetchone()
            rejection = dict(row) if row else None
        return {"group_id": group_id, "group": group, "pair": pair, "rejection": rejection}

    def _restore(self, state, version):
        group_id, group, pair = state["group_id"], state["group"], state["pair"]
        if group is not None:
            for member in group["members"]:
                if self.membership(member) not in (None, group_id):
                    raise Conflict("An affected name now belongs to another group.")
            for i, first in enumerate(group["members"]):
                if any(self.rejected(first, second) for second in group["members"][i + 1 :]):
                    raise Conflict("An affected pair is now rejected.")
        if group_id is not None:
            self.connection.execute("DELETE FROM name_groups WHERE group_id=?", (group_id,))
            if group is not None:
                self.connection.execute(
                    "INSERT INTO name_groups VALUES(?,?,?,?,?)",
                    (
                        group_id,
                        group["preferred_name"],
                        group["description"],
                        version,
                        normalize_name(group["preferred_name"]),
                    ),
                )
                self.connection.executemany(
                    "INSERT INTO group_members VALUES(?,?)", [(name, group_id) for name in group["members"]]
                )
        if pair is not None:
            rejection = state["rejection"]
            if (
                rejection is not None
                and self.membership(pair[0]) is not None
                and self.membership(pair[0]) == self.membership(pair[1])
            ):
                raise Conflict("An affected pair is now grouped together.")
            self.connection.execute("DELETE FROM rejected_pairs WHERE first_name=? AND second_name=?", pair)
            if rejection is not None:
                self.connection.execute(
                    "INSERT INTO rejected_pairs VALUES(:first_name,:second_name,:reason)", rejection
                )

    def mutate(self, action, payload, *, operation_id, actor, expected_revision, authorize=None):
        """Authorize inside the write transaction; duplicate delivery returns its saved result."""
        operation_id = str(uuid.UUID(_identifier(operation_id, "operation_id")))
        actor = _text(actor, "actor", 128)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer.")
        request_hash = hashlib.sha256(_json([action, payload, actor, expected_revision]).encode()).hexdigest()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            previous = self.connection.execute(
                "SELECT * FROM curation_changes WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if previous is not None:
                if previous["request_hash"] != request_hash:
                    raise Conflict("Operation ID was already used for a different request.")
                self.connection.rollback()
                return json.loads(previous["result_json"])
            revision = self.metadata()["revision"]
            if revision != expected_revision:
                raise Conflict("Curation revision changed; refresh before editing.")
            if not isinstance(payload, dict):
                raise ValueError("Expected an operation object.")
            pair = None
            group_id = None
            if action == "create_group":
                group_id = uuid.uuid4().hex
            elif action in ("update_group", "delete_group"):
                group_id = _identifier(payload.get("group_id"), "group_id")
            elif action in ("reject_pair", "reopen_pair"):
                pair = sorted(normalize_name(_text(payload.get(k), k)) for k in ("first", "second"))
            elif action == "undo_change":
                target = self.connection.execute(
                    "SELECT * FROM curation_changes WHERE operation_id=?",
                    (str(uuid.UUID(_identifier(payload.get("operation_id"), "operation_id"))),),
                ).fetchone()
                if target is None:
                    raise Conflict("Unknown change to undo.")
                saved_after = json.loads(target["after_json"])
                group_id, pair = saved_after["group_id"], saved_after["pair"]
                if self._capture(group_id, pair) != saved_after:
                    raise Conflict("Affected records changed; edit explicitly instead.")
            before = self._capture(group_id, pair)
            if action == "undo_change":
                if set(payload) != {"operation_id"}:
                    raise ValueError("Supply only operation_id for undo.")
                self._restore(json.loads(target["before_json"]), revision + 1)
            else:
                self._apply(action, payload, revision + 1, group_id)
            result = {"operation_id": operation_id, "revision": revision + 1, "group_id": group_id}
            after = self._capture(group_id, pair)
            self.connection.execute("UPDATE curation_metadata SET revision=? WHERE id=1", (revision + 1,))
            self.connection.execute(
                "INSERT INTO curation_changes(operation_id,actor,request_hash,action,"
                "before_json,after_json,revision,result_json) VALUES(?,?,?,?,?,?,?,?)",
                (operation_id, actor, request_hash, action, _json(before), _json(after), revision + 1, _json(result)),
            )
            if authorize is not None:
                authorize()
            self.connection.commit()
            return result
        except BaseException:
            self.connection.rollback()
            raise

    def _apply(self, action, payload, version, group_id=None):
        if not isinstance(payload, dict):
            raise ValueError("Expected an operation object.")
        if action in ("create_group", "update_group"):
            allowed = {"group_id", "preferred_name", "description", "members"}
            if set(payload) - allowed:
                raise ValueError("Unknown group fields.")
            if action == "update_group":
                self.group(group_id)
            preferred = _text(payload.get("preferred_name"), "preferred_name")
            description = payload.get("description", "")
            if not isinstance(description, str) or len(description) > 2000:
                raise ValueError("description must be text of at most 2000 characters.")
            members = payload.get("members")
            if not isinstance(members, list) or not 1 <= len(members) <= MAX_MEMBERS:
                raise ValueError("Supply 1-100 members.")
            members = sorted({normalize_name(_text(n, "member")) for n in members})
            for member in members:
                owner = self.membership(member)
                if owner is not None and owner != group_id:
                    raise Conflict("A name already belongs to another group; remove it there first.")
            for i, first in enumerate(members):
                if any(self.rejected(first, second) for second in members[i + 1 :]):
                    raise Conflict("Reopen a rejected pair before grouping its names.")
            self.connection.execute(
                "INSERT INTO name_groups VALUES(?,?,?,?,?) ON CONFLICT(group_id) DO UPDATE SET "
                "preferred_name=excluded.preferred_name,description=excluded.description,version=excluded.version,"
                "preferred_key=excluded.preferred_key",
                (group_id, preferred, description, version, normalize_name(preferred)),
            )
            self.connection.execute("DELETE FROM group_members WHERE group_id=?", (group_id,))
            self.connection.executemany("INSERT INTO group_members VALUES(?,?)", [(name, group_id) for name in members])
            return group_id
        if action in ("reject_pair", "reopen_pair"):
            if set(payload) - {"first", "second", "reason"}:
                raise ValueError("Unknown pair fields.")
            first, second = sorted(normalize_name(_text(payload.get(k), k)) for k in ("first", "second"))
            if first == second:
                raise ValueError("A pair needs distinct normalized names.")
            reason = _text(payload.get("reason"), "reason")
            if action == "reject_pair":
                if self.membership(first) is not None and self.membership(first) == self.membership(second):
                    raise Conflict("Remove the shared group membership before rejecting the pair.")
                self.connection.execute(
                    "INSERT INTO rejected_pairs VALUES(?,?,?) ON CONFLICT(first_name,second_name) "
                    "DO UPDATE SET reason=excluded.reason",
                    (first, second, reason),
                )
            else:
                self.connection.execute(
                    "DELETE FROM rejected_pairs WHERE first_name=? AND second_name=?", (first, second)
                )
            return None
        if action == "delete_group":
            if set(payload) != {"group_id"}:
                raise ValueError("Supply only group_id for dissolution.")
            self.group(group_id)
            self.connection.execute("DELETE FROM name_groups WHERE group_id=?", (group_id,))
            return group_id
        raise ValueError("Unsupported curation operation.")
