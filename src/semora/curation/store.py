"""Persistent, transactional name groups. Only this store is writable."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from semora.ner.names import normalize_name

from .provenance import annotation, autonomous_annotation, normalize_group

NORMALIZATION = "nfc-casefold-whitespace-v1"
MAX_MEMBERS = 100


class Conflict(ValueError):
    """The caller must refresh state before retrying a different operation."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def membership_key(members):
    """Stable count identity, independent of display metadata and audit revisions."""
    return hashlib.sha256(_json(sorted(members)).encode()).hexdigest()


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
        new_store = not self.path.exists()
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
                metadata["schema_version"] not in (1, 2)
                or metadata["corpus_id"] != corpus_id
                or metadata["normalization"] != NORMALIZATION
            ):
                raise ValueError("Curation store schema, corpus or normalization differs.")
            if not read_only and metadata["schema_version"] == 1:
                self._upgrade_provenance(backup=not new_store)
            if read_only:
                self.connection.execute("PRAGMA query_only=ON")
        except BaseException:
            self.close()
            raise

    def _upgrade_provenance(self, *, backup=True):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            # Interactive/count lanes can open together. Recheck after taking the writer lock.
            if self.metadata()["schema_version"] != 1:
                self.connection.rollback()
                return
            if backup:
                path = self.path.with_name(self.path.name + ".schema1.backup-" + uuid.uuid4().hex)
                # A separate read connection sees the committed schema1 snapshot while
                # this connection holds the write reservation; self-backup would block.
                with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as source:
                    with closing(sqlite3.connect(path)) as target:
                        source.backup(target)
            for statement in (
                "CREATE TABLE group_provenance (group_id TEXT PRIMARY KEY REFERENCES name_groups(group_id) "
                "ON DELETE CASCADE, origin TEXT NOT NULL, "
                "human_reviewed INTEGER NOT NULL CHECK(human_reviewed IN(0,1)), "
                "provenance_json TEXT NOT NULL)",
                "CREATE TABLE member_provenance (normalized_name TEXT PRIMARY KEY REFERENCES "
                "group_members(normalized_name) ON DELETE CASCADE, origin TEXT NOT NULL, "
                "human_reviewed INTEGER NOT NULL CHECK(human_reviewed IN(0,1)), provenance_json TEXT NOT NULL)",
                "CREATE TABLE curation_serving (id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL "
                "CHECK(enabled IN(0,1)), settings_revision INTEGER NOT NULL CHECK(settings_revision>=0))",
                "CREATE TABLE curation_setting_receipts (operation_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, "
                "result_json TEXT NOT NULL)",
            ):
                self.connection.execute(statement)
            # Only an exact current snapshot with an explicit admin actor proves review.
            # This visits the small sidecar audit, never the article corpus.
            known = {}
            for row in self.connection.execute("SELECT actor,after_json FROM curation_changes ORDER BY revision"):
                state = json.loads(row["after_json"])
                group = state.get("group")
                if group is not None:
                    known[state["group_id"]] = (group, annotation(row["actor"]))
            for row in self.connection.execute("SELECT group_id FROM name_groups"):
                group_id = row[0]
                current = self._legacy_group(group_id)
                prior = known.get(group_id)
                value = prior[1] if prior and prior[0] == current else annotation("")
                self._save_annotations(group_id, value, {name: value for name in current["members"]})
            self.connection.execute("UPDATE curation_metadata SET schema_version=2 WHERE id=1")
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
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

    def _legacy_group(self, group_id):
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

    def group(self, group_id):
        group = self._legacy_group(group_id)
        default = annotation("")
        if self.metadata()["schema_version"] == 1:
            return {**group, **default, "member_provenance": {name: default for name in group["members"]}}
        row = self.connection.execute("SELECT * FROM group_provenance WHERE group_id=?", (group_id,)).fetchone()
        value = self._annotation_row(row) if row else default
        members = {name: default for name in group["members"]}
        for row in self.connection.execute(
            "SELECT p.* FROM group_members m JOIN member_provenance p "
            "ON p.normalized_name=m.normalized_name WHERE m.group_id=?",
            (group_id,),
        ):
            members[row["normalized_name"]] = self._annotation_row(row)
        origins = {item["origin"] for item in members.values()}
        origin = next(iter(origins)) if len(origins) == 1 else "mixed"
        return {
            **group,
            **value,
            "origin": origin,
            "human_reviewed": value["human_reviewed"] and all(item["human_reviewed"] for item in members.values()),
            "member_provenance": members,
        }

    @staticmethod
    def _annotation_row(row):
        return {
            "origin": row["origin"],
            "human_reviewed": bool(row["human_reviewed"]),
            "provenance": json.loads(row["provenance_json"]),
        }

    def _save_annotations(self, group_id, value, members):
        self.connection.execute(
            "INSERT OR REPLACE INTO group_provenance VALUES(?,?,?,?)",
            (group_id, value["origin"], int(value["human_reviewed"]), _json(value["provenance"])),
        )
        for name, item in members.items():
            self.connection.execute(
                "INSERT OR REPLACE INTO member_provenance VALUES(?,?,?,?)",
                (name, item["origin"], int(item["human_reviewed"]), _json(item["provenance"])),
            )

    def serving_enabled(self):
        """None preserves the gateway's existing environment-based behavior."""
        if self.metadata()["schema_version"] == 1:
            return None
        row = self.connection.execute("SELECT enabled FROM curation_serving WHERE id=1").fetchone()
        return bool(row[0]) if row else None

    def configure_serving(self, enabled, settings_revision, operation_id, authorize=None):
        if type(enabled) is not bool or type(settings_revision) is not int or settings_revision < 0:
            raise ValueError("Expected a Boolean enabled flag and nonnegative settings revision.")
        operation_id = str(uuid.UUID(_identifier(operation_id, "operation_id")))
        request_hash = hashlib.sha256(_json([enabled, settings_revision]).encode()).hexdigest()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            prior = self.connection.execute(
                "SELECT * FROM curation_setting_receipts WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if prior:
                if prior["request_hash"] != request_hash:
                    raise Conflict("Operation ID was already used for different settings.")
                result = json.loads(prior["result_json"])
            else:
                current = self.connection.execute("SELECT * FROM curation_serving WHERE id=1").fetchone()
                if (
                    current
                    and settings_revision == current["settings_revision"]
                    and enabled != bool(current["enabled"])
                ):
                    raise Conflict("Settings revision was reused with different settings.")
                if current is None or settings_revision > current["settings_revision"]:
                    self.connection.execute(
                        "INSERT OR REPLACE INTO curation_serving VALUES(1,?,?)", (int(enabled), settings_revision)
                    )
                    result = {"applied_settings_revision": settings_revision, "serving_enabled": enabled}
                else:
                    result = {
                        "applied_settings_revision": current["settings_revision"],
                        "serving_enabled": bool(current["enabled"]),
                    }
                self.connection.execute(
                    "INSERT INTO curation_setting_receipts VALUES(?,?,?)", (operation_id, request_hash, _json(result))
                )
            if authorize is not None:
                authorize()
            self.connection.commit()
            return result
        except BaseException:
            self.connection.rollback()
            raise

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
            current = self.group(result["group_id"])
            if "membership_key" in result:
                unchanged = result["membership_key"] == membership_key(current["members"])
            else:
                unchanged = current["version"] == result["group_version"]
            if not unchanged:
                raise Conflict("Curation changed while counting; retry the count.")
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
                normalized = normalize_group(group)
                self._save_annotations(group_id, normalized, normalized["member_provenance"])
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
        """Explicit manual operation; existing request hashes remain replayable."""
        if action == "apply_autonomous":
            raise ValueError("Use the dedicated autonomous publication method.")
        return self._mutate(
            action,
            payload,
            operation_id=operation_id,
            actor=actor,
            expected_revision=expected_revision,
            authorize=authorize,
        )

    def apply_autonomous(self, payload, *, provenance, operation_id, validate, authorize=None):
        """Append safe memberships after bounded affected-state validation under the writer lock."""
        if not callable(validate):
            raise ValueError("Autonomous publication requires affected-state validation.")
        value = autonomous_annotation(provenance)
        if not isinstance(payload, dict):
            raise ValueError("Expected an operation object.")
        payload = json.loads(_json(payload))
        payload.setdefault("group_id", _identifier(operation_id, "operation_id"))
        return self._mutate(
            "apply_autonomous",
            payload,
            operation_id=operation_id,
            actor="system:autonomous-curation",
            expected_revision=None,
            authorize=authorize,
            autonomous=value,
            validate=validate,
        )

    def _mutate(
        self, action, payload, *, operation_id, actor, expected_revision, authorize=None, autonomous=None, validate=None
    ):
        """Authorize inside the write transaction; duplicate delivery returns its saved result."""
        operation_id = str(uuid.UUID(_identifier(operation_id, "operation_id")))
        actor = _text(actor, "actor", 128)
        if autonomous is None and (type(expected_revision) is not int or expected_revision < 0):
            raise ValueError("expected_revision must be a nonnegative integer.")
        material = [action, payload, actor, expected_revision]
        if autonomous is not None:
            material.append(autonomous)
        request_hash = hashlib.sha256(_json(material).encode()).hexdigest()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if autonomous is not None:
                settings = self.connection.execute("SELECT * FROM curation_serving WHERE id=1").fetchone()
                settings_revision = autonomous["provenance"].get("settings_revision")
                if (
                    settings is None
                    or not settings["enabled"]
                    or type(settings_revision) is not int
                    or settings_revision != settings["settings_revision"]
                ):
                    raise Conflict("Autonomous publication settings are disabled or stale.")
            previous = self.connection.execute(
                "SELECT * FROM curation_changes WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if previous is not None:
                if previous["request_hash"] != request_hash:
                    raise Conflict("Operation ID was already used for a different request.")
                if autonomous is not None and authorize is not None:
                    authorize()
                self.connection.rollback()
                return json.loads(previous["result_json"])
            revision = self.metadata()["revision"]
            if autonomous is not None:
                validate()
            if autonomous is None and revision != expected_revision:
                raise Conflict("Curation revision changed; refresh before editing.")
            if not isinstance(payload, dict):
                raise ValueError("Expected an operation object.")
            pair = None
            group_id = None
            if action == "create_group":
                group_id = uuid.uuid4().hex
            elif action in ("update_group", "delete_group", "apply_autonomous"):
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
                current = self._capture(group_id, pair)
                if saved_after["group"] is not None and "member_provenance" not in saved_after["group"]:
                    current["group"] = self._legacy_group(group_id) if current["group"] else None
                if current != saved_after:
                    raise Conflict("Affected records changed; edit explicitly instead.")
            before = self._capture(group_id, pair)
            if action == "undo_change":
                if set(payload) != {"operation_id"}:
                    raise ValueError("Supply only operation_id for undo.")
                self._restore(json.loads(target["before_json"]), revision + 1)
            else:
                applied_action = action
                if action == "apply_autonomous":
                    applied_action = "update_group" if before["group"] is not None else "create_group"
                self._apply(applied_action, payload, revision + 1, group_id, autonomous=autonomous, actor=actor)
            result = {"operation_id": operation_id, "revision": revision + 1, "group_id": group_id}
            after = self._capture(group_id, pair)
            result["group_exists"] = after["group"] is not None
            if (
                before["group"] is not None
                and after["group"] is not None
                and before["group"]["members"] == after["group"]["members"]
            ):
                # Promote valid counts atomically, preserving monotonic record revisions for undo.
                self.connection.execute(
                    "INSERT OR IGNORE INTO group_counts SELECT group_id,publication,?,article_count "
                    "FROM group_counts WHERE group_id=? AND revision=?",
                    (after["group"]["version"], group_id, before["group"]["version"]),
                )
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

    def _apply(self, action, payload, version, group_id=None, *, autonomous=None, actor=""):
        if not isinstance(payload, dict):
            raise ValueError("Expected an operation object.")
        if action in ("create_group", "update_group"):
            allowed = {"group_id", "preferred_name", "description", "members"}
            if set(payload) - allowed:
                raise ValueError("Unknown group fields.")
            prior = self.group(group_id) if action == "update_group" else None
            preferred = _text(payload.get("preferred_name"), "preferred_name")
            description = payload.get("description", "")
            if not isinstance(description, str) or len(description) > 2000:
                raise ValueError("description must be text of at most 2000 characters.")
            members = payload.get("members")
            if not isinstance(members, list) or not 1 <= len(members) <= MAX_MEMBERS:
                raise ValueError("Supply 1-100 members.")
            members = sorted({normalize_name(_text(n, "member")) for n in members})
            if autonomous is not None and len(members) < 2:
                raise ValueError("Autonomous groups require at least two spellings.")
            if autonomous is not None and prior is not None:
                if set(prior["members"]) == set(members):
                    raise Conflict("No additional spelling was selected.")
                if not set(prior["members"]) <= set(members):
                    raise Conflict("Autonomous publication cannot remove existing memberships.")
                protected = any(
                    item["origin"] != "autonomous" or item["human_reviewed"]
                    for item in prior["member_provenance"].values()
                )
                if protected and (preferred != prior["preferred_name"] or description != prior["description"]):
                    raise Conflict("Autonomous publication cannot overwrite reviewed or unknown group metadata.")
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
            value = autonomous if autonomous is not None else annotation(actor)
            member_values = {name: value for name in members}
            if prior is not None:
                if autonomous is not None:
                    member_values.update(prior["member_provenance"])
                elif value["human_reviewed"]:
                    for name in set(members) & prior["member_provenance"].keys():
                        member_values[name] = {**prior["member_provenance"][name], "human_reviewed": True}
                    value = {"origin": prior["origin"], "human_reviewed": True, "provenance": prior["provenance"]}
            self._save_annotations(group_id, value, member_values)
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
