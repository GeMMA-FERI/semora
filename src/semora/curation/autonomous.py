"""Bounded autonomous vocabulary work; model output is never publication authority."""

from __future__ import annotations

import copy
import json
import time
import unicodedata
import uuid
from types import SimpleNamespace
from typing import Any

from semora.ner.names import normalize_name

from .frequency import FrequencyNotReady, frequency_page
from .proposals import PROMPT_VERSION, RecordedState, digest, prepare_task, review_response, validate_response
from .store import Conflict

POLICY_VERSION = "conservative-ocr-v1"
MAX_FRONTIER = 512
COOLDOWN_SECONDS = 30 * 24 * 3600
MAX_ATTEMPTS = 3
RESERVATION_SECONDS = 3600


def initialize_autonomous(connection):
    """Called only on a writable sidecar; never modifies the published dictionary."""
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS autonomous_cursors (
            generation TEXT NOT NULL, policy_version TEXT NOT NULL, after_entity_id INTEGER NOT NULL DEFAULT 0,
            exhausted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(generation,policy_version));
        CREATE TABLE IF NOT EXISTS autonomous_frequency_cursors (
            generation TEXT NOT NULL, policy_version TEXT NOT NULL, frequency_identity TEXT NOT NULL,
            after_rank INTEGER NOT NULL DEFAULT 0,
            exhausted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(generation,policy_version));
        CREATE TABLE IF NOT EXISTS autonomous_released_names(normalized_name TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS autonomous_frontier (
            generation TEXT NOT NULL, policy_version TEXT NOT NULL, normalized_name TEXT NOT NULL,
            article_count INTEGER NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            next_due REAL NOT NULL DEFAULT 0, operation_id TEXT, reason TEXT,
            PRIMARY KEY(generation,policy_version,normalized_name));
        CREATE INDEX IF NOT EXISTS autonomous_frontier_priority ON autonomous_frontier
            (generation,policy_version,article_count DESC,normalized_name) WHERE status='queued';
        CREATE INDEX IF NOT EXISTS autonomous_frontier_due ON autonomous_frontier
            (generation,policy_version,status,next_due,normalized_name);
        CREATE TABLE IF NOT EXISTS autonomous_discovery_receipts (
            operation_id TEXT PRIMARY KEY, generation TEXT NOT NULL, policy_version TEXT NOT NULL,
            anchor TEXT, request_hash TEXT NOT NULL, result_json TEXT);
        CREATE TABLE IF NOT EXISTS autonomous_evaluations (
            evaluation_id TEXT PRIMARY KEY, result_json TEXT NOT NULL);
    """)


def _operation(value):
    try:
        return uuid.UUID(value).hex
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("An autonomous operation requires a UUID.") from error


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _initialize(store):
    if store.connection.in_transaction:
        raise ValueError("Autonomous discovery must own its short sidecar transactions.")
    if not getattr(store, "_autonomous_initialized", False):
        initialize_autonomous(store.connection)
        store._autonomous_initialized = True


def _store_result(store, operation, result):
    connection = store.connection
    with connection:
        prior = connection.execute(
            "SELECT result_json FROM autonomous_discovery_receipts WHERE operation_id=?", (operation,)
        ).fetchone()
        if prior and prior[0]:
            return json.loads(prior[0])
        connection.execute(
            "UPDATE autonomous_discovery_receipts SET result_json=? WHERE operation_id=?", (_json(result), operation)
        )
    return result


def _defer(store, generation, anchor, reason, *, now, operation_id):
    with store.connection:
        store.connection.execute(
            "UPDATE autonomous_frontier SET status=CASE WHEN attempts>=? THEN 'exhausted' ELSE 'deferred' END,"
            "next_due=?,reason=? WHERE generation=? AND policy_version=? AND normalized_name=? "
            "AND operation_id=? AND status IN ('preparing','prepared')",
            (MAX_ATTEMPTS, now + COOLDOWN_SECONDS, reason[:500], generation, POLICY_VERSION, anchor, operation_id),
        )


def _empty_result(store, operation, generation, reason=None):
    cursor = store.connection.execute(
        "SELECT exhausted FROM autonomous_frequency_cursors WHERE generation=? AND policy_version=?",
        (generation, POLICY_VERSION),
    ).fetchone()
    queued = store.connection.execute(
        "SELECT 1 FROM autonomous_frontier WHERE generation=? AND policy_version=? AND status='queued' LIMIT 1",
        (generation, POLICY_VERSION),
    ).fetchone()
    released = store.connection.execute("SELECT 1 FROM autonomous_released_names LIMIT 1").fetchone()
    result = {
        "status": "empty", "discovery_id": operation,
        "has_more": bool(queued or released or (cursor and not cursor[0])),
    }
    if reason:
        result["reason"] = reason[:500]
    return result


def _grouped_names(connection, names):
    """Check one bounded page against the authoritative name-membership primary key."""
    names = list(dict.fromkeys(names))
    if not names:
        return set()
    return {row[0] for row in connection.execute(
        "SELECT normalized_name FROM group_members WHERE normalized_name IN ("
        + ",".join("?" for _ in names) + ")", names
    )}


def _stage_released(catalogue, generation, limit):
    """Stage a bounded queue of names released by deletion/undo without resetting the rank cursor."""
    if limit <= 0:
        return 0
    connection = catalogue.store.connection
    names = [row[0] for row in connection.execute(
        "SELECT normalized_name FROM autonomous_released_names ORDER BY normalized_name LIMIT ?", (limit,)
    )]
    if not names:
        return 0
    grouped = _grouped_names(connection, names)
    placeholders = ",".join("?" for _ in names)
    counts = {row[0]: row[1] for row in catalogue.connection.execute(
        f"SELECT normalized_name,article_count FROM names WHERE normalized_name IN ({placeholders})", names
    )}
    prior = {row[0]: (row[1], row[2]) for row in connection.execute(
        "SELECT normalized_name,status,attempts FROM autonomous_frontier WHERE generation=? AND policy_version=? "
        f"AND normalized_name IN ({placeholders})", (generation, POLICY_VERSION, *names)
    )}
    staged, removed = [], []
    for name in names:
        if name in grouped or name not in counts:
            removed.append(name)
        else:
            status, attempts = prior.get(name, (None, 0))
            if status in ("preparing", "prepared"):
                continue
            # A release does not reset retries or shorten a previously recorded
            # cooldown. Those rows are already managed by the normal frontier.
            if status in (None, "queued", "grouped", "published") and attempts < MAX_ATTEMPTS:
                staged.append((generation, POLICY_VERSION, name, counts[name]))
            removed.append(name)
    connection.executemany(
        "INSERT INTO autonomous_frontier(generation,policy_version,normalized_name,article_count,status) "
        "VALUES(?,?,?,?,'queued') ON CONFLICT(generation,policy_version,normalized_name) DO UPDATE SET "
        "article_count=excluded.article_count,status='queued',next_due=0", staged
    )
    connection.executemany(
        "DELETE FROM autonomous_released_names WHERE normalized_name=?", [(name,) for name in removed]
    )
    return sum(prior.get(row[2], (None, 0))[0] != "queued" for row in staged)


def discover_task(
    catalogue: Any, *, operation_id: str, scan_limit: int = 128, candidate_limit: int = 8, now: float | None = None
) -> dict[str, Any]:
    """Stage globally common names from a prepared rank index using bounded keyset pages."""
    if type(scan_limit) is not int or not 1 <= scan_limit <= 512:
        raise ValueError("Use a discovery scan of 1-512 names.")
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 12:
        raise ValueError("Use 1-12 autonomous proposal candidates.")
    generation = catalogue.entities.generation
    if not isinstance(generation, str) or not generation:
        raise ValueError("Autonomous discovery requires a published entity generation.")
    operation = _operation(operation_id)
    request_hash = digest({"scan_limit": scan_limit, "candidate_limit": candidate_limit})
    now = time.time() if now is None else now
    store = catalogue.store
    _initialize(store)
    connection = store.connection
    connection.execute("BEGIN IMMEDIATE")
    try:
        expired = connection.execute(
            "SELECT normalized_name FROM autonomous_frontier WHERE generation=? AND policy_version=? "
            "AND status IN ('preparing','prepared') AND next_due<=? ORDER BY next_due,normalized_name LIMIT ?",
            (generation, POLICY_VERSION, now, scan_limit),
        ).fetchall()
        connection.executemany(
            "UPDATE autonomous_frontier SET status=CASE WHEN attempts>=? THEN 'exhausted' ELSE 'deferred' END,"
            "next_due=?,reason='Abandoned pipeline reservation expired.' WHERE generation=? AND policy_version=? "
            "AND normalized_name=?",
            [(MAX_ATTEMPTS, now + COOLDOWN_SECONDS, generation, POLICY_VERSION, row[0]) for row in expired],
        )
        receipt = connection.execute(
            "SELECT * FROM autonomous_discovery_receipts WHERE operation_id=?", (operation,)
        ).fetchone()
        if receipt:
            if (
                receipt["generation"] != generation
                or receipt["policy_version"] != POLICY_VERSION
                or receipt["request_hash"] != request_hash
            ):
                raise Conflict("Autonomous discovery operation belongs to another publication or policy.")
            if receipt["result_json"]:
                connection.rollback()
                return json.loads(receipt["result_json"])
            anchor = receipt["anchor"]
            reservation = (
                connection.execute(
                    "SELECT status,operation_id FROM autonomous_frontier WHERE generation=? AND policy_version=? "
                    "AND normalized_name=?",
                    (generation, POLICY_VERSION, anchor),
                ).fetchone()
                if anchor
                else None
            )
            if anchor and (
                not reservation
                or reservation["operation_id"] != operation
                or reservation["status"] not in ("preparing", "prepared")
            ):
                connection.commit()
                return _store_result(
                    store,
                    operation,
                    _empty_result(store, operation, generation, "The original discovery reservation expired."),
                )
        else:
            try:
                frequency_page(catalogue.entities, 0, 0)
            except FrequencyNotReady as error:
                result = {"status": "empty", "discovery_id": operation, "has_more": True,
                          "reason": str(error)}
                connection.execute(
                    "INSERT INTO autonomous_discovery_receipts VALUES(?,?,?,?,?,?)",
                    (operation, generation, POLICY_VERSION, None, request_hash, _json(result)),
                )
                connection.commit()
                return result
            new_order = connection.execute(
                "INSERT OR IGNORE INTO autonomous_frequency_cursors"
                "(generation,policy_version,frequency_identity) VALUES(?,?,?)",
                (generation, POLICY_VERSION, catalogue.entities.frequency_identity),
            ).rowcount
            if new_order:
                # Old ID-ordered staging has no reservations. Refill it from rank zero,
                # preserving every in-flight lease, receipt, cooldown and published group.
                connection.execute(
                    "DELETE FROM autonomous_frontier WHERE generation=? AND policy_version=? AND status='queued' "
                    "AND attempts=0 AND operation_id IS NULL",
                    (generation, POLICY_VERSION),
                )
            # Membership is the checkmark. Retire only queued work; keep saved
            # receipts and active reservations unchanged for replay/reconciliation.
            connection.execute(
                "UPDATE autonomous_frontier SET status='grouped' WHERE generation=? AND policy_version=? "
                "AND status='queued' AND EXISTS(SELECT 1 FROM group_members m "
                "WHERE m.normalized_name=autonomous_frontier.normalized_name)",
                (generation, POLICY_VERSION),
            )
            active = connection.execute(
                "SELECT COUNT(*) FROM autonomous_frontier WHERE generation=? AND policy_version=? "
                "AND status IN ('queued','preparing','prepared')",
                (generation, POLICY_VERSION),
            ).fetchone()[0]
            free = max(0, MAX_FRONTIER - active)
            free -= _stage_released(catalogue, generation, min(scan_limit, free))
            due = connection.execute(
                "SELECT normalized_name FROM autonomous_frontier WHERE generation=? AND policy_version=? "
                "AND status='deferred' AND next_due<=? ORDER BY next_due,normalized_name LIMIT ?",
                (generation, POLICY_VERSION, now, min(scan_limit, free)),
            ).fetchall()
            grouped = _grouped_names(connection, [row[0] for row in due])
            connection.executemany(
                "UPDATE autonomous_frontier SET status='grouped' WHERE generation=? AND policy_version=? "
                "AND normalized_name=?",
                [(generation, POLICY_VERSION, name) for name in grouped],
            )
            due = [row for row in due if row[0] not in grouped]
            connection.executemany(
                "UPDATE autonomous_frontier SET status='queued' WHERE generation=? AND policy_version=? "
                "AND normalized_name=?",
                [(generation, POLICY_VERSION, row[0]) for row in due],
            )
            cursor = connection.execute(
                "SELECT after_rank,exhausted,frequency_identity FROM autonomous_frequency_cursors "
                "WHERE generation=? AND policy_version=?",
                (generation, POLICY_VERSION),
            ).fetchone()
            if cursor["frequency_identity"] != catalogue.entities.frequency_identity:
                raise Conflict("Frequency order changed within the published entity generation.")
            capacity = min(scan_limit, max(0, free - len(due)))
            if capacity and not cursor["exhausted"]:
                rows = frequency_page(catalogue.entities, cursor["after_rank"], capacity)
                grouped = _grouped_names(connection, [row["normalized_name"] for row in rows])
                connection.executemany(
                    "INSERT OR IGNORE INTO autonomous_frontier"
                    "(generation,policy_version,normalized_name,article_count,status) "
                    "VALUES(?,?,?,?,'queued')",
                    [(generation, POLICY_VERSION, row["normalized_name"], row["article_count"])
                     for row in rows if row["normalized_name"] not in grouped],
                )
                connection.execute(
                    "UPDATE autonomous_frequency_cursors SET after_rank=?,exhausted=? "
                    "WHERE generation=? AND policy_version=?",
                    (
                        rows[-1]["rank"] if rows else cursor["after_rank"],
                        int(len(rows) < capacity),
                        generation,
                        POLICY_VERSION,
                    ),
                )
            selected = connection.execute(
                "SELECT normalized_name FROM autonomous_frontier WHERE generation=? AND policy_version=? "
                "AND status='queued' AND NOT EXISTS(SELECT 1 FROM group_members m "
                "WHERE m.normalized_name=autonomous_frontier.normalized_name) "
                "ORDER BY article_count DESC,normalized_name LIMIT 1",
                (generation, POLICY_VERSION),
            ).fetchone()
            anchor = selected[0] if selected else None
            connection.execute(
                "INSERT INTO autonomous_discovery_receipts VALUES(?,?,?,?,?,NULL)",
                (operation, generation, POLICY_VERSION, anchor, request_hash),
            )
            if anchor:
                connection.execute(
                    "UPDATE autonomous_frontier SET status='preparing',attempts=attempts+1,operation_id=?,next_due=? "
                    "WHERE generation=? AND policy_version=? AND normalized_name=?",
                    (operation, now + RESERVATION_SECONDS, generation, POLICY_VERSION, anchor),
                )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    if anchor is None:
        return _store_result(store, operation, _empty_result(store, operation, generation))
    try:
        group_id = store.membership(anchor)
        if group_id:
            # Another pipeline can publish between anchor selection and preparation.
            # There is no model work to do once that spelling has a group.
            with connection:
                connection.execute(
                    "UPDATE autonomous_frontier SET status='grouped' WHERE generation=? AND policy_version=? "
                    "AND normalized_name=? AND operation_id=? AND status='preparing'",
                    (generation, POLICY_VERSION, anchor, operation),
                )
            return _store_result(store, operation, _empty_result(
                store, operation, generation, "Discovery anchor already belongs to a curation group."
            ))
        group = None
        prepared = prepare_task(
            catalogue, anchor, group_id=group_id, limit=candidate_limit, include_autonomous_groups=True
        )
        stats = {}
        for name in [anchor, *prepared["context"]["candidates"]]:
            row = catalogue.connection.execute(
                "SELECT name,article_count FROM names WHERE normalized_name=?", (name,)
            ).fetchone()
            if row is None:
                raise Conflict("An autonomous candidate is absent from the published dictionary.")
            stats[name] = dict(row)
        prepared["context"]["autonomous"] = {
            "policy_version": POLICY_VERSION,
            "discovery_id": operation,
            "stats": stats,
            "preferred_name": group["preferred_name"] if group_id else None,
        }
        prepared["fingerprint"] = digest({key: prepared[key] for key in ("context", "task")})
        result = {"status": "prepared", "discovery_id": operation, "prepared": prepared}
        with connection:
            connection.execute(
                "UPDATE autonomous_frontier SET status='prepared' WHERE generation=? AND policy_version=? "
                "AND normalized_name=? AND operation_id=?",
                (generation, POLICY_VERSION, anchor, operation),
            )
        return _store_result(store, operation, result)
    except ValueError as error:
        if isinstance(error, Conflict):
            raise
        _defer(store, generation, anchor, str(error), now=now, operation_id=operation)
        return _store_result(store, operation, _empty_result(store, operation, generation, str(error)))


def _letters(value):
    return "".join(
        c for c in unicodedata.normalize("NFKD", normalize_name(value)) if c.isalnum() and not unicodedata.combining(c)
    )


def _ocr_signature(value):
    return _letters(value).replace("rn", "m").translate(str.maketrans({"l": "i", "0": "o", "1": "i"}))


def _policy_accepts(anchor, name, stats, *, symmetric=True):
    first, second = _letters(anchor), _letters(name)
    if first == second and len(first) >= 4:
        return True
    # Restricted confusion-equivalent spellings, not arbitrary edit-distance edges.
    # Frequency is a conservative tie-breaker, not a claim of semantic certainty.
    first_count, second_count = stats[anchor]["article_count"], stats[name]["article_count"]
    dominant = max(first_count, second_count) if symmetric else first_count
    subordinate = min(first_count, second_count) if symmetric else second_count
    return (
        len(first) >= 8
        and len(second) >= 8
        and first[0] == second[0]
        and _ocr_signature(anchor) == _ocr_signature(name)
        and dominant >= 10
        and dominant >= 5 * max(1, subordinate)
    )


def _policy_representatives(catalogue, prepared, response):
    """Use group canonicals instead of rare aliases when checking automatic unions."""
    context = prepared["context"]
    stats = dict(context["autonomous"]["stats"])
    names = [context["anchor"], *[d["name"] for d in response["decisions"] if d["decision"] == "include"]]
    representatives, groups = {}, {}
    nodes = []
    for name in names:
        owner = catalogue.store.membership(name)
        if owner is None:
            representatives[name] = name
            if name == context["anchor"]:
                nodes.append(name)
            continue
        if owner not in groups:
            group = catalogue.store.group(owner, include_provenance=False)
            if group.get("origin") != "autonomous" or group.get("human_reviewed") is not False:
                raise Conflict("Autonomous union cannot modify a protected group.")
            representative = normalize_name(group["preferred_name"])
            snapshot = context.get("autonomous_groups", {}).get(owner)
            if snapshot:
                if representative != snapshot["representative"]:
                    raise Conflict("Autonomous group representative changed during inference.")
                value = snapshot["representative_stats"]
            else:
                # Only a concurrently created owner reaches this path. The pinned
                # dictionary is immutable; a bounded UNIQUE lookup may reject an
                # old model decision without changing its original operation hash.
                row = catalogue.connection.execute(
                    "SELECT name,article_count FROM names WHERE normalized_name=?", (representative,)
                ).fetchone()
                if row is None:
                    raise Conflict("Autonomous group representative is absent from the publication.")
                value = dict(row)
            if not isinstance(catalogue.store, RecordedState) and representative not in group["members"]:
                raise Conflict("Autonomous group representative is not a group member.")
            groups[owner] = representative
            stats[representative] = value
            nodes.append(representative)
        representatives[name] = groups[owner]
    nodes = list(dict.fromkeys(nodes))
    compatible = all(
        _policy_accepts(first, second, stats)
        for index, first in enumerate(nodes) for second in nodes[index + 1:]
    )
    baseline = max(nodes, key=lambda name: stats[name]["article_count"])
    return representatives, stats, baseline, compatible


def _final_result(catalogue, prepared, response):
    context = prepared["context"]
    overlap = context.get("overlap_version") == "unreviewed-union-v1"
    checked = validate_response(
        prepared, response, catalogue.store, catalogue.entities.generation, allow_autonomous_overlap=overlap
    )
    automatic = context.get("autonomous", {})
    if automatic.get("policy_version") != POLICY_VERSION or context["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Unsupported autonomous policy or prompt version.")
    stats = automatic["stats"]
    response = copy.deepcopy(json.loads(response) if isinstance(response, str) else response)
    if overlap:
        representatives, policy_stats, baseline, compatible = _policy_representatives(catalogue, prepared, response)
    added = []
    for decision in response["decisions"]:
        if decision["decision"] == "include":
            allowed = (
                compatible and (
                    baseline == representatives[decision["name"]]
                    or _policy_accepts(baseline, representatives[decision["name"]], policy_stats)
                ) if overlap else _policy_accepts(context["anchor"], decision["name"], stats, symmetric=False)
            )
            if not allowed:
                decision.update(
                    decision="uncertain", reason="Deferred by conservative autonomous spelling policy.", evidence_ids=[]
                )
            else:
                added.append(decision["name"])
    if context["group_id"]:
        response["preferred_name"] = automatic["preferred_name"]
    elif overlap:
        # A newly discovered rare alias may point to a much more common spelling.
        # Choose only among policy-accepted members using the pinned counts, so
        # RecordedState and the live writer derive the same immutable payload.
        accepted = set(added)
        members = [context["anchor"], *[name for name in context["candidates"] if name in accepted]]
        highest = max(stats[name]["article_count"] for name in members)
        preferred = normalize_name(response["preferred_name"])
        if preferred not in members or stats[preferred]["article_count"] != highest:
            common = next(name for name in members if stats[name]["article_count"] == highest)
            response["preferred_name"] = stats[common]["name"]
    elif normalize_name(response["preferred_name"]) != context["anchor"]:
        response["preferred_name"] = stats[context["anchor"]]["name"]
    checked = validate_response(
        prepared, response, catalogue.store, catalogue.entities.generation, allow_autonomous_overlap=overlap
    )
    result = {
        "status": "accepted" if added else "deferred",
        "prepared": prepared,
        "response": response,
        "validated": checked,
    }
    if not added:
        result["reason"] = "No new member passed the autonomous spelling policy."
    return result


def evaluate_response(catalogue: Any, prepared: dict[str, Any], response: Any) -> dict[str, Any]:
    """Reuse bounded evidence follow-up, then independently gate publication."""
    _initialize(catalogue.store)
    if prepared["context"]["generation"] != catalogue.entities.generation:
        raise Conflict("Autonomous proposal belongs to another entity generation.")
    key = digest({"fingerprint": prepared.get("fingerprint"), "response": response, "policy": POLICY_VERSION})
    row = catalogue.store.connection.execute(
        "SELECT result_json FROM autonomous_evaluations WHERE evaluation_id=?", (key,)
    ).fetchone()
    if row:
        return json.loads(row[0])
    checked = review_response(
        catalogue, prepared, response,
        allow_autonomous_overlap=prepared["context"].get("overlap_version") == "unreviewed-union-v1",
    )
    if "follow_up" in checked:
        result = {"status": "follow_up", "prepared": checked["follow_up"]}
    else:
        prepared = checked.get("prepared", prepared)
        response = checked.get("response", response)
        result = _final_result(catalogue, prepared, response)
        if result["status"] == "deferred":
            _defer(
                catalogue.store,
                prepared["context"]["generation"],
                prepared["context"]["anchor"],
                result["reason"],
                now=time.time(),
                operation_id=prepared["context"]["autonomous"]["discovery_id"],
            )
    with catalogue.store.connection:
        catalogue.store.connection.execute(
            "INSERT OR IGNORE INTO autonomous_evaluations VALUES(?,?)", (key, _json(result))
        )
    return result


def apply_autonomous(
    catalogue: Any,
    prepared: dict[str, Any],
    response: Any,
    *,
    operation_id: str,
    provenance: dict[str, Any],
    authorize=None,
) -> dict[str, Any]:
    """Publish through the store's dedicated system writer, never a human-review API."""
    _initialize(catalogue.store)
    context = prepared["context"]
    # Derive a stable payload without reading today's state. The store replays an
    # existing receipt before invoking validation, so lost acknowledgements are safe.
    from .proposals import RecordedState

    recorded = SimpleNamespace(store=RecordedState(context), entities=SimpleNamespace(generation=context["generation"]))
    accepted = _final_result(recorded, prepared, response)
    if accepted["status"] != "accepted":
        raise ValueError("No new member passed the autonomous spelling policy.")
    selected = accepted["validated"]
    preferred = selected["preferred_name"]
    description = context.get("description") or f"OCR and spelling variants of {preferred}."
    payload = {
        "group_id": context["group_id"] or _operation(operation_id),
        "preferred_name": preferred,
        "description": description,
        "members": selected["members"],
    }
    if not isinstance(provenance, dict):
        raise ValueError("Autonomous provenance must be an object.")
    decisions = []
    for decision in selected["decisions"]:
        if decision["decision"] != "include":
            continue
        reason = decision["reason"].encode("utf-8")
        recorded_decision = {
            "name": decision["name"],
            "reason": reason[:240].decode("utf-8", errors="ignore"),
            "evidence_ids": list(decision["evidence_ids"]),
        }
        if len(reason) > 240:
            recorded_decision["reason_truncated"] = True
        decisions.append(recorded_decision)
    # Retain direct decision references without duplicating source passages. Full
    # reasons and evidence remain in evaluation receipts; the store enforces 16KiB.
    provenance = {
        **provenance,
        "policy_version": POLICY_VERSION,
        "prompt_version": context["prompt_version"],
        "generation": context["generation"],
        "prepared_fingerprint": prepared["fingerprint"],
        "accepted_decisions": decisions,
        "evidence_ids": sorted({reference for decision in decisions for reference in decision["evidence_ids"]}),
    }

    if context.get("overlap_version") == "unreviewed-union-v1":
        provenance["overlap_version"] = context["overlap_version"]

    def validate():
        reservation = catalogue.store.connection.execute(
            "SELECT status,operation_id,next_due FROM autonomous_frontier WHERE generation=? AND policy_version=? "
            "AND normalized_name=?",
            (context["generation"], POLICY_VERSION, context["anchor"]),
        ).fetchone()
        if (
            not reservation
            or reservation["operation_id"] != context["autonomous"]["discovery_id"]
            or reservation["status"] != "prepared"
            or reservation["next_due"] <= time.time()
        ):
            raise Conflict("Autonomous pipeline reservation expired or was superseded.")
        current = _final_result(catalogue, prepared, response)
        if current["status"] != "accepted" or set(current["validated"]["members"]) != set(selected["members"]):
            raise Conflict("Autonomous group changes invalidate the accepted model decisions.")
        return current

    result = catalogue.store.apply_autonomous(
        payload, provenance=provenance, operation_id=operation_id, validate=validate, authorize=authorize
    )
    with catalogue.store.connection:
        catalogue.store.connection.execute(
            "UPDATE autonomous_frontier SET status='published' WHERE generation=? AND policy_version=? "
            "AND normalized_name=? "
            "AND operation_id=? AND status='prepared'",
            (context["generation"], POLICY_VERSION, context["anchor"], context["autonomous"]["discovery_id"]),
        )
    return result
