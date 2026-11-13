"""Transactional convergence of overlapping autonomous groups in the local sidecar."""

from __future__ import annotations

from semora.ner.names import normalize_name


def eligible(group):
    return (
        group["origin"] == "autonomous"
        and not group["human_reviewed"]
        and all(
            item["origin"] == "autonomous" and not item["human_reviewed"]
            for item in group["member_provenance"].values()
        )
    )


def plan_union(store, payload, *, display_versions=None):
    """Read only selected memberships and affected groups under the caller's writer."""
    from .store import MAX_MEMBERS, Conflict, _identifier, _text

    if set(payload) - {"group_id", "preferred_name", "description", "members"}:
        raise ValueError("Unknown group fields.")
    raw = payload.get("members")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_MEMBERS:
        raise ValueError("Supply 1-100 members.")
    members = list(dict.fromkeys(normalize_name(_text(name, "member")) for name in raw))
    marks = ",".join("?" for _ in members)
    owners = dict(
        store.connection.execute(
            f"SELECT normalized_name,group_id FROM group_members WHERE normalized_name IN ({marks})", members
        )
    )
    ids = list(dict.fromkeys(owners[name] for name in members if name in owners))
    proposed = store.canonical_group_id(_identifier(payload.get("group_id"), "group_id"))
    if not ids:
        return {"payload": payload, "target": proposed, "groups": {}, "sources": [], "noop": False}
    marks = ",".join("?" for _ in ids)
    # Reject oversized unions through the covering membership index before
    # hydrating potentially large provenance records for every source group.
    prior_members = {
        row[0]
        for row in store.connection.execute(
            f"SELECT normalized_name FROM group_members WHERE group_id IN ({marks}) LIMIT ?", [*ids, MAX_MEMBERS + 1]
        )
    }
    if len(prior_members | set(members)) > MAX_MEMBERS:
        raise Conflict("Merged group exceeds the 100-member limit.")
    rows = store.connection.execute(f"SELECT * FROM name_groups WHERE group_id IN ({marks})", ids).fetchall()
    groups = {group["group_id"]: group for group in store._hydrate_groups(rows)}
    if ids == [proposed] and proposed in groups and not eligible(groups[proposed]):
        # Preserve the existing explicitly addressed extension API. Its writer
        # still rejects metadata replacement or removal of protected members.
        return {"payload": payload, "target": proposed, "groups": {}, "sources": [], "noop": False}
    if len(groups) != len(ids) or any(not eligible(group) for group in groups.values()):
        raise Conflict("Autonomous overlaps cannot change reviewed or unknown groups.")
    target = proposed if proposed in groups else ids[0]
    selected = set(members)
    for group in groups.values():
        selected.update(group["members"])
    if len(selected) > MAX_MEMBERS:
        raise Conflict("Merged group exceeds the 100-member limit.")
    if store._members_rejected(selected):
        raise Conflict("Reopen a rejected pair before grouping its names.")
    prior = groups[target]
    may_rename = (
        isinstance(display_versions, dict)
        and type(display_versions.get(target)) is int
        and display_versions[target] == prior["version"]
    )
    preferred = _text(payload.get("preferred_name"), "preferred_name", 200) if may_rename else prior["preferred_name"]
    merged = {
        **payload,
        "group_id": target,
        "preferred_name": preferred,
        "description": prior["description"],
        "members": sorted(selected),
    }
    sources = [group_id for group_id in ids if group_id != target]
    return {
        "payload": merged,
        "target": target,
        "groups": groups,
        "sources": sources,
        "noop": not sources and selected == set(prior["members"]) and preferred == prior["preferred_name"],
    }


def capture(store, group_ids):
    """Snapshots are limited to the groups and aliases touched by this union."""
    ids = list(group_ids)
    marks = ",".join("?" for _ in ids)
    rows = store.connection.execute(f"SELECT * FROM name_groups WHERE group_id IN ({marks})", ids).fetchall()
    groups = {group["group_id"]: group for group in store._hydrate_groups(rows)}
    return {
        "groups": {group_id: groups.get(group_id) for group_id in ids},
        "redirects": dict(
            store.connection.execute(
                f"SELECT retired_group_id,group_id FROM group_redirects WHERE group_id IN ({marks}) "
                f"OR retired_group_id IN ({marks})",
                [*ids, *ids],
            )
        ),
    }


def retire_sources(store, plan):
    sources = plan["sources"]
    if not sources:
        return
    marks = ",".join("?" for _ in sources)
    store.connection.execute(f"DELETE FROM name_groups WHERE group_id IN ({marks})", sources)
    # All aliases point directly at a live canonical group, never at another alias.
    store.connection.execute(
        f"UPDATE group_redirects SET group_id=? WHERE group_id IN ({marks})", [plan["target"], *sources]
    )
    store.connection.executemany(
        "INSERT INTO group_redirects(retired_group_id,group_id) VALUES(?,?)",
        [(group_id, plan["target"]) for group_id in sources],
    )


def preserve_members(store, plan):
    """Moving a spelling does not turn the current inference into its original source."""
    if (
        not plan["sources"] and plan["groups"]
        and set(plan["payload"]["members"]) == set(plan["groups"][plan["target"]]["members"])
    ):
        # Metadata-only changes preserve the exact count for version promotion.
        return
    for group in plan["groups"].values():
        for name, value in group["member_provenance"].items():
            from .store import _json

            store.connection.execute(
                "INSERT OR REPLACE INTO member_provenance VALUES(?,?,?,?)",
                (name, value["origin"], int(value["human_reviewed"]), _json(value["provenance"])),
            )
    ids = list(plan["groups"])
    if ids:
        marks = ",".join("?" for _ in ids)
        store.connection.execute(f"DELETE FROM group_counts WHERE group_id IN ({marks})", ids)


def restore(store, state, version):
    from .provenance import normalize_group
    from .store import Conflict

    groups = state["groups"]
    ids = list(groups)
    for group in groups.values():
        if group is None:
            continue
        for member in group["members"]:
            if store.membership(member) not in (None, *ids):
                raise Conflict("An affected name now belongs to another group.")
        if store._members_rejected(group["members"]):
            raise Conflict("An affected pair is now rejected.")
    marks = ",".join("?" for _ in ids)
    store.connection.execute(f"DELETE FROM name_groups WHERE group_id IN ({marks})", ids)
    store.connection.execute(
        f"DELETE FROM group_redirects WHERE group_id IN ({marks}) OR retired_group_id IN ({marks})", [*ids, *ids]
    )
    store.connection.execute(f"DELETE FROM group_counts WHERE group_id IN ({marks})", ids)
    for group_id, group in groups.items():
        if group is None:
            continue
        store.connection.execute(
            "INSERT INTO name_groups VALUES(?,?,?,?,?)",
            (group_id, group["preferred_name"], group["description"], version, normalize_name(group["preferred_name"])),
        )
        store.connection.executemany(
            "INSERT INTO group_members VALUES(?,?)", [(name, group_id) for name in group["members"]]
        )
        normalized = normalize_group(group)
        store._save_annotations(group_id, normalized, normalized["member_provenance"])
    store.connection.executemany("INSERT INTO group_redirects VALUES(?,?)", state["redirects"].items())


def enqueue_released(store, before):
    """Queue only names that became unowned; discovery resolves their current frequency."""
    groups = before.get("merge", {}).get("groups", {before["group_id"]: before["group"]})
    candidates = {name for group in groups.values() if group for name in group["members"]}
    if not candidates:
        return
    marks = ",".join("?" for _ in candidates)
    owned = {
        row[0]
        for row in store.connection.execute(
            f"SELECT normalized_name FROM group_members WHERE normalized_name IN ({marks})", list(candidates)
        )
    }
    store.connection.executemany(
        "INSERT OR IGNORE INTO autonomous_released_names(normalized_name) VALUES(?)",
        [(name,) for name in sorted(candidates - owned)],
    )
