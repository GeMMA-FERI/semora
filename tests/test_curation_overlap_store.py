"""Storage-level convergence: model-approved overlap uses one local writer."""

import concurrent.futures
import json
import sqlite3
import threading
import uuid
from types import SimpleNamespace

import pytest

from semora.curation import CurationStore
from semora.curation.retrieval import CuratedRetrieval
from semora.curation.store import Conflict, membership_key


def identifier():
    return uuid.uuid4().hex


@pytest.fixture
def store(tmp_path):
    with CurationStore(tmp_path / "curation.sqlite", corpus_id="corpus", initialize=True) as store:
        store.configure_serving(True, 1, identifier())
        yield store


def automatic(store, members, *, operation=None, name="Canonical", validate=lambda: None, authorize=None):
    return store.apply_autonomous(
        {"preferred_name": name, "description": name + " description", "members": members},
        provenance={"policy_version": "ocr-v1", "settings_revision": 1, "source": name},
        operation_id=operation or identifier(),
        validate=validate,
        authorize=authorize,
    )


def manual(store, action, payload):
    return store.mutate(
        action, payload, operation_id=identifier(), actor="admin:1", expected_revision=store.metadata()["revision"]
    )


def test_overlap_extends_existing_group_and_keeps_first_metadata_and_provenance(store):
    first = automatic(store, ["alpha", "a1"], name="First")
    original = store.group(first["group_id"])
    result = automatic(store, ["alpha", "a2"], name="Second")
    assert result["group_id"] == first["group_id"]
    group = store.group(result["group_id"])
    assert group["members"] == ["a1", "a2", "alpha"]
    assert group["preferred_name"] == "First" and group["description"] == "First description"
    assert group["member_provenance"]["alpha"] == original["member_provenance"]["alpha"]
    assert group["member_provenance"]["a2"]["provenance"]["source"] == "Second"
    assert len(store.groups()) == 1


def test_two_existing_groups_merge_with_redirects_and_complete_audit(store):
    first = automatic(store, ["alpha", "a1"], name="First")
    second = automatic(store, ["beta", "b1"], name="Second")
    prior = store.group(second["group_id"])
    merged = automatic(store, ["alpha", "beta", "extra"])
    assert merged["group_id"] == first["group_id"]
    assert store.group(second["group_id"])["group_id"] == first["group_id"]
    group = store.group(first["group_id"])
    assert group["members"] == ["a1", "alpha", "b1", "beta", "extra"]
    assert group["member_provenance"]["beta"] == prior["member_provenance"]["beta"]
    assert len(store.groups()) == 1
    row = store.connection.execute(
        "SELECT before_json,after_json FROM curation_changes WHERE operation_id=?", (merged["operation_id"],)
    ).fetchone()
    before, after = (json.loads(value) for value in row)
    assert before["merge"]["groups"][second["group_id"]]["members"] == ["b1", "beta"]
    assert after["merge"]["groups"][second["group_id"]] is None
    assert after["merge"]["redirects"] == {second["group_id"]: first["group_id"]}


def test_duplicate_ack_replays_original_request_even_after_group_growth(store):
    first = automatic(store, ["alpha", "a1"])
    operation = identifier()
    result = automatic(store, ["alpha", "a2"], operation=operation)
    automatic(store, ["alpha", "a3"])
    revision = store.metadata()["revision"]
    assert (
        automatic(
            store,
            ["alpha", "a2"],
            operation=operation,
            validate=lambda: pytest.fail("Receipt must replay before live validation"),
        )
        == result
    )
    assert store.metadata()["revision"] == revision
    assert result["group_id"] == first["group_id"]


def test_already_applied_overlap_is_successful_receipt_without_changing_group_version(store):
    first = automatic(store, ["alpha", "a1"])
    version = store.group(first["group_id"])["version"]
    result = automatic(store, ["alpha", "a1"])
    assert result["already_grouped"] is True
    assert result["group_id"] == first["group_id"]
    assert store.group(first["group_id"])["version"] == version


def test_merge_undo_restores_both_groups_aliases_and_releases_new_names(store):
    first = automatic(store, ["alpha", "a1"], name="First")
    second = automatic(store, ["beta", "b1"], name="Second")
    merged = automatic(store, ["alpha", "beta", "extra"])
    manual(store, "undo_change", {"operation_id": merged["operation_id"]})
    assert store.group(first["group_id"])["members"] == ["a1", "alpha"]
    assert store.group(second["group_id"])["members"] == ["b1", "beta"]
    assert store.connection.execute("SELECT COUNT(*) FROM group_redirects").fetchone()[0] == 0
    assert [row[0] for row in store.connection.execute("SELECT * FROM autonomous_released_names")] == ["extra"]


def test_undo_refuses_changed_merge_descendants(store):
    automatic(store, ["alpha", "a1"])
    automatic(store, ["beta", "b1"])
    merged = automatic(store, ["alpha", "beta"])
    automatic(store, ["alpha", "extra"])
    with pytest.raises(Conflict, match="Affected records changed"):
        manual(store, "undo_change", {"operation_id": merged["operation_id"]})


def test_redirects_flatten_on_later_merges_and_readonly_links_resolve(store):
    first = automatic(store, ["alpha", "a1"])
    second = automatic(store, ["beta", "b1"])
    third = automatic(store, ["gamma", "g1"])
    automatic(store, ["alpha", "beta"])
    merged = automatic(store, ["gamma", "alpha"])
    assert merged["group_id"] == third["group_id"]
    for retired in (first["group_id"], second["group_id"]):
        assert store.canonical_group_id(retired) == third["group_id"]
    with CurationStore(store.path, corpus_id="corpus", read_only=True) as reader:
        assert reader.group(second["group_id"])["group_id"] == third["group_id"]


def test_human_overlap_and_rejected_cross_group_pair_cannot_be_merged(store):
    human = manual(store, "create_group", {"preferred_name": "Human", "members": ["human", "h1"]})
    with pytest.raises(Conflict, match="reviewed or unknown"):
        automatic(store, ["human", "extra"])
    assert store.group(human["group_id"])["members"] == ["h1", "human"]
    automatic(store, ["alpha", "a1"])
    automatic(store, ["beta", "b1"])
    manual(store, "reject_pair", {"first": "a1", "second": "b1", "reason": "Distinct places"})
    with pytest.raises(Conflict, match="rejected"):
        automatic(store, ["alpha", "beta"])
    assert store.membership("alpha") != store.membership("beta")


def test_authorization_loss_rolls_back_union_aliases_provenance_and_audit(store):
    first = automatic(store, ["alpha", "a1"])
    second = automatic(store, ["beta", "b1"])
    revision = store.metadata()["revision"]

    def denied():
        raise PermissionError("revoked")

    with pytest.raises(PermissionError):
        automatic(store, ["alpha", "beta", "extra"], authorize=denied)
    assert store.metadata()["revision"] == revision
    assert store.group(first["group_id"])["members"] == ["a1", "alpha"]
    assert store.group(second["group_id"])["members"] == ["b1", "beta"]
    assert store.connection.execute("SELECT COUNT(*) FROM group_redirects").fetchone()[0] == 0
    assert store.membership("extra") is None


def test_group_removal_queues_only_released_names_and_undo_keeps_queue_deduped(store):
    result = automatic(store, ["alpha", "a1"])
    removed = manual(store, "delete_group", {"group_id": result["group_id"]})
    assert {row[0] for row in store.connection.execute("SELECT * FROM autonomous_released_names")} == {"alpha", "a1"}
    manual(store, "undo_change", {"operation_id": removed["operation_id"]})
    assert store.membership("alpha") == result["group_id"]
    assert store.connection.execute("SELECT COUNT(*) FROM autonomous_released_names").fetchone()[0] == 2


def test_old_count_receipts_cannot_poison_merged_group_cache(store):
    first = automatic(store, ["alpha", "a1"])
    second = automatic(store, ["beta", "b1"])
    old = store.group(second["group_id"])
    stale = {
        "group_id": second["group_id"],
        "generation": "g",
        "group_version": old["version"],
        "membership_key": membership_key(old["members"]),
        "article_count": 7,
    }
    store.save_count(stale)
    automatic(store, ["alpha", "beta"])
    with pytest.raises(Conflict, match="changed while counting"):
        store.save_count(stale)
    merged = store.group(second["group_id"])
    store.save_count({**stale, "membership_key": membership_key(merged["members"]), "article_count": 11})
    saved = tuple(store.connection.execute("SELECT group_id,article_count FROM group_counts").fetchone())
    assert saved == (first["group_id"], 11)


def test_alias_count_recomputes_distinct_article_union(store):
    first = automatic(store, ["alpha", "a1"])
    second = automatic(store, ["beta", "b1"])
    automatic(store, ["alpha", "beta"])
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE names(entity_id INTEGER PRIMARY KEY,normalized_name TEXT UNIQUE);
        CREATE TABLE entity_articles(entity_id INTEGER,article_id TEXT,content_sha256 TEXT);
        ATTACH DATABASE ':memory:' AS corpus;
        CREATE TABLE corpus.article_ner_state(article_id TEXT,ner_run_id TEXT,content_sha256 TEXT);
        CREATE TABLE corpus.articles(article_id TEXT,is_valid INTEGER,char_start INTEGER,char_end INTEGER,
                                     line_start INTEGER,line_end INTEGER);
    """)
    db.executemany("INSERT INTO names VALUES(?,?)", [(1, "alpha"), (2, "a1"), (3, "beta"), (4, "b1")])
    db.executemany("INSERT INTO entity_articles VALUES(?,?,'hash')", [(1, "shared"), (3, "shared"), (2, "extra")])
    db.executemany("INSERT INTO corpus.article_ner_state VALUES(?,'ner','hash')", [("shared",), ("extra",)])
    db.executemany("INSERT INTO corpus.articles VALUES(?,1,0,1,1,2)", [("shared",), ("extra",)])
    db.commit()
    try:
        reader = CuratedRetrieval(SimpleNamespace(connection=db, generation="g", run_id="ner"), store)
        result = reader.count_articles(second["group_id"])
        assert result["group_id"] == first["group_id"]
        assert result["article_count"] == 2
    finally:
        db.close()


def test_concurrent_connections_converge_without_a_backend_reservation(store):
    barrier = threading.Barrier(2)

    def publish(member):
        with CurationStore(store.path, corpus_id="corpus") as writer:
            barrier.wait(timeout=3)
            return automatic(writer, ["alpha", member])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(publish, "a1")
        second = pool.submit(publish, "a2")
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert results[0]["group_id"] == results[1]["group_id"]
    assert store.group(results[0]["group_id"])["members"] == ["a1", "a2", "alpha"]
    assert len(store.groups()) == 1


def test_oversized_union_rejects_before_hydrating_group_provenance(store, monkeypatch):
    automatic(store, [f"first-{i}" for i in range(60)])
    automatic(store, [f"second-{i}" for i in range(60)])
    monkeypatch.setattr(store, "_hydrate_groups", lambda *a, **k: pytest.fail("Oversized group hydrated"))
    with pytest.raises(Conflict, match="100-member"):
        automatic(store, ["first-0", "second-0"])


def test_readonly_old_sidecar_without_redirect_table_still_reads_groups(store):
    created = automatic(store, ["alpha", "a1"])
    store.connection.execute("DROP TABLE group_redirects")
    with CurationStore(store.path, corpus_id="corpus", read_only=True) as reader:
        assert reader.group(created["group_id"])["members"] == ["a1", "alpha"]


def test_failed_manual_removal_cannot_leave_a_release_notification(store):
    created = automatic(store, ["alpha", "a1"])

    def denied():
        raise PermissionError("revoked")

    with pytest.raises(PermissionError):
        store.mutate(
            "delete_group",
            {"group_id": created["group_id"]},
            operation_id=identifier(),
            actor="admin:1",
            expected_revision=store.metadata()["revision"],
            authorize=denied,
        )
    assert store.membership("alpha") == created["group_id"]
    assert store.connection.execute("SELECT COUNT(*) FROM autonomous_released_names").fetchone()[0] == 0


def test_overlap_queries_use_existing_covering_membership_and_alias_indexes(store):
    membership_plan = " ".join(
        row[3]
        for row in store.connection.execute(
            "EXPLAIN QUERY PLAN SELECT normalized_name FROM group_members WHERE group_id IN (?,?) LIMIT 101",
            [identifier(), identifier()],
        )
    )
    redirects_plan = " ".join(
        row[3]
        for row in store.connection.execute(
            "EXPLAIN QUERY PLAN SELECT retired_group_id,group_id FROM group_redirects WHERE group_id IN (?,?) "
            "OR retired_group_id IN (?,?)",
            [identifier() for _ in range(4)],
        )
    )
    assert "COVERING INDEX curation_group_members" in membership_plan
    assert "group_redirect_target" in redirects_plan
    assert "sqlite_autoindex_group_redirects" in redirects_plan


def test_explicit_reviewed_extension_through_retired_id_preserves_replay(store):
    first = automatic(store, ["alpha", "a1"])
    second = automatic(store, ["beta", "b1"])
    automatic(store, ["alpha", "beta"])
    group = store.group(first["group_id"])
    manual(
        store,
        "update_group",
        {
            "group_id": first["group_id"],
            "preferred_name": "Reviewed",
            "description": "Reviewed city",
            "members": group["members"],
        },
    )
    payload = {
        "group_id": second["group_id"],
        "preferred_name": "Reviewed",
        "description": "Reviewed city",
        "members": [*group["members"], "addition"],
    }
    operation = identifier()

    def apply():
        return store.apply_autonomous(
            payload,
            operation_id=operation,
            validate=lambda: None,
            provenance={"policy_version": "ocr-v1", "settings_revision": 1},
        )

    result = apply()
    assert result["group_id"] == first["group_id"]
    assert payload["group_id"] == second["group_id"]
    assert store.group(first["group_id"])["preferred_name"] == "Reviewed"
    assert store.group(first["group_id"])["member_provenance"]["addition"]["human_reviewed"] is False
    assert apply() == result


def test_bounded_rejection_probe_handles_empty_singleton_and_hundred_members(store):
    members = [f"name-{index:03}" for index in range(100)]
    statements = []
    store.connection.set_trace_callback(statements.append)
    try:
        assert store._members_rejected([]) is False
        assert store._members_rejected(["single"]) is False
        assert statements == []
        assert store._members_rejected(members) is False
        assert len(statements) == 1
        store.connection.execute("INSERT INTO rejected_pairs VALUES(?,?,?)", (members[-2], members[-1], "Distinct"))
        statements.clear()
        assert store._members_rejected(members) is True
        assert len(statements) == 1
        assert store._members_rejected([*members[:-1], "outside"]) is False
    finally:
        store.connection.set_trace_callback(None)


def test_hundred_member_publication_uses_one_rejection_query(store):
    statements = []
    store.connection.set_trace_callback(statements.append)
    try:
        automatic(store, [f"name-{index:03}" for index in range(100)])
    finally:
        store.connection.set_trace_callback(None)
    probes = [sql for sql in statements if "SELECT 1 FROM rejected_pairs" in sql]
    assert len(probes) == 1
    plan = " ".join(
        row[3]
        for row in store.connection.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM rejected_pairs WHERE first_name IN (?,?) "
            "AND second_name IN (?,?) LIMIT 1",
            ["a", "b", "c", "d"],
        )
    )
    assert "COVERING INDEX sqlite_autoindex_rejected_pairs" in plan
