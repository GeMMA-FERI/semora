import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from semora.curation import CurationStore
from semora.curation.autonomous import POLICY_VERSION, apply_autonomous, discover_task, evaluate_response
from semora.curation.catalogue import CurationCatalogue
from semora.curation.frequency import prepare_frequency
from semora.curation.store import Conflict


@pytest.fixture
def catalogue(tmp_path):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        "CREATE TABLE names(entity_id INTEGER PRIMARY KEY, normalized_name TEXT UNIQUE,"
        "name TEXT,article_count INTEGER);"
    )
    connection.executemany(
        "INSERT INTO names VALUES(?,?,?,?)",
        [
            (1, "ljubljana", "Ljubljana", 100),
            (2, "ljubijana", "Ljubijana", 2),
            (3, "marina", "Marina", 20),
            (4, "marija", "Marija", 10),
        ],
    )
    connection.commit()
    with CurationStore(tmp_path / "curation.sqlite", corpus_id="c", initialize=True) as store:
        frequency = prepare_frequency(connection, tmp_path / "frequency", "dictionary")
        entities = SimpleNamespace(connection=connection, generation="g", run_id="ner",
                                   frequency_path=frequency["path"], frequency_identity=frequency["identity"])
        entities.search_entities = lambda q: {
            "entities": [
                {"name": row["name"], "article_count": row["article_count"], "distance": 1}
                for row in connection.execute("SELECT * FROM names")
            ]
        }
        yield CurationCatalogue(entities, store)
    connection.close()


def answer(prepared, *, decision="include"):
    return {
        "preferred_name": "Ljubljana",
        "decisions": [
            {"name": name, "decision": decision, "reason": "Spelling judgment.", "evidence_ids": []}
            for name in prepared["context"]["candidates"]
        ],
    }


def test_discovery_replay_priority_cursor_and_request_identity(catalogue):
    operation = uuid.uuid4().hex
    first = discover_task(catalogue, operation_id=operation, scan_limit=2)
    assert first["status"] == "prepared"
    assert first["prepared"]["context"]["anchor"] == "ljubljana"
    cursor = catalogue.store.connection.execute("SELECT after_rank FROM autonomous_frequency_cursors").fetchone()[0]
    assert cursor == 2
    assert discover_task(catalogue, operation_id=operation, scan_limit=2) == first
    assert catalogue.store.connection.execute("SELECT after_rank FROM autonomous_frequency_cursors").fetchone()[0] == 2
    with pytest.raises(Conflict):
        discover_task(catalogue, operation_id=operation, scan_limit=3)
    catalogue.entities.generation = "other"
    with pytest.raises(Conflict):
        discover_task(catalogue, operation_id=operation, scan_limit=2)


def test_policy_accepts_ljubljana_ocr_but_defers_distinct_names(catalogue):
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    prepared = result["prepared"]
    checked = evaluate_response(catalogue, prepared, answer(prepared))
    assert checked["status"] == "accepted"
    assert checked["validated"]["members"] == ["ljubijana", "ljubljana"]
    assert {d["name"] for d in checked["response"]["decisions"] if d["decision"] == "uncertain"} == {"marija", "marina"}
    assert catalogue.store.groups() == []
    assert evaluate_response(catalogue, prepared, answer(prepared)) == checked


def test_policy_requires_dominant_anchor_and_no_singleton_publication(catalogue):
    catalogue.connection.execute("UPDATE names SET article_count=2 WHERE normalized_name='ljubljana'")
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    response = answer(prepared)
    response["preferred_name"] = prepared["context"]["anchor"]
    result = evaluate_response(catalogue, prepared, response)
    assert result["status"] == "deferred"
    assert len(result["validated"]["members"]) == 1


def test_stale_human_rejection_and_generation_block_application(catalogue):
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    response = answer(prepared)
    catalogue.store.mutate(
        "reject_pair",
        {"first": "ljubljana", "second": "ljubijana", "reason": "Human decision"},
        operation_id=uuid.uuid4().hex,
        actor="admin:1",
        expected_revision=0,
    )
    with pytest.raises(Conflict):
        evaluate_response(catalogue, prepared, response)
    catalogue.entities.generation = "next"
    with pytest.raises(Conflict):
        evaluate_response(catalogue, prepared, response)


def test_evidence_followup_is_bounded_and_replayed(catalogue):
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    response = answer(prepared, decision="uncertain")
    response["decisions"][0]["decision"] = "needs_evidence"
    calls = []
    catalogue.evidence = lambda name, **kw: (
        calls.append(name) or {"sampling": "bounded", "passages": [{"passage": name}]}
    )
    result = evaluate_response(catalogue, prepared, response)
    assert result["status"] == "follow_up"
    assert len(calls) == 2
    assert evaluate_response(catalogue, prepared, response) == result
    assert len(calls) == 2


def test_cooldown_and_exhaustion_do_not_rescan_names(catalogue):
    catalogue.suggest_variants = lambda *args, **kw: {"candidates": []}
    first = discover_task(catalogue, operation_id=uuid.uuid4().hex, now=0)
    assert first["status"] == "empty" and first["has_more"]
    assert (
        catalogue.store.connection.execute(
            "SELECT COUNT(*) FROM autonomous_frontier WHERE status='deferred'"
        ).fetchone()[0]
        == 1
    )
    for _ in range(4):
        discover_task(catalogue, operation_id=uuid.uuid4().hex, now=1)
    cursor = catalogue.store.connection.execute(
        "SELECT after_rank,exhausted FROM autonomous_frequency_cursors"
    ).fetchone()
    assert tuple(cursor) == (4, 1)
    assert catalogue.store.connection.execute("SELECT MAX(attempts) FROM autonomous_frontier").fetchone()[0] == 1


def test_due_promotion_respects_total_active_frontier_cap(catalogue):
    discover_task(catalogue, operation_id=uuid.uuid4().hex)
    connection = catalogue.store.connection
    with connection:
        connection.execute("DELETE FROM autonomous_frontier")
        connection.executemany(
            "INSERT INTO autonomous_frontier VALUES(?,?,?,1,?,1,?,NULL,NULL)",
            [("g", POLICY_VERSION, f"active{i}", "prepared", 99999999999) for i in range(512)]
            + [("g", POLICY_VERSION, f"due{i}", "deferred", 0) for i in range(128)],
        )
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex, now=1)
    assert result["status"] == "empty"
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM autonomous_frontier WHERE status IN ('queued','preparing','prepared')"
        ).fetchone()[0]
        == 512
    )


def test_application_is_dedicated_autonomous_and_idempotent(catalogue):
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    operation = uuid.uuid4().hex
    catalogue.store.configure_serving(True, settings_revision=1, operation_id=uuid.uuid4().hex)
    result = apply_autonomous(
        catalogue,
        prepared,
        answer(prepared),
        operation_id=operation,
        provenance={"model_configuration_id": "model-1", "settings_revision": 1},
    )
    assert (
        apply_autonomous(
            catalogue,
            prepared,
            answer(prepared),
            operation_id=operation,
            provenance={"model_configuration_id": "model-1", "settings_revision": 1},
        )
        == result
    )
    group = catalogue.store.group(result["group_id"])
    assert group["origin"] == "autonomous" and not group["human_reviewed"]
    assert group["members"] == ["ljubijana", "ljubljana"]
    assert group["description"] == "OCR and spelling variants of Ljubljana."
    receipt = catalogue.store.connection.execute(
        "SELECT actor FROM curation_changes WHERE operation_id=?", (str(uuid.UUID(operation)),)
    ).fetchone()
    assert receipt[0] == "system:autonomous-curation"


def test_orphan_reservation_expires_without_changing_saved_receipt(catalogue, monkeypatch):
    operation = uuid.uuid4().hex
    first = discover_task(catalogue, operation_id=operation, now=100)
    assert first["status"] == "prepared"
    monkeypatch.setattr("semora.curation.autonomous.time.time", lambda: 4000)
    discover_task(catalogue, operation_id=uuid.uuid4().hex, now=4000)
    assert discover_task(catalogue, operation_id=operation, now=4000) == first
    row = catalogue.store.connection.execute(
        "SELECT status FROM autonomous_frontier WHERE normalized_name='ljubljana'"
    ).fetchone()
    assert row[0] == "deferred"
    catalogue.store.configure_serving(True, settings_revision=1, operation_id=uuid.uuid4().hex)
    with pytest.raises(Conflict, match="reservation"):
        apply_autonomous(
            catalogue,
            first["prepared"],
            answer(first["prepared"]),
            operation_id=uuid.uuid4().hex,
            provenance={"settings_revision": 1},
        )
    assert catalogue.store.groups() == []


def test_human_group_anchor_is_not_autonomously_extended(catalogue):
    catalogue.store.mutate(
        "create_group",
        {"preferred_name": "Ljubljana", "members": ["ljubljana"]},
        operation_id=uuid.uuid4().hex,
        actor="admin:1",
        expected_revision=0,
    )
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    assert result["status"] == "prepared"
    assert result["prepared"]["context"]["anchor"] == "marina"
    assert "ljubljana" not in result["prepared"]["context"]["candidates"]
    assert catalogue.store.group(catalogue.store.membership("ljubljana"))["members"] == ["ljubljana"]


def test_late_evaluation_does_not_defer_successor_reservation(catalogue):
    first = discover_task(catalogue, operation_id=uuid.uuid4().hex, now=100)
    connection = catalogue.store.connection
    with connection:
        connection.execute("UPDATE autonomous_frontier SET status='queued' WHERE normalized_name='ljubljana'")
    second = discover_task(catalogue, operation_id=uuid.uuid4().hex, now=101)
    response = answer(first["prepared"], decision="uncertain")
    assert evaluate_response(catalogue, first["prepared"], response)["status"] == "deferred"
    row = connection.execute(
        "SELECT status,operation_id FROM autonomous_frontier WHERE normalized_name='ljubljana'"
    ).fetchone()
    assert tuple(row) == ("prepared", second["discovery_id"])


def test_lost_apply_ack_does_not_publish_successor_reservation(catalogue):
    first = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    catalogue.store.configure_serving(True, settings_revision=1, operation_id=uuid.uuid4().hex)
    operation = uuid.uuid4().hex
    metadata = {"model_configuration_id": "model-1", "settings_revision": 1}
    result = apply_autonomous(
        catalogue, first["prepared"], answer(first["prepared"]), operation_id=operation, provenance=metadata
    )
    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljub1jana','Ljub1jana',1)")
    connection = catalogue.store.connection
    with connection:
        # A newly encountered ungrouped alias should extend the established group,
        # while the old grouped anchor is never scheduled for new model work.
        connection.execute("UPDATE autonomous_frontier SET status='queued' WHERE normalized_name='ljubljana'")
        connection.execute("UPDATE autonomous_frontier SET status='deferred',next_due=1e12 "
                           "WHERE normalized_name IN ('marina','marija')")
        connection.execute(
            "INSERT INTO autonomous_frontier(generation,policy_version,normalized_name,article_count,status) "
            "VALUES(?,?,'ljub1jana',1,'queued')", (catalogue.entities.generation, POLICY_VERSION)
        )
    second = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    assert second["status"] == "prepared"
    assert second["prepared"]["context"]["anchor"] == "ljub1jana"
    assert (
        apply_autonomous(
            catalogue, first["prepared"], answer(first["prepared"]), operation_id=operation, provenance=metadata
        )
        == result
    )
    row = connection.execute(
        "SELECT status,operation_id FROM autonomous_frontier WHERE normalized_name='ljub1jana'"
    ).fetchone()
    assert tuple(row) == ("prepared", second["discovery_id"])

    before = catalogue.store.group(result["group_id"])
    successor_operation = uuid.uuid4().hex
    successor_metadata = {"model_configuration_id": "model-2", "settings_revision": 1}
    extended = apply_autonomous(
        catalogue,
        second["prepared"],
        answer(second["prepared"]),
        operation_id=successor_operation,
        provenance=successor_metadata,
    )
    assert extended["group_id"] == result["group_id"]
    after = catalogue.store.group(extended["group_id"])
    assert after["preferred_name"] == before["preferred_name"] == "Ljubljana"
    assert after["description"] == before["description"]
    assert after["members"] == ["ljub1jana", "ljubijana", "ljubljana"]
    assert after["version"] == before["version"] + 1
    for member in before["members"]:
        assert after["member_provenance"][member] == before["member_provenance"][member]
    assert after["member_provenance"]["ljub1jana"]["provenance"]["model_configuration_id"] == "model-2"
    assert after["member_provenance"]["ljub1jana"]["human_reviewed"] is False
    assert (
        apply_autonomous(
            catalogue,
            second["prepared"],
            answer(second["prepared"]),
            operation_id=successor_operation,
            provenance=successor_metadata,
        )
        == extended
    )
    assert catalogue.store.group(extended["group_id"]) == after


@pytest.mark.parametrize("distinct", ["ljubljanca", "ljubljanasociety", "ljubljan", "ljubljanica"])
def test_nearby_places_organizations_and_substrings_are_deferred(catalogue, distinct):
    catalogue.connection.execute("INSERT INTO names VALUES(5,?,?,1)", (distinct, distinct))
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    result = evaluate_response(catalogue, prepared, answer(prepared))
    assert distinct not in result["validated"]["members"]


def test_evidence_does_not_override_restricted_spelling_policy(catalogue):
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    response = answer(prepared, decision="uncertain")
    next(d for d in response["decisions"] if d["name"] == "marina")["decision"] = "needs_evidence"
    catalogue.evidence = lambda name, **kw: {"sampling": "bounded", "passages": [{"passage": name}]}
    follow_up = evaluate_response(catalogue, prepared, response)["prepared"]
    response = answer(follow_up, decision="uncertain")
    target = next(d for d in response["decisions"] if d["name"] == "marina")
    target.update(decision="include", evidence_ids=["e0-0", "e1-0"])
    result = evaluate_response(catalogue, follow_up, response)
    assert result["status"] == "deferred" and "marina" not in result["validated"]["members"]


def test_published_members_retain_bounded_decisions_and_evidence_references(catalogue):
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    response = answer(prepared, decision="uncertain")
    next(d for d in response["decisions"] if d["name"] == "ljubijana")["decision"] = "needs_evidence"
    catalogue.evidence = lambda name, **kw: {
        "sampling": "bounded",
        "passages": [{"passage": f"source passage for {name}"}],
    }
    follow_up = evaluate_response(catalogue, prepared, response)["prepared"]
    response = answer(follow_up, decision="uncertain")
    next(d for d in response["decisions"] if d["name"] == "ljubijana").update(
        decision="include", reason="\u00e9" * 240, evidence_ids=["e0-0", "e1-0"]
    )
    evaluated = evaluate_response(catalogue, follow_up, response)
    assert evaluated["status"] == "accepted"
    catalogue.store.configure_serving(True, settings_revision=1, operation_id=uuid.uuid4().hex)
    operation = uuid.uuid4().hex
    metadata = {"model_configuration_id": "model-1", "settings_revision": 1}
    result = apply_autonomous(catalogue, follow_up, response, operation_id=operation, provenance=metadata)
    group = catalogue.store.group(result["group_id"])
    retained = group["member_provenance"]["ljubijana"]["provenance"]
    assert retained["accepted_decisions"] == [
        {
            "name": "ljubijana",
            "reason": "\u00e9" * 120,
            "reason_truncated": True,
            "evidence_ids": ["e0-0", "e1-0"],
        }
    ]
    assert retained["evidence_ids"] == ["e0-0", "e1-0"]
    assert "passages" not in retained and "evidence" not in retained
    assert len(evaluated["response"]["decisions"][0]["reason"]) == 240
    assert apply_autonomous(catalogue, follow_up, response, operation_id=operation, provenance=metadata) == result


def test_frequency_upgrade_resets_only_unreserved_staging_and_preserves_history(catalogue):
    import json

    from semora.curation.autonomous import initialize_autonomous
    from semora.curation.proposals import digest

    connection = catalogue.store.connection
    initialize_autonomous(connection)
    old_operation = uuid.uuid4().hex
    old_result = {"status": "prepared", "prepared": {"context": {"anchor": "old-active"}}}
    with connection:
        connection.execute("INSERT INTO autonomous_cursors VALUES(?,?,999,1)", ("g", POLICY_VERSION))
        connection.executemany(
            "INSERT INTO autonomous_frontier(generation,policy_version,normalized_name,article_count,status) "
            "VALUES(?,?,?,1,'queued')",
            [("g", POLICY_VERSION, f"old-unreserved-{i}") for i in range(512)],
        )
        connection.executemany(
            "INSERT INTO autonomous_frontier VALUES(?,?,?,1,?,?,?,?,'history')",
            [("g", POLICY_VERSION, "old-active", "prepared", 1, 99999999999, old_operation),
             ("g", POLICY_VERSION, "old-retry", "queued", 2, 0, "previous-attempt"),
             ("g", POLICY_VERSION, "old-deferred", "deferred", 2, 99999999999, "previous-deferred"),
             ("g", POLICY_VERSION, "old-published", "published", 1, 0, "previous-publication")],
        )
        connection.execute("INSERT INTO autonomous_discovery_receipts VALUES(?,?,?,?,?,?)",
                           (old_operation, "g", POLICY_VERSION, "old-active",
                            digest({"scan_limit": 128, "candidate_limit": 8}), json.dumps(old_result)))
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex, now=1)
    assert result["prepared"]["context"]["anchor"] == "ljubljana"
    assert connection.execute(
        "SELECT COUNT(*) FROM autonomous_frontier WHERE normalized_name LIKE 'old-unreserved-%'"
    ).fetchone()[0] == 0
    for name, status, attempts in [("old-active", "prepared", 1), ("old-retry", "queued", 2),
                                   ("old-deferred", "deferred", 2), ("old-published", "published", 1)]:
        row = connection.execute(
            "SELECT status,attempts,reason FROM autonomous_frontier WHERE normalized_name=?", (name,)
        ).fetchone()
        assert tuple(row) == (status, attempts, "history")
    assert tuple(connection.execute("SELECT after_entity_id,exhausted FROM autonomous_cursors").fetchone()) == (999, 1)
    assert connection.execute("SELECT after_rank FROM autonomous_frequency_cursors").fetchone()[0] == 4
    assert discover_task(catalogue, operation_id=old_operation, now=2) == old_result


def test_missing_frequency_is_explicit_and_manual_reads_still_work(catalogue):
    original = catalogue.entities.frequency_path
    catalogue.entities.frequency_path = str(original) + ".missing"
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    assert result["status"] == "empty" and result["has_more"]
    assert "not ready" in result["reason"]
    assert catalogue.find_names("lju")["names"]
    assert catalogue.store.connection.execute("SELECT COUNT(*) FROM autonomous_frequency_cursors").fetchone()[0] == 0


def test_rank_cursor_is_bound_to_artifact_identity(catalogue, tmp_path):
    discover_task(catalogue, operation_id=uuid.uuid4().hex)
    original_rank = catalogue.store.connection.execute(
        "SELECT after_rank FROM autonomous_frequency_cursors"
    ).fetchone()[0]
    other = prepare_frequency(catalogue.connection, tmp_path / "other", "different-dictionary")
    catalogue.entities.frequency_path = other["path"]
    catalogue.entities.frequency_identity = other["identity"]
    with pytest.raises(Conflict, match="Frequency order changed"):
        discover_task(catalogue, operation_id=uuid.uuid4().hex)
    assert catalogue.store.connection.execute(
        "SELECT after_rank FROM autonomous_frequency_cursors"
    ).fetchone()[0] == original_rank


def test_discovery_starts_with_common_high_id_before_rare_low_id(catalogue, tmp_path):
    with catalogue.connection:
        catalogue.connection.execute("UPDATE names SET entity_id=999 WHERE normalized_name='ljubljana'")
        catalogue.connection.execute("INSERT INTO names VALUES(1,'#rare','#rare',1)")
    ranked = prepare_frequency(catalogue.connection, tmp_path / "high-id", "high-id-dictionary")
    catalogue.entities.frequency_path = ranked["path"]
    catalogue.entities.frequency_identity = ranked["identity"]
    traced = []
    catalogue.connection.set_trace_callback(traced.append)
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex, scan_limit=1)
    assert result["prepared"]["context"]["anchor"] == "ljubljana"
    assert not any("ORDER BY article_count" in sql for sql in traced)


def test_grouped_rank_page_advances_without_model_work(catalogue):
    catalogue.store.mutate(
        "create_group", {"preferred_name": "Ljubljana", "members": ["ljubljana"]},
        operation_id=uuid.uuid4().hex, actor="admin:1", expected_revision=0,
    )
    catalogue.suggest_variants = lambda *a, **k: pytest.fail("Grouped names must not reach proposal preparation")
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex, scan_limit=1)
    assert result["status"] == "empty" and result["has_more"]
    assert catalogue.store.connection.execute("SELECT after_rank FROM autonomous_frequency_cursors").fetchone()[0] == 1
    assert catalogue.store.connection.execute("SELECT COUNT(*) FROM autonomous_frontier").fetchone()[0] == 0


def test_grouped_queued_retirement_preserves_active_work_and_retry_history(catalogue):
    first = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    catalogue.store.mutate(
        "create_group", {"preferred_name": "Marina", "members": ["marina"]},
        operation_id=uuid.uuid4().hex, actor="admin:1", expected_revision=0,
    )
    connection = catalogue.store.connection
    with connection:
        connection.execute("UPDATE autonomous_frontier SET attempts=2,operation_id='old',next_due=123 "
                           "WHERE normalized_name='marina'")
    second = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    assert second["prepared"]["context"]["anchor"] == "marija"
    retired = connection.execute("SELECT status,attempts,operation_id,next_due FROM autonomous_frontier "
                                 "WHERE normalized_name='marina'").fetchone()
    assert tuple(retired) == ("grouped", 2, "old", 123)
    active = connection.execute("SELECT status,operation_id FROM autonomous_frontier "
                                "WHERE normalized_name='ljubljana'").fetchone()
    assert tuple(active) == ("prepared", first["discovery_id"])


def test_membership_page_lookup_uses_primary_key(catalogue):
    plan = " ".join(row[3] for row in catalogue.store.connection.execute(
        "EXPLAIN QUERY PLAN SELECT normalized_name FROM group_members WHERE normalized_name IN (?,?)",
        ("ljubljana", "marina"),
    ))
    assert "SEARCH group_members USING COVERING INDEX" in plan
    assert "SCAN group_members" not in plan
