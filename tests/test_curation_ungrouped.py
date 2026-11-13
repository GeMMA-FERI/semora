"""Grouped spellings are processed once; concurrent automatic work converges safely."""

import copy
import uuid
from types import SimpleNamespace

import pytest

from semora.curation.autonomous import (
    POLICY_VERSION,
    _final_result,
    apply_autonomous,
    discover_task,
    evaluate_response,
    initialize_autonomous,
)
from semora.curation.frequency import prepare_frequency
from semora.curation.proposals import SYSTEM, RecordedState, digest, prepare_task, validate_response
from semora.curation.store import Conflict

from .test_curation_autonomous import answer
from .test_curation_autonomous import catalogue as catalogue


def legacy_display_task(prepared):
    """Retain explicit v1 receipt tests when new discovery opts into model labels."""
    prepared = copy.deepcopy(prepared)
    prepared["context"].pop("display_name_version", None)
    prepared["task"]["messages"][0]["content"] = SYSTEM
    prepared["fingerprint"] = digest({key: prepared[key] for key in ("context", "task")})
    return prepared


def automatic_group(store, members, preferred=None):
    store.configure_serving(True, settings_revision=1, operation_id=uuid.uuid4().hex)
    return store.apply_autonomous(
        {"preferred_name": preferred or members[0], "members": members},
        provenance={"policy_version": POLICY_VERSION, "settings_revision": 1},
        operation_id=uuid.uuid4().hex,
        validate=lambda: None,
    )


def fresh_order(catalogue, tmp_path):
    catalogue.connection.commit()
    frequency = prepare_frequency(catalogue.connection, tmp_path / "next", "dictionary-next")
    catalogue.entities.frequency_path = frequency["path"]
    catalogue.entities.frequency_identity = frequency["identity"]


def human_group(store, members):
    return store.mutate(
        "create_group",
        {"preferred_name": members[0], "members": members},
        operation_id=uuid.uuid4().hex,
        actor="admin:1",
        expected_revision=store.metadata()["revision"],
    )


def test_automatic_prepare_exposes_only_unreviewed_group_candidates(catalogue):
    group = automatic_group(catalogue.store, ["ljubljana", "ljubijana"])
    manual = prepare_task(catalogue, "marina")
    assert "ljubljana" not in manual["context"]["candidates"]
    automatic = prepare_task(catalogue, "marina", include_autonomous_groups=True)
    context = automatic["context"]
    assert {"ljubljana", "ljubijana"} <= set(context["candidates"])
    snapshot = context["autonomous_groups"][group["group_id"]]
    assert snapshot["representative"] == "ljubljana"
    assert snapshot["representative_stats"]["article_count"] == 100
    assert context["overlap_version"] == "unreviewed-union-v1"


def test_new_rare_alias_can_join_an_existing_common_automatic_group(catalogue, tmp_path):
    group = automatic_group(catalogue.store, ["ljubljana", "ljubijana"], "Ljubljana")
    human_group(catalogue.store, ["marina", "marija"])
    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljub1jana','Ljub1jana',1)")
    fresh_order(catalogue, tmp_path)
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    assert prepared["context"]["anchor"] == "ljub1jana"
    response = answer(prepared)
    assert evaluate_response(catalogue, prepared, response)["status"] == "accepted"
    result = apply_autonomous(
        catalogue, prepared, response, operation_id=uuid.uuid4().hex, provenance={"settings_revision": 1}
    )
    assert result["group_id"] == group["group_id"]
    assert catalogue.store.group(group["group_id"])["members"] == ["ljub1jana", "ljubijana", "ljubljana"]


def test_two_concurrent_raw_proposals_converge_through_review_and_apply(catalogue):
    human_group(catalogue.store, ["marina", "marija"])
    first = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    second = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    assert first["context"]["anchor"] == "ljubljana"
    assert second["context"]["anchor"] == "ljubijana"
    catalogue.store.configure_serving(True, settings_revision=1, operation_id=uuid.uuid4().hex)
    one = apply_autonomous(
        catalogue, first, answer(first), operation_id=uuid.uuid4().hex, provenance={"settings_revision": 1}
    )
    assert evaluate_response(catalogue, second, answer(second))["status"] == "accepted"
    operation = uuid.uuid4().hex
    two = apply_autonomous(
        catalogue, second, answer(second), operation_id=operation, provenance={"settings_revision": 1}
    )
    assert two["group_id"] == one["group_id"]
    assert (
        apply_autonomous(catalogue, second, answer(second), operation_id=operation, provenance={"settings_revision": 1})
        == two
    )
    assert len([g for g in catalogue.store.groups() if g["origin"] == "autonomous"]) == 1


def test_manual_validation_stays_strict_after_automatic_owner_appears(catalogue):
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    automatic_group(catalogue.store, ["ljubljana", "ljubijana"])
    with pytest.raises(Conflict):
        validate_response(prepared, answer(prepared), catalogue.store, catalogue.entities.generation)
    assert evaluate_response(catalogue, prepared, answer(prepared))["status"] == "accepted"


def test_rare_member_cannot_bridge_equally_common_group_representatives(catalogue, tmp_path):
    automatic_group(catalogue.store, ["ljubljana", "ljubijana"], "Ljubljana")
    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljub1jana','Ljub1jana',100)")
    fresh_order(catalogue, tmp_path)
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    assert prepared["context"]["anchor"] == "ljub1jana"
    response = answer(prepared)
    response["preferred_name"] = "Ljub1jana"
    checked = evaluate_response(catalogue, prepared, response)
    assert checked["status"] == "deferred"
    assert checked["validated"]["members"] == ["ljub1jana"]


def test_live_writer_rejects_new_unsafe_group_after_cached_model_validation(catalogue, tmp_path):
    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljub1jana','Ljub1jana',200)")
    fresh_order(catalogue, tmp_path)
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    response = answer(prepared)
    response["preferred_name"] = "Ljub1jana"
    for decision in response["decisions"]:
        decision["decision"] = "include" if decision["name"] == "ljubijana" else "exclude"
    assert evaluate_response(catalogue, prepared, response)["status"] == "accepted"
    old = automatic_group(catalogue.store, ["ljubljana", "ljubijana"], "Ljubljana")
    with pytest.raises(Conflict, match="invalidate"):
        apply_autonomous(
            catalogue, prepared, response, operation_id=uuid.uuid4().hex, provenance={"settings_revision": 1}
        )
    assert catalogue.store.membership("ljub1jana") is None
    assert catalogue.store.group(old["group_id"])["members"] == ["ljubijana", "ljubljana"]


def test_legacy_task_keeps_original_directional_policy(catalogue):
    human_group(catalogue.store, ["marina", "marija"])
    discover_task(catalogue, operation_id=uuid.uuid4().hex)
    prepared = discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"]
    assert prepared["context"]["anchor"] == "ljubijana"
    legacy = copy.deepcopy(prepared)
    legacy["context"].pop("overlap_version")
    legacy["context"].pop("autonomous_groups")
    legacy["context"].pop("display_name_version", None)
    legacy["fingerprint"] = digest({k: legacy[k] for k in ("context", "task")})
    assert evaluate_response(catalogue, legacy, answer(legacy))["status"] == "deferred"


@pytest.mark.parametrize("status,attempts", [("deferred", 2), ("exhausted", 3), ("published", 3)])
def test_released_queue_does_not_reset_cooldown_or_attempt_budget(catalogue, status, attempts):
    initialize_autonomous(catalogue.store.connection)
    connection = catalogue.store.connection
    with connection:
        connection.execute(
            "INSERT INTO autonomous_frontier(generation,policy_version,normalized_name,"
            "article_count,status,attempts,next_due) "
            "VALUES(?,?,'ljubljana',100,?,?,1e12)",
            (catalogue.entities.generation, POLICY_VERSION, status, attempts),
        )
        connection.execute("INSERT INTO autonomous_released_names VALUES('ljubljana')")
    discover_task(catalogue, operation_id=uuid.uuid4().hex)
    row = connection.execute(
        "SELECT status,attempts,next_due FROM autonomous_frontier WHERE normalized_name='ljubljana'"
    ).fetchone()
    assert tuple(row) == (status, attempts, 1e12)
    assert connection.execute("SELECT COUNT(*) FROM autonomous_released_names").fetchone()[0] == 0


def test_released_name_is_eligible_after_the_rank_cursor_is_exhausted(catalogue):
    first = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    connection = catalogue.store.connection
    with connection:
        connection.execute("UPDATE autonomous_frontier SET status='grouped'")
        connection.execute("UPDATE autonomous_frequency_cursors SET exhausted=1,after_rank=4")
        connection.execute("INSERT INTO autonomous_released_names VALUES('ljubljana')")
    second = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    assert second["prepared"]["context"]["anchor"] == "ljubljana"
    assert second["discovery_id"] != first["discovery_id"]
    assert connection.execute("SELECT COUNT(*) FROM autonomous_released_names").fetchone()[0] == 0
    assert discover_task(catalogue, operation_id=first["discovery_id"]) == first


def test_released_queue_is_not_drained_when_frontier_is_full(catalogue):
    initialize_autonomous(catalogue.store.connection)
    connection = catalogue.store.connection
    with connection:
        connection.executemany(
            "INSERT INTO autonomous_frontier(generation,policy_version,normalized_name,article_count,status,next_due) "
            "VALUES(?,?,?,1,'prepared',1e12)",
            [(catalogue.entities.generation, POLICY_VERSION, f"held-{i}") for i in range(512)],
        )
        connection.execute("INSERT INTO autonomous_released_names VALUES('ljubljana')")
    result = discover_task(catalogue, operation_id=uuid.uuid4().hex)
    assert result["status"] == "empty"
    assert connection.execute("SELECT COUNT(*) FROM autonomous_released_names").fetchone()[0] == 1
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM autonomous_frontier WHERE status IN ('queued','preparing','prepared')"
        ).fetchone()[0]
        == 512
    )


def test_empty_release_batch_keeps_later_release_backlog_visible(catalogue):
    discover_task(catalogue, operation_id=uuid.uuid4().hex)
    connection = catalogue.store.connection
    with connection:
        connection.execute("UPDATE autonomous_frontier SET status='grouped'")
        connection.execute("UPDATE autonomous_frequency_cursors SET exhausted=1,after_rank=4")
        connection.executemany(
            "INSERT INTO autonomous_released_names VALUES(?)", [("aaa-absent",), ("aab-absent",), ("ljubljana",)]
        )
    first = discover_task(catalogue, operation_id=uuid.uuid4().hex, scan_limit=2)
    assert first["status"] == "empty" and first["has_more"]
    assert connection.execute("SELECT COUNT(*) FROM autonomous_released_names").fetchone()[0] == 1
    second = discover_task(catalogue, operation_id=uuid.uuid4().hex, scan_limit=2)
    assert second["prepared"]["context"]["anchor"] == "ljubljana"


def test_new_group_prefers_common_member_and_replays_identical_payload(catalogue):

    human_group(catalogue.store, ["marina", "marija"])

    discover_task(catalogue, operation_id=uuid.uuid4().hex)

    prepared = legacy_display_task(discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"])

    assert prepared["context"]["anchor"] == "ljubijana"

    response = answer(prepared)

    response["preferred_name"] = "Ljubijana"

    recorded = SimpleNamespace(store=RecordedState(prepared["context"]), entities=catalogue.entities)

    before = _final_result(recorded, prepared, response)

    live = evaluate_response(catalogue, prepared, response)

    assert before["validated"]["preferred_name"] == live["validated"]["preferred_name"] == "Ljubljana"

    catalogue.store.configure_serving(True, settings_revision=1, operation_id=uuid.uuid4().hex)

    operation = uuid.uuid4().hex

    result = apply_autonomous(
        catalogue, prepared, response, operation_id=operation, provenance={"settings_revision": 1}
    )

    assert catalogue.store.group(result["group_id"])["preferred_name"] == "Ljubljana"

    assert _final_result(catalogue, prepared, response)["validated"]["preferred_name"] == "Ljubljana"

    assert (
        apply_autonomous(catalogue, prepared, response, operation_id=operation, provenance={"settings_revision": 1})
        == result
    )


def test_equal_frequency_retains_model_case_or_uses_stable_member_order(catalogue, tmp_path):

    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljubljána','Ljubljána',100)")

    fresh_order(catalogue, tmp_path)

    prepared = legacy_display_task(discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"])

    response = answer(prepared)

    response["preferred_name"] = "LJUBLJÁNA"

    assert _final_result(catalogue, prepared, response)["validated"]["preferred_name"] == "LJUBLJÁNA"

    response["preferred_name"] = "Ljubijana"

    for ordered in (response["decisions"], list(reversed(response["decisions"]))):
        response["decisions"] = ordered

        checked = _final_result(catalogue, prepared, response)

        assert checked["validated"]["preferred_name"] == "Ljubljana"


def test_excluded_higher_frequency_candidate_is_not_canonical(catalogue, tmp_path):

    human_group(catalogue.store, ["marina", "marija"])

    catalogue.connection.execute("INSERT INTO names VALUES(5,'maribor','Maribor',1000)")

    fresh_order(catalogue, tmp_path)

    discover_task(catalogue, operation_id=uuid.uuid4().hex)

    discover_task(catalogue, operation_id=uuid.uuid4().hex)

    prepared = legacy_display_task(discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"])

    assert prepared["context"]["anchor"] == "ljubijana"

    response = answer(prepared)

    response["preferred_name"] = "Ljubijana"

    for decision in response["decisions"]:
        decision["decision"] = "include" if decision["name"] == "ljubljana" else "exclude"

    checked = evaluate_response(catalogue, prepared, response)

    assert checked["status"] == "accepted"

    assert checked["validated"]["preferred_name"] == "Ljubljana"

    assert "maribor" not in checked["validated"]["members"]


def test_tied_nonanchor_canonical_uses_prepared_order_not_response_order(catalogue, tmp_path):
    human_group(catalogue.store, ["marina", "marija"])
    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljublj\u00e1na','Ljublj\u00e1na',100)")
    fresh_order(catalogue, tmp_path)
    discover_task(catalogue, operation_id=uuid.uuid4().hex)
    discover_task(catalogue, operation_id=uuid.uuid4().hex)
    prepared = legacy_display_task(discover_task(catalogue, operation_id=uuid.uuid4().hex)["prepared"])
    context = prepared["context"]
    assert context["anchor"] == "ljubijana"
    stats = context["autonomous"]["stats"]
    winners = [name for name in context["candidates"] if stats[name]["article_count"] == 100]
    assert len(winners) == 2
    expected = stats[winners[0]]["name"]
    response = answer(prepared)
    response["preferred_name"] = "Ljubijana"
    original = response["decisions"]
    for ordered in (original, list(reversed(original))):
        response["decisions"] = ordered
        checked = _final_result(catalogue, prepared, response)
        assert checked["validated"]["preferred_name"] == expected
