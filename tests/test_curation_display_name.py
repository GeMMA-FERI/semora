"""Model labels are display metadata, never matching or membership authority."""

import uuid
from types import SimpleNamespace

import pytest

from semora.curation.autonomous import (
    _final_result,
    _policy_representatives,
    apply_autonomous,
    discover_task,
    evaluate_response,
)
from semora.curation.proposals import (
    MODEL_DISPLAY_SYSTEM,
    SYSTEM,
    RecordedState,
    group_representative,
    prepare_task,
    validate_response,
)
from semora.curation.provenance import MODEL_DISPLAY_VERSION
from semora.curation.store import Conflict, membership_key

from .test_curation_autonomous import answer
from .test_curation_autonomous import catalogue as catalogue
from .test_curation_overlap_store import automatic, identifier, manual
from .test_curation_overlap_store import store as store
from .test_curation_ungrouped import automatic_group, fresh_order, human_group


def publication(catalogue, prepared, response, operation=None):
    catalogue.store.configure_serving(True, 1, identifier())
    return apply_autonomous(
        catalogue, prepared, response, operation_id=operation or identifier(), provenance={"settings_revision": 1}
    )


def rename(store, group, name, *, operation=None, versions=None, members=None, authorize=None):
    return store.apply_autonomous(
        {
            "group_id": group["group_id"],
            "preferred_name": name,
            "description": group["description"],
            "members": members or group["members"],
        },
        provenance={
            "policy_version": "ocr-v1",
            "settings_revision": 1,
            "display_name_version": MODEL_DISPLAY_VERSION,
            "display_target_versions": versions if versions is not None else {group["group_id"]: group["version"]},
        },
        operation_id=operation or identifier(),
        validate=lambda: None,
        authorize=authorize,
    )


def test_new_model_label_can_use_nominative_form_absent_from_members(catalogue, tmp_path):
    catalogue.connection.execute("UPDATE names SET normalized_name='ljubljani',name='Ljubljani' WHERE entity_id=1")
    catalogue.connection.execute("UPDATE names SET normalized_name='ljubijani',name='Ljubijani' WHERE entity_id=2")
    fresh_order(catalogue, tmp_path)
    prepared = discover_task(catalogue, operation_id=identifier())["prepared"]
    assert prepared["context"]["anchor"] == "ljubljani"
    assert prepared["context"]["display_name_version"] == MODEL_DISPLAY_VERSION
    assert prepared["task"]["messages"][0]["content"] == MODEL_DISPLAY_SYSTEM
    response = answer(prepared)
    checked = evaluate_response(catalogue, prepared, response)
    assert checked["validated"]["preferred_name"] == "Ljubljana"
    assert checked["validated"]["members"] == ["ljubijani", "ljubljani"]
    result = publication(catalogue, prepared, response)
    group = catalogue.store.group(result["group_id"])
    assert group["preferred_name"] == "Ljubljana" and "ljubljana" not in group["members"]


def test_recorded_and_live_labels_replay_same_payload(catalogue):
    prepared = discover_task(catalogue, operation_id=identifier())["prepared"]
    response = answer(prepared)
    response["preferred_name"] = "Ljubljana (Slovenia)"
    recorded = SimpleNamespace(store=RecordedState(prepared["context"]), entities=catalogue.entities)
    assert _final_result(recorded, prepared, response)["validated"]["preferred_name"] == "Ljubljana (Slovenia)"
    operation = identifier()
    result = publication(catalogue, prepared, response, operation)
    before = catalogue.store.metadata()["revision"]
    assert publication(catalogue, prepared, response, operation) == result
    assert catalogue.store.metadata()["revision"] == before


def test_nonmember_existing_label_does_not_block_new_variant_growth(catalogue, tmp_path):
    original = automatic_group(catalogue.store, ["ljubljana", "ljubijana"], "City of Ljubljana")
    human_group(catalogue.store, ["marina", "marija"])
    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljub1jana','Ljub1jana',1)")
    fresh_order(catalogue, tmp_path)
    prepared = discover_task(catalogue, operation_id=identifier())["prepared"]
    snapshot = prepared["context"]["autonomous_groups"][original["group_id"]]
    assert snapshot["representative"] == "ljubljana"
    response = answer(prepared)
    response["preferred_name"] = "Ljubljana"
    result = publication(catalogue, prepared, response)
    group = catalogue.store.group(result["group_id"])
    assert result["group_id"] == original["group_id"]
    assert group["preferred_name"] == "Ljubljana"
    assert "ljub1jana" in group["members"]


def test_metadata_only_rename_keeps_id_members_provenance_and_exact_count_through_undo(store):
    first = automatic(store, ["ljubljani", "ljubijani"], name="Ljubljani")
    original = store.group(first["group_id"])
    key = membership_key(original["members"])
    store.save_count({"group_id": original["group_id"], "generation": "g", "membership_key": key, "article_count": 7})
    operation = identifier()
    result = rename(store, original, "Ljubljana", operation=operation)
    renamed = store.group(first["group_id"])
    assert renamed["preferred_name"] == "Ljubljana" and renamed["version"] > original["version"]
    assert renamed["group_id"] == original["group_id"] and renamed["members"] == original["members"]
    assert renamed["member_provenance"] == original["member_provenance"]
    assert membership_key(renamed["members"]) == key
    assert (
        store.connection.execute(
            "SELECT article_count FROM group_counts WHERE group_id=? AND publication='g' AND revision=?",
            (renamed["group_id"], renamed["version"]),
        ).fetchone()[0]
        == 7
    )
    assert rename(store, original, "Ljubljana", operation=operation) == result
    manual(store, "undo_change", {"operation_id": operation})
    restored = store.group(first["group_id"])
    assert restored["preferred_name"] == original["preferred_name"]
    assert restored["members"] == original["members"] and restored["member_provenance"] == original["member_provenance"]
    assert (
        store.connection.execute(
            "SELECT article_count FROM group_counts WHERE group_id=? AND publication='g' AND revision=?",
            (restored["group_id"], restored["version"]),
        ).fetchone()[0]
        == 7
    )


def test_stale_display_version_preserves_newer_label_but_still_unions_members(store):
    first = automatic(store, ["alpha", "a1"], name="Old label")
    original = store.group(first["group_id"])
    rename(store, original, "New proper label")
    result = rename(store, original, "Stale model label", members=["alpha", "extra"])
    group = store.group(result["group_id"])
    assert group["preferred_name"] == "New proper label" and "extra" in group["members"]
    missing = rename(store, group, "Unbound label", versions={}, members=["alpha", "another"])
    assert store.group(missing["group_id"])["preferred_name"] == "New proper label"


def test_identical_display_names_do_not_merge_distinct_groups(store):
    first = automatic(store, ["alpha", "a1"], name="Example")
    second = automatic(store, ["beta", "b1"], name="Example")
    assert first["group_id"] != second["group_id"] and len(store.groups()) == 2


def test_protected_group_label_cannot_change_through_display_marker(store):
    created = manual(store, "create_group", {"preferred_name": "Reviewed", "members": ["alpha", "a1"]})
    group = store.group(created["group_id"])
    with pytest.raises(Conflict):
        rename(store, group, "Overwritten")
    assert store.group(group["group_id"]) == group


def test_authorization_loss_rolls_back_label_and_audit(store):
    first = automatic(store, ["alpha", "a1"], name="Original")
    group = store.group(first["group_id"])
    operation = identifier()

    def deny():
        raise PermissionError("Lease lost")

    with pytest.raises(PermissionError):
        rename(store, group, "New label", operation=operation, authorize=deny)
    assert store.group(group["group_id"]) == group
    assert (
        store.connection.execute(
            "SELECT 1 FROM curation_changes WHERE operation_id=?", (str(uuid.UUID(operation)),)
        ).fetchone()
        is None
    )


def test_manual_proposal_and_legacy_prompt_still_require_selected_spelling(catalogue):
    prepared = prepare_task(catalogue, "ljubljana")
    response = answer(prepared)
    response["preferred_name"] = "City of Ljubljana"
    with pytest.raises(ValueError, match="selected spellings"):
        validate_response(prepared, response, catalogue.store, catalogue.entities.generation)
    assert prepared["task"]["messages"][0]["content"] == SYSTEM
    assert "display_name_version" not in prepared["context"]


@pytest.mark.parametrize("label", ["", " ", "x" * 201, "Bad\x00label"])
def test_invalid_new_display_names_fail_before_publication(catalogue, label):
    prepared = discover_task(catalogue, operation_id=identifier())["prepared"]
    response = answer(prepared)
    response["preferred_name"] = label
    with pytest.raises(ValueError, match="display name"):
        evaluate_response(catalogue, prepared, response)
    assert catalogue.store.groups() == []


def test_policy_representative_uses_bounded_member_index_not_display_text(catalogue):
    members = [f"variant-{i:03}" for i in range(100)]
    catalogue.connection.executemany(
        "INSERT INTO names VALUES(?,?,?,?)", [(10 + i, name, name.title(), i) for i, name in enumerate(members)]
    )
    statements = []
    catalogue.connection.set_trace_callback(statements.append)
    representative, stats = group_representative(catalogue, {"members": members, "preferred_name": "Unrelated label"})
    catalogue.connection.set_trace_callback(None)
    assert representative == "variant-099" and stats["article_count"] == 99
    assert len([sql for sql in statements if sql.startswith("SELECT")]) == 1
    plan = catalogue.connection.execute(
        "EXPLAIN QUERY PLAN SELECT normalized_name,name,article_count FROM names WHERE normalized_name IN (?,?)",
        members[:2],
    ).fetchall()
    assert any("INDEX" in row[3] and "normalized_name=?" in row[3] for row in plan)
    assert not any("SCAN names" in row[3] or "TEMP B-TREE" in row[3] for row in plan)


def test_group_without_current_published_member_fails_closed(catalogue):
    with pytest.raises(Conflict, match="published representative"):
        group_representative(catalogue, {"members": ["not-in-dictionary"], "preferred_name": "Ljubljana"})


def test_concurrent_label_change_preserves_newer_label_without_blocking_union(catalogue, tmp_path):
    original = automatic_group(catalogue.store, ["ljubljana", "ljubijana"], "Original")
    human_group(catalogue.store, ["marina", "marija"])
    catalogue.connection.execute("INSERT INTO names VALUES(5,'ljub1jana','Ljub1jana',1)")
    fresh_order(catalogue, tmp_path)
    prepared = discover_task(catalogue, operation_id=identifier())["prepared"]
    response = answer(prepared)
    rename(catalogue.store, catalogue.store.group(original["group_id"]), "Newer proper label")
    result = publication(catalogue, prepared, response)
    group = catalogue.store.group(result["group_id"])
    assert group["preferred_name"] == "Newer proper label" and "ljub1jana" in group["members"]
    assert publication(catalogue, prepared, response, result["operation_id"]) == result


def test_undo_removed_dominant_representative_cannot_authorize_old_policy(catalogue, tmp_path):
    catalogue.connection.executemany(
        "INSERT INTO names VALUES(?,?,?,?)", [(5, "ljub1jana", "Ljub1jana", 1), (6, "ljubljana.", "Ljubljana.", 1)]
    )
    original = automatic_group(catalogue.store, ["ljubijana", "ljub1jana"], "Low-frequency aliases")
    extension = automatic_group(catalogue.store, ["ljubijana", "ljubljana"], "Ljubljana")
    human_group(catalogue.store, ["marina", "marija"])
    fresh_order(catalogue, tmp_path)
    prepared = discover_task(catalogue, operation_id=identifier())["prepared"]
    assert prepared["context"]["anchor"] == "ljubljana."
    assert prepared["context"]["autonomous_groups"][original["group_id"]]["representative"] == "ljubljana"
    response = answer(prepared)
    recorded = SimpleNamespace(store=RecordedState(prepared["context"]), entities=catalogue.entities)
    assert _final_result(recorded, prepared, response)["status"] == "accepted"
    manual(catalogue.store, "undo_change", {"operation_id": extension["operation_id"]})
    assert "ljubljana" not in catalogue.store.group(original["group_id"])["members"]
    representatives, _, baseline, compatible = _policy_representatives(catalogue, prepared, response)
    assert representatives["ljubijana"] == "ljubijana" and baseline == "ljubijana"
    assert not compatible
    with pytest.raises(Conflict):
        publication(catalogue, prepared, response)
    assert "ljubljana." not in catalogue.store.group(original["group_id"])["members"]
