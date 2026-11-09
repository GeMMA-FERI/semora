import copy
from types import SimpleNamespace

import pytest

from semora.curation import CurationStore
from semora.curation.proposals import prepare_task, training_example, validate_response
from semora.curation.store import Conflict


@pytest.fixture
def case(tmp_path):
    with CurationStore(tmp_path / "c.sqlite", corpus_id="c", initialize=True) as store:
        catalogue = SimpleNamespace(
            store=store,
            entities=SimpleNamespace(generation="g"),
            suggest_variants=lambda *a, **k: {
                "candidates": [
                    {"normalized_name": "ljubijana", "group_id": None},
                    {"normalized_name": "ljubljani", "group_id": "other"},
                ]
            },
            evidence=lambda name, **k: {
                "sampling": "bounded",
                "passages": [{"article_id": name, "content_sha256": "hash", "passage": name}],
            },
        )
        prepared = prepare_task(catalogue, "Ljubljana")
        response = {
            "preferred_name": "Ljubljana",
            "decisions": [
                {
                    "name": "ljubijana",
                    "decision": "include",
                    "reason": "Same place in passages.",
                    "evidence_ids": ["e0-0", "e1-0"],
                }
            ],
        }
        yield store, prepared, response


def test_prepare_bounds_and_provenance(case):
    store, prepared, response = case
    assert prepared["context"]["candidates"] == ["ljubijana"]
    assert prepared["context"]["evidence"][0]["passages"][0]["content_sha256"] == "hash"
    validated = validate_response(prepared, response, store, "g")
    assert validated["members"] == ["ljubijana", "ljubljana"]
    assert store.groups() == []


@pytest.mark.parametrize("change", ["missing", "duplicate", "invented", "no_anchor", "preferred", "fields"])
def test_reject_malformed_or_unsupported_decisions(case, change):
    store, prepared, response = case
    if change == "missing":
        response["decisions"] = []
    if change == "duplicate":
        response["decisions"] *= 2
    if change == "invented":
        response["decisions"][0]["evidence_ids"] = ["invented"]
    if change == "no_anchor":
        response["decisions"][0]["evidence_ids"] = ["e1-0"]
    if change == "preferred":
        response["preferred_name"] = "A society"
    if change == "fields":
        response["auto_apply"] = True
    with pytest.raises(ValueError):
        validate_response(prepared, response, store, "g")


def test_stale_and_tampered_tasks(case):
    store, prepared, response = case
    with pytest.raises(Conflict):
        validate_response(prepared, response, store, "new-publication")
    altered = copy.deepcopy(prepared)
    altered["context"]["candidates"].append("injected")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_response(altered, response, store, "g")
    store.connection.execute("INSERT INTO rejected_pairs VALUES('ljubijana','ljubljana','reviewed separately')")
    with pytest.raises(Conflict):
        validate_response(prepared, response, store, "g")


def test_training_targets_require_human_review(case):
    _, prepared, response = case
    with pytest.raises(ValueError):
        training_example(prepared, response, review_status="unreviewed", proposal_id="p", attempt_id="a")
    row = training_example(prepared, response, review_status="edited", proposal_id="p", attempt_id="a")
    assert row["messages"][-1]["role"] == "assistant"
    assert row["attempt_id"] == "a"


def test_export_rejects_invalid_target(case):
    _, prepared, response = case
    response["preferred_name"] = "invented"
    with pytest.raises(ValueError):
        training_example(prepared, response, review_status="accepted", proposal_id="p", attempt_id="a")


def test_existing_group_uses_actual_anchor_evidence(tmp_path):
    import uuid

    with CurationStore(tmp_path / "c.sqlite", corpus_id="c", initialize=True) as store:
        group_id = uuid.uuid4().hex
        store.mutate(
            "create_group",
            {"group_id": group_id, "preferred_name": "Zulu", "members": ["alpha", "zulu"]},
            operation_id=uuid.uuid4().hex,
            actor="admin:1",
            expected_revision=0,
        )
        catalogue = SimpleNamespace(
            store=store,
            entities=SimpleNamespace(generation="g"),
            suggest_variants=lambda *a, **k: {"candidates": [{"normalized_name": "zulv", "group_id": None}]},
            evidence=lambda name, **k: {
                "sampling": "bounded",
                "passages": [{"passage": name}] if name != "alpha" else [],
            },
        )
        assert prepare_task(catalogue, "Zulu")["context"]["evidence"][0]["name"] == "zulu"
        catalogue.evidence = lambda name, **k: {
            "sampling": "bounded",
            "passages": [] if name == "zulu" else [{"passage": name}],
        }
        with pytest.raises(ValueError, match="anchor"):
            prepare_task(catalogue, "Zulu")


def test_review_application_is_atomic_and_idempotent(case):
    import uuid

    from semora.curation.proposals import apply_reviewed

    store, prepared, response = case
    operation = uuid.uuid4().hex

    def deny():
        raise PermissionError("Revoked")

    with pytest.raises(PermissionError):
        apply_reviewed(
            prepared, response, store, "g", operation_id=operation, actor="admin:1", expected_revision=0, authorize=deny
        )
    assert store.groups() == []
    receipt = apply_reviewed(
        prepared, response, store, "g", operation_id=operation, actor="admin:1", expected_revision=0
    )
    assert store.metadata()["revision"] == 1
    assert (
        apply_reviewed(prepared, response, store, "g", operation_id=operation, actor="admin:1", expected_revision=0)
        == receipt
    )
    with pytest.raises(Conflict):
        apply_reviewed(
            prepared, response, store, "g", operation_id=uuid.uuid4().hex, actor="admin:1", expected_revision=0
        )
    with pytest.raises(Conflict):
        apply_reviewed(prepared, response, store, "g", operation_id=operation, actor="admin:2", expected_revision=0)


def test_all_uncertain_proposal_is_not_applied(case):
    import uuid

    from semora.curation.proposals import apply_reviewed

    store, prepared, response = case
    response["decisions"][0]["decision"] = "uncertain"
    with pytest.raises(ValueError, match="No additional"):
        apply_reviewed(
            prepared, response, store, "g", operation_id=uuid.uuid4().hex, actor="admin:1", expected_revision=0
        )
    assert store.groups() == []


def test_unrelated_approvals_preserve_proposals_and_replay(case):
    import uuid

    from semora.curation.proposals import apply_reviewed

    store, prepared, response = case
    store.mutate(
        "create_group",
        {"preferred_name": "Maribor", "members": ["maribor", "mariboru"]},
        operation_id=uuid.uuid4().hex,
        actor="admin:1",
        expected_revision=0,
    )
    assert validate_response(prepared, response, store, "g")["expected_revision"] == 1
    operation = uuid.uuid4().hex
    receipt = apply_reviewed(
        prepared, response, store, "g", operation_id=operation, actor="admin:1", expected_revision=0
    )
    assert receipt["revision"] == 2
    assert (
        apply_reviewed(prepared, response, store, "g", operation_id=operation, actor="admin:1", expected_revision=0)
        == receipt
    )


def test_relevant_membership_change_invalidates_proposal(case):
    import uuid

    store, prepared, response = case
    store.mutate(
        "create_group",
        {"preferred_name": "Ljubijana", "members": ["ljubijana"]},
        operation_id=uuid.uuid4().hex,
        actor="admin:1",
        expected_revision=0,
    )
    with pytest.raises(Conflict):
        validate_response(prepared, response, store, "g")


def test_validation_rejects_edit_during_relevant_state_read(case, monkeypatch):
    from semora.curation import proposals

    store, prepared, response = case
    original = proposals.relevant_state

    def interleaved(*args):
        state = original(*args)
        store.connection.execute("UPDATE curation_metadata SET revision=revision+1")
        return state

    monkeypatch.setattr(proposals, "relevant_state", interleaved)
    with pytest.raises(Conflict, match="during proposal validation"):
        validate_response(prepared, response, store, "g")


def test_preparation_guards_state_capture(case, monkeypatch):
    from semora.curation import proposals

    store, _, _ = case
    catalogue = SimpleNamespace(
        store=store,
        entities=SimpleNamespace(generation="g"),
        suggest_variants=lambda *a, **k: {"candidates": [{"normalized_name": "ljubijana", "group_id": None}]},
        evidence=lambda name, **k: {"sampling": "bounded", "passages": [{"passage": name}]},
    )
    original = proposals.relevant_state

    def interleaved(*args):
        state = original(*args)
        store.connection.execute("UPDATE curation_metadata SET revision=revision+1")
        return state

    monkeypatch.setattr(proposals, "relevant_state", interleaved)
    with pytest.raises(Conflict, match="during task preparation"):
        prepare_task(catalogue, "Ljubljana")


def test_json_text_training_target_is_an_object(case):
    import json

    _, prepared, response = case
    example = training_example(
        prepared, json.dumps(response), review_status="accepted", proposal_id="p", attempt_id="a"
    )
    assert json.loads(example["messages"][-1]["content"]) == response
