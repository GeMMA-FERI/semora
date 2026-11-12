import concurrent.futures
import json
import sqlite3
import threading
import uuid
from contextlib import closing

import pytest

from semora.curation import Conflict, CurationStore


def identifier():
    return uuid.uuid4().hex


def manual(store, members=None, actor="admin:1", **payload):
    action = "update_group" if payload.get("group_id") else "create_group"
    return store.mutate(
        action,
        {"preferred_name": "Ljubljana", "members": members or ["Ljubljana", "Ljubijana"], **payload},
        operation_id=identifier(),
        actor=actor,
        expected_revision=store.metadata()["revision"],
    )


@pytest.fixture
def store(tmp_path):
    with CurationStore(tmp_path / "curation.sqlite", corpus_id="c", initialize=True) as value:
        yield value


def enable(store, revision=1):
    return store.configure_serving(True, revision, identifier())


def automatic(store, members=None, provenance=None, **kwargs):
    return store.apply_autonomous(
        {"preferred_name": "Ljubljana", "members": members or ["Ljubljana", "Ljubijana"]},
        operation_id=kwargs.pop("operation_id", identifier()),
        provenance=provenance or {"policy_version": "ocr-v1", "settings_revision": 1},
        validate=kwargs.pop("validate", lambda: None),
        **kwargs,
    )


def legacy_store(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with CurationStore(path, corpus_id="c", initialize=True) as value:
        operation = identifier()
        payload = {"preferred_name": "Ljubljana", "members": ["Ljubljana", "Ljubijana"]}
        result = value.mutate("create_group", payload, operation_id=operation, actor="admin:1", expected_revision=0)
    with closing(sqlite3.connect(path)) as conn, conn:
        for table in ("member_provenance", "group_provenance", "curation_serving", "curation_setting_receipts"):
            conn.execute("DROP TABLE " + table)
        conn.execute("UPDATE curation_metadata SET schema_version=1")
        for column in ("before_json", "after_json"):
            state = json.loads(conn.execute("SELECT " + column + " FROM curation_changes").fetchone()[0])
            if state["group"]:
                for key in ("origin", "human_reviewed", "provenance", "member_provenance"):
                    state["group"].pop(key)
            conn.execute("UPDATE curation_changes SET " + column + "=?", (json.dumps(state),))
    return path, payload, operation, result


def test_autonomous_requires_explicit_current_serving_and_bounded_provenance(store):
    assert store.serving_enabled() is None
    with pytest.raises(Conflict, match="settings"):
        automatic(store)
    enable(store)
    with pytest.raises(Conflict, match="settings"):
        automatic(store, provenance={"policy_version": "ocr-v1", "settings_revision": 0})
    with pytest.raises(ValueError, match="policy_version"):
        automatic(store, provenance={"settings_revision": 1})
    with pytest.raises(ValueError, match="JSON"):
        automatic(store, provenance={"policy_version": "ocr-v1", "bad": float("nan")})
    with pytest.raises(ValueError, match="16 KiB"):
        automatic(store, provenance={"policy_version": "ocr-v1", "large": "x" * 20000})
    assert store.groups() == []


def test_serving_settings_are_monotonic_idempotent_and_authorized(store):
    operation = identifier()
    result = store.configure_serving(True, 4, operation)
    assert result == {"applied_settings_revision": 4, "serving_enabled": True}
    assert store.configure_serving(True, 4, operation) == result
    with pytest.raises(Conflict):
        store.configure_serving(False, 4, operation)
    with pytest.raises(Conflict):
        store.configure_serving(False, 4, identifier())
    assert store.configure_serving(False, 3, identifier()) == result

    def denied():
        raise PermissionError("disabled")

    with pytest.raises(PermissionError):
        store.configure_serving(False, 5, identifier(), denied)
    assert store.serving_enabled() is True
    with pytest.raises(PermissionError):
        store.configure_serving(True, 4, operation, denied)
    store.configure_serving(False, 5, identifier())
    with pytest.raises(Conflict, match="settings"):
        automatic(store)


def test_autonomous_receipt_replays_without_revalidating_but_reauthorizes(store):
    enable(store)
    calls = []
    operation = identifier()
    result = automatic(
        store,
        operation_id=operation,
        validate=lambda: calls.append("validate"),
        authorize=lambda: calls.append("authorize"),
    )
    assert (
        automatic(
            store,
            operation_id=operation,
            validate=lambda: calls.append("validate"),
            authorize=lambda: calls.append("authorize"),
        )
        == result
    )
    assert calls == ["validate", "authorize", "authorize"]
    group = store.group(result["group_id"])
    assert group["origin"] == "autonomous" and group["human_reviewed"] is False
    assert all(
        item["origin"] == "autonomous" and item["human_reviewed"] is False
        for item in group["member_provenance"].values()
    )
    assert group["provenance"]["policy_version"] == "ocr-v1"
    with pytest.raises(Conflict):
        automatic(store, operation_id=operation, provenance={"policy_version": "different", "settings_revision": 1})


def test_dedicated_autonomous_method_cannot_be_bypassed(store):
    with pytest.raises(ValueError, match="dedicated"):
        store.mutate("apply_autonomous", {}, operation_id=identifier(), actor="admin:1", expected_revision=0)


def test_human_members_and_metadata_are_protected_and_undo_restores_tags(store):
    created = manual(store, description="Reviewed city")
    group_id = created["group_id"]
    before = store.group(group_id)
    enable(store)
    payload = {
        "group_id": group_id,
        "preferred_name": "Ljubljana",
        "description": "Reviewed city",
        "members": [*before["members"], "LjubIjanna"],
    }
    result = store.apply_autonomous(
        payload,
        provenance={"policy_version": "ocr-v1", "settings_revision": 1},
        operation_id=identifier(),
        validate=lambda: None,
    )
    group = store.group(group_id)
    assert group["origin"] == "mixed" and group["human_reviewed"] is False
    for name in before["members"]:
        assert group["member_provenance"][name] == before["member_provenance"][name]
    assert group["member_provenance"]["ljubijanna"]["human_reviewed"] is False
    with pytest.raises(Conflict, match="metadata"):
        store.apply_autonomous(
            {**payload, "preferred_name": "LjubIjanna", "members": [*payload["members"], "Ljubljiana"]},
            provenance={"policy_version": "ocr-v1", "settings_revision": 1},
            operation_id=identifier(),
            validate=lambda: None,
        )
    with pytest.raises(Conflict, match="remove"):
        store.apply_autonomous(
            {**payload, "members": ["Ljubljana", "Ljubljiana"]},
            provenance={"policy_version": "ocr-v1", "settings_revision": 1},
            operation_id=identifier(),
            validate=lambda: None,
        )
    store.mutate(
        "undo_change",
        {"operation_id": result["operation_id"]},
        operation_id=identifier(),
        actor="admin:1",
        expected_revision=store.metadata()["revision"],
    )
    restored = store.group(group_id)
    assert restored["member_provenance"] == before["member_provenance"]
    assert restored["origin"] == "human" and restored["human_reviewed"] is True


def test_human_review_preserves_autonomous_origin(store):
    enable(store)
    created = automatic(store)
    group = store.group(created["group_id"])
    manual(store, group_id=created["group_id"], members=group["members"])
    reviewed = store.group(created["group_id"])
    assert reviewed["origin"] == "autonomous" and reviewed["human_reviewed"] is True
    assert all(
        item["origin"] == "autonomous" and item["human_reviewed"] for item in reviewed["member_provenance"].values()
    )


def test_unrelated_group_progress_does_not_invalidate_autonomous_apply(store):
    enable(store)
    manual(store, members=["Maribor", "Mariboru"], preferred_name="Maribor")
    assert automatic(store)["revision"] == 2


def test_validation_and_authorization_failures_roll_back_auto_publication(store):
    enable(store)

    def denied():
        raise PermissionError("revoked")

    for kwargs in ({"validate": denied}, {"authorize": denied}):
        with pytest.raises(PermissionError):
            automatic(store, **kwargs)
    assert store.groups() == [] and store.metadata()["revision"] == 0
    assert store.connection.execute("SELECT COUNT(*) FROM curation_changes").fetchone()[0] == 0


def test_rejected_pairs_and_unknown_metadata_are_protected(store):
    created = manual(store, actor="import:legacy")
    assert store.group(created["group_id"])["origin"] == "unknown"
    enable(store)
    store.mutate(
        "reject_pair",
        {"first": "Ljubljana", "second": "Ljubljiana", "reason": "Different place"},
        operation_id=identifier(),
        actor="admin:1",
        expected_revision=1,
    )
    with pytest.raises(Conflict, match="rejected"):
        store.apply_autonomous(
            {
                "group_id": created["group_id"],
                "preferred_name": "Ljubljana",
                "members": ["Ljubljana", "Ljubijana", "Ljubljiana"],
            },
            provenance={"policy_version": "ocr-v1", "settings_revision": 1},
            operation_id=identifier(),
            validate=lambda: None,
        )


def test_schema1_readonly_upgrade_backup_replay_and_undo(tmp_path):
    path, payload, operation, result = legacy_store(tmp_path)
    with CurationStore(path, corpus_id="c", read_only=True) as reader:
        assert reader.metadata()["schema_version"] == 1
        assert reader.group(result["group_id"])["origin"] == "unknown"
        assert reader.serving_enabled() is None
    with CurationStore(path, corpus_id="c") as store:
        assert store.metadata()["schema_version"] == 2
        assert store.group(result["group_id"])["origin"] == "human"
        assert store.group(result["group_id"])["human_reviewed"] is True
        assert (
            store.mutate("create_group", payload, operation_id=operation, actor="admin:1", expected_revision=0)
            == result
        )
        store.mutate(
            "undo_change", {"operation_id": operation}, operation_id=identifier(), actor="admin:1", expected_revision=1
        )
        assert store.groups() == []
    backups = list(tmp_path.glob("legacy.sqlite.schema1.backup-*"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as conn:
        assert conn.execute("SELECT schema_version FROM curation_metadata").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM name_groups").fetchone()[0] == 1
    with CurationStore(path, corpus_id="c"):
        pass
    assert len(list(tmp_path.glob("legacy.sqlite.schema1.backup-*"))) == 1


def test_concurrent_schema1_openers_upgrade_once(tmp_path, monkeypatch):
    path, _, _, _ = legacy_store(tmp_path)
    barrier = threading.Barrier(2)
    original = CurationStore._upgrade_provenance

    def synchronized(self, **kwargs):
        barrier.wait(timeout=5)
        original(self, **kwargs)

    monkeypatch.setattr(CurationStore, "_upgrade_provenance", synchronized)

    def open_store():
        with CurationStore(path, corpus_id="c") as store:
            return store.metadata()["schema_version"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: open_store(), range(2))) == [2, 2]
    assert len(list(tmp_path.glob("legacy.sqlite.schema1.backup-*"))) == 1


def test_migration_failure_rolls_back_and_preserves_backup(tmp_path, monkeypatch):
    path, _, _, _ = legacy_store(tmp_path)

    def fail(*args):
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(CurationStore, "_save_annotations", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        CurationStore(path, corpus_id="c")
    with closing(sqlite3.connect(path)) as conn, conn:
        assert conn.execute("SELECT schema_version FROM curation_metadata").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM name_groups").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='group_provenance'").fetchone()[0] == 0
    assert len(list(tmp_path.glob("legacy.sqlite.schema1.backup-*"))) == 1
