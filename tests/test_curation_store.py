import json
import sqlite3
import uuid

import pytest

from semora.curation import Conflict, CurationStore


@pytest.fixture
def store(tmp_path):
    with CurationStore(tmp_path / "curation.sqlite", corpus_id="corpus", initialize=True) as value:
        yield value


def mutate(store, action="create_group", payload=None, **kwargs):
    return store.mutate(
        action,
        payload or {"preferred_name": "Ljubljana", "members": ["LJUBLJANA", "Ljubljani"]},
        operation_id=kwargs.pop("operation_id", str(uuid.uuid4())),
        actor="admin:1",
        expected_revision=kwargs.pop("expected_revision", store.metadata()["revision"]),
        **kwargs,
    )


def test_group_normalization_replay_and_conflicts(store):
    operation_id = str(uuid.uuid4())
    created = mutate(store, operation_id=operation_id)
    assert store.group(created["group_id"])["members"] == ["ljubljana", "ljubljani"]
    assert mutate(store, operation_id=operation_id, expected_revision=0) == created
    assert store.metadata()["revision"] == 1
    with pytest.raises(Conflict):
        mutate(store)
    with pytest.raises(Conflict):
        mutate(store, expected_revision=0)
    with pytest.raises(Conflict):
        mutate(store, operation_id=operation_id)
    assert store.metadata()["revision"] == 1


def test_update_move_reject_and_reopen(store):
    group = mutate(store)["group_id"]
    mutate(store, "update_group", {"group_id": group, "preferred_name": "Ljubljana", "members": ["Ljubljana"]})
    pair = {"first": "Ljubljana", "second": "Ljubljani", "reason": "Different referent"}
    mutate(store, "reject_pair", pair)
    assert store.rejected("LJUBLJANI", "Ljubljana")
    with pytest.raises(Conflict):
        mutate(
            store,
            "update_group",
            {"group_id": group, "preferred_name": "Ljubljana", "members": ["Ljubljana", "Ljubljani"]},
        )
    mutate(store, "reopen_pair", pair)
    mutate(
        store, "update_group", {"group_id": group, "preferred_name": "Ljubljana", "members": ["Ljubljana", "Ljubljani"]}
    )
    with pytest.raises(Conflict):
        mutate(store, "reject_pair", pair)


def test_undo_is_audited_and_rejects_changed_records(store):
    created = mutate(store)
    changed = mutate(
        store, "update_group", {"group_id": created["group_id"], "preferred_name": "City", "members": ["Ljubljana"]}
    )
    with pytest.raises(Conflict):
        mutate(store, "undo_change", {"operation_id": created["operation_id"]})
    mutate(store, "undo_change", {"operation_id": changed["operation_id"]})
    assert store.group(created["group_id"])["preferred_name"] == "Ljubljana"
    assert store.metadata()["revision"] == 3
    assert store.connection.execute("SELECT COUNT(*) FROM curation_changes").fetchone()[0] == 3


def test_authorization_failure_rolls_back_everything(store):
    def expired():
        raise PermissionError("Lease expired")

    with pytest.raises(PermissionError):
        mutate(store, authorize=expired)
    assert store.groups() == []
    assert store.metadata()["revision"] == 0
    assert store.connection.execute("SELECT COUNT(*) FROM curation_changes").fetchone()[0] == 0


def test_audit_insert_failure_rolls_back(store):
    store.connection.execute(
        "CREATE TRIGGER fail_audit BEFORE INSERT ON curation_changes BEGIN SELECT RAISE(ABORT,'audit'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        mutate(store)
    assert store.groups() == []
    assert store.metadata()["revision"] == 0


def test_boundary_validation(store):
    for members in ([], ["x"] * 101, [" "], [42]):
        with pytest.raises(ValueError):
            mutate(store, payload={"preferred_name": "City", "members": members})
    assert store.groups() == []


def test_binding_and_read_only(store):
    with pytest.raises(ValueError, match="differs"):
        CurationStore(store.path, corpus_id="other")
    with CurationStore(store.path, corpus_id="corpus", read_only=True) as reader:
        assert reader.metadata()["revision"] == 0
        with pytest.raises(sqlite3.OperationalError):
            mutate(reader)


def test_independent_connections_cannot_overwrite_stale_revision(store):
    with CurationStore(store.path, corpus_id="corpus") as other:
        mutate(store)
        with pytest.raises(Conflict):
            mutate(other, payload={"preferred_name": "Maribor", "members": ["Maribor"]}, expected_revision=0)


def test_cli_initialization_and_replay(tmp_path, capsys, monkeypatch):
    from semora.curation.cli import main

    args = ["curation", "--store", str(tmp_path / "curation.sqlite"), "--corpus-id", "corpus"]
    monkeypatch.setattr("sys.argv", args + ["init"])
    main()
    assert json.loads(capsys.readouterr().out)["revision"] == 0
    envelope = dict(
        action="create_group",
        payload=dict(preferred_name="Ljubljana", members=["Ljubljana"]),
        operation_id=str(uuid.uuid4()),
        actor="admin:1",
        expected_revision=0,
    )
    request = tmp_path / "request.json"
    request.write_text(json.dumps(envelope), encoding="utf-8")
    for _ in range(2):
        monkeypatch.setattr("sys.argv", args + ["mutate", "--request", str(request)])
        main()
        assert json.loads(capsys.readouterr().out)["revision"] == 1


def test_dissolution_and_undo_after_unrelated_edit(store):
    first = mutate(store)
    mutate(store, payload={"preferred_name": "Maribor", "members": ["Maribor"]})
    undone = mutate(store, "undo_change", {"operation_id": first["operation_id"]})
    assert store.membership("Ljubljana") is None
    assert len(store.groups()) == 1
    mutate(store, "undo_change", {"operation_id": undone["operation_id"]})
    dissolved = mutate(store, "delete_group", {"group_id": first["group_id"]})
    assert store.membership("Ljubljana") is None
    mutate(store, payload={"preferred_name": "Other", "members": ["Ljubljana"]})
    with pytest.raises(Conflict):
        mutate(store, "undo_change", {"operation_id": dissolved["operation_id"]})
    audit = store.connection.execute(
        "SELECT before_json FROM curation_changes WHERE operation_id=?", (dissolved["operation_id"],)
    ).fetchone()[0]
    assert "maribor" not in audit


def test_receipts_distinguish_group_creation_from_undo_dissolution(store):
    created = mutate(store)
    assert created["group_exists"] is True
    undone = mutate(store, "undo_change", {"operation_id": created["operation_id"]})
    assert undone["group_id"] == created["group_id"]
    assert undone["group_exists"] is False
