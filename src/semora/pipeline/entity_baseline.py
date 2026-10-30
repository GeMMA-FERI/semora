"""Read-only evidence audit of a legacy entity dictionary, including partial coverage."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from semora.ner.names import normalize_name
from semora.ner.spellfix import extension_path, load_spellfix
from semora.pipeline.manifest import _invalid_constant, _unique_object
from semora.pipeline.ner_baseline import _config as ner_run_config
from semora.storage.connection import open_connection

NORMALIZATION = "nfc-casefold-whitespace-v1"
_ERROR_LIMIT = 50
_PAIR_BATCH = 200
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_mentions(rows: Iterable[Any]) -> dict[str, int]:
    """Return distinct normalized names and earliest source-relative offsets."""
    result: dict[str, int] = {}
    for row in rows:
        text, start = row["text"], row["char_start"]
        if not isinstance(text, str) or type(start) is not int or start < 0:
            raise ValueError("Mention text/start must be a string and nonnegative integer.")
        name = normalize_name(text)
        if name:
            result[name] = min(start, result.get(name, start))
    return result


def expected_names_sha256(names: dict[str, int]) -> str:
    if any(
        not name or name != normalize_name(name) or type(start) is not int or start < 0 for name, start in names.items()
    ):
        raise ValueError("Expected names must be normalized with nonnegative offsets.")
    return _sha(_json(sorted(names.items())))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path, run_id: str) -> tuple[dict, Path, str]:
    path = root / "manifest.json"
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Entity manifest is not a supported bounded regular file.")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict) or set(value) != {"generation", "ner_run_id", "names"}:
        raise ValueError("Unknown legacy entity manifest fields.")
    generation = value["generation"]
    if not isinstance(generation, str) or not re.fullmatch("[0-9a-f]{32}", generation):
        raise ValueError("Invalid legacy entity generation identifier.")
    if value["ner_run_id"] != run_id or type(value["names"]) is not int or value["names"] < 0:
        raise ValueError("Entity manifest run/count does not match the requested NER run.")
    dictionary = root / f"{generation}.sqlite"
    if dictionary.is_symlink() or not dictionary.resolve(strict=True).is_relative_to(root) or not dictionary.is_file():
        raise ValueError("Entity generation must be a contained regular SQLite file.")
    return value, dictionary.resolve(), hashlib.sha256(raw).hexdigest()


def _table(connection: sqlite3.Connection, name: str, expected: list[tuple]) -> None:
    actual = [
        (r["name"], r["type"].upper(), r["notnull"], r["pk"]) for r in connection.execute(f"PRAGMA table_info({name})")
    ]
    if actual != expected:
        raise ValueError(f"Unsupported entity dictionary schema: {name}")


def _schemas(connection: sqlite3.Connection) -> None:
    _table(
        connection,
        "names",
        [
            ("entity_id", "INTEGER", 0, 1),
            ("normalized_name", "TEXT", 1, 0),
            ("name", "TEXT", 1, 0),
            ("article_count", "INTEGER", 1, 0),
        ],
    )
    _table(
        connection,
        "entity_articles",
        [
            ("entity_id", "INTEGER", 1, 1),
            ("article_id", "TEXT", 1, 2),
            ("char_start", "INTEGER", 1, 0),
            ("content_sha256", "TEXT", 1, 0),
        ],
    )
    _table(connection, "metadata", [("generation", "TEXT", 0, 0), ("ner_run_id", "TEXT", 0, 0)])
    _table(
        connection,
        "name_spellfix_vocab",
        [
            ("id", "INTEGER", 0, 1),
            ("rank", "INT", 0, 0),
            ("langid", "INT", 0, 0),
            ("word", "TEXT", 0, 0),
            ("k1", "TEXT", 0, 0),
            ("k2", "TEXT", 0, 0),
        ],
    )
    unique = False
    for index in connection.execute("PRAGMA index_list(names)"):
        if index["unique"] and not index["partial"]:
            escaped = index["name"].replace('"', '""')
            columns = [r for r in connection.execute(f'PRAGMA index_xinfo("{escaped}")') if r["key"]]
            if len(columns) == 1 and columns[0]["name"] == "normalized_name" and columns[0]["coll"] == "BINARY":
                unique = True
    if not unique:
        raise ValueError("Entity normalized names need the supported unique BINARY index.")
    row = connection.execute("SELECT sql FROM sqlite_master WHERE name='name_spellfix' AND type='table'").fetchone()
    ddl = row[0] if row else ""
    if not re.fullmatch(
        r'\s*CREATE\s+VIRTUAL\s+TABLE\s+(?:name_spellfix|"name_spellfix"|`name_spellfix`|\[name_spellfix\])'
        r"\s+USING\s+spellfix1\s*;?\s*",
        ddl,
        re.IGNORECASE,
    ):
        raise ValueError("Unsupported spellfix virtual-table schema/options.")


def _dictionary_rows(connection: sqlite3.Connection, report: dict, error, progress) -> None:
    counts = report["counts"]
    counts["names"] = connection.execute("SELECT COUNT(*) FROM names").fetchone()[0]
    counts["existing_memberships"] = connection.execute("SELECT COUNT(*) FROM entity_articles").fetchone()[0]
    if counts["names"] != report["manifest"]["names"]:
        error("Entity manifest name count differs from dictionary rows.")
    metadata = connection.execute("SELECT * FROM metadata").fetchmany(2)
    if (
        len(metadata) != 1
        or metadata[0]["generation"] != report["manifest"]["generation"]
        or (metadata[0]["ner_run_id"] != report["run_id"])
    ):
        error("Dictionary metadata does not identify exactly the manifest generation/run.")
    # The supported spellfix shadow uses an INTEGER PRIMARY KEY. Its direct
    # ordered scan avoids the virtual table's global ORDER BY temporary B-tree.
    spelling = iter(
        connection.execute(
            "SELECT id,word,rank,langid,k1,k2,lower(spellfix1_translit(word)) AS transliterated,"
            "spellfix1_phonehash(COALESCE(k1,word)) AS phone FROM name_spellfix_vocab ORDER BY id"
        )
    )
    word = next(spelling, None)
    fingerprint = hashlib.sha256()
    rows = connection.execute(
        "SELECT n.*,m.article_id,m.char_start,m.content_sha256 FROM names n "
        "LEFT JOIN entity_articles m ON m.entity_id=n.entity_id ORDER BY n.entity_id,m.article_id"
    )
    visited = 0
    for number, (entity_id, members) in enumerate(itertools.groupby(rows, lambda r: r["entity_id"]), 1):
        first = next(members)
        name = first["normalized_name"]
        if (
            type(entity_id) is not int
            or entity_id <= 0
            or not isinstance(name, str)
            or not name
            or (
                name != normalize_name(name)
                or not isinstance(first["name"], str)
                or normalize_name(first["name"]) != name
            )
        ):
            error("Dictionary name/normalization/identity is invalid.", entity_id=entity_id)
        found = 0
        for member in itertools.chain((first,), members):
            if member["article_id"] is not None:
                found += 1
                visited += 1
                if (
                    not isinstance(member["article_id"], str)
                    or not member["article_id"]
                    or type(member["char_start"]) is not int
                    or member["char_start"] < 0
                    or not isinstance(member["content_sha256"], str)
                    or _DIGEST.fullmatch(member["content_sha256"]) is None
                ):
                    error("Dictionary membership fields are invalid.", entity_id=entity_id)
                fingerprint.update(
                    _json([entity_id, member["article_id"], member["char_start"], member["content_sha256"]]).encode(
                        "utf-8"
                    )
                )
        if found == 0 or type(first["article_count"]) is not int or first["article_count"] != found:
            error("Dictionary article_count differs from actual distinct memberships.", entity_id=entity_id)
        while word is not None and word["id"] < entity_id:
            error("Spellfix has a vocabulary row without its dictionary name.", entity_id=word["id"])
            counts["spellfix_rows"] += 1
            word = next(spelling, None)
        if word is None or word["id"] != entity_id:
            error("Dictionary name is missing from spellfix.", entity_id=entity_id)
        else:
            counts["spellfix_rows"] += 1
            if word["word"] != name or type(word["rank"]) is not int or word["rank"] != found or word["langid"] != 0:
                error("Spellfix rowid/word/rank/language disagrees with dictionary names.", entity_id=entity_id)
            if (word["k1"] or word["word"]) != word["transliterated"] or word["k2"] != word["phone"]:
                error("Spellfix phonetic payload disagrees with its word.", entity_id=entity_id)
            fingerprint.update(
                _json(
                    [
                        entity_id,
                        name,
                        first["name"],
                        found,
                        word["word"],
                        word["rank"],
                        word["langid"],
                        word["k1"],
                        word["k2"],
                    ]
                ).encode("utf-8")
            )
            word = next(spelling, None)
        if progress and (number % 1000 == 0 or number == counts["names"]):
            progress({"event": "entity_baseline_names_audited", "checked": number, "total": counts["names"]})
    while word is not None:
        error("Spellfix has a vocabulary row without its dictionary name.", entity_id=word["id"])
        counts["spellfix_rows"] += 1
        word = next(spelling, None)
    counts["orphan_name_memberships"] = counts["existing_memberships"] - visited
    if counts["orphan_name_memberships"]:
        error("Dictionary memberships refer to missing names.")
    report["dictionary_content_fingerprint"] = fingerprint.hexdigest()


def _expected_memberships(connection: sqlite3.Connection, report: dict, error, progress) -> None:
    counts, run_id = report["counts"], report["run_id"]
    counts["ner_completed_articles"] = connection.execute(
        "SELECT COUNT(*) FROM corpus.article_ner_state WHERE ner_run_id=?",
        (run_id,),
    ).fetchone()[0]
    counts["ner_pending_articles"] = connection.execute(
        "SELECT COUNT(*) FROM corpus.articles a WHERE a.is_valid=1 AND TRIM(a.content)<>'' AND NOT EXISTS "
        "(SELECT 1 FROM corpus.article_ner_state s WHERE s.ner_run_id=? AND s.article_id=a.article_id)",
        (run_id,),
    ).fetchone()[0]
    mentions = iter(
        connection.execute(
            "SELECT * FROM corpus.article_entities WHERE ner_run_id=? ORDER BY article_id,entity_index",
            (run_id,),
        )
    )
    mention = next(mentions, None)
    pending: list[tuple] = []
    active: dict[str, dict] = {}
    fingerprint = hashlib.sha256()

    def flush() -> None:
        if not pending:
            return
        values = ",".join("(?,?,?,?)" for _ in pending)
        parameters = tuple(value for pair in pending for value in pair)
        results = connection.execute(
            "WITH expected(article_id,normalized_name,first_start,source_hash) AS (VALUES " + values + ") "
            "SELECT e.*,m.article_id AS found_article,m.char_start,m.content_sha256 FROM expected e "
            "LEFT JOIN names n ON n.normalized_name=e.normalized_name "
            "LEFT JOIN entity_articles m ON m.entity_id=n.entity_id AND m.article_id=e.article_id",
            parameters,
        )
        for row in results:
            state = active[row["article_id"]]
            if row["found_article"] is None:
                counts["pending_memberships"] += 1
                state["missing"] += 1
            else:
                counts["matched_memberships"] += 1
                if row["char_start"] != row["first_start"] or row["content_sha256"] != row["source_hash"]:
                    error("Dictionary membership hash/earliest offset disagrees with current NER.", row["article_id"])
                    state["valid"] = False
                else:
                    counts["verified_memberships"] += 1
            state["remaining"] -= 1
            if state["remaining"] == 0:
                if state["valid"] and state["missing"] == 0:
                    counts["complete_articles"] += 1
                else:
                    counts["pending_articles"] += 1
                del active[row["article_id"]]
        pending.clear()

    def orphan() -> None:
        nonlocal mention
        if mention is None:
            raise ValueError("NER mention cursor unexpectedly ended.")
        counts["orphan_ner_mentions"] += 1
        error("NER mention has no completion row.", mention["article_id"])
        mention = next(mentions, None)

    states = connection.execute(
        "SELECT s.*,a.content,a.is_valid FROM corpus.article_ner_state s LEFT JOIN corpus.articles a USING(article_id) "
        "WHERE s.ner_run_id=? ORDER BY s.article_id",
        (run_id,),
    )
    for checked, state in enumerate(states, 1):
        before = report["error_count"]
        article_id, text = state["article_id"], state["content"]
        valid_source = isinstance(text, str) and text.strip(" ") != "" and state["is_valid"] == 1
        if not valid_source or state["content_sha256"] != _sha(text):
            error("NER completion is not bound to an eligible current article hash.", article_id)
        found = 0

        def article_mentions(article_id=article_id, text=text, valid_source=valid_source):
            nonlocal mention, found
            while mention is not None and mention["article_id"] < article_id:
                orphan()
            while mention is not None and mention["article_id"] == article_id:
                entity = mention
                mention = next(mentions, None)
                start, end = entity["char_start"], entity["char_end"]
                valid = (
                    valid_source
                    and type(start) is int
                    and type(end) is int
                    and 0 <= start < end <= len(text)
                    and text[start:end] == entity["text"]
                )
                if entity["entity_index"] != found:
                    error("NER mention indices are not contiguous.", article_id)
                if not valid or not isinstance(entity["label"], str) or not entity["label"].strip():
                    error("NER mention does not match its current source/label.", article_id)
                    valid = False
                if entity["score"] is not None and not math.isfinite(entity["score"]):
                    error("NER mention score is not finite.", article_id)
                    valid = False
                found += 1
                if valid:
                    yield entity

        expected = normalized_mentions(article_mentions())
        if found != state["entity_count"] or type(state["entity_count"]) is not int or found < 0:
            error("NER mention count differs from its completion record.", article_id)
        valid_state = report["error_count"] == before
        counts["expected_memberships"] += len(expected)
        fingerprint.update(
            _json([article_id, state["content_sha256"], expected_names_sha256(expected)]).encode("utf-8")
        )
        if not expected:
            counts["zero_name_articles"] += 1
            counts["complete_articles" if valid_state else "pending_articles"] += 1
        else:
            active[article_id] = {"valid": valid_state, "remaining": len(expected), "missing": 0}
            for name, start in sorted(expected.items()):
                pending.append((article_id, name, start, state["content_sha256"]))
                if len(pending) == _PAIR_BATCH:
                    flush()
        if progress and (checked % 1000 == 0 or checked == counts["ner_completed_articles"]):
            progress(
                {
                    "event": "entity_baseline_articles_audited",
                    "checked": checked,
                    "total": counts["ner_completed_articles"],
                }
            )
    while mention is not None:
        orphan()
    flush()
    counts["unexpected_memberships"] = counts["existing_memberships"] - counts["matched_memberships"]
    if counts["unexpected_memberships"]:
        error("Dictionary has memberships not present in the current completed eligible NER output.")
    report["ner_content_fingerprint"] = fingerprint.hexdigest()


def audit_entity_baseline(
    database_path: str | Path,
    entity_dir: str | Path,
    *,
    run_id: str = "classla-sl",
    spellfix_path: str | Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Inspect legacy files in read-only snapshots; missing expected pairs remain pending."""
    report: dict[str, Any] = {
        "format": "semora.entity-baseline.v1",
        "run_id": run_id,
        "normalization": NORMALIZATION,
        "eligible": False,
        "complete": False,
        "pending_memberships": 0,
        "compatibility_verified": False,
        "dictionary_path": None,
        "manifest": None,
        "errors": [],
        "error_count": 0,
        "counts": dict.fromkeys(
            (
                "names",
                "existing_memberships",
                "spellfix_rows",
                "orphan_name_memberships",
                "ner_completed_articles",
                "ner_pending_articles",
                "expected_memberships",
                "matched_memberships",
                "verified_memberships",
                "pending_memberships",
                "unexpected_memberships",
                "complete_articles",
                "pending_articles",
                "zero_name_articles",
                "orphan_ner_mentions",
            ),
            0,
        ),
        "limitations": [
            "Legacy dictionaries do not record corpus UUID or portable model/normalization provenance.",
            "This audit verifies dictionary output against stored NER, not historical model execution.",
            "Missing expected memberships are pending additions, never proof of complete dictionary coverage.",
            "Evidence fingerprints stream logical rows; no 15GB dictionary copy or physical file hash is created.",
        ],
    }

    def error(reason: str, article_id: str | None = None, *, entity_id: int | None = None) -> None:
        report["error_count"] += 1
        if len(report["errors"]) < _ERROR_LIMIT:
            report["errors"].append({"reason": reason, "article_id": article_id, "entity_id": entity_id})

    try:
        if not isinstance(run_id, str) or not run_id.strip() or run_id != run_id.strip():
            raise ValueError("NER run ID must be a nonempty trimmed string.")
        source, root = Path(database_path).resolve(strict=True), Path(entity_dir).resolve(strict=True)
        manifest, dictionary, manifest_sha = _manifest(root, run_id)
        if dictionary.samefile(source):
            raise ValueError("Dictionary cannot alias the corpus database.")
        report.update({"manifest": manifest, "manifest_sha256": manifest_sha, "dictionary_path": str(dictionary)})
        library = (
            Path(spellfix_path).resolve(strict=True)
            if spellfix_path is not None
            else extension_path(source.parent.parent)
        )
        report["spellfix_library_sha256"] = _file_hash(library)
        with closing(open_connection(dictionary, read_only=True)) as connection:
            load_spellfix(connection, library)
            connection.execute("ATTACH DATABASE ? AS corpus", (source.as_uri() + "?mode=ro",))
            connection.execute("BEGIN")
            versions = {r[0] for r in connection.execute("SELECT version FROM corpus.schema_migrations")}
            if not set(range(1, 14)).issubset(versions):
                raise ValueError("Entity audit requires the known Semora corpus/NER schema.")
            run = connection.execute("SELECT config_json FROM corpus.ner_runs WHERE ner_run_id=?", (run_id,)).fetchone()
            if run is None:
                raise ValueError("Requested historical NER run does not exist.")
            config, evidence = ner_run_config(run[0])
            report.update({"ner_config_sha256": _sha(_json(config)), "model_evidence": evidence})
            _schemas(connection)
            _dictionary_rows(connection, report, error, progress)
            _expected_memberships(connection, report, error, progress)
    except (ValueError, OSError, sqlite3.DatabaseError, KeyError, TypeError) as exc:
        error(str(exc))
    report["pending_memberships"] = report["counts"]["pending_memberships"]
    report["eligible"] = report["error_count"] == 0
    report["complete"] = (
        report["eligible"] and not report["pending_memberships"] and not report["counts"]["ner_pending_articles"]
    )
    report["evidence_fingerprint"] = _sha(_json(report))
    return report
