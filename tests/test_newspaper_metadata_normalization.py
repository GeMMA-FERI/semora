from __future__ import annotations

import json
import pathlib

from semora.storage import Database, Newspaper, Run

FIXTURE_DIRECTORY = pathlib.Path(__file__).resolve().parent / "fixtures" / "newspapers"
SAMPLE_STEMS = (
    "URN_NBN_SI_doc-003KUXI2",
    "URN_NBN_SI_doc-004FX1J8",
    "URN_NBN_SI_DOC-19CB23YW",
)


def test_newspaper_schema_contains_normalized_metadata_columns_and_indexes(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "newspapers.sqlite")
    try:
        db.initialize()

        columns = {
            row["name"] for row in db.conn.execute("PRAGMA table_info(newspapers)")
        }
        assert {
            "metadata_json",
            "date",
            "publisher",
            "source",
            "rights",
            "title",
            "urn",
            "issue",
            "volume",
        } <= columns

        indexes = {
            row["name"] for row in db.conn.execute("PRAGMA index_list(newspapers)")
        }
        assert {
            "idx_newspapers_date",
            "idx_newspapers_publisher",
            "idx_newspapers_source_date",
            "idx_newspapers_rights",
            "idx_newspapers_title",
            "idx_newspapers_urn",
            "idx_newspapers_source_volume_issue",
        } <= indexes
    finally:
        db.close()


def test_smoke_imports_normalized_metadata_from_tracked_fixtures(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "newspapers.sqlite")
    try:
        db.initialize()
        db.insert_run(Run(run_id="newspaper-smoke", run_type="newspapers"))

        original_metadata: dict[str, dict] = {}
        for stem in SAMPLE_STEMS:
            metadata_path = FIXTURE_DIRECTORY / f"{stem}_metadata.json"
            assert metadata_path.is_file()

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            original_metadata[stem] = metadata
            db.insert_newspaper(
                Newspaper(
                    newspaper_id=stem,
                    run_id="newspaper-smoke",
                    content=f"Fixture content for {stem}",
                    metadata=metadata,
                )
            )

        rows = {row["newspaper_id"]: row for row in db.get_newspapers()}
        assert len(rows) == 3

        multilingual_title = rows["URN_NBN_SI_doc-003KUXI2"]
        assert multilingual_title["date"] == "1953-01-30"
        assert multilingual_title["publisher"] == "J. Debevec"
        assert multilingual_title["source"] == "Ameriška domovina"
        assert multilingual_title["rights"] == "InC"
        assert multilingual_title["title"] == "American home"
        assert multilingual_title["urn"] == "URN:NBN:SI:doc-003KUXI2"
        assert multilingual_title["issue"] == "21"
        assert multilingual_title["volume"] == "53"

        scalar_title = rows["URN_NBN_SI_doc-004FX1J8"]
        assert scalar_title["date"] == "1935-03-06"
        assert scalar_title["title"] == "Nemški zrakoplov &quot;Grof Zeppelin&quot;"
        assert scalar_title["issue"] == "10"
        assert scalar_title["volume"] == "69"

        typed_urn = rows["URN_NBN_SI_DOC-19CB23YW"]
        assert typed_urn["date"] == "1929"
        assert typed_urn["title"] == "Ženski svet"
        assert typed_urn["urn"] == "URN:NBN:SI:doc-19CB23YW"

        for stem, metadata in original_metadata.items():
            assert json.loads(rows[stem]["metadata_json"]) == metadata
    finally:
        db.close()
