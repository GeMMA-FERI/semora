-- Register even unidentified issues so adoption coverage is explicit.
CREATE TABLE IF NOT EXISTS pipeline_source_identities (
    newspaper_id TEXT PRIMARY KEY REFERENCES newspapers(newspaper_id),
    identifier_key TEXT UNIQUE,
    source_identifier TEXT,
    CHECK ((identifier_key IS NULL) = (source_identifier IS NULL))
);
