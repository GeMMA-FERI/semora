-- Batch membership is independent of corpus run IDs and hash ordering.
CREATE TABLE IF NOT EXISTS pipeline_batches (
    batch_id TEXT PRIMARY KEY,
    manifest_fingerprint TEXT NOT NULL,
    config_json TEXT NOT NULL,
    issue_count INTEGER NOT NULL CHECK (issue_count > 0),
    state TEXT NOT NULL CHECK (state IN ('running', 'failed', 'ingested')),
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_batch_issues (
    batch_id TEXT NOT NULL REFERENCES pipeline_batches(batch_id),
    relative_path TEXT NOT NULL,
    newspaper_id TEXT NOT NULL REFERENCES newspapers(newspaper_id),
    action TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    source_identifier TEXT,
    source_identifier_key TEXT,
    PRIMARY KEY (batch_id, relative_path)
);
CREATE INDEX IF NOT EXISTS pipeline_batch_issues_newspaper ON pipeline_batch_issues(newspaper_id);
CREATE INDEX IF NOT EXISTS pipeline_batch_issues_identifier ON pipeline_batch_issues(source_identifier_key, newspaper_id);

CREATE TABLE IF NOT EXISTS pipeline_issue_stages (
    newspaper_id TEXT NOT NULL REFERENCES newspapers(newspaper_id),
    stage TEXT NOT NULL CHECK (stage IN ('articles', 'chunks')),
    input_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    output_count INTEGER NOT NULL CHECK (output_count >= 0),
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (newspaper_id, stage)
);
