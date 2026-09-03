CREATE TABLE openai_embedding_batches (
    batch_job_id TEXT PRIMARY KEY,
    embedding_run_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    input_file_path TEXT NOT NULL,
    input_file_id TEXT,
    provider_batch_id TEXT UNIQUE,
    output_file_id TEXT,
    error_file_id TEXT,
    request_count INTEGER NOT NULL,
    input_count INTEGER NOT NULL,
    completed_request_count INTEGER NOT NULL DEFAULT 0,
    failed_request_count INTEGER NOT NULL DEFAULT 0,
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    submitted_at TEXT,
    last_polled_at TEXT,
    completed_at TEXT,
    imported_at TEXT,
    FOREIGN KEY (embedding_run_id) REFERENCES embedding_runs(embedding_run_id),
    UNIQUE (embedding_run_id, batch_index)
);

CREATE TABLE openai_embedding_batch_items (
    batch_job_id TEXT NOT NULL,
    embedding_run_id TEXT NOT NULL,
    custom_id TEXT NOT NULL,
    input_index INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    imported_at TEXT,
    PRIMARY KEY (batch_job_id, custom_id, input_index),
    FOREIGN KEY (batch_job_id) REFERENCES openai_embedding_batches(batch_job_id),
    FOREIGN KEY (embedding_run_id) REFERENCES embedding_runs(embedding_run_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);

CREATE INDEX idx_openai_embedding_batches_run_status
ON openai_embedding_batches(embedding_run_id, status, batch_index);

CREATE INDEX idx_openai_embedding_batch_items_batch_status
ON openai_embedding_batch_items(batch_job_id, status, custom_id, input_index);

CREATE INDEX idx_openai_embedding_batch_items_run_chunk_status
ON openai_embedding_batch_items(embedding_run_id, chunk_id, status);
