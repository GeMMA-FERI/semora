CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE newspapers (
    newspaper_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,
    date TEXT,
    publisher TEXT,
    source TEXT,
    rights TEXT,
    title TEXT,
    urn TEXT,
    issue TEXT,
    volume TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE articles (
    article_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    newspaper_id TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata_json TEXT,
    is_valid INTEGER DEFAULT NULL,
    cleaning_reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (newspaper_id) REFERENCES newspapers(newspaper_id)
);

CREATE TABLE chunking_runs (
    chunking_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    method TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    chunking_run_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    method TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (article_id) REFERENCES articles(article_id),
    FOREIGN KEY (chunking_run_id) REFERENCES chunking_runs(chunking_run_id),
    UNIQUE (chunking_run_id, article_id, chunk_index)
);

CREATE TABLE embedding_runs (
    embedding_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE embeddings (
    embedding_id TEXT PRIMARY KEY,
    embedding_run_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    tensor_blob BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (embedding_run_id) REFERENCES embedding_runs(embedding_run_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);

CREATE INDEX idx_logs_run_id ON logs(run_id);
CREATE INDEX idx_newspapers_run_id ON newspapers(run_id);
CREATE INDEX idx_newspapers_date ON newspapers(date);
CREATE INDEX idx_newspapers_publisher ON newspapers(publisher);
CREATE INDEX idx_newspapers_source_date ON newspapers(source, date);
CREATE INDEX idx_newspapers_rights ON newspapers(rights);
CREATE INDEX idx_newspapers_title ON newspapers(title);
CREATE INDEX idx_newspapers_urn ON newspapers(urn);
CREATE INDEX idx_newspapers_source_volume_issue
ON newspapers(source, volume, issue);
CREATE INDEX idx_articles_run_id ON articles(run_id);
CREATE INDEX idx_articles_newspaper_id ON articles(newspaper_id);
CREATE INDEX idx_chunking_runs_run_id ON chunking_runs(run_id);
CREATE INDEX idx_chunks_run_id ON chunks(run_id);
CREATE INDEX idx_chunks_article_id ON chunks(article_id);
CREATE INDEX idx_chunks_chunking_run_id ON chunks(chunking_run_id);
CREATE INDEX idx_embedding_runs_run_id ON embedding_runs(run_id);
CREATE INDEX idx_embeddings_embedding_run_id ON embeddings(embedding_run_id);
CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
