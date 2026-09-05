CREATE INDEX idx_chunk_fts_map_chunk_id
ON chunk_fts_map(chunk_id);

CREATE VIRTUAL TABLE chunk_lemma_fts USING fts5(
    title,
    text,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE lemma_index_state (
    state_id INTEGER PRIMARY KEY CHECK (state_id = 1),
    last_article_id TEXT NOT NULL DEFAULT '',
    surface_chunks INTEGER NOT NULL,
    processed_articles INTEGER NOT NULL DEFAULT 0,
    indexed_chunks INTEGER NOT NULL DEFAULT 0,
    pipeline_type TEXT NOT NULL,
    complete INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
