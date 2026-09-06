DROP TABLE chunk_lemma_fts;
DROP TABLE lemma_index_state;
DROP TABLE lemma_index_articles;
DROP TABLE chunk_fts;
DROP TABLE chunk_fts_map;

CREATE VIRTUAL TABLE article_fts USING fts5(
    title,
    text,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE article_fts_map (
    fts_id INTEGER PRIMARY KEY,
    article_id TEXT NOT NULL UNIQUE,
    FOREIGN KEY (article_id) REFERENCES articles(article_id)
);

CREATE TABLE article_fts_state (
    state_id INTEGER PRIMARY KEY CHECK (state_id = 1),
    last_article_id TEXT NOT NULL DEFAULT '',
    indexed_articles INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE article_lemma_fts USING fts5(
    title,
    text,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE article_lemma_index_state (
    state_id INTEGER PRIMARY KEY CHECK (state_id = 1),
    last_article_id TEXT NOT NULL DEFAULT '',
    surface_articles INTEGER NOT NULL,
    processed_articles INTEGER NOT NULL DEFAULT 0,
    indexed_articles INTEGER NOT NULL DEFAULT 0,
    pipeline_type TEXT NOT NULL,
    complete INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_article_fts_map_article_id
ON article_fts_map(article_id);

CREATE INDEX idx_articles_valid_article_id
ON articles(article_id)
WHERE is_valid = 1;
