DROP TABLE chunk_fts;

CREATE VIRTUAL TABLE chunk_fts USING fts5(
    title,
    text,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE chunk_fts_map (
    fts_id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
