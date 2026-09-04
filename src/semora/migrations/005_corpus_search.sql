ALTER TABLE newspapers ADD COLUMN relative_path TEXT;

ALTER TABLE articles ADD COLUMN char_start INTEGER;
ALTER TABLE articles ADD COLUMN char_end INTEGER;
ALTER TABLE articles ADD COLUMN line_start INTEGER;
ALTER TABLE articles ADD COLUMN line_end INTEGER;

ALTER TABLE chunks ADD COLUMN char_start INTEGER;
ALTER TABLE chunks ADD COLUMN char_end INTEGER;
ALTER TABLE chunks ADD COLUMN line_start INTEGER;
ALTER TABLE chunks ADD COLUMN line_end INTEGER;

CREATE UNIQUE INDEX idx_newspapers_relative_path
ON newspapers(relative_path)
WHERE relative_path IS NOT NULL;

CREATE INDEX idx_articles_source_span
ON articles(newspaper_id, line_start, line_end);

CREATE INDEX idx_chunks_source_span
ON chunks(article_id, line_start, line_end);

CREATE VIRTUAL TABLE chunk_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);
