CREATE TABLE ner_runs (
    ner_run_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- A row also records successful extraction when no entities were found.
CREATE TABLE article_ner_state (
    ner_run_id TEXT NOT NULL REFERENCES ner_runs(ner_run_id) ON DELETE CASCADE,
    article_id TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    content_sha256 TEXT NOT NULL,
    entity_count INTEGER NOT NULL CHECK (entity_count >= 0),
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ner_run_id, article_id)
);
CREATE INDEX idx_article_ner_state_article ON article_ner_state(article_id);

CREATE TABLE article_entities (
    ner_run_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    entity_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    label TEXT NOT NULL,
    char_start INTEGER NOT NULL CHECK (char_start >= 0),
    char_end INTEGER NOT NULL CHECK (char_end > char_start),
    score REAL,
    PRIMARY KEY (ner_run_id, article_id, entity_index),
    FOREIGN KEY (ner_run_id, article_id)
        REFERENCES article_ner_state(ner_run_id, article_id) ON DELETE CASCADE
);
CREATE INDEX idx_article_entities_name ON article_entities(ner_run_id, text, label);
CREATE INDEX idx_article_entities_article ON article_entities(article_id);

-- Offsets refer to articles.content, so changed text must be extracted again.
CREATE TRIGGER invalidate_article_ner AFTER UPDATE OF content, is_valid ON articles
WHEN NEW.content IS NOT OLD.content OR NEW.is_valid IS NOT OLD.is_valid
BEGIN
    DELETE FROM article_ner_state WHERE article_id = OLD.article_id;
END;
