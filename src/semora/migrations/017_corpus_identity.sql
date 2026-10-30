-- Logical archive identity survives relocation and consistent file backups.
CREATE TABLE IF NOT EXISTS pipeline_corpus_identity (
    identity_id INTEGER PRIMARY KEY CHECK (identity_id=1),
    corpus_id TEXT NOT NULL UNIQUE CHECK (length(corpus_id)=32)
);
INSERT OR IGNORE INTO pipeline_corpus_identity(identity_id,corpus_id)
VALUES (1,lower(hex(randomblob(16))));
