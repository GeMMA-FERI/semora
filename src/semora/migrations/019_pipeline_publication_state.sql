-- The serving pointer records this epoch. A writer advances it before changing
-- any mutable corpus or auxiliary index, so an interrupted run fails closed.
CREATE TABLE pipeline_publication_state (
    state_id INTEGER PRIMARY KEY CHECK (state_id = 1),
    epoch INTEGER NOT NULL CHECK (epoch >= 0)
);
INSERT INTO pipeline_publication_state (state_id, epoch) VALUES (1, 0);
