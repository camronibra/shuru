-- Model registry and the labels a model gives to documents.

CREATE TABLE models (
    id         INTEGER PRIMARY KEY,
    created_at TEXT    NOT NULL,
    -- File name inside the models folder.
    file_name  TEXT    NOT NULL,
    -- SHA-256 of the model file, checked every time the model is loaded.
    sha256     TEXT    NOT NULL UNIQUE,
    labels     TEXT    NOT NULL,  -- JSON array, in weight-row order
    params     TEXT    NOT NULL,  -- JSON object: hash_bits, n-gram range, epochs, ...
    metrics    TEXT    NOT NULL,  -- JSON object: per-split scores, baselines, cutoff
    is_active  INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    note       TEXT    NOT NULL DEFAULT ''
) STRICT;

-- At most one active model.
CREATE UNIQUE INDEX models_one_active ON models (is_active) WHERE is_active = 1;

-- Model predictions for documents, kept apart from documents.label (the distant
-- label) so the two can be compared and the corpus filtered on either.
CREATE TABLE doc_scores (
    document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    model_id    INTEGER NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,  -- probability of the predicted label
    margin      REAL    NOT NULL,  -- gap to the runner-up; small = uncertain
    scored_at   TEXT    NOT NULL,
    PRIMARY KEY (document_id, model_id)
) STRICT, WITHOUT ROWID;
