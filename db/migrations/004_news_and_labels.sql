-- Live headlines, their scores, drift tracking and hand labels.
--
-- News feeds are separate from the training sources on purpose: the model has
-- never seen these outlets, so its behaviour here is an honest check.

CREATE TABLE news_feeds (
    id             INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL UNIQUE,
    url            TEXT    NOT NULL,
    -- What the feed's outlet is assumed to publish, if known. '' = unknown.
    expected_label TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL
) STRICT;

CREATE TABLE news_items (
    id           INTEGER PRIMARY KEY,
    feed_id      INTEGER NOT NULL REFERENCES news_feeds (id),
    -- The feed's guid for the entry, or its link.
    guid         TEXT    NOT NULL,
    url          TEXT    NOT NULL DEFAULT '',
    title        TEXT    NOT NULL,
    published_at TEXT,
    fetched_at   TEXT    NOT NULL,
    UNIQUE (feed_id, guid)
) STRICT;

CREATE INDEX news_items_fetched ON news_items (fetched_at);

CREATE TABLE news_scores (
    item_id    INTEGER NOT NULL REFERENCES news_items (id) ON DELETE CASCADE,
    model_id   INTEGER NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    label      TEXT    NOT NULL,
    confidence REAL    NOT NULL,
    margin     REAL    NOT NULL,
    scored_at  TEXT    NOT NULL,
    PRIMARY KEY (item_id, model_id)
) STRICT, WITHOUT ROWID;

-- The labelling queue asks for "lowest margin first" for one model.
CREATE INDEX news_scores_margin ON news_scores (model_id, margin);

-- One row per day, feed and model: enough to rebuild any window without
-- re-reading every score. Recomputed for recent days on each news run.
CREATE TABLE drift_daily (
    day          TEXT    NOT NULL,
    feed_id      INTEGER NOT NULL REFERENCES news_feeds (id),
    model_id     INTEGER NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    n            INTEGER NOT NULL,
    conf_sum     REAL    NOT NULL,
    conf_hist    TEXT    NOT NULL,  -- JSON array: counts of confidence in 10 equal bins
    label_counts TEXT    NOT NULL,  -- JSON object: predicted label -> count
    -- How often the prediction matched the feed's expected label (0 if none is set).
    n_expected   INTEGER NOT NULL DEFAULT 0,
    n_agree      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, feed_id, model_id)
) STRICT, WITHOUT ROWID;

-- Headlines waiting for a person to label them, least certain first.
CREATE TABLE label_queue (
    item_id   INTEGER PRIMARY KEY REFERENCES news_items (id) ON DELETE CASCADE,
    model_id  INTEGER NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    margin    REAL    NOT NULL,
    queued_at TEXT    NOT NULL,
    status    TEXT    NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'labelled', 'skipped'))
) STRICT;

CREATE INDEX label_queue_status ON label_queue (status, margin);

-- Labels given by a person. Kept apart from distant labels (documents.label,
-- news_feeds.expected_label) so each kind can be measured against the other.
CREATE TABLE human_labels (
    item_id     INTEGER PRIMARY KEY REFERENCES news_items (id) ON DELETE CASCADE,
    label       TEXT    NOT NULL,
    annotator   TEXT    NOT NULL DEFAULT '',
    note        TEXT    NOT NULL DEFAULT '',
    labelled_at TEXT    NOT NULL
) STRICT;
