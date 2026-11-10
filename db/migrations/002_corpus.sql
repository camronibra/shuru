-- The corpus: sources, the URLs seen, the documents kept, and the indexes that
-- make duplicate checks and search cheap.

-- Mirror of the configured sources, refreshed at the start of every ingest run.
-- Keeping it in the database means a snapshot is self-describing: training on
-- another machine needs the database only, not the config that produced it.
CREATE TABLE sources (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL UNIQUE,
    kind       TEXT    NOT NULL,
    -- Distant-supervision label: the language or variety this source is assumed
    -- to publish in. Every document from the source inherits it.
    label      TEXT    NOT NULL,
    licence    TEXT    NOT NULL DEFAULT '',
    -- 1 = never used for training; kept aside to test on an unseen source.
    holdout    INTEGER NOT NULL DEFAULT 0 CHECK (holdout IN (0, 1)),
    created_at TEXT    NOT NULL
) STRICT;

CREATE TABLE documents (
    id           INTEGER PRIMARY KEY,
    source_id    INTEGER NOT NULL REFERENCES sources (id),
    url          TEXT    NOT NULL UNIQUE,
    title        TEXT    NOT NULL DEFAULT '',
    -- Cleaned, lightly normalised text. Paragraphs are separated by newlines.
    text         TEXT    NOT NULL,
    -- SHA-256 of the folded text. UNIQUE makes exact-duplicate detection an index lookup.
    text_hash    TEXT    NOT NULL UNIQUE,
    -- Copied from the source at ingest time, so later config edits do not relabel history.
    label        TEXT    NOT NULL,
    published_at TEXT,
    fetched_at   TEXT    NOT NULL,
    -- Day used for date-based splits: published date if the page gave one, else fetch date.
    doc_date     TEXT    NOT NULL,
    n_chars      INTEGER NOT NULL,
    n_tokens     INTEGER NOT NULL,
    -- MinHash signature: num_perm unsigned 32-bit integers, little-endian.
    signature    BLOB    NOT NULL
) STRICT;

CREATE INDEX documents_source ON documents (source_id);
CREATE INDEX documents_date   ON documents (doc_date);

-- LSH index for near-duplicate lookup. A signature is cut into bands; two documents
-- that agree on any whole band are candidates. One row per (document, band).
CREATE TABLE doc_bands (
    band   INTEGER NOT NULL,
    hash   INTEGER NOT NULL,
    doc_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    PRIMARY KEY (band, hash, doc_id)
) STRICT, WITHOUT ROWID;

-- Every article URL the pipeline has tried, with what happened to it. A URL in
-- this table is not fetched again (except errors, which are retried a few times).
CREATE TABLE urls (
    url           TEXT    PRIMARY KEY,
    source_id     INTEGER NOT NULL REFERENCES sources (id),
    status        TEXT    NOT NULL CHECK (status IN (
                      'stored',          -- became a document
                      'duplicate',       -- same text as an existing document
                      'near_duplicate',  -- almost the same text as an existing document
                      'rejected',        -- extracted but failed cleaning (too short, wrong script)
                      'blocked',         -- robots.txt or an off-site redirect
                      'error'            -- network or HTTP error
                  )),
    detail        TEXT    NOT NULL DEFAULT '',
    http_status   INTEGER,
    -- The document stored, or the existing document this URL duplicates.
    document_id   INTEGER REFERENCES documents (id) ON DELETE SET NULL,
    tries         INTEGER NOT NULL DEFAULT 1,
    first_seen_at TEXT    NOT NULL,
    last_tried_at TEXT    NOT NULL
) STRICT, WITHOUT ROWID;

CREATE INDEX urls_source_status ON urls (source_id, status);

-- Validators for conditional requests on feeds, sitemaps and listing pages.
-- Sending them back lets the server answer "304 Not Modified" with no body.
CREATE TABLE http_cache (
    url           TEXT PRIMARY KEY,
    etag          TEXT,
    last_modified TEXT,
    checked_at    TEXT NOT NULL
) STRICT, WITHOUT ROWID;

-- Full-text search over the corpus. An external-content table: the index stores
-- terms only and reads the text back from `documents`; triggers keep it in step.
CREATE VIRTUAL TABLE documents_fts USING fts5 (
    title,
    text,
    content = 'documents',
    content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER documents_fts_insert AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts (rowid, title, text) VALUES (new.id, new.title, new.text);
END;

CREATE TRIGGER documents_fts_delete AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts (documents_fts, rowid, title, text)
    VALUES ('delete', old.id, old.title, old.text);
END;

CREATE TRIGGER documents_fts_update AFTER UPDATE OF title, text ON documents BEGIN
    INSERT INTO documents_fts (documents_fts, rowid, title, text)
    VALUES ('delete', old.id, old.title, old.text);
    INSERT INTO documents_fts (rowid, title, text) VALUES (new.id, new.title, new.text);
END;
