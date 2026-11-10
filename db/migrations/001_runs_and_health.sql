-- Runs and health.
--
-- Every job writes one row to `runs` when it starts and updates it when it ends,
-- so "what ran, when, did it work and how long did each step take" is one query.
-- All timestamps in the database are UTC, ISO 8601 with a trailing Z.

CREATE TABLE runs (
    id          INTEGER PRIMARY KEY,
    job         TEXT    NOT NULL,
    status      TEXT    NOT NULL CHECK (status IN ('running', 'ok', 'failed')),
    started_at  TEXT    NOT NULL,
    finished_at TEXT,
    duration_s  REAL,
    -- Short error summary for failed runs. Never holds scraped page content.
    error       TEXT,
    -- JSON object of counters and per-step timings (keys like "t_fetch_s").
    metrics     TEXT    NOT NULL DEFAULT '{}'
) STRICT;

CREATE INDEX runs_job_started ON runs (job, started_at);

-- One row per health check. NULL means the metric is not available on this host
-- (temperature and throttle flags exist only on a Raspberry Pi).
CREATE TABLE health_samples (
    id               INTEGER PRIMARY KEY,
    run_id           INTEGER NOT NULL REFERENCES runs (id),
    taken_at         TEXT    NOT NULL,
    mem_total_mb     INTEGER,
    mem_available_mb INTEGER,
    swap_used_mb     INTEGER,
    load1            REAL,
    load5            REAL,
    load15           REAL,
    cpus             INTEGER NOT NULL,
    disk_total_mb    INTEGER NOT NULL,
    disk_free_mb     INTEGER NOT NULL,
    db_mb            REAL    NOT NULL,
    wal_mb           REAL    NOT NULL,
    cpu_temp_c       REAL,
    -- Raw bit flags from `vcgencmd get_throttled` (under-voltage, throttling).
    throttled        INTEGER,
    -- JSON object of failing checks, e.g. {"disk": "disk free 900 MB < 2000 MB"}.
    problems         TEXT    NOT NULL DEFAULT '{}'
) STRICT;

CREATE INDEX health_samples_taken ON health_samples (taken_at);
