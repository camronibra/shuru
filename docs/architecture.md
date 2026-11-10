# Architecture

How shuru works, from a URL in a config file to a label on a headline. Each section
names the module that does the work, so you can read the code next to this page.

## The big picture

```
 config/sources.toml ─┐
                      ▼
 ┌──────────── shuru ingest ─────────────────────────────────────────────┐
 │ discover ─► known? ─► fetch ─► extract ─► clean ─► duplicate? ─► store │
 │ (feeds,     (skip     (robots,  (HTML to   (normalise,  (hash,    (SQLite)
 │  sitemaps,   before    delay,    text)      quality,     MinHash)        │
 │  listings)   download) caps)               hash)                        │
 └───────────────────────────────────────────────────────────────────────┘
                      │
                      ▼  documents, labelled by source (distant supervision)
 ┌─ shuru train (PC) ─┐      ┌─ shuru score ──────────┐     ┌─ shuru export ─┐
 │ sentences, split by │ ──► │ model label + confidence│ ──► │ filtered       │
 │ source and date     │     │ for every document      │     │ snapshot       │
 └─────────────────────┘     └─────────────────────────┘     └────────────────┘
                      │
 config/news_feeds.toml ─► shuru news ─► score headlines ─► drift ─► labelling queue
                                                                        │
                                       shuru label export / apply ◄─────┘
```

Every box is one `shuru` sub-command (see `src/lab/cli.py`). Each runs, does one job,
records itself in the runs table and exits. There is no daemon.

## Where things live

| What | Where |
|---|---|
| Code, config, docs | this repository |
| Database, models, exports, reports, backups | the data root: `~/.local/share/shuru` (or `LAB_ROOT`, `/srv/lab` on a server) |
| Your own sources and feeds | `config/sources.local.toml`, `config/news_feeds.local.toml` (gitignored) |
| Secrets (`NTFY_TOPIC`) | the environment, or `.env` locally, `/etc/shuru/env` on a server |

Nothing scraped is ever inside the repository. `src/lab/config.py` refuses any data path
that would resolve outside the data root.

## Collecting: `shuru ingest`

Module: `src/lab/ingest/job.py`. Read its docstring first.

**1. Discover** (`ingest/discover.py`). Each source says how its articles are found:

- `rss`: an RSS or Atom feed lists recent articles, often with title and date.
- `sitemap`: an XML sitemap (or index of sitemaps, newest first, at most 10 followed).
- `listing`: an ordinary HTML page; links matching `article_pattern` are articles.
- `file`: a local JSON Lines dataset you downloaded yourself.

URLs are made canonical (lower-case host, no `#fragment`, no `utm_*` tracking
parameters) so one article has one URL.

**2. Skip what is known** (`corpus/store.py: filter_new_urls`). Every URL ever tried is
in the `urls` table with its outcome. Known URLs are dropped *before* anything is
downloaded. A failed URL is retried after `retry_after_hours`, up to `max_tries`.
This is the main reason a routine run is cheap.

Feeds themselves are fetched with **conditional requests**: the `ETag` and
`Last-Modified` from last time are sent back, and an unchanged feed answers
`304 Not Modified` with no body. The demo shows a second run fetching nothing but
six empty 304s.

**3. Fetch politely** (`ingest/fetch.py`). Every request goes through `Fetcher.get`:

- only `http(s)`, and only the source's own hosts. Redirects are followed by hand so
  every hop is checked again (a redirect to another site or a forbidden path stops);
- robots.txt (`ingest/robots.py`, RFC 9309 with `*` and `$` wildcards, which the
  standard library parser does not support). A missing robots.txt allows everything;
  a 401/403/5xx or network error means fetch nothing this run;
- a per-site delay: the largest of the default (2 s), the source's `delay_s` and
  the site's `Crawl-delay`;
- hard caps: response size (checked after decompression, so a small gzip bomb is
  stopped), total time, text content types only;
- retries for network errors, 429 and 5xx, waiting twice as long each time and
  honouring `Retry-After`. A site that keeps failing or refusing is left alone for
  the rest of the run.

**4. Extract** (`ingest/extract.py`). The HTML is parsed (never executed or rendered)
with selectolax, a compiled parser. Scripts, styles, navigation, headers, footers,
asides, forms and figures are removed. The body comes from the source's
`content_selector`, or from the page's largest `<article>`/`<main>`. Title and
publication date come from the page's metadata.

**5. Clean** (`corpus/clean.py: prepare`). A pure function, so easy to test and to
benchmark:

```
raw text ─► normalise ─► too short? wrong script? ─► fold ─► SHA-256 ─► tokens ─► MinHash
```

See "Text processing" below for normalise and fold.

**6. Deduplicate and store** (`corpus/store.py`).

- *Exact duplicates*: the SHA-256 of the folded text has a UNIQUE index, so "have we
  got this text already?" is one index lookup.
- *Near duplicates* (the same wire story with an extra sentence): MinHash, explained
  in `src/lab/text/dedup.py`. Each document's signature is cut into 16 bands, stored in
  `doc_bands`; any document sharing a band is a candidate, and candidates are
  compared in full. The default threshold is 0.8 estimated similarity.

Each URL's outcome is recorded: `stored`, `duplicate`, `near_duplicate`, `rejected`
(too short, wrong script, no text), `blocked` (robots, off-site) or `error`.
`shuru stats` turns those into "what was kept, and why not".

### Threads, and why there is only one writer

```
crawl thread per site ──┐
crawl thread per site ──┼──► bounded queue (200) ──► main thread: dedupe + write
crawl thread per site ──┘
```

Most of a crawl is waiting: for the network, and for the polite delay. One thread per
**site** (not per source: two sources on one site share a thread, so the site sees one
visitor) lets many sites wait at the same time.

Crawl threads only *read* the database (their own read-only connection). The main
thread is the **only writer**. It takes finished pages off the queue, checks them for
duplicates and writes them in batches of `batch_size` per transaction. That design
avoids lock contention and makes duplicate checks race-free: two threads can never
both decide the same wire story is new. The queue is bounded, so if writing falls
behind, crawlers wait instead of filling memory.

The parsing and hashing do run in the crawl threads, so with Python's GIL only one
of them computes at a time. That is fine: the benchmark shows CPU is under 1% of a
crawl's time. If it ever was not, this is the spot to change.

## Text processing

Modules: `src/lab/text/`. All pure Python, mostly calling C underneath
(`str.translate`, compiled regexes, `zlib.crc32`).

**Normalise** (`normalise.py`). The same word arrives as different code points
depending on keyboard and CMS: Arabic yeh ي against Persian yeh ی, Arabic kaf ك against
Persian kaf ک, three digit sets, optional vowel marks, zero-width non-joiners (ZWNJ)
used by some newsrooms and not others. None of that is language. Left alone it would
split word counts, hide duplicates and teach the classifier "this outlet uses an Arabic
keyboard" instead of "this is Dari".

Two levels:

- `normalise()` is what the corpus stores and readers see: yeh and kaf unified,
  invisible characters removed, whitespace tidied, ZWNJ kept only between two letters.
  Digits and vowel marks stay. It is idempotent.
- `fold()` is for comparing only (hashes, duplicate detection, classifier features):
  it also drops vowel marks, tatweel and ZWNJ, maps all digits to ASCII and lower-cases
  Latin. Never stored.

**Tokenise** (`tokenise.py`). One regex with script-specific rules: an Arabic-script
word may contain vowel marks and ZWNJ (`می‌شود` stays one token), a Latin word may
contain an apostrophe, a number may use any digit set and inner separators (`۳٫۵`).
`script_ratio()` decides whether a document is in the expected script.

**MinHash** (`dedup.py`). Explained step by step in the module docstring. Hash
parameters are derived from SHA-256, not `random`, so signatures are identical on every
machine and Python version. A test pins exact values.

## The classifier

The goal is to tell closely related varieties apart, the first case being Dari
(Afghanistan) and Iranian Persian. The labels and the model are generic.

### Labels: distant supervision

Nobody labels the training text. Each document inherits its source's label: text from
an Afghan outlet is `prs` (Dari), from an Iranian outlet `pes` (Iranian Persian). This
costs nothing and it is noisy, in ways worth being able to explain:

- outlets quote each other, run wire copy and sometimes publish in both varieties;
- the classifier can learn an outlet's house style, boilerplate, keyboard habits or
  topics (Afghan news mentions Kabul) instead of the variety;
- a source's label is a claim about the outlet, not about each sentence.

The design limits the damage and, more importantly, **measures** it:

| Defence | Where |
|---|---|
| Normalise and fold before features, so keyboards and spelling habits carry no signal | `text/normalise.py` |
| Remove duplicates, so wire copy does not count twice or appear on both sides of a split | ingest |
| Repeated sentences (bylines, "follow us") are used once | `model/dataset.py` |
| Cap examples per source, so one big outlet cannot dominate | `max_examples_per_source` |
| Equal examples per label, so the model does not learn "how much we scraped" | `balance_labels` |
| Split by source and date, never randomly (below) | `model/dataset.py` |
| Hand labels, stored apart from distant labels, to measure the noise directly | `labelling/` |

### Splits

A random split would put sentences from the same article, day and outlet on both
sides, and the test score would mostly measure memorisation. Instead:

- **`test_date`**: documents from training sources dated on or after a cutoff day (by
  default the newest 20%). *Does the model still work on later text?*
- **`test_source`**: every document from sources marked `holdout = true`. *Does it
  work on an outlet it has never seen?*
- **`human`**: hand-labelled headlines with odd ids (even ids may be used for
  training with `--human-labels`). The one test set that does not depend on the
  assumption being tested.

A big gap between `test_date` and `test_source` means the model learned outlets, not
language. In the demo, the per-source table under `test_source` shows exactly this kind
of gap between two held-out sources.

### Model

Character 1–4-grams, hashed into 2^18 buckets, with multinomial logistic regression
(`model/features.py`, `model/train.py`, `model/predict.py`).

- **Character n-grams** catch suffixes, spellings and loanwords without a tokeniser,
  and still work on a five-word headline.
- **Hashing** means no vocabulary: memory is fixed (2 labels × 262,144 floats = 2 MB)
  and nothing but the weights has to be shipped.
- **Logistic regression**, trained by plain SGD with L2 shrinkage, in about forty lines
  of `fit()`. Its probabilities are meaningful, which the labelling queue and drift
  tracking depend on.

Every model is compared with two **baselines** on the same test sentences
(`corpus/langid_baselines.py`): always-the-majority-label, and a hand-written marker
word list (`config/markers.toml`, e.g. پوهنتون against دانشگاه for "university").

The model file is one JSON header line plus raw float32 weights. It is never
unpickled, so loading a model cannot run code. The registry (`model/registry.py`,
table `models`) stores each file's SHA-256 and checks it on every load.

**Training runs on a PC**, not on the server. `shuru train` refuses on a host with less
than 3 GB of RAM unless forced. The hand-off: `shuru backup` on the server, copy the
snapshot to the PC, train there, copy the `.shuru` file back, `shuru model install`.

### Using the model as a filter

`shuru score` gives every document a predicted label and confidence (table
`doc_scores`, kept apart from the distant label). `shuru export --model-label prs
--min-confidence 0.9` then writes only what the model is at least 90% sure is Dari.
Each export gets a manifest: filters, counts per source and label, licences, SHA-256.

## Live headlines, drift and hand labels

**`shuru news`** (`news/poll.py`, `score.py`, `drift.py`, `labelling/queue.py`) runs hourly:

1. Poll the feeds in `config/news_feeds.toml`. They must be outlets the model was not
   trained on (it warns if a feed shares a site with a training source), so this is
   unseen text. Only headlines are stored.
2. Score new headlines with the active model.
3. Update `drift_daily`: per day and feed, the count, mean confidence, a 10-bin
   confidence histogram, label counts, and agreement with the feed's expected label.
4. Compare the model's first week with the latest week using the Population Stability
   Index (formula in `drift.py`). Above 0.25 sends one alert per day.
5. Top up the **labelling queue** with the headlines the model is least sure of
   (smallest gap between its top two probabilities). This is uncertainty sampling, the
   simplest form of active learning: the labels a person gives teach the most.

**Hand labelling** is a spreadsheet round trip: `shuru label export` writes a TSV file
(untrusted titles are made safe for spreadsheets), a person fills in the `label` column,
`shuru label apply` stores it in `human_labels`. `shuru label report` then answers two
questions:

- *How noisy are distant labels?* Feed's expected label vs hand label.
- *How good is the model really?* Model vs hand label. This is measured on its least
  certain headlines, so it understates its accuracy in general.

## Observability

**The runs table** (`runs.py`). Every job runs inside `with run(conn, "ingest") as r:`.
It records start, end, status, duration, an error summary, counters and per-step
timings (as JSON). A job that raises is marked failed and sends an alert. A job that was
killed outright is closed as "interrupted" by its next run. `shuru runs` lists runs;
`shuru runs --id N` shows where the time went:

```
time per step (summed over threads):
  fetch      12.03s  69.0%
  db_write    2.49s  14.3%
  minhash     2.00s  11.4%
```

Waiting steps (network, delay, database) are measured in wall-clock time. Computing
steps are measured in per-thread CPU time, so time a thread spent waiting for the GIL
doesn't make the parser look slow.

**Alerts** (`notify.py`) go to ntfy: one HTTP POST, which can land on your phone.
No topic configured means alerts are only logged. Sending never raises.

**Health** (`health/collect.py`, every 15 minutes) samples memory, swap, load, disk, the
database and WAL size, and on a Raspberry Pi the CPU temperature and firmware throttle
flags, plus any systemd units listed in `[health].watch_services`. It alerts when the
set of failing checks *changes*, not on every sample.

**Logs** go to stderr: readable text in a terminal, JSON lines under systemd. Every
string field is length-capped and control characters are escaped, because log lines
can contain scraped text.

## The database

SQLite in WAL mode: readers never block the writer, and a power cut can lose the last
commits but not corrupt the file. Schema changes are numbered SQL files in
`db/migrations/`. Each is applied once, in its own transaction, and recorded in
`schema_migrations`. `db/schema.sql` is the generated overview; a test fails if it is
out of date.

| Table | Holds |
|---|---|
| `runs`, `health_samples` | every job execution; health samples |
| `sources`, `documents`, `urls` | the corpus and the outcome of every URL tried |
| `doc_bands` | MinHash LSH index for near-duplicate lookup |
| `http_cache` | ETag / Last-Modified for conditional requests |
| `documents_fts` | full-text search (`shuru search`), kept in step by triggers |
| `models`, `doc_scores` | model registry; model labels for documents |
| `news_feeds`, `news_items`, `news_scores`, `drift_daily` | headlines and drift |
| `label_queue`, `human_labels` | hand labelling |

Tables are `STRICT` (types are enforced) and every timestamp is UTC.

Nothing loads a whole table into memory. Large reads page through the table by id
(`WHERE id > ? LIMIT n`), and the page cache is fixed at 8 MB. Measured at 20,000
documents, every job except training peaks under 50 MB (`bench/RESULTS.md`).

## Server mode

Optional: the same commands, run by systemd timers on a small Linux host. The reference
is a Raspberry Pi 4 with 2 GB of RAM, which may also be running something important such
as a DNS server.

`scripts/bootstrap_server.sh` sets it up: a `shuru` system user with no login, code in
`/opt/shuru` owned by root (the jobs cannot change it), data in `/srv/lab`, Python
packages installed by pip from `requirements.lock` with every hash checked, and the units
from `deploy/systemd/`.

| Timer | When | Job |
|---|---|---|
| `lab-health` | every 15 min | health check (may run alongside others) |
| `lab-ingest` | every 6 h | collect |
| `lab-news` | hourly | headlines, scores, drift, queue |
| `lab-score` | daily | label new documents with the active model |
| `lab-stats` | daily | corpus report |
| `lab-backup` | daily | database snapshot (keeps 7) |

**Heavy jobs never overlap.** They take a lock file (`db/jobs.lock`) and wait up to
30–60 minutes for it (`--lock-wait`). The lock is held by the kernel, so a killed job
cannot leave it stuck.

**Other services are not starved.** Every unit runs at `Nice=19`, with
`IOSchedulingClass=idle`, a `CPUQuota`, `MemoryHigh`/`MemoryMax` (well above measured
peaks) and no swap.

**A compromised job can do little.** Scraped content is treated as hostile, so the
units assume the process might be subverted: `NoNewPrivileges`, `ProtectSystem=strict`
with write access only to the folders that job writes, no home directories, private
`/tmp` and `/dev`, no capabilities, a system-call filter, no executable memory.
`systemd-analyze security` rates the ingest unit 1.3 out of 10 for exposure, where lower
is better. An optional `IPAddressDeny` line blocks LAN addresses too; it's off by default
because it would also block a local DNS server.

**Reaching it.** Over Tailscale only: `deploy/hardening/tailscale-acl.hujson` lets your
devices SSH in and lets the server reach nothing on the tailnet.
`deploy/hardening/sshd_10-hardening.conf` allows key-only SSH with no root login.

**Failures outside Python** (killed for memory, timed out) trigger `lab-alert@.service`
through `OnFailure=`, which sends the same kind of alert.

### One database owner, not one user per job

An earlier plan was one Linux user per job, with a separate writer service owning the
database. It's simpler, and as safe in practice, for all jobs to run as one `shuru` user:

- every job writes to the same SQLite file, so separate users would all need write
  access to it anyway, or a daemon to write on their behalf (a new network-facing
  component and protocol);
- what separate users would buy, limiting what a subverted job can touch, is done
  per unit by the systemd sandbox: each job can write only to its own folders;
- SQLite with WAL, a single writer thread per job and the job lock already give
  "one writer at a time".

If jobs ever needed different secrets or network access, splitting users would be
the next step.
