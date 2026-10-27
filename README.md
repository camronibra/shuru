# shuru

*A small data pipeline on a mission: stopping machines from misclassifying "low-resource" languages spoken by millions.*

> **Status: early development (Stage 0 of 7).** The project skeleton, configuration and logging interfaces are in place. Most modules are still specified stubs. The original prototype scraper, which does run, is described under [Legacy prototype](#legacy-prototype-friendlyscraper).

## About

NLP tools routinely misclassify low-resource languages or lump them in with a dominant relative. Fixing that starts with data: clean, deduplicated text corpora where closely related languages and varieties are told apart.

shuru is a general tool for collecting text in low-resource languages. It gathers text from sources you configure, then extracts, cleans, deduplicates and labels it and stores it as a corpus for training models elsewhere. A lightweight classifier can be trained to tell a language from its close relatives and used to filter the corpus. Its least certain predictions are queued for hand labelling, so its quality can be measured and improved.

The first problem it is being built to solve, is Dari. It's a language spoken by millions in Afghanistan, it is routinely labelled as Persian (Farsi), so the first classifier separates Dari from Iranian Persian. The pipeline itself is not tied to those languages.

## How it runs

- **Locally:** anyone with a terminal can clone it and run it. You choose and configure your own sources; none are bundled.
- **Server mode (optional):** unattended on a small Linux host with systemd. The reference target is a Raspberry Pi 4 Model B with **2 GB RAM** and an SSD, so everything is built to fit that budget: streaming instead of loading and capped response sizes. No model training happens on the Pi. It only collects, cleans and stores text, and training is done on a separate PC from exported corpus snapshots.

## Planned pipeline

| Stage | What it adds |
|---|---|
| 0 | Project skeleton: packaging, config, structured logging, server bootstrap ← *current* |
| 1 | SQLite storage (WAL), schema migrations, a runs table recording every job, phone alerts on failure, health monitoring |
| 2 | Ingestion: config-driven sources, robots.txt, polite fetching, text extraction, corpus storage and cleaning |
| 3 | Language-script text processing (normalisation, tokenisation, near-duplicate detection) with benchmarks |
| 4 | Corpus statistics, language-ID baselines, classifier training on a PC (first case: Dari vs Iranian Persian) and a model registry |
| 5 | Scoring live news headlines as unseen data, plus drift tracking |
| 6 | A labelling queue for the least certain sentences |
| 7 | systemd hardening, backups, CI and documentation |

## Quick start (development)

Requires Python 3.11+, SQLite 3.38+ and [uv](https://docs.astral.sh/uv/).

```bash
make lock    # first time only: resolve dependencies into uv.lock
make sync    # install the locked dependencies
make check   # lint, type-check and test
```

Settings live in `config/settings.toml`. Copy `.env.example` to `.env` for local overrides.

## Design principles

- **Nothing scraped is ever committed.** The repo holds the code, config and docs only. Data, models and the database would live outside it, on whichever local machine it's being run on.
- **Polite, config-driven collection.** Sources are defined in TOML, not code.
- **Labels from source metadata (distant supervision).** Each source is tagged with the language or variety it represents and its text inherits that label. In the first case, text from Afghan outlets is labelled Dari and text from Iranian outlets is labelled Persian. This label is noisy because outlets usually quote each other, publish in both varieties and carry wire copy. A classifier can also learn an outlet's keyboard habits or house style instead of the language variety. Script normalisation, deduplication and splitting train/test data by source and by date are there as tools to assist in limitting this. Hand labels are stored separately, so their effect can be measured.
- **Honest evaluation.** Train/test splits are by source and by date, never random. News feeds used for scoring are separate from the training sources.
- **Measure before optimising.** The pipeline is expected to be network-bound. Per-step timings are recorded for every run and a compiled (Rust) component will be added only if a CPU-bound step shows up in measurements.
- **Scraped content is hostile input.** It is truncated, sanitised and never executed or rendered as HTML.
- **Every job is observable.** Each run is recorded with its status and metrics, and failures send a phone alert.

## Legacy prototype: friendlyscraper

The project started as `friendlyscraper`, a small rate-limited scraper. Its files (`scraper.py`, `source.py`, `models.py`) are still at the repo root for reference and will be replaced by the Stage 2 ingestion pipeline.

```bash
pip install -r requirements.txt
python scraper.py
```

It runs against [books.toscrape.com](https://books.toscrape.com/), a site intended for scraping practice, and writes a JSONL file (one JSON object per line: url, text). For each source you supply the category pages, an article URL pattern (regex) and a CSS selector. To add your own, copy the example from `source.py` into `sources_local.py`, which is gitignored and takes priority.

The prototype checks robots.txt before every request, sends a descriptive user agent, waits at least 2 seconds between requests (longer if the site sets a crawl delay) and stores the source directory. It has no retry/backoff and no support for JavaScript-rendered sites.

## Future direction

Live scraping is the small-scale mode. A later direction is finding text in target languages inside web archives (e.g. Common Crawl), using the same extraction and classification logic at a much larger scale.

## Disclaimer

Not affiliated with any website scraped with it. Use only where a site's terms of service and the law permit. You are responsible for your own compliance, including copyright and dataset licences. Provided as is, without warranty.
