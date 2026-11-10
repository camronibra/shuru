"""The ``shuru`` command line. Every job and every report starts here.

Each sub-command is a small function, ``cmd_<name>``, that wires settings, the
database and the runs table to one of the modules. The modules do the work; this
file only connects and prints.

Jobs that change data (ingest, news, score, train, stats, backup, reindex) take the
job lock, so two of them never run at once, and are recorded in the runs table.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from lab import __version__
from lab.backup import BackupError, backup_database
from lab.config import ConfigError, Settings, ensure_dirs, load_dotenv, load_settings
from lab.corpus import store
from lab.corpus.export import export_corpus
from lab.corpus.stats import corpus_stats, format_stats, write_report
from lab.db import DatabaseError, connect, dump_schema, file_size_mb, migrate, open_database
from lab.health.collect import read_meminfo, run_health
from lab.ingest.fetch import Fetcher
from lab.ingest.job import make_hasher, run_ingest
from lab.ingest.models import load_sources
from lab.labelling.apply import apply_sheet, known_labels, label_report
from lab.labelling.queue import export_sheet, refresh_queue
from lab.logs import configure_logging, get_logger
from lab.model import registry
from lab.model.predict import ModelError, score_documents
from lab.model.train import TrainError, format_report, train_and_register
from lab.news.drift import drift_report, update_daily
from lab.news.drift import format_report as format_drift
from lab.news.poll import load_feeds, overlapping_feeds, poll_feeds
from lab.news.score import score_headlines
from lab.notify import Notifier
from lab.runs import JobBusyError, Run, job_lock, recent_runs, run
from lab.text.normalise import normalise
from lab.text.tokenise import tokenise
from lab.timeutil import days_ago, utc_now

log = get_logger("lab.cli")

# Training belongs on a PC. Below this much RAM, `shuru train` refuses unless forced.
MIN_TRAIN_RAM_MB = 3000
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_BUSY = 75  # EX_TEMPFAIL: "try again later"


def _open(settings: Settings) -> sqlite3.Connection:
    ensure_dirs(settings.paths)
    return open_database(settings.paths.db_file)


@contextmanager
def _job(
    settings: Settings, name: str, args: argparse.Namespace, *, exclusive: bool = True
) -> Iterator[tuple[sqlite3.Connection, Run, Notifier]]:
    """Open the database, take the job lock and record a run around the block."""
    conn = _open(settings)
    notifier = Notifier.from_config(settings.notify)
    try:
        if exclusive:
            with (
                job_lock(settings.paths.lock_file, args.lock_wait),
                run(conn, name, notifier) as current,
            ):
                yield conn, current, notifier
        else:
            with run(conn, name, notifier) as current:
                yield conn, current, notifier
    finally:
        conn.close()


# --- setup and inspection -------------------------------------------------------------


def cmd_init(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    conn.close()
    print(f"data root: {settings.paths.root}")
    print(f"database:  {settings.paths.db_file} (schema up to date)")
    return 0


def cmd_sources(args: argparse.Namespace, settings: Settings) -> int:
    sources = load_sources(settings.config_dir)
    if not sources:
        print("No sources configured. Add [[source]] tables to config/sources.local.toml.")
        return 0
    print(f"{'name':<26}{'kind':<9}{'label':<8}{'holdout':<9}{'enabled':<9}site")
    for source in sources:
        print(
            f"{source.name:<26}{source.kind:<9}{source.label:<8}"
            f"{'yes' if source.holdout else 'no':<9}{'yes' if source.enabled else 'no':<9}"
            f"{source.site or source.urls[0]}"
        )
    return 0


def cmd_runs(args: argparse.Namespace, settings: Settings) -> int:
    conn = connect(settings.paths.db_file, read_only=True)
    if args.id is not None:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            print(f"no run with id {args.id}", file=sys.stderr)
            return EXIT_ERROR
        print(
            f"run {row['id']}: {row['job']} {row['status']} started {row['started_at']} "
            f"took {row['duration_s']}s"
        )
        if row["error"]:
            print(f"error: {row['error']}")
        metrics = json.loads(row["metrics"])
        timings = {k: v for k, v in metrics.items() if k.startswith("t_") and k.endswith("_s")}
        for key, value in sorted(metrics.items()):
            if key not in timings:
                print(f"  {key:<24}{value}")
        if timings:
            total = sum(timings.values()) or 1.0
            print("  time per step (summed over threads):")
            for key, value in sorted(timings.items(), key=lambda item: -item[1]):
                print(f"    {key[2:-2]:<18}{value:>10.3f}s {value / total:>7.1%}")
        return 0
    print(f"{'id':>5}  {'job':<9}{'status':<8}{'started (UTC)':<22}{'seconds':>9}  summary")
    for row in recent_runs(conn, args.limit):
        metrics = json.loads(row["metrics"])
        counters = " ".join(
            f"{key}={value}" for key, value in sorted(metrics.items()) if not key.startswith("t_")
        )
        summary = row["error"] or counters
        duration = "" if row["duration_s"] is None else f"{row['duration_s']:.1f}"
        print(
            f"{row['id']:>5}  {row['job']:<9}{row['status']:<8}{row['started_at']:<22}"
            f"{duration:>9}  {summary[:110]}"
        )
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    """Full-text search. The query is normalised like the corpus and matched as words."""
    words = tokenise(normalise(args.query))
    if not words:
        print("nothing to search for", file=sys.stderr)
        return EXIT_ERROR
    # Quoting each word keeps FTS5 from reading punctuation as query syntax.
    match = " ".join('"' + word.replace('"', "") + '"' for word in words)
    conn = connect(settings.paths.db_file, read_only=True)
    rows = conn.execute(
        "SELECT d.id, s.name, d.label, d.doc_date, d.title, "
        "snippet(documents_fts, 1, '[', ']', ' … ', 16) AS snippet "
        "FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid "
        "JOIN sources s ON s.id = d.source_id "
        "WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?",
        (match, args.limit),
    ).fetchall()
    for row in rows:
        print(f"#{row['id']} {row['name']} [{row['label']}] {row['doc_date']}  {row['title']}")
        print(f"    {' '.join(row['snippet'].split())}")
    print(f"{len(rows)} result(s)")
    return 0


def cmd_schema(args: argparse.Namespace, settings: Settings) -> int:
    """Print the schema that the migrations produce (used to refresh db/schema.sql)."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    migrate(conn)
    header = (
        "-- Generated by `shuru schema` from db/migrations/. Do not edit by hand.\n"
        "-- A read-only overview of the whole database; the migrations are the source.\n\n"
    )
    print(header + dump_schema(conn), end="")
    return 0


# --- jobs ---------------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    sources = load_sources(settings.config_dir)
    with _job(settings, "ingest", args) as (conn, current, notifier):
        reports = run_ingest(
            settings,
            sources,
            conn,
            current,
            notifier=notifier,
            only=set(args.source) if args.source else None,
            max_pages=args.max_pages,
        )
        stored = current.metrics.get("stored", 0)
    print(f"ingest finished: {len(reports)} source(s), {stored} new document(s)")
    return 0


def cmd_stats(args: argparse.Namespace, settings: Settings) -> int:
    sources = load_sources(settings.config_dir)
    with _job(settings, "stats", args) as (conn, current, _notifier):
        stats = corpus_stats(conn, settings, sources)
        path = write_report(stats, settings.paths.dir("reports"))
        current.set("documents", stats["documents"])
        current.set("chars", stats["chars"])
    print(json.dumps(stats, ensure_ascii=False, indent=2) if args.json else format_stats(stats))
    print(f"\nsaved to {path}", file=sys.stderr)
    return 0


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    total_ram = read_meminfo().get("MemTotal")
    if total_ram is not None and total_ram < MIN_TRAIN_RAM_MB and not args.force:
        print(
            f"This host has {total_ram} MB of RAM. Training is meant to run on a PC: copy a\n"
            "database snapshot there (see `shuru backup`) and train on it. Use --force to\n"
            "train here anyway.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    with _job(settings, "train", args) as (conn, current, _notifier):
        model_id, model = train_and_register(
            settings,
            conn,
            current,
            use_human_labels=args.human_labels,
            activate=not args.no_activate,
            note=args.note,
        )
    print(f"model {model_id} trained" + ("" if args.no_activate else " and activated"))
    print(format_report(model))
    return 0


def cmd_score(args: argparse.Namespace, settings: Settings) -> int:
    with _job(settings, "score", args) as (conn, current, _notifier):
        active = registry.load_active(conn, settings.paths.dir("models"))
        if active is None:
            raise ModelError("no active model: run `shuru train` or `shuru model install`")
        model_id, model = active
        scored = score_documents(conn, model_id, model, settings.model, current)
    print(f"model {model_id} scored {scored} document(s)")
    return 0


def cmd_news(args: argparse.Namespace, settings: Settings) -> int:
    feeds = load_feeds(settings.config_dir)
    if not feeds:
        print("No feeds configured. Add [[feed]] tables to config/news_feeds.local.toml.")
        return 0
    for name in overlapping_feeds(feeds, load_sources(settings.config_dir)):
        log.warning("feed %s shares a site with a training source, so it is not unseen data", name)
    with _job(settings, "news", args) as (conn, current, notifier):
        fetcher = Fetcher(settings.http)
        try:
            new_items = poll_feeds(settings, feeds, conn, current, fetcher)
        finally:
            fetcher.close()
        active = registry.load_active(conn, settings.paths.dir("models"))
        if active is None:
            print(f"{new_items} new headline(s) stored. No active model yet, so nothing scored.")
            return 0
        model_id, model = active
        scored = score_headlines(conn, model_id, model, current)
        # Recent days are recomputed; older daily rows are already final.
        update_daily(conn, model_id, days_ago(2))
        report = drift_report(conn, model_id, settings.drift)
        queued = refresh_queue(conn, model_id, settings.labelling)
        current.set("queued", queued)
        if report.psi is not None:
            current.set("psi", report.psi)
        if report.alert:
            current.set("drift_alert", 1)
            already = conn.execute(
                "SELECT 1 FROM runs WHERE job = 'news' AND id != ? AND started_at >= ? "
                "AND json_extract(metrics, '$.drift_alert') = 1",
                (current.id, utc_now()[:10]),
            ).fetchone()
            if not already:  # one alert per day is enough
                notifier.send("shuru: classifier drift", format_drift(report), priority=3)
    print(f"{new_items} new headline(s), {scored} scored, {queued} added to the labelling queue")
    print(format_drift(report))
    return 0


def cmd_drift(args: argparse.Namespace, settings: Settings) -> int:
    conn = connect(settings.paths.db_file, read_only=True)
    model_id = registry.active_model_id(conn)
    if model_id is None:
        print("no active model", file=sys.stderr)
        return EXIT_ERROR
    print(format_drift(drift_report(conn, model_id, settings.drift)))
    print(f"\n{'day':<12}{'feed':<24}{'n':>6}{'mean conf':>11}{'agree':>8}  labels")
    for row in conn.execute(
        "SELECT d.day, f.name, d.n, d.conf_sum, d.n_expected, d.n_agree, d.label_counts "
        "FROM drift_daily d JOIN news_feeds f ON f.id = d.feed_id "
        "WHERE d.model_id = ? ORDER BY d.day DESC, f.name LIMIT ?",
        (model_id, args.limit),
    ):
        agree = f"{row['n_agree'] / row['n_expected']:.2f}" if row["n_expected"] else "-"
        print(
            f"{row['day']:<12}{row['name']:<24}{row['n']:>6}{row['conf_sum'] / row['n']:>11.3f}"
            f"{agree:>8}  {row['label_counts']}"
        )
    return 0


def cmd_health(args: argparse.Namespace, settings: Settings) -> int:
    # Not exclusive: a health check must be able to run while a long job holds the lock.
    with _job(settings, "health", args, exclusive=False) as (conn, current, notifier):
        sample, problems = run_health(settings, conn, current, notifier)
    print(
        f"memory available: {sample.mem_available_mb} MB of {sample.mem_total_mb} MB   "
        f"swap used: {sample.swap_used_mb} MB   load: {sample.load1} {sample.load5} "
        f"{sample.load15} ({sample.cpus} CPUs)"
    )
    print(
        f"disk free: {sample.disk_free_mb} MB of {sample.disk_total_mb} MB   "
        f"database: {sample.db_mb} MB   WAL: {sample.wal_mb} MB   "
        f"CPU temperature: {sample.cpu_temp_c if sample.cpu_temp_c is not None else 'n/a'}"
    )
    for message in problems.values():
        print(f"PROBLEM: {message}")
    if not problems:
        print("all checks pass")
    return EXIT_ERROR if problems else 0


def cmd_backup(args: argparse.Namespace, settings: Settings) -> int:
    with _job(settings, "backup", args) as (conn, current, _notifier):
        path = backup_database(conn, settings.paths.dir("backups"), settings.backup.keep)
        current.set("snapshot_mb", file_size_mb(path))
    print(f"snapshot written: {path}")
    return 0


def cmd_reindex(args: argparse.Namespace, settings: Settings) -> int:
    with _job(settings, "reindex", args) as (conn, current, _notifier):
        done = store.reindex(conn, make_hasher(settings))
        current.set("documents", done)
    print(f"recomputed signatures for {done} document(s)")
    return 0


def cmd_export(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    model_id = registry.active_model_id(conn)
    if (args.model_label or args.min_confidence > 0) and model_id is None:
        print("filtering by model needs an active model", file=sys.stderr)
        return EXIT_ERROR
    if args.model_label or args.min_confidence > 0:
        unscored = conn.execute(
            "SELECT COUNT(*) FROM documents d WHERE NOT EXISTS ("
            "SELECT 1 FROM doc_scores s WHERE s.document_id = d.id AND s.model_id = ?)",
            (model_id,),
        ).fetchone()[0]
        if unscored:
            print(
                f"note: {unscored} document(s) have no score from model {model_id} and are "
                "left out; run `shuru score` first to include them",
                file=sys.stderr,
            )
    result = export_corpus(
        conn,
        settings.paths.dir("exports"),
        label=args.label,
        model_id=model_id,
        model_label=args.model_label,
        min_confidence=args.min_confidence,
    )
    print(f"{result.documents} document(s), {result.chars:,} characters")
    print(f"data:     {result.path}")
    print(f"manifest: {result.manifest_path}")
    return 0


# --- model registry -----------------------------------------------------------------------


def cmd_model(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    models_dir = settings.paths.dir("models")
    if args.action == "list":
        header = f"{'id':>4} {'active':<7}{'created (UTC)':<22}{'labels':<12}"
        print(f"{header}{'date':>7}{'source':>8}  file")
        for row in registry.list_models(conn):
            splits = json.loads(row["metrics"]).get("splits", {})
            by_date = splits.get("test_date", {}).get("accuracy")
            by_source = splits.get("test_source", {}).get("accuracy")
            print(
                f"{row['id']:>4} {'yes' if row['is_active'] else '':<7}{row['created_at']:<22}"
                f"{','.join(json.loads(row['labels'])):<12}"
                f"{'-' if by_date is None else format(by_date, '.3f'):>7}"
                f"{'-' if by_source is None else format(by_source, '.3f'):>8}  {row['file_name']}"
            )
        print("\n'date' and 'source' are accuracy on the two test sets (see `shuru model show`).")
        return 0
    if args.action == "install":
        model_id = registry.install(conn, Path(args.target), models_dir)
        registry.activate(conn, model_id)
        print(f"model {model_id} installed and activated")
        return 0
    model_id = int(args.target) if args.target else (registry.active_model_id(conn) or 0)
    if args.action == "activate":
        registry.activate(conn, model_id)
        print(f"model {model_id} is now active")
        return 0
    model = registry.load_model(conn, model_id, models_dir)  # action == "show"
    print(
        json.dumps(model.metrics, ensure_ascii=False, indent=2)
        if args.json
        else format_report(model)
    )
    return 0


# --- hand labelling -----------------------------------------------------------------------


def cmd_label(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    if args.action == "export":
        out = (
            Path(args.target)
            if args.target
            else (settings.paths.dir("labelling") / f"to-label-{utc_now()[:10]}.tsv")
        )
        count = export_sheet(conn, out, args.limit)
        print(f"{count} headline(s) written to {out}")
        print(f"Fill in the 'label' column ({', '.join(sorted(known_labels(conn)))} or skip),")
        print(f"then run: shuru label apply {out}")
        return 0
    if args.action == "apply":
        if not args.target:
            print("usage: shuru label apply FILE", file=sys.stderr)
            return EXIT_ERROR
        result = apply_sheet(conn, Path(args.target), known_labels(conn), args.annotator)
        print(
            f"{result.applied} label(s) stored, {result.skipped} skipped, {result.blank} left blank"
        )
        for error in result.errors:
            print(f"  not applied: {error}", file=sys.stderr)
        return EXIT_ERROR if result.errors else 0
    report = label_report(conn, registry.active_model_id(conn))  # action == "report"
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    print(f"hand-labelled headlines: {report['n_labelled']} {report['by_label']}")
    print(f"waiting in the queue:    {report['pending']}")
    for key, title in (
        ("distant_vs_human", "feed's expected label vs hand label (how noisy are distant labels?)"),
        ("model_vs_human", "model vs hand label (measured on the model's least certain headlines)"),
    ):
        scores = report.get(key, {})
        if scores.get("n"):
            print(
                f"{title}:\n  n={scores['n']} accuracy={scores['accuracy']:.3f} "
                f"macro-F1={scores['macro_f1']:.3f}"
            )
    return 0


def cmd_alert(args: argparse.Namespace, settings: Settings) -> int:
    """Send one alert. Used by systemd when a unit fails outside Python (for example OOM)."""
    sent = Notifier.from_config(settings.notify).send(args.title, args.message, priority=4)
    return 0 if sent or not settings.notify.topic else EXIT_ERROR


# --- argument parsing ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shuru", description="Collect, clean and label text in low-resource languages."
    )
    parser.add_argument("--version", action="version", version=f"shuru {__version__}")
    parser.add_argument("--settings", type=Path, help="settings file (default: LAB_SETTINGS)")
    parser.add_argument(
        "--lock-wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="how long to wait if another job is running (default: do not wait)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(
        name: str, handler: Callable[[argparse.Namespace, Settings], int], help_text: str
    ) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    add("init", cmd_init, "create the data folders and the database")
    add("sources", cmd_sources, "list the configured sources")

    sub = add("ingest", cmd_ingest, "fetch new articles from the sources and store them")
    sub.add_argument("--source", action="append", metavar="NAME", help="only this source")
    sub.add_argument("--max-pages", type=int, metavar="N", help="pages per source this run")

    sub = add("stats", cmd_stats, "corpus statistics")
    sub.add_argument("--json", action="store_true")

    sub = add("search", cmd_search, "full-text search in the corpus")
    sub.add_argument("query")
    sub.add_argument("--limit", type=int, default=10)

    sub = add("train", cmd_train, "train a classifier on the corpus and register it")
    sub.add_argument("--human-labels", action="store_true", help="also train on hand labels")
    sub.add_argument("--no-activate", action="store_true", help="register without activating")
    sub.add_argument("--note", default="", help="free text stored with the model")
    sub.add_argument("--force", action="store_true", help="train even on a low-memory host")

    sub = add("model", cmd_model, "list, show, activate or install models")
    sub.add_argument("action", choices=("list", "show", "activate", "install"))
    sub.add_argument("target", nargs="?", help="model id, or a model file for install")
    sub.add_argument("--json", action="store_true")

    add("score", cmd_score, "label every document with the active model")
    add("news", cmd_news, "poll news feeds, score headlines, update drift and the queue")

    sub = add("drift", cmd_drift, "show the drift report for the active model")
    sub.add_argument("--limit", type=int, default=30, help="daily rows to show")

    sub = add("label", cmd_label, "hand labelling: export a sheet, apply it, report")
    sub.add_argument("action", choices=("export", "apply", "report"))
    sub.add_argument("target", nargs="?", help="sheet file")
    sub.add_argument("--limit", type=int, default=50, help="headlines per sheet")
    sub.add_argument("--annotator", default="", help="who labelled the sheet")
    sub.add_argument("--json", action="store_true")

    sub = add("export", cmd_export, "write a corpus snapshot (JSON Lines, gzip)")
    sub.add_argument("--label", help="keep only this distant label")
    sub.add_argument("--model-label", help="keep only documents the model gave this label")
    sub.add_argument("--min-confidence", type=float, default=0.0, metavar="P")

    add("health", cmd_health, "check memory, load, disk and database size")
    add("backup", cmd_backup, "write a compressed snapshot of the database")
    add("reindex", cmd_reindex, "rebuild duplicate-detection data after changing [dedup]")

    sub = add("runs", cmd_runs, "show recent job runs, or one run in detail")
    sub.add_argument("--limit", type=int, default=20)
    sub.add_argument("--id", type=int, help="show metrics and step timings of one run")

    add("schema", cmd_schema, "print the database schema built by the migrations")

    sub = add("alert", cmd_alert, "send an alert (used by systemd on unit failure)")
    sub.add_argument("title")
    sub.add_argument("message")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(".env"))
    try:
        settings = load_settings(args.settings)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    logging_config = settings.logging
    configure_logging(
        args.command, logging_config.level, logging_config.max_field_chars, logging_config.format
    )
    try:
        return int(args.handler(args, settings))
    except JobBusyError as exc:
        print(f"busy: {exc}", file=sys.stderr)
        return EXIT_BUSY
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except (DatabaseError, ModelError, TrainError, BackupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:  # output piped into `head`
        return 0


if __name__ == "__main__":
    sys.exit(main())
