"""Run the whole pipeline end to end on synthetic data. Nothing leaves this machine.

    uv run python scripts/demo.py            # temporary folder, removed afterwards
    uv run python scripts/demo.py --keep DIR # keep config, database and outputs in DIR

What it does:

1. Builds eight fake news sites (random Persian-script text, see src/lab/synth.py)
   and serves them from localhost, each on its own port.
2. Writes a config folder that points at them: four training sources, two holdout
   sources and two news feeds.
3. Runs the real commands, in the order you would run them: ingest (twice), stats,
   train, score, news, label export / apply / report, export, backup, health, runs.

The text is synthetic, so the classifier scores printed here say nothing about real
Dari or Iranian Persian. The point is to watch every part of the system work, and
to have a database to poke at afterwards (use --keep).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

from lab import synth
from lab.cli import main as shuru
from lab.config import REPO_ROOT

ARTICLES_PER_SITE = 60
HEADLINES_PER_FEED = 120

SETTINGS = """\
# Demo settings: localhost sites, so the politeness delay can be tiny.
[http]
default_delay_s = 0.01
[ingest]
max_pages_per_source = 500
[labelling]
queue_size = 40
max_per_feed = 20
[health]
min_disk_free_mb = 100
"""


def build_sites(seed: int) -> tuple[list[synth.Site], dict[str, str], dict[str, str]]:
    """Returns (sites, label of each training source, label of each feed)."""
    rng = random.Random(seed)
    today = synth.fixed_today()
    # Wire copy: the same three stories carried by several outlets.
    wire_style = synth.Style(synth.VARIETY_A, marker_rate=0.0, style_rate=0.0)
    wire = [
        (synth.make_title(rng, wire_style), synth.make_paragraphs(rng, wire_style, 6))
        for _ in range(3)
    ]
    # The same stories with a sentence added by the local desk: near-duplicates.
    edited = [
        (title, [*paragraphs, synth.make_sentence(rng, wire_style)]) for title, paragraphs in wire
    ]
    plan = [
        ("af-one", synth.VARIETY_A, wire),
        ("af-two", synth.VARIETY_A, edited),
        ("af-held-out", synth.VARIETY_A, None),
        ("ir-one", synth.VARIETY_B, wire),
        ("ir-two", synth.VARIETY_B, None),
        ("ir-held-out", synth.VARIETY_B, None),
    ]
    sites: list[synth.Site] = []
    source_labels: dict[str, str] = {}
    for name, variety, shared in plan:
        style = synth.make_style(rng, variety)
        sites.append(
            synth.build_news_site(rng, name, style, ARTICLES_PER_SITE, today=today, shared=shared)
        )
        source_labels[name] = variety
    feed_labels = {"af-headlines": synth.VARIETY_A, "ir-headlines": synth.VARIETY_B}
    for name, variety in feed_labels.items():
        style = synth.make_style(rng, variety)
        sites.append(synth.build_headline_feed(rng, name, style, HEADLINES_PER_FEED, today=today))
    return sites, source_labels, feed_labels


def write_config(
    config_dir: Path, urls: dict[str, str], sources: dict[str, str], feeds: dict[str, str]
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.toml").write_text(SETTINGS, encoding="utf-8")
    shutil.copyfile(REPO_ROOT / "config" / "markers.toml", config_dir / "markers.toml")
    source_tables = [
        f'[[source]]\nname = "{name}"\nkind = "rss"\nlabel = "{label}"\n'
        f'urls = ["{urls[name]}/feed.xml"]\ncontent_selector = "div.body"\n'
        f'licence = "synthetic demo text"\nholdout = {str("held-out" in name).lower()}\n'
        for name, label in sources.items()
    ]
    (config_dir / "sources.toml").write_text("\n".join(source_tables), encoding="utf-8")
    feed_tables = [
        f'[[feed]]\nname = "{name}"\nurl = "{urls[name]}/feed.xml"\nlabel = "{label}"\n'
        for name, label in feeds.items()
    ]
    (config_dir / "news_feeds.toml").write_text("\n".join(feed_tables), encoding="utf-8")


def fill_in_sheet(sheet: Path, feeds: dict[str, str], rng: random.Random) -> None:
    """Stand in for the person: label most rows with the feed's variety, skip a few."""
    with sheet.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["label"] = "skip" if rng.random() < 0.1 else feeds[row["feed"]]
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def step(title: str, *command: str, settings: Path) -> None:
    print(f"\n{'=' * 78}\n$ shuru {' '.join(command)}\n# {title}\n{'=' * 78}", flush=True)
    code = shuru(["--settings", str(settings), *command])
    if code != 0:
        raise SystemExit(f"`shuru {' '.join(command)}` exited with {code}")


def run_demo(root: Path, seed: int) -> None:
    config_dir, data_root = root / "config", root / "data"
    os.environ["LAB_ROOT"] = str(data_root)
    os.environ.pop("NTFY_TOPIC", None)  # the demo never sends real alerts
    settings = config_dir / "settings.toml"
    sites, source_labels, feed_labels = build_sites(seed)

    with synth.serve(sites) as urls:
        write_config(config_dir, urls, source_labels, feed_labels)
        step("create the data folders and the database", "init", settings=settings)
        step("the configured sources", "sources", settings=settings)
        step("first crawl: everything is new", "ingest", settings=settings)
        step("second crawl: feeds answer 304, nothing is fetched", "ingest", settings=settings)
        step("what was collected, and what was thrown away", "stats", settings=settings)
        step("full-text search", "search", "گزارش اقتصادی", "--limit", "3", settings=settings)
        step("train, evaluate on both test sets, register", "train", settings=settings)
        step("label every document with the model", "score", settings=settings)
        step("poll headline feeds, score them, fill the queue", "news", settings=settings)

        sheet = data_root / "labelling" / "demo-sheet.tsv"
        step("write the least certain headlines to a sheet", "label", "export", str(sheet),
             settings=settings)  # fmt: skip
        fill_in_sheet(sheet, feed_labels, random.Random(seed))
        print("\n(the demo filled in the label column, standing in for a person)")
        step("store the hand labels", "label", "apply", str(sheet), "--annotator", "demo",
             settings=settings)  # fmt: skip
        step("what the hand labels show", "label", "report", settings=settings)
        step("train again, now also on hand labels", "train", "--human-labels",
             "--note", "with hand labels", settings=settings)  # fmt: skip
        step("the registry: two models to compare", "model", "list", settings=settings)
        step("score the documents with the new active model", "score", settings=settings)
        step("snapshot of what the model is 90% sure is variety A", "export",
             "--model-label", synth.VARIETY_A, "--min-confidence", "0.9",
             settings=settings)  # fmt: skip
        step("database snapshot", "backup", settings=settings)
        step("host health", "health", settings=settings)
        step("every job that ran", "runs", settings=settings)
        step("where the first crawl spent its time", "runs", "--id", "1", settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--keep", type=Path, metavar="DIR", help="keep everything in DIR")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.keep is not None:
        if args.keep.exists() and any(args.keep.iterdir()):
            print(f"{args.keep} is not empty; choose a new folder", file=sys.stderr)
            return 1
        run_demo(args.keep.resolve(), args.seed)
        print(f"\nKept in {args.keep}. To explore it:")
        print(f"  export LAB_ROOT={args.keep.resolve() / 'data'}")
        print(f"  export LAB_SETTINGS={args.keep.resolve() / 'config' / 'settings.toml'}")
        print("  uv run shuru stats      (the fake sites are gone, so ingest will find nothing)")
        return 0
    with tempfile.TemporaryDirectory(prefix="shuru-demo-") as temporary:
        run_demo(Path(temporary), args.seed)
    print("\nDemo finished. Nothing was kept; use --keep DIR to explore the result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
