"""Benchmark the CPU-bound steps of the pipeline, to decide where (if anywhere) to optimise.

    uv run python bench/bench_text.py                 # default: 300 documents
    uv run python bench/bench_text.py --docs 1000 --json results.json

What is measured, per document unless noted:

    extract     HTML page -> article text (selectolax, compiled C)
    normalise   storage normalisation (str.translate + regex, C)
    fold        comparison folding (C)
    tokenise    regex tokeniser (C)
    minhash     64-permutation MinHash signature + LSH bands (pure Python)
    features    hashed character 1-4-grams (pure Python, CRC-32 in C)
    predict     model score for one document: features + weighted sum + softmax
    predict_headline   the same for a 12-word headline (what the news job scores)
    train_100   one training epoch, per 100 sentences

And for context, the cost the pipeline cannot avoid:

    polite_delay   the minimum wait between two requests to one site (settings.toml)

The text is synthetic (see src/lab/synth.py) with realistic length and character mix.
Numbers are the median of several repeats, measured with time.process_time so other
programs on the machine disturb them less. Peak memory comes from the OS (ru_maxrss).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import random
import resource
import statistics
import sys
import time
from array import array
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lab import synth
from lab.config import load_settings
from lab.ingest.extract import extract
from lab.model.features import hashed_features
from lab.model.predict import Model
from lab.model.train import fit
from lab.text.dedup import MinHasher
from lab.text.normalise import fold, normalise
from lab.text.tokenise import tokenise

REPEATS = 5


def median_seconds(work: Callable[[], object], repeats: int = REPEATS) -> float:
    """Median CPU seconds for one call of work, over several repeats."""
    times = []
    for _ in range(repeats):
        started = time.process_time()
        work()
        times.append(time.process_time() - started)
    return statistics.median(times)


def per_item_ms(work: Callable[[], object], n: int) -> float:
    return round(median_seconds(work) / n * 1000, 4)


def machine() -> dict[str, Any]:
    model = ""
    with contextlib.suppress(OSError):  # Raspberry Pi and some ARM boards name themselves here
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n")
    if not model:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            model = platform.processor()
    return {
        "machine": model or platform.machine(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
    }


def run(n_docs: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    styles = [synth.make_style(rng, synth.VARIETY_A), synth.make_style(rng, synth.VARIETY_B)]
    documents = [synth.make_document(rng, styles[i % 2]) for i in range(n_docs)]
    pages = [
        synth.article_html(synth.make_title(rng, styles[0]), doc.split("\n"), None)
        for doc in documents
    ]
    headlines = [synth.make_title(rng, styles[i % 2]) for i in range(n_docs)]
    normalised = [normalise(doc) for doc in documents]
    folded = [fold(doc) for doc in normalised]
    tokens = [tokenise(doc) for doc in folded]

    settings = load_settings(env={"LAB_ROOT": "/tmp/shuru-bench-unused"})  # noqa: S108
    dedup, model_config = settings.dedup, settings.model
    hasher = MinHasher(dedup.num_perm, dedup.bands, dedup.shingle_size)
    bits, n_min, n_max = model_config.hash_bits, model_config.ngram_min, model_config.ngram_max
    rows = [array("f", [rng.uniform(-1, 1) for _ in range(1 << bits)]) for _ in range(2)]
    model = Model(("pes", "prs"), bits, n_min, n_max, rows, [0.0, 0.0])
    sentences = [s for doc in normalised[:60] for s in doc.split(".") if len(s) > 30][:400]
    encoded = [(hashed_features(s, n_min, n_max, bits), i % 2) for i, s in enumerate(sentences)]

    def each(fn: Callable[[Any], object], items: list[Any]) -> Callable[[], None]:
        """A callable that applies fn to every item: one timed repeat."""

        def work() -> None:
            for item in items:
                fn(item)

        return work

    steps = {
        "extract": per_item_ms(each(lambda page: extract(page, "div.body"), pages), n_docs),
        "normalise": per_item_ms(each(normalise, documents), n_docs),
        "fold": per_item_ms(each(fold, normalised), n_docs),
        "tokenise": per_item_ms(each(tokenise, folded), n_docs),
        "minhash": per_item_ms(
            each(lambda t: hasher.band_hashes(hasher.signature(t)), tokens), n_docs
        ),
        "features": per_item_ms(
            each(lambda d: hashed_features(d, n_min, n_max, bits, 2000), normalised), n_docs
        ),
        "predict": per_item_ms(each(lambda d: model.predict(d, 2000), normalised), n_docs),
        "predict_headline": per_item_ms(each(model.predict, headlines), n_docs),
        "train_100": per_item_ms(
            lambda: fit(encoded, 2, bits, epochs=1, learning_rate=0.5, l2=1e-6, seed=1),
            len(encoded),
        )
        * 100,
    }
    per_doc_ingest = sum(steps[k] for k in ("extract", "normalise", "fold", "tokenise", "minhash"))
    delay_ms = settings.http.default_delay_s * 1000
    return {
        **machine(),
        "documents": n_docs,
        "mean_chars": round(sum(map(len, normalised)) / n_docs),
        "mean_tokens": round(sum(map(len, tokens)) / n_docs),
        "ms": {k: round(v, 3) for k, v in steps.items()},
        "ingest_cpu_ms_per_doc": round(per_doc_ingest, 3),
        "polite_delay_ms": delay_ms,
        "cpu_share_of_one_site": round(per_doc_ingest / (per_doc_ingest + delay_ms), 4),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def report(result: dict[str, Any]) -> str:
    ms = result["ms"]
    lines = [
        f"{result['machine']} ({result['arch']}), Python {result['python']}",
        f"{result['documents']} documents, mean {result['mean_chars']} chars / "
        f"{result['mean_tokens']} tokens",
        "",
        f"{'step':<20}{'ms per item':>12}",
    ]
    lines += [f"{name:<20}{value:>12.3f}" for name, value in ms.items()]
    cpu = result["ingest_cpu_ms_per_doc"]
    lines += [
        "",
        f"ingest CPU per document (extract..minhash): {cpu:.2f} ms",
        f"polite delay per request to one site:        {result['polite_delay_ms']:.0f} ms",
        f"CPU is {result['cpu_share_of_one_site']:.2%} of the time one site's crawl takes",
        f"one core keeps up with about {result['polite_delay_ms'] / cpu:.0f} sites crawled at once",
        f"peak memory: {result['peak_rss_mb']} MB",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the text pipeline.")
    parser.add_argument("--docs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=Path, help="also write the raw numbers here")
    args = parser.parse_args()
    result = run(args.docs, args.seed)
    print(report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
