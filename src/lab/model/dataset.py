"""Build training and test examples from the corpus.

**Labels are distant.** Nobody labelled these sentences. Each one inherits the
label of the source it came from ("published by an Afghan outlet, so Dari"). That
is cheap and it is noisy: outlets quote each other, run wire copy and publish in
more than one variety.

**Splits are never random.** A random split would put sentences from the same
article, the same day and the same outlet on both sides, and the test score would
mostly measure memory. Instead there are two test sets, each asking a harder and
more honest question:

``test_date``
    Documents from training sources dated on or after a cutoff day. Question:
    does the model still work on *later* text from outlets it knows?

``test_source``
    Every document from sources marked ``holdout = true``. Question: does it work
    on an outlet it has *never seen*? If this score is far below ``test_date``,
    the model has learned outlets (house style, boilerplate) rather than language.

Further guards against learning the outlet instead of the language: a cap on
examples per source, equal numbers of training examples per label, and repeated
sentences (bylines, "follow us on ...") are used only once.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

from lab.config import ModelConfig
from lab.corpus.store import iter_documents
from lab.logs import get_logger
from lab.text.normalise import fold
from lab.text.tokenise import split_sentences

log = get_logger(__name__)

TRAIN, TEST_DATE, TEST_SOURCE, HUMAN = "train", "test_date", "test_source", "human"
# Stop remembering sentence hashes beyond this many (bounds memory on huge corpora).
MAX_REMEMBERED_SENTENCES = 5_000_000


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    label: str
    source: str
    split: str


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """How documents are divided. ``cutoff`` is None when no date split is possible."""

    cutoff: str | None
    first_day: str | None
    last_day: str | None
    reason: str = ""


def plan_split(conn: sqlite3.Connection, config: ModelConfig) -> SplitPlan:
    """Choose the cutoff day: documents dated on or after it are test, never train.

    With ``test_from`` set, that day is the cutoff. Otherwise the cutoff is the day
    that leaves about ``test_date_fraction`` of the training sources' documents
    after it. Holdout sources are not counted: they are test material regardless.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n, MIN(d.doc_date) AS first, MAX(d.doc_date) AS last "
        "FROM documents d JOIN sources s ON s.id = d.source_id WHERE s.holdout = 0"
    ).fetchone()
    count, first_day, last_day = row["n"], row["first"], row["last"]
    if not count:
        return SplitPlan(None, None, None, "no documents from training sources")
    if config.test_from:
        cutoff = config.test_from
    else:
        offset = min(int(count * (1 - config.test_date_fraction)), count - 1)
        cutoff = conn.execute(
            "SELECT d.doc_date FROM documents d JOIN sources s ON s.id = d.source_id "
            "WHERE s.holdout = 0 ORDER BY d.doc_date LIMIT 1 OFFSET ?",
            (offset,),
        ).fetchone()["doc_date"]
    if cutoff <= first_day:
        return SplitPlan(
            None,
            first_day,
            last_day,
            "all documents fall on or after the cutoff day, so nothing would be left to "
            "train on; collect over more days to get a date-based test set",
        )
    return SplitPlan(cutoff, first_day, last_day)


def assign_split(holdout: bool, doc_date: str, cutoff: str | None) -> str:
    """Which side a document is on. Depends only on its source and its date."""
    if holdout:
        return TEST_SOURCE
    if cutoff is not None and doc_date >= cutoff:
        return TEST_DATE
    return TRAIN


def sentences(text: str, config: ModelConfig) -> Iterator[str]:
    """Sentences of a usable length; over-long ones are cut."""
    for sentence in split_sentences(text):
        if len(sentence) >= config.min_sentence_chars:
            yield sentence[: config.max_sentence_chars]


def iter_examples(
    conn: sqlite3.Connection, config: ModelConfig, plan: SplitPlan
) -> Iterator[Example]:
    """Every sentence in the corpus with its label and split, streamed document by document.

    A sentence already seen anywhere (after folding) is skipped.
    """
    seen: set[int] = set()
    for row in iter_documents(conn):
        split = assign_split(bool(row["holdout"]), row["doc_date"], plan.cutoff)
        for sentence in sentences(row["text"], config):
            key = hash(fold(sentence))
            if key in seen:
                continue
            if len(seen) < MAX_REMEMBERED_SENTENCES:
                seen.add(key)
            yield Example(sentence, row["label"], row["source"], split)


def iter_human_examples(conn: sqlite3.Connection, *, for_training: bool) -> Iterator[Example]:
    """Hand-labelled headlines.

    They are divided by item id: even ids may be added to training, odd ids are
    only ever used for evaluation. A model trained with hand labels and one trained
    without can then be compared on the same untouched half.
    """
    parity = 0 if for_training else 1
    rows = conn.execute(
        "SELECT i.title, h.label FROM human_labels h JOIN news_items i ON i.id = h.item_id "
        "WHERE h.item_id % 2 = ? ORDER BY h.item_id",
        (parity,),
    )
    for row in rows:
        yield Example(row["title"], row["label"], "human", TRAIN if for_training else HUMAN)


def sample_examples(examples: Iterator[Example], config: ModelConfig) -> dict[str, list[Example]]:
    """Keep at most ``max_examples_per_source`` examples per source and split.

    Reservoir sampling: every example of a source has the same chance of being
    kept, in one pass and without knowing the total in advance.
    """
    rng = random.Random(config.seed)
    cap = config.max_examples_per_source
    reservoirs: dict[tuple[str, str], list[Example]] = {}
    seen_counts: dict[tuple[str, str], int] = {}
    for example in examples:
        key = (example.split, example.source)
        seen = seen_counts.get(key, 0) + 1
        seen_counts[key] = seen
        reservoir = reservoirs.setdefault(key, [])
        if len(reservoir) < cap:
            reservoir.append(example)
        else:
            slot = rng.randrange(seen)
            if slot < cap:
                reservoir[slot] = example
    by_split: dict[str, list[Example]] = {}
    for (split, _source), reservoir in sorted(reservoirs.items()):
        by_split.setdefault(split, []).extend(reservoir)
    return by_split


def balance_labels(examples: list[Example], max_total: int, seed: int) -> list[Example]:
    """Equal numbers of examples per label (the size of the smallest), shuffled.

    Without this the model would lean towards whichever label has more text, which
    reflects how much was scraped and not how likely each variety is.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)
    if not by_label:
        return []
    per_label = min(min(len(group) for group in by_label.values()), max_total // len(by_label))
    balanced: list[Example] = []
    for _label, group in sorted(by_label.items()):
        rng.shuffle(group)
        balanced.extend(group[:per_label])
    rng.shuffle(balanced)
    return balanced
