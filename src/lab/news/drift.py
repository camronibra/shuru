"""Drift: is the model seeing the same kind of text, and behaving the same, as before?

A model is trained once and the world moves on: outlets change their style, new
topics bring new vocabulary. Nothing fails loudly when that happens; the scores
just get quietly worse. Drift tracking makes it visible.

**What is tracked**, per day and feed (table ``drift_daily``):

- how many headlines were scored and their mean confidence,
- a histogram of confidence (10 bins),
- how many headlines got each label,
- how often the prediction matched the feed's expected label, where one is set.

**How drift is measured.** Two windows are compared: the *reference* window (the
first ``window_days`` days this model scored headlines) and the *current* window
(the latest ``window_days`` days). The Population Stability Index compares their
confidence histograms::

    PSI = sum over bins of (current% - reference%) * ln(current% / reference%)

0 means identical. By the usual rule of thumb, below 0.1 is stable, 0.1 to 0.25 is
worth a look, and above 0.25 is a real shift. PSI says *that* the distribution of
confidence moved, not *why*: the labelling queue is how you find out.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from lab.config import DriftConfig
from lab.db import transaction
from lab.model.evaluate import HISTOGRAM_BINS


@dataclass(slots=True)
class WindowStats:
    """Aggregate of drift_daily rows over a range of days."""

    first_day: str
    last_day: str
    n: int = 0
    conf_sum: float = 0.0
    hist: list[int] = field(default_factory=lambda: [0] * HISTOGRAM_BINS)
    label_counts: dict[str, int] = field(default_factory=dict)
    n_expected: int = 0
    n_agree: int = 0

    @property
    def mean_confidence(self) -> float | None:
        return self.conf_sum / self.n if self.n else None

    @property
    def agreement(self) -> float | None:
        """Share of predictions matching the feed's expected label (None if no feed has one)."""
        return self.n_agree / self.n_expected if self.n_expected else None

    def label_share(self) -> dict[str, float]:
        return {
            label: round(count / self.n, 4) for label, count in sorted(self.label_counts.items())
        }


@dataclass(frozen=True, slots=True)
class DriftReport:
    model_id: int
    reference: WindowStats | None
    current: WindowStats | None
    psi: float | None
    alert: bool
    reason: str = ""


def psi(reference: Sequence[int], current: Sequence[int]) -> float:
    """Population Stability Index between two histograms with the same bins.

    Half a count is added to every bin first, so an empty bin does not send the
    logarithm to infinity.
    """
    if len(reference) != len(current):
        raise ValueError("histograms must have the same number of bins")
    ref_total = sum(reference) + 0.5 * len(reference)
    cur_total = sum(current) + 0.5 * len(current)
    total = 0.0
    for ref_count, cur_count in zip(reference, current, strict=True):
        ref_share = (ref_count + 0.5) / ref_total
        cur_share = (cur_count + 0.5) / cur_total
        total += (cur_share - ref_share) * math.log(cur_share / ref_share)
    return total


def update_daily(conn: sqlite3.Connection, model_id: int, since_day: str) -> int:
    """Recompute the daily aggregates for days from since_day on. Returns rows written."""
    groups: dict[tuple[str, int], WindowStats] = {}
    rows = conn.execute(
        "SELECT substr(i.fetched_at, 1, 10) AS day, i.feed_id, s.label, s.confidence, "
        "f.expected_label FROM news_scores s "
        "JOIN news_items i ON i.id = s.item_id JOIN news_feeds f ON f.id = i.feed_id "
        "WHERE s.model_id = ? AND i.fetched_at >= ?",
        (model_id, since_day),
    )
    for row in rows:
        stats = groups.setdefault((row["day"], row["feed_id"]), WindowStats(row["day"], row["day"]))
        stats.n += 1
        stats.conf_sum += row["confidence"]
        stats.hist[min(int(row["confidence"] * HISTOGRAM_BINS), HISTOGRAM_BINS - 1)] += 1
        stats.label_counts[row["label"]] = stats.label_counts.get(row["label"], 0) + 1
        if row["expected_label"]:
            stats.n_expected += 1
            stats.n_agree += row["label"] == row["expected_label"]
    with transaction(conn):
        for (day, feed_id), stats in groups.items():
            conn.execute(
                "INSERT OR REPLACE INTO drift_daily (day, feed_id, model_id, n, conf_sum, "
                "conf_hist, label_counts, n_expected, n_agree) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    day,
                    feed_id,
                    model_id,
                    stats.n,
                    stats.conf_sum,
                    json.dumps(stats.hist),
                    json.dumps(stats.label_counts, sort_keys=True),
                    stats.n_expected,
                    stats.n_agree,
                ),
            )
    return len(groups)


def window(conn: sqlite3.Connection, model_id: int, first_day: str, last_day: str) -> WindowStats:
    """Add up the daily rows of all feeds between two days (inclusive)."""
    stats = WindowStats(first_day, last_day)
    rows = conn.execute(
        "SELECT n, conf_sum, conf_hist, label_counts, n_expected, n_agree FROM drift_daily "
        "WHERE model_id = ? AND day BETWEEN ? AND ?",
        (model_id, first_day, last_day),
    )
    for row in rows:
        stats.n += row["n"]
        stats.conf_sum += row["conf_sum"]
        for index, count in enumerate(json.loads(row["conf_hist"])):
            stats.hist[index] += count
        for label, count in json.loads(row["label_counts"]).items():
            stats.label_counts[label] = stats.label_counts.get(label, 0) + count
        stats.n_expected += row["n_expected"]
        stats.n_agree += row["n_agree"]
    return stats


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def drift_report(conn: sqlite3.Connection, model_id: int, config: DriftConfig) -> DriftReport:
    """Compare the model's first window of headlines with its latest one."""
    span = conn.execute(
        "SELECT MIN(day) AS first, MAX(day) AS last FROM drift_daily WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    if span["first"] is None:
        return DriftReport(model_id, None, None, None, False, "no scored headlines yet")
    width = config.window_days
    reference_end = min(_shift(span["first"], width - 1), span["last"])
    reference = window(conn, model_id, span["first"], reference_end)
    current = window(conn, model_id, _shift(span["last"], -(width - 1)), span["last"])
    if current.first_day <= _shift(span["first"], width - 1):
        return DriftReport(
            model_id,
            reference,
            None,
            None,
            False,
            f"not enough history: need at least {2 * width} days of headlines "
            "to compare two windows",
        )
    if reference.n < config.min_items or current.n < config.min_items:
        return DriftReport(
            model_id,
            reference,
            current,
            None,
            False,
            f"too few headlines to compare (need {config.min_items} per window)",
        )
    value = psi(reference.hist, current.hist)
    return DriftReport(model_id, reference, current, round(value, 4), value >= config.psi_alert)


def format_report(report: DriftReport) -> str:
    lines = [f"model {report.model_id}"]
    for name, stats in (("reference", report.reference), ("current", report.current)):
        if stats is None:
            continue
        mean = stats.mean_confidence
        agreement = stats.agreement
        lines.append(
            f"  {name:<10}{stats.first_day} to {stats.last_day}  n={stats.n:<6}"
            f" mean confidence={'-' if mean is None else format(mean, '.3f')}"
            f"  agreement with expected={'-' if agreement is None else format(agreement, '.3f')}"
            f"  labels={stats.label_share()}"
        )
    if report.psi is not None:
        verdict = "ALERT: confidence distribution has shifted" if report.alert else "stable"
        lines.append(f"  PSI={report.psi:.4f}  ({verdict})")
    if report.reason:
        lines.append(f"  {report.reason}")
    return "\n".join(lines)
