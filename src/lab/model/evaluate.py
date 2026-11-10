"""Scoring a classifier against labelled examples.

The same function scores the model and the baselines, so their numbers are
comparable: it only needs (true label, predicted label, source) triples.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

HISTOGRAM_BINS = 10


def score_predictions(triples: Iterable[tuple[str, str, str]]) -> dict[str, Any]:
    """Summarise (true label, predicted label, source name) triples.

    Returns a JSON-ready dict:

    - ``accuracy``: share of predictions that are right.
    - ``per_label``: precision, recall and F1 for each label.
      Precision: of everything predicted X, how much really was X.
      Recall: of everything that really was X, how much was found.
      F1: their harmonic mean.
    - ``macro_f1``: plain average of the per-label F1 scores, so a small label
      counts as much as a large one.
    - ``confusion``: confusion[true][predicted] = count.
    - ``per_source``: accuracy for each source. One source far below the others
      points at mislabelled or unusual text there.
    """
    confusion: dict[str, dict[str, int]] = {}
    per_source: dict[str, list[int]] = {}
    total = correct = 0
    for truth, predicted, source in triples:
        row = confusion.setdefault(truth, {})
        row[predicted] = row.get(predicted, 0) + 1
        tally = per_source.setdefault(source, [0, 0])
        tally[0] += 1
        total += 1
        if truth == predicted:
            tally[1] += 1
            correct += 1
    if not total:
        return {"n": 0}

    labels = sorted(set(confusion) | {p for row in confusion.values() for p in row})
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = confusion.get(label, {}).get(label, 0)
        predicted_as = sum(confusion.get(other, {}).get(label, 0) for other in labels)
        actual = sum(confusion.get(label, {}).values())
        precision = true_positive / predicted_as if predicted_as else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n": actual,
        }
    # Average over labels that actually occur, so a stray predicted label does not drag it down.
    present = [per_label[label]["f1"] for label in labels if per_label[label]["n"]]
    return {
        "n": total,
        "accuracy": round(correct / total, 4),
        "macro_f1": round(sum(present) / len(present), 4),
        "per_label": per_label,
        "confusion": confusion,
        "per_source": {
            source: {"n": n, "accuracy": round(right / n, 4)}
            for source, (n, right) in sorted(per_source.items())
        },
    }


def confidence_histogram(confidences: Iterable[float], bins: int = HISTOGRAM_BINS) -> list[int]:
    """Counts of confidence values in equal-width bins over [0, 1]."""
    counts = [0] * bins
    for value in confidences:
        counts[min(int(value * bins), bins - 1)] += 1
    return counts
