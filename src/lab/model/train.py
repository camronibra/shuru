"""Training: multinomial logistic regression by stochastic gradient descent.

**The model.** For each label there is one weight per feature bucket and one bias.
The score of a label for a text is the sum of the weights of the text's buckets
(scaled so the feature vector has length 1) plus the bias. Softmax turns the scores
into probabilities.

**The training loop.** Start with all weights at zero. For each training sentence:

1. Compute the probabilities with the current weights.
2. The error for a label is ``p - 1`` if it is the true label, else ``p``.
   (This is the gradient of the cross-entropy loss with respect to the score.)
3. Move the weights of the sentence's buckets a small step against the error.
   Right answers given with confidence change almost nothing; confident mistakes
   change a lot.

Repeat over the shuffled data for a few epochs, with a smaller step each epoch.

**Regularisation.** Each update also shrinks the touched weights slightly towards
zero (L2). Without it, weights on separable data grow without limit and the model
becomes overconfident, which would make "least certain" and the drift histogram
meaningless. Shrinking only the touched weights is an approximation of true L2
that keeps each update proportional to the sentence length, not the table size.

Pure Python on purpose: the whole algorithm is the forty lines of ``fit``. It
takes about a minute per 100,000 sentences on a PC, which is where training runs.
"""

from __future__ import annotations

import math
import random
import sqlite3
from array import array
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lab.config import Settings
from lab.corpus.langid_baselines import MajorityBaseline, MarkerBaseline, load_markers
from lab.logs import get_logger
from lab.model import registry
from lab.model.dataset import (
    HUMAN,
    TEST_DATE,
    TEST_SOURCE,
    TRAIN,
    Example,
    balance_labels,
    iter_examples,
    iter_human_examples,
    plan_split,
    sample_examples,
)
from lab.model.evaluate import confidence_histogram, score_predictions
from lab.model.features import hashed_features
from lab.model.predict import Model, softmax
from lab.runs import Run
from lab.timeutil import utc_now

log = get_logger(__name__)

# Errors smaller than this are not worth an update.
_MIN_ERROR = 1e-4


class TrainError(Exception):
    """Training cannot go ahead with the data available."""


def fit(
    examples: Sequence[tuple[array[int], int]],
    n_labels: int,
    hash_bits: int,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[list[array[float]], list[float]]:
    """Fit weights and biases. examples are (feature buckets, label index) pairs.

    Returns (weights, bias): one float array of 2**hash_bits per label, one bias per label.
    """
    weights = [array("f", bytes(4 << hash_bits)) for _ in range(n_labels)]  # all zeros
    bias = [0.0] * n_labels
    order = list(range(len(examples)))
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(order)
        rate = learning_rate / (1 + epoch)
        shrink = 1.0 - rate * l2
        loss = 0.0
        for index in order:
            features, truth = examples[index]
            if not features:
                continue
            scale = 1.0 / math.sqrt(len(features))
            probs = softmax(
                [
                    bias[k] + scale * sum(map(weights[k].__getitem__, features))
                    for k in range(n_labels)
                ]
            )
            loss -= math.log(max(probs[truth], 1e-12))
            for k in range(n_labels):
                error = probs[k] - (1.0 if k == truth else 0.0)
                if abs(error) < _MIN_ERROR:
                    continue
                step = rate * error * scale
                row = weights[k]
                for bucket in features:
                    row[bucket] = row[bucket] * shrink - step
                bias[k] -= rate * error
        if on_epoch is not None:
            on_epoch(epoch + 1, loss / max(len(order), 1))
    return weights, bias


def _evaluate_model(model: Model, examples: Sequence[Example]) -> dict[str, Any]:
    triples: list[tuple[str, str, str]] = []
    confidences: list[float] = []
    for example in examples:
        prediction = model.predict(example.text)
        triples.append((example.label, prediction.label, example.source))
        confidences.append(prediction.confidence)
    scores = score_predictions(triples)
    if confidences:
        scores["mean_confidence"] = round(sum(confidences) / len(confidences), 4)
        scores["confidence_hist"] = confidence_histogram(confidences)
    return scores


def _evaluate_baseline(
    predict: Callable[[str], str], examples: Sequence[Example]
) -> dict[str, Any]:
    scores = score_predictions((e.label, predict(e.text), e.source) for e in examples)
    # Baselines are compared on the headline numbers only.
    return {key: scores[key] for key in ("n", "accuracy", "macro_f1") if key in scores}


def train_and_register(
    settings: Settings,
    conn: sqlite3.Connection,
    run: Run,
    *,
    use_human_labels: bool = False,
    activate: bool = True,
    note: str = "",
) -> tuple[int, Model]:
    """Build the dataset, train, evaluate against the baselines, save and register.

    Returns (model id, model). The model's ``metrics`` hold everything measured.
    """
    config = settings.model
    plan = plan_split(conn, config)
    if plan.first_day is None:
        raise TrainError("the corpus is empty: run `shuru ingest` first")
    if plan.cutoff is None:
        log.warning("no date-based test set: %s", plan.reason)

    with run.step("dataset", cpu=True):
        by_split = sample_examples(iter_examples(conn, config, plan), config)
        train_pool = by_split.get(TRAIN, [])
        if use_human_labels:
            train_pool = train_pool + list(iter_human_examples(conn, for_training=True))
        train = balance_labels(train_pool, config.max_examples, config.seed)
        tests: dict[str, list[Example]] = {
            TEST_DATE: by_split.get(TEST_DATE, []),
            TEST_SOURCE: by_split.get(TEST_SOURCE, []),
            HUMAN: list(iter_human_examples(conn, for_training=False)),
        }
    labels = tuple(sorted({example.label for example in train}))
    if len(labels) < 2:
        found = sorted({example.label for example in train_pool})
        raise TrainError(
            "training needs sentences from at least two labels on the training side of the "
            f"split; found {found or 'none'}. Add sources for another label (not all holdout)."
        )
    label_index = {label: index for index, label in enumerate(labels)}
    log.info(
        "dataset: train=%d %s cutoff=%s",
        len(train),
        " ".join(f"{name}={len(items)}" for name, items in tests.items()),
        plan.cutoff,
    )

    with run.step("features", cpu=True):
        encoded = [
            (
                hashed_features(e.text, config.ngram_min, config.ngram_max, config.hash_bits),
                label_index[e.label],
            )
            for e in train
        ]

    losses: list[float] = []

    def on_epoch(epoch: int, loss: float) -> None:
        losses.append(round(loss, 4))
        log.info("epoch %d/%d: mean loss %.4f", epoch, config.epochs, loss)

    with run.step("fit", cpu=True):
        weights, bias = fit(
            encoded,
            len(labels),
            config.hash_bits,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            l2=config.l2,
            seed=config.seed,
            on_epoch=on_epoch,
        )
    del encoded
    model = Model(
        labels=labels,
        hash_bits=config.hash_bits,
        ngram_min=config.ngram_min,
        ngram_max=config.ngram_max,
        weights=weights,
        bias=bias,
        created_at=utc_now(),
    )

    with run.step("evaluate", cpu=True):
        majority = MajorityBaseline(example.label for example in train)
        markers = load_markers(settings.config_dir)
        marker_baseline = (
            MarkerBaseline({k: v for k, v in markers.items() if k in labels}, majority.label)
            if any(label in markers for label in labels)
            else None
        )
        splits: dict[str, Any] = {}
        baselines: dict[str, dict[str, Any]] = {"majority": {}, "markers": {}}
        for name, examples in tests.items():
            # A test sentence with a label the model was never trained on cannot be scored fairly.
            usable = [example for example in examples if example.label in label_index]
            if not usable:
                continue
            splits[name] = _evaluate_model(model, usable)
            baselines["majority"][name] = _evaluate_baseline(majority.predict, usable)
            if marker_baseline is not None:
                baselines["markers"][name] = _evaluate_baseline(marker_baseline.predict, usable)
        # Score on (part of) the training data too: far above the test scores = overfitting.
        splits["train_sample"] = _evaluate_model(model, train[:2000])

    train_sources = sorted({example.source for example in train})
    model.metrics = {
        "n_train": len(train),
        "train_per_label": len(train) // len(labels),
        "train_sources": train_sources,
        "cutoff": plan.cutoff,
        "first_day": plan.first_day,
        "last_day": plan.last_day,
        "used_human_labels": use_human_labels,
        "epoch_loss": losses,
        "settings": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "l2": config.l2,
            "seed": config.seed,
        },
        "splits": splits,
        "baselines": {name: value for name, value in baselines.items() if value},
    }

    models_dir = settings.paths.dir("models")
    stamp = model.created_at.replace("-", "").replace(":", "")
    path = _unused_path(models_dir, f"model-{stamp}")
    model.save(path)
    model_id = registry.register(conn, path, model, note)
    if activate:
        registry.activate(conn, model_id)
    run.set("model_id", model_id)
    run.set("n_train", len(train))
    for name in (TEST_DATE, TEST_SOURCE, HUMAN):
        if name in splits:
            run.set(f"{name}_accuracy", splits[name]["accuracy"])
    return model_id, model


def _unused_path(folder: Path, stem: str) -> Path:
    path = folder / f"{stem}.shuru"
    counter = 1
    while path.exists():
        counter += 1
        path = folder / f"{stem}-{counter}.shuru"
    return path


def format_report(model: Model) -> str:
    """The model's metrics as a small text table: model against baselines, per test set."""
    metrics = model.metrics
    lines = [
        f"labels: {', '.join(model.labels)}   training sentences: {metrics.get('n_train', '?')} "
        f"({metrics.get('train_per_label', '?')} per label)",
        f"date cutoff: {metrics.get('cutoff') or 'none (no date-based test set)'}   "
        f"training sources: {', '.join(metrics.get('train_sources', []))}",
        "",
        f"{'test set':<14}{'n':>7}{'model acc':>11}{'macro-F1':>10}{'majority':>10}{'markers':>9}",
    ]
    descriptions = {
        TEST_DATE: "later text, known sources",
        TEST_SOURCE: "sources never trained on",
        HUMAN: "hand-labelled headlines",
        "train_sample": "training data (sanity check)",
    }
    baselines = metrics.get("baselines", {})
    for name, description in descriptions.items():
        scores = metrics.get("splits", {}).get(name)
        if not scores or not scores.get("n"):
            continue

        def baseline(kind: str, split: str = name) -> str:
            value = baselines.get(kind, {}).get(split, {}).get("accuracy")
            return f"{value:.3f}" if isinstance(value, float) else "-"

        lines.append(
            f"{name:<14}{scores['n']:>7}{scores['accuracy']:>11.3f}{scores['macro_f1']:>10.3f}"
            f"{baseline('majority'):>10}{baseline('markers'):>9}   {description}"
        )
    for name in (TEST_SOURCE, TEST_DATE):
        per_source = metrics.get("splits", {}).get(name, {}).get("per_source")
        if per_source:
            lines.append("")
            lines.append(f"accuracy by source ({name}):")
            for source, scores in per_source.items():
                lines.append(f"  {source:<30}{scores['n']:>7}{scores['accuracy']:>8.3f}")
    return "\n".join(lines)
