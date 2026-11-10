"""The trained model: loading it, saving it and scoring text.

A model is a table of numbers: for each label, one weight per feature bucket, plus
one bias. Scoring a text means

1. hash its character n-grams to buckets (``features.hashed_features``),
2. for each label, add up the weights of those buckets (plus the bias),
3. turn the sums into probabilities with softmax.

That is all multinomial logistic regression does at prediction time, and it needs
nothing but the standard library. Memory use is fixed: labels x 2**hash_bits floats
(2 labels at 18 bits = 2 MB).

**File format.** One line of JSON (labels, parameters, biases, metrics), a newline,
then the weights as raw little-endian float32, label after label. The file holds
only numbers and is never executed or unpickled, so loading a model cannot run code.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.config import ModelConfig
from lab.db import transaction
from lab.model.features import hashed_features
from lab.runs import Run
from lab.timeutil import utc_now

MAGIC = "shuru-model-1"
MAX_HEADER_BYTES = 1_000_000


class ModelError(Exception):
    """A model file is missing, damaged or not what the registry recorded."""


@dataclass(frozen=True, slots=True)
class Prediction:
    label: str
    confidence: float  # probability of the predicted label
    margin: float  # confidence minus the runner-up's probability; near 0 = a coin toss
    probabilities: dict[str, float]


def softmax(scores: list[float]) -> list[float]:
    """Scores to probabilities that sum to 1. Subtracting the max avoids overflow."""
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass(slots=True)
class Model:
    labels: tuple[str, ...]
    hash_bits: int
    ngram_min: int
    ngram_max: int
    weights: list[array[float]]  # one array of 2**hash_bits floats per label
    bias: list[float]
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def featurise(self, text: str, max_chars: int | None = None) -> array[int]:
        return hashed_features(text, self.ngram_min, self.ngram_max, self.hash_bits, max_chars)

    def probabilities(self, features: array[int]) -> list[float]:
        """Probability of each label, in ``labels`` order, for a set of feature buckets."""
        if not features:
            return softmax(list(self.bias))
        scale = 1.0 / math.sqrt(len(features))  # unit-length feature vector
        return softmax(
            [
                bias + scale * sum(map(weights.__getitem__, features))
                for weights, bias in zip(self.weights, self.bias, strict=True)
            ]
        )

    def predict(self, text: str, max_chars: int | None = None) -> Prediction:
        probs = self.probabilities(self.featurise(text, max_chars))
        ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
        best = ranked[0]
        runner_up = probs[ranked[1]] if len(ranked) > 1 else 0.0
        return Prediction(
            label=self.labels[best],
            confidence=probs[best],
            margin=probs[best] - runner_up,
            probabilities=dict(zip(self.labels, probs, strict=True)),
        )

    def save(self, path: Path) -> str:
        """Write the model file. Returns its SHA-256."""
        header = {
            "format": MAGIC,
            "labels": list(self.labels),
            "hash_bits": self.hash_bits,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "bias": self.bias,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n")
            for weights in self.weights:
                handle.write(_little_endian(weights).tobytes())
        temporary.replace(path)  # atomic: a reader never sees half a file
        return sha256_file(path)

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> Model:
        """Read a model file, checking its hash (if given) and its shape."""
        if not path.is_file():
            raise ModelError(f"model file not found: {path}")
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise ModelError(f"{path.name}: file does not match the hash in the registry")
        with path.open("rb") as handle:
            first_line = handle.readline(MAX_HEADER_BYTES)
            try:
                header = json.loads(first_line)
                if header.get("format") != MAGIC:
                    raise ModelError(f"{path.name}: not a shuru model file")
                labels = tuple(str(label) for label in header["labels"])
                bits = int(header["hash_bits"])
                bias = [float(value) for value in header["bias"]]
                model = cls(
                    labels=labels,
                    hash_bits=bits,
                    ngram_min=int(header["ngram_min"]),
                    ngram_max=int(header["ngram_max"]),
                    weights=[],
                    bias=bias,
                    metrics=dict(header.get("metrics", {})),
                    created_at=str(header.get("created_at", "")),
                )
            except (ValueError, KeyError, TypeError, AttributeError):
                raise ModelError(f"{path.name}: unreadable model header") from None
            if not 10 <= bits <= 24 or len(labels) < 2 or len(bias) != len(labels):
                raise ModelError(f"{path.name}: implausible model shape")
            size = 4 << bits
            for _label in labels:
                chunk = handle.read(size)
                if len(chunk) != size:
                    raise ModelError(f"{path.name}: weights are truncated")
                weights: array[float] = array("f")
                weights.frombytes(chunk)
                model.weights.append(_little_endian(weights))
            if handle.read(1):
                raise ModelError(f"{path.name}: unexpected data after the weights")
        return model


def _little_endian(values: array[float]) -> array[float]:
    """Files are little-endian. On a big-endian machine, swap a copy."""
    if sys.byteorder == "little":
        return values
    swapped = array("f", values)  # pragma: no cover
    swapped.byteswap()  # pragma: no cover
    return swapped  # pragma: no cover


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score_documents(
    conn: sqlite3.Connection, model_id: int, model: Model, config: ModelConfig, run: Run
) -> int:
    """Give every document that this model has not scored yet a label and a confidence.

    The corpus is read a batch at a time and each batch is written in one statement,
    so memory stays flat however large the corpus is. Safe to interrupt and re-run.
    """
    scored = 0
    last_id = 0
    while True:
        rows = conn.execute(
            "SELECT d.id, substr(d.text, 1, ?) AS head FROM documents d "
            "WHERE d.id > ? AND NOT EXISTS ("
            "  SELECT 1 FROM doc_scores s WHERE s.document_id = d.id AND s.model_id = ?) "
            "ORDER BY d.id LIMIT 200",
            (config.max_score_chars, last_id, model_id),
        ).fetchall()
        if not rows:
            break
        last_id = rows[-1]["id"]
        now = utc_now()
        with run.step("score", cpu=True):
            batch = []
            for row in rows:
                prediction = model.predict(row["head"])
                batch.append(
                    (
                        row["id"],
                        model_id,
                        prediction.label,
                        prediction.confidence,
                        prediction.margin,
                        now,
                    )
                )
        with run.step("db_write"), transaction(conn):
            conn.executemany(
                "INSERT OR REPLACE INTO doc_scores "
                "(document_id, model_id, label, confidence, margin, scored_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
        scored += len(batch)
    run.count("documents_scored", scored)
    return scored
