"""Baselines: the simplest things that could work, to compare the classifier against.

A classifier's accuracy means little alone. "94%" is impressive if the obvious
approach gets 60% and unremarkable if it gets 93%. Two baselines are scored on
exactly the same test sentences as the model:

``MajorityBaseline``
    Always answers with the most common training label. Any real model must beat it.

``MarkerBaseline``
    A hand-written word list per label (config/markers.toml): words used in one
    variety and not the other, such as پوهنتون against دانشگاه for "university".
    It answers with the label whose markers appear most, and falls back to the
    majority label when none appear, which for short sentences is most of the time.
    It shows how far explicit vocabulary knowledge gets you.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from lab.config import ConfigError, read_toml
from lab.text.normalise import fold
from lab.text.tokenise import tokenise


class MajorityBaseline:
    def __init__(self, training_labels: Iterable[str]) -> None:
        counts = Counter(training_labels)
        if not counts:
            raise ValueError("no training labels")
        # Ties resolve alphabetically so the result does not depend on input order.
        self.label = min(counts, key=lambda label: (-counts[label], label))

    def predict(self, text: str) -> str:
        return self.label


class MarkerBaseline:
    def __init__(self, markers: Mapping[str, Iterable[str]], fallback: str) -> None:
        # Markers are folded like the text they are matched against.
        self._label_of: dict[str, str] = {}
        for label, words in markers.items():
            for word in words:
                self._label_of[fold(word)] = label
        self._fallback = fallback

    def predict(self, text: str) -> str:
        hits: Counter[str] = Counter()
        for token in tokenise(fold(text)):
            label = self._label_of.get(token)
            if label is not None:
                hits[label] += 1
        if not hits:
            return self._fallback
        best = max(hits.values())
        winners = sorted(label for label, count in hits.items() if count == best)
        return winners[0] if len(winners) == 1 else self._fallback


def load_markers(config_dir: Path) -> dict[str, list[str]]:
    """Read config/markers.toml: ``[markers]`` with one list of words per label."""
    path = config_dir / "markers.toml"
    if not path.exists():
        return {}
    table = read_toml(path).get("markers", {})
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: [markers] must be a table of word lists")
    markers: dict[str, list[str]] = {}
    for label, words in table.items():
        if not isinstance(words, list) or not all(isinstance(word, str) for word in words):
            raise ConfigError(f"{path}: markers.{label} must be a list of words")
        markers[label] = [word for word in words if word.strip()]
    return markers
