"""Turn extracted text into a corpus-ready document, or say why it was rejected.

``prepare`` is the whole text side of the pipeline for one document:

    raw text -> normalise -> quality checks -> fold -> hash -> tokenise -> MinHash

It is a pure function (no network, no database), so it is easy to test and it is
exactly what ``bench/bench_text.py`` measures.
"""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass

from lab.config import CleanConfig
from lab.runs import Timings
from lab.text.dedup import MinHasher
from lab.text.normalise import fold, normalise
from lab.text.tokenise import SCRIPT_RANGES, script_ratio, tokenise


@dataclass(frozen=True, slots=True)
class Prepared:
    """A document that passed cleaning, with everything needed to store and dedupe it."""

    text: str  # normalised, what the corpus stores
    text_hash: str  # SHA-256 of the folded text: equal hash = same text
    n_chars: int
    n_tokens: int
    signature: array[int]  # MinHash signature, for near-duplicates
    bands: list[int]  # LSH band hashes of the signature


@dataclass(frozen=True, slots=True)
class Rejected:
    reason: str  # "too_short" or "wrong_script"


def truncate(text: str, max_chars: int) -> str:
    """Cut at the last paragraph break before max_chars, or hard at max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", max_chars // 2, max_chars)
    return text[: cut if cut != -1 else max_chars].rstrip()


def prepare(
    raw_text: str,
    config: CleanConfig,
    hasher: MinHasher,
    timings: Timings | None = None,
) -> Prepared | Rejected:
    """Clean one document and compute its hashes. See the module docstring."""
    timings = timings if timings is not None else Timings()

    with timings.step("normalise", cpu=True):
        text = truncate(normalise(raw_text), config.max_chars)

    with timings.step("quality", cpu=True):
        if len(text) < config.min_chars:
            return Rejected("too_short")
        if (
            config.min_script_ratio > 0
            and config.script in SCRIPT_RANGES
            and script_ratio(text, config.script) < config.min_script_ratio
        ):
            return Rejected("wrong_script")

    with timings.step("hash", cpu=True):
        folded = fold(text)
        text_hash = hashlib.sha256(folded.encode("utf-8")).hexdigest()

    with timings.step("tokenise", cpu=True):
        tokens = tokenise(folded)

    with timings.step("minhash", cpu=True):
        signature = hasher.signature(tokens)
        bands = hasher.band_hashes(signature)

    return Prepared(
        text=text,
        text_hash=text_hash,
        n_chars=len(text),
        n_tokens=len(tokens),
        signature=signature,
        bands=bands,
    )
