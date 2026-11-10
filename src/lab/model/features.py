"""Character n-gram features with the hashing trick.

**Why character n-grams?** Closely related varieties differ in small things: a
suffix, a loanword, a spelling habit. Short character sequences capture those
without a tokeniser or a dictionary, and they still work on a five-word headline.

**Why hashing?** A normal bag-of-n-grams keeps a vocabulary: a dict from every
n-gram ever seen to a column number. It grows with the corpus and has to be shipped
with the model. Here each n-gram is hashed straight to one of ``2**bits`` buckets
(CRC-32, then keep the low bits). Memory is fixed up front, there is nothing to
ship, and the same function gives the same buckets at training and at scoring time.
The price is collisions: two n-grams can share a bucket. With 262,144 buckets and
a linear model that costs very little accuracy.

A text is represented as the *set* of buckets its n-grams fall into (present or
not, no counts), scaled to unit length. That keeps long and short texts comparable.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator
from zlib import crc32

from lab.text.normalise import fold


def char_ngrams(text: str, n_min: int, n_max: int) -> Iterator[str]:
    """The n-grams of text after folding, with one space of padding on each side.

    The padding marks word boundaries at the ends, so "کتاب" yields " ک" and "ب ".
    This readable version exists for tests and for explaining the idea;
    ``hashed_features`` computes the same thing faster.
    """
    padded = f" {fold(text)} "
    for n in range(n_min, n_max + 1):
        for start in range(len(padded) - n + 1):
            yield padded[start : start + n]


def hashed_features(
    text: str, n_min: int, n_max: int, bits: int, max_chars: int | None = None
) -> array[int]:
    """Bucket indices (unique, unordered) of every character n-gram in text.

    The text is folded first, so spelling variants of the same word hit the same
    buckets and the classifier cannot key on them.
    """
    folded = fold(text if max_chars is None else text[:max_chars])
    if not folded:
        return array("I")
    # UTF-32 gives every character exactly 4 bytes, so an n-gram is a plain byte slice.
    data = f" {folded} ".encode("utf-32-le")
    length = len(data) // 4
    mask = (1 << bits) - 1
    buckets: set[int] = set()
    for n in range(n_min, n_max + 1):
        width = 4 * n
        buckets.update(
            [
                crc32(data[start : start + width]) & mask
                for start in range(0, 4 * (length - n + 1), 4)
            ]
        )
    return array("I", buckets)
