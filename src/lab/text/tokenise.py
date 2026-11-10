"""Script-aware tokeniser, sentence splitter and script statistics.

"Script-aware" means the rules depend on the writing system of each stretch of text:
an Arabic-script word may contain vowel marks and ZWNJs (می‌شود is one word), a Latin
word may contain an apostrophe or hyphen, and a number may be written in any of three
digit sets. One compiled regex does the work, so tokenising runs at C speed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import NamedTuple

# Letter ranges per script, written as the inside of a regex character class.
SCRIPT_RANGES: dict[str, str] = {
    # Base letters, extended letters (Persian, Pashto, Urdu, ...), presentation forms.
    "arabic": (
        "\u0620-\u064a\u066e-\u06d3\u06d5\u06ee\u06ef\u06fa-\u06ff"
        "\u0750-\u077f\u08a0-\u08c9\ufb50-\ufdff\ufe70-\ufefc"
    ),
    "latin": "A-Za-z\u00c0-\u024f",
    "cyrillic": "\u0400-\u052f",
    "devanagari": "\u0900-\u097f",
}

_ARABIC = SCRIPT_RANGES["arabic"]
_LATIN = SCRIPT_RANGES["latin"]
_MARKS = "\u064b-\u065f\u0670"
_DIGITS = "0-9\u0660-\u0669\u06f0-\u06f9"

# An Arabic-script word: letters and marks, with single ZWNJs allowed inside.
_ARABIC_WORD = f"[{_ARABIC}][{_ARABIC}{_MARKS}]*(?:\u200c[{_ARABIC}][{_ARABIC}{_MARKS}]*)*"
_LATIN_WORD = f"[{_LATIN}]+(?:['’\\-][{_LATIN}]+)*"
# Digits with inner separators: 1,200.5  ۱۲/۳  12:30  (٫ and ٬ are the Arabic separators).
_NUMBER = f"[{_DIGITS}]+(?:[.,/:\u066b\u066c][{_DIGITS}]+)*"
_OTHER_WORD = r"[^\W\d_]+"  # letters of any other script
_PUNCT = r"[^\w\s]"

_TOKEN_RE = re.compile(
    f"(?P<arabic>{_ARABIC_WORD})|(?P<latin>{_LATIN_WORD})|(?P<number>{_NUMBER})"
    f"|(?P<other>{_OTHER_WORD})|(?P<punct>{_PUNCT})"
)
_WORD_RE = re.compile(f"{_ARABIC_WORD}|{_LATIN_WORD}|{_NUMBER}|{_OTHER_WORD}")
_SCRIPT_RUNS = {name: re.compile(f"[{ranges}]+") for name, ranges in SCRIPT_RANGES.items()}
# Sentence enders: . ! ? and the Arabic question mark ؟, Urdu full stop ۔ and ellipsis.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟۔…])\s+|\n+")


class Token(NamedTuple):
    text: str
    kind: str  # "arabic", "latin", "number", "other" or "punct"


def iter_tokens(text: str) -> Iterator[Token]:
    """Every token with its kind, in order. Whitespace is skipped."""
    for match in _TOKEN_RE.finditer(text):
        kind = match.lastgroup
        assert kind is not None  # noqa: S101 - every alternative is a named group
        yield Token(match.group(), kind)


def tokenise(text: str) -> list[str]:
    """Words and numbers only, no punctuation. The fast path used by the pipeline."""
    return _WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace, and on newlines.

    "3.5" stays whole because no whitespace follows the dot.
    """
    return [part for part in (piece.strip() for piece in _SENTENCE_BREAK.split(text)) if part]


def script_ratio(text: str, script: str = "arabic") -> float:
    """Share of the letters in text that belong to script (0.0 if there are no letters).

    Digits, punctuation and spaces are ignored, so "۲۰۲۴: کابل" is 100% Arabic-script.
    """
    letters = sum(map(str.isalpha, text))
    if not letters:
        return 0.0
    in_script = sum(map(len, _SCRIPT_RUNS[script].findall(text)))
    return min(in_script / letters, 1.0)
