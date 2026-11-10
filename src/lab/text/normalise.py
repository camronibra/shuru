"""Persian-script normalisation.

The same word can reach us as different code points depending on the keyboard and
the publishing system behind a site: Arabic yeh (ي) or Persian yeh (ی), Arabic kaf
(ك) or Persian kaf (ک), three different digit sets, optional vowel marks, and a
zero-width non-joiner (ZWNJ) that is typed in some newsrooms and left out in others.
None of that says anything about the language. Left alone it would split one word
into several in every count, hide duplicates, and let a classifier learn "this
outlet uses Arabic keyboards" instead of the language variety.

Two levels:

``normalise(text)``
    Conservative. The result is what the corpus stores and what a reader sees:
    yeh and kaf are unified, invisible junk is removed, whitespace is tidied.
    Digits, vowel marks and meaningful ZWNJs are kept.

``fold(text)``
    Aggressive, for comparing only: hashing, duplicate detection and classifier
    features. Also drops vowel marks, tatweel and ZWNJ, maps all digits to ASCII
    and lower-cases Latin. Never stored, never shown.

Both run in C almost entirely (``str.translate`` and compiled regexes).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

ZWNJ = "\u200c"

ARABIC_YEH = "\u064a"  # ي
ALEF_MAKSURA = "\u0649"  # ى
PERSIAN_YEH = "\u06cc"  # ی
ARABIC_KAF = "\u0643"  # ك
PERSIAN_KAF = "\u06a9"  # ک
TATWEEL = "\u0640"  # ـ  (stretches a word for looks)

# Invisible formatting characters that carry no text: zero-width space and joiner,
# word joiner, byte-order mark, soft hyphen, and the bidi direction marks.
_INVISIBLE = (
    "\u200b\u200d\u2060\ufeff\u00ad\u200e\u200f\u061c"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)
# Space characters other than the plain space.
_SPACES = (
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u202f\u205f\u3000\t\v\f"
)
_LINE_BREAKS = "\r\u0085\u2028\u2029"

# Arabic combining marks: Quranic signs, vowel marks (harakat), superscript alef.
_DIACRITICS = [
    *range(0x0610, 0x061B),
    *range(0x064B, 0x0660),
    0x0670,
    *range(0x06D6, 0x06DD),
    *range(0x06DF, 0x06E5),
    0x06E7,
    0x06E8,
    *range(0x06EA, 0x06EE),
]


def _storage_table() -> dict[int, str | None]:
    table: dict[int, str | None] = {}
    # C0 and C1 control characters are removed, except newline.
    for code in (*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)):
        table[code] = None
    table[ord("\n")] = "\n"
    for ch in _INVISIBLE:
        table[ord(ch)] = None
    for ch in _SPACES:
        table[ord(ch)] = " "
    for ch in _LINE_BREAKS:
        table[ord(ch)] = "\n"
    table[ord(ARABIC_YEH)] = PERSIAN_YEH
    table[ord(ALEF_MAKSURA)] = PERSIAN_YEH
    table[ord(ARABIC_KAF)] = PERSIAN_KAF
    return table


def _fold_table() -> dict[int, str | None]:
    table: dict[int, str | None] = dict.fromkeys(_DIACRITICS)
    table[ord(TATWEEL)] = None
    table[ord(ZWNJ)] = None
    table[ord("\n")] = " "
    for digit in range(10):
        table[0x0660 + digit] = str(digit)  # Arabic-Indic ٠-٩
        table[0x06F0 + digit] = str(digit)  # Persian ۰-۹
    return table


_STORAGE_TABLE = _storage_table()
_FOLD_TABLE = _fold_table()

_ARABIC_BLOCK = "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
_ZWNJ_RUN = re.compile(f"{ZWNJ}{{2,}}")
# A ZWNJ only means something between two Arabic-script characters.
_ZWNJ_STRAY = re.compile(f"(?<![{_ARABIC_BLOCK}]){ZWNJ}|{ZWNJ}(?![{_ARABIC_BLOCK}])")
_SPACE_RUN = re.compile(" {2,}")
_SPACE_AROUND_NEWLINE = re.compile(" *\n *")
_BLANK_LINES = re.compile("\n{3,}")


def _clean(text: str, form: Literal["NFC", "NFKC"]) -> str:
    text = unicodedata.normalize(form, text).replace("\r\n", "\n")
    text = text.translate(_STORAGE_TABLE)
    if ZWNJ in text:
        text = _ZWNJ_RUN.sub(ZWNJ, text)
        text = _ZWNJ_STRAY.sub("", text)
    text = _SPACE_RUN.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def normalise(text: str) -> str:
    """Canonical form for storage. Idempotent: normalise(normalise(x)) == normalise(x).

    Steps: Unicode NFC, remove control and invisible characters, turn odd spaces
    into plain spaces, unify yeh and kaf, drop ZWNJs that join nothing, collapse
    runs of spaces and blank lines.
    """
    return _clean(text, "NFC")


def fold(text: str) -> str:
    """Comparison key: two strings a reader would call the same text fold to the same key.

    On top of ``normalise``: Unicode NFKC (so presentation-form ligatures become plain
    letters), no vowel marks, no tatweel, no ZWNJ, ASCII digits, lower-case Latin,
    and all whitespace (newlines included) as single spaces.
    """
    text = _clean(text, "NFKC").translate(_FOLD_TABLE).lower()
    return _SPACE_RUN.sub(" ", text).strip()
