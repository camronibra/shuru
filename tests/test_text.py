from __future__ import annotations

import random

from lab import synth
from lab.text.dedup import (
    MinHasher,
    jaccard,
    signature_from_blob,
    signature_to_blob,
    similarity,
)
from lab.text.normalise import (
    ALEF_MAKSURA,
    ARABIC_KAF,
    ARABIC_YEH,
    PERSIAN_KAF,
    PERSIAN_YEH,
    TATWEEL,
    ZWNJ,
    fold,
    normalise,
)
from lab.text.tokenise import Token, iter_tokens, script_ratio, split_sentences, tokenise

ZWJ = "\N{ZERO WIDTH JOINER}"
NBSP = "\N{NO-BREAK SPACE}"
RLM = "\N{RIGHT-TO-LEFT MARK}"
FATHATAN = "\N{ARABIC FATHATAN}"
KASRA = "\N{ARABIC KASRA}"


# --- normalise ------------------------------------------------------------------------


def test_arabic_yeh_and_kaf_become_persian() -> None:
    typed_on_arabic_keyboard = f"{ARABIC_KAF}تاب عل{ARABIC_YEH} مصطف{ALEF_MAKSURA}"
    assert (
        normalise(typed_on_arabic_keyboard) == f"{PERSIAN_KAF}تاب عل{PERSIAN_YEH} مصطف{PERSIAN_YEH}"
    )


def test_invisible_characters_and_odd_spaces_are_cleaned() -> None:
    raw = f"﻿سلام{NBSP}{NBSP}دنیا{RLM}\t خوب{ZWJ}  \r\n\r\n\r\n\r\nپایان\x00\x07 "
    assert normalise(raw) == "سلام دنیا خوب\n\nپایان"


def test_zwnj_is_kept_inside_words_and_dropped_elsewhere() -> None:
    assert normalise(f"می{ZWNJ}شود") == f"می{ZWNJ}شود"  # joins two parts of one word
    assert normalise(f"می{ZWNJ}{ZWNJ}{ZWNJ}شود") == f"می{ZWNJ}شود"  # repeated by a CMS
    assert normalise(f"کتاب{ZWNJ} خوب") == "کتاب خوب"  # before a space: joins nothing
    assert normalise(f"{ZWNJ}کتاب{ZWNJ}") == "کتاب"
    assert normalise(f"abc{ZWNJ}def") == "abcdef"  # meaningless in Latin text


def test_normalise_keeps_what_a_reader_sees() -> None:
    text = f"قیمت ۱۲۳ و ٤٥ شد؛ کتا{FATHATAN}ب"
    assert normalise(text) == text  # digits and vowel marks are content, not noise


def test_normalise_is_idempotent_on_noisy_text() -> None:
    rng = random.Random(3)
    style = synth.make_style(rng, synth.VARIETY_A)
    for _ in range(50):
        once = normalise(synth.make_document(rng, style))
        assert normalise(once) == once


# --- fold -----------------------------------------------------------------------------


def test_fold_makes_spelling_variants_equal() -> None:
    variants = [
        f"می{ZWNJ}شود ۱۲ کتاب",
        "میشود 12 کتاب",
        f"م{KASRA}یشود ١٢ {ARABIC_KAF}تاب",
        f"میشود ۱۲ کتـ{TATWEEL}ـاب",
        "میشود  12\nکتاب ",
    ]
    assert {fold(variant) for variant in variants} == {"میشود 12 کتاب"}


def test_fold_lowercases_latin_and_unfolds_ligatures() -> None:
    assert fold("BBC News") == "bbc news"
    assert fold("\N{ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM}") == "لا"


def test_fold_does_not_merge_different_words() -> None:
    assert fold("کتاب") != fold("کباب")
    assert fold("شیر") != fold("شر")


# --- tokenise ---------------------------------------------------------------------------


def test_tokenise_keeps_zwnj_words_whole_and_drops_punctuation() -> None:
    text = f"او گفت: «می{ZWNJ}خواهم ۲۵ کتاب بخرم»، سپس رفت."
    assert tokenise(text) == ["او", "گفت", f"می{ZWNJ}خواهم", "۲۵", "کتاب", "بخرم", "سپس", "رفت"]


def test_tokens_know_their_kind() -> None:
    tokens = list(iter_tokens("کابل 2024/05/01 BBC's ۳٫۵ Москва؟"))
    assert tokens == [
        Token("کابل", "arabic"),
        Token("2024/05/01", "number"),
        Token("BBC's", "latin"),
        Token("۳٫۵", "number"),
        Token("Москва", "other"),
        Token("؟", "punct"),
    ]


def test_split_sentences() -> None:
    text = "قیمت ۳.۵ درصد بالا رفت. چرا؟ کسی نمی‌داند!\nخط تازه"
    assert split_sentences(text) == ["قیمت ۳.۵ درصد بالا رفت.", "چرا؟", "کسی نمی‌داند!", "خط تازه"]
    assert split_sentences("  \n ") == []


def test_script_ratio_ignores_digits_and_punctuation() -> None:
    assert script_ratio("۲۰۲۴: کابل!") == 1.0
    assert script_ratio("Kabul کابل") == 4 / 9
    assert script_ratio("hello world") == 0.0
    assert script_ratio("123 ... !") == 0.0
    assert script_ratio("hello мир", "cyrillic") == 3 / 8


# --- MinHash ------------------------------------------------------------------------------


def _tokens(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(synth.COMMON_WORDS) for _ in range(n)]


def test_signature_is_stable_across_runs_and_machines() -> None:
    """Signatures are stored and compared later, so they must never change."""
    hasher = MinHasher(num_perm=8, bands=4, shingle_size=3)
    signature = hasher.signature(
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    )
    assert list(signature) == [
        541823785, 3453030638, 1054022729, 1686154700,
        2555656977, 1516522803, 3803306831, 3317937807,
    ]  # fmt: skip
    assert hasher.band_hashes(signature) == [2920661046, 3847371508, 432891601, 3236534917]


def test_similarity_estimates_jaccard() -> None:
    rng = random.Random(11)
    hasher = MinHasher(num_perm=256, bands=64)
    base = _tokens(rng, 400)
    for keep in (350, 250, 120):
        other = base[:keep] + _tokens(rng, 400 - keep)
        exact = jaccard(hasher.shingle_hashes(base), hasher.shingle_hashes(other))
        estimate = similarity(hasher.signature(base), hasher.signature(other))
        assert abs(estimate - exact) < 0.1, (keep, exact, estimate)


def test_near_duplicates_share_a_band_and_unrelated_texts_do_not() -> None:
    rng = random.Random(5)
    hasher = MinHasher()
    original = _tokens(rng, 300)
    edited = [*original[:150], "ویرایش", *original[150:]]  # one word inserted
    unrelated = _tokens(rng, 300)
    bands = hasher.band_hashes(hasher.signature(original))

    def shares_a_band(tokens: list[str]) -> bool:
        other = hasher.band_hashes(hasher.signature(tokens))
        return any(a == b for a, b in zip(bands, other, strict=True))

    assert similarity(hasher.signature(original), hasher.signature(edited)) > 0.9
    assert shares_a_band(edited)
    assert similarity(hasher.signature(original), hasher.signature(unrelated)) < 0.2
    assert not shares_a_band(unrelated)


def test_short_and_empty_inputs() -> None:
    hasher = MinHasher()
    assert len(hasher.signature([])) == 64
    assert similarity(hasher.signature([]), hasher.signature([])) == 1.0
    assert hasher.signature(["تنها"]) == hasher.signature(["تنها"])
    assert similarity(hasher.signature(["یک"]), hasher.signature(["دو"])) < 0.2


def test_signature_blob_roundtrip() -> None:
    signature = MinHasher().signature(["الف", "ب", "پ", "ت"])
    blob = signature_to_blob(signature)
    assert len(blob) == 64 * 4
    assert signature_from_blob(blob) == signature
