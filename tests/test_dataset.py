"""Training data, the classifier, the registry and the baselines."""

from __future__ import annotations

import random
import sqlite3
from array import array
from pathlib import Path

import pytest

from lab import synth
from lab.config import REPO_ROOT, ModelConfig, Settings
from lab.corpus.export import export_corpus
from lab.corpus.langid_baselines import MajorityBaseline, MarkerBaseline, load_markers
from lab.corpus.stats import corpus_stats, format_stats
from lab.model import registry
from lab.model.dataset import (
    TEST_DATE,
    TEST_SOURCE,
    TRAIN,
    Example,
    assign_split,
    balance_labels,
    iter_examples,
    plan_split,
    sample_examples,
)
from lab.model.evaluate import confidence_histogram, score_predictions
from lab.model.features import char_ngrams, hashed_features
from lab.model.predict import Model, ModelError, score_documents, softmax
from lab.model.train import TrainError, fit, train_and_register
from lab.runs import Run
from lab.text.dedup import MinHasher, signature_to_blob


def add_source(conn: sqlite3.Connection, name: str, label: str, holdout: bool = False) -> int:
    cursor = conn.execute(
        "INSERT INTO sources (name, kind, label, holdout, created_at) VALUES (?, 'rss', ?, ?, 'x')",
        (name, label, int(holdout)),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def add_document(conn: sqlite3.Connection, source_id: int, label: str, day: str, text: str) -> None:
    number = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    signature = signature_to_blob(MinHasher().signature(text.split()))
    conn.execute(
        "INSERT INTO documents (source_id, url, text, text_hash, label, fetched_at, doc_date, "
        "n_chars, n_tokens, signature) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source_id, f"https://d.test/{number}", text, f"h{number}", label, day + "T00:00:00Z",
         day, len(text), len(text.split()), signature),
    )  # fmt: skip


@pytest.fixture
def corpus(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Two training sources and one holdout source per label, over 20 days."""
    rng = random.Random(9)
    for name, label, holdout in (
        ("af-1", "prs", False), ("af-2", "prs", False), ("af-x", "prs", True),
        ("ir-1", "pes", False), ("ir-2", "pes", False), ("ir-x", "pes", True),
    ):  # fmt: skip
        source_id = add_source(conn, name, label, holdout)
        style = synth.make_style(rng, label)
        for day in range(1, 21):
            add_document(
                conn, source_id, label, f"2026-03-{day:02d}", synth.make_document(rng, style)
            )
    return conn


# --- features ---------------------------------------------------------------------------


def test_char_ngrams_are_padded_and_folded() -> None:
    assert list(char_ngrams("اب", 1, 2)) == [" ", "ا", "ب", " ", " ا", "اب", "ب "]
    assert list(char_ngrams("كتاب", 4, 4)) == list(char_ngrams("کتاب", 4, 4))


def test_hashed_features_match_the_readable_version_and_stay_in_range() -> None:
    from zlib import crc32

    text = "می‌خواهم ۲۵ کتاب بخرم"
    bits = 12
    expected = {
        crc32(gram.encode("utf-32-le")) & ((1 << bits) - 1) for gram in char_ngrams(text, 1, 4)
    }
    features = hashed_features(text, 1, 4, bits)
    assert set(features) == expected and len(features) == len(expected)
    assert max(features) < 1 << bits
    assert hashed_features("", 1, 4, bits) == array("I")
    # Spelling variants land on exactly the same buckets.
    assert set(hashed_features("كتاب ٢٥", 1, 4, bits)) == set(
        hashed_features("کتاب 25", 1, 4, bits)
    )


# --- splits -----------------------------------------------------------------------------


def test_split_depends_only_on_source_and_date() -> None:
    assert assign_split(True, "2020-01-01", "2026-03-15") == TEST_SOURCE
    assert assign_split(False, "2026-03-15", "2026-03-15") == TEST_DATE
    assert assign_split(False, "2026-03-14", "2026-03-15") == TRAIN
    assert assign_split(False, "2026-03-14", None) == TRAIN


def test_plan_split_picks_a_cutoff_from_training_sources(corpus: sqlite3.Connection) -> None:
    plan = plan_split(corpus, ModelConfig(test_date_fraction=0.25))
    assert plan.cutoff == "2026-03-16"  # the last 5 of 20 days
    assert (plan.first_day, plan.last_day) == ("2026-03-01", "2026-03-20")
    assert plan_split(corpus, ModelConfig(test_from="2026-03-11")).cutoff == "2026-03-11"


def test_no_cutoff_when_everything_is_one_day(conn: sqlite3.Connection) -> None:
    source_id = add_source(conn, "a", "prs")
    for _ in range(5):
        add_document(conn, source_id, "prs", "2026-03-01", "یک دو سه")
    plan = plan_split(conn, ModelConfig())
    assert plan.cutoff is None and "collect over more days" in plan.reason


def test_examples_never_cross_the_split(corpus: sqlite3.Connection) -> None:
    config = ModelConfig(test_date_fraction=0.25)
    plan = plan_split(corpus, config)
    examples = list(iter_examples(corpus, config, plan))
    by_split: dict[str, set[str]] = {}
    for example in examples:
        by_split.setdefault(example.split, set()).add(example.source)
        assert config.min_sentence_chars <= len(example.text) <= config.max_sentence_chars
    assert by_split[TRAIN] == {"af-1", "af-2", "ir-1", "ir-2"}
    assert by_split[TEST_SOURCE] == {"af-x", "ir-x"}
    assert by_split[TEST_DATE] <= by_split[TRAIN]  # later text from the same sources
    texts = [example.text for example in examples]
    assert len(texts) == len(set(texts))  # repeated sentences are used once


def test_sampling_caps_each_source_and_balancing_equalises_labels() -> None:
    examples = [Example(f"s{n}", "prs", "big", TRAIN) for n in range(500)]
    examples += [Example(f"t{n}", "pes", "small", TRAIN) for n in range(40)]
    sampled = sample_examples(iter(examples), ModelConfig(max_examples_per_source=100))[TRAIN]
    assert sum(e.source == "big" for e in sampled) == 100
    assert sum(e.source == "small" for e in sampled) == 40
    balanced = balance_labels(sampled, max_total=1000, seed=1)
    assert sum(e.label == "prs" for e in balanced) == sum(e.label == "pes" for e in balanced) == 40
    assert len(balance_labels(sampled, max_total=20, seed=1)) == 20


# --- the classifier ---------------------------------------------------------------------


def test_softmax() -> None:
    assert softmax([0.0, 0.0]) == [0.5, 0.5]
    assert softmax([1000.0, 0.0])[0] == pytest.approx(1.0)  # no overflow


def test_fit_separates_two_obvious_classes() -> None:
    bits = 10
    rng = random.Random(0)
    examples = [(array("I", rng.sample(range(0, 100), 10)), 0) for _ in range(100)]
    examples += [(array("I", rng.sample(range(500, 600), 10)), 1) for _ in range(100)]
    weights, bias = fit(examples, 2, bits, epochs=3, learning_rate=0.5, l2=1e-6, seed=1)
    model = Model(("a", "b"), bits, 1, 1, weights, bias)
    assert model.probabilities(array("I", [1, 2, 3, 4]))[0] > 0.8
    assert model.probabilities(array("I", [501, 502, 503]))[1] > 0.8
    assert model.probabilities(array("I", [900]))[0] == pytest.approx(0.5, abs=0.05)  # unseen


def test_model_file_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    weights = [array("f", [0.0] * 1024), array("f", [0.5] * 1024)]
    model = Model(("pes", "prs"), 10, 1, 4, weights, [0.1, -0.1], {"note": "x"}, "2026-01-01")
    path = tmp_path / "m.shuru"
    digest = model.save(path)
    loaded = Model.load(path, expected_sha256=digest)
    assert loaded.labels == model.labels and loaded.bias == model.bias
    assert loaded.weights == weights and loaded.metrics == {"note": "x"}
    assert loaded.predict("کتاب").probabilities == model.predict("کتاب").probabilities

    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ModelError, match="does not match"):
        Model.load(path, expected_sha256=digest)
    with pytest.raises(ModelError, match="truncated"):
        Model.load(path)
    (tmp_path / "junk.shuru").write_bytes(b"\x80\x04 pickle?\n")
    with pytest.raises(ModelError, match="unreadable"):
        Model.load(tmp_path / "junk.shuru")


def test_training_end_to_end(corpus: sqlite3.Connection, settings: Settings, run: Run) -> None:
    corpus.execute("INSERT INTO runs (id, job, status, started_at) VALUES (0, 'train', 'ok', 'x')")
    model_id, model = train_and_register(settings, corpus, run)
    metrics = model.metrics
    assert model.labels == ("pes", "prs")
    assert set(metrics["splits"]) >= {TEST_DATE, TEST_SOURCE, "train_sample"}
    for split in (TEST_DATE, TEST_SOURCE):
        accuracy = metrics["splits"][split]["accuracy"]
        # Small synthetic corpus: the claim tested is "clearly beats guessing".
        assert accuracy > metrics["baselines"]["majority"][split]["accuracy"] + 0.1, split
    assert "af-x" not in metrics["train_sources"]
    assert registry.active_model_id(corpus) == model_id

    loaded = registry.load_model(corpus, model_id, settings.paths.dir("models"))
    assert (
        loaded.predict("پوهنتون کابل").probabilities == model.predict("پوهنتون کابل").probabilities
    )
    assert score_documents(corpus, model_id, model, settings.model, run) == 120
    assert score_documents(corpus, model_id, model, settings.model, run) == 0  # already done

    export = export_corpus(
        corpus,
        settings.paths.dir("exports"),
        model_id=model_id,
        model_label="prs",
        min_confidence=0.6,
    )
    kept = corpus.execute(
        "SELECT COUNT(*) FROM doc_scores "
        "WHERE model_id = ? AND label = 'prs' AND confidence >= 0.6",
        (model_id,),
    ).fetchone()[0]
    assert export.documents == kept > 0
    everything = export_corpus(corpus, settings.paths.dir("exports"), label="pes")
    assert everything.documents == 60


def test_training_needs_two_labels(conn: sqlite3.Connection, settings: Settings, run: Run) -> None:
    source_id = add_source(conn, "only", "prs")
    for day in range(1, 6):
        add_document(
            conn, source_id, "prs", f"2026-03-0{day}", "این یک جمله آزمایشی طولانی است. " * 5
        )
    with pytest.raises(TrainError, match="at least two labels"):
        train_and_register(settings, conn, run)


def test_install_rejects_a_conflicting_file(tmp_path: Path, conn: sqlite3.Connection) -> None:
    model = Model(("a", "b"), 10, 1, 2, [array("f", [0.0] * 1024)] * 2, [0.0, 0.0])
    elsewhere = tmp_path / "pc" / "model.shuru"
    elsewhere.parent.mkdir()
    model.save(elsewhere)
    models_dir = tmp_path / "models"
    first = registry.install(conn, elsewhere, models_dir)
    assert registry.install(conn, elsewhere, models_dir) == first  # same file: no-op
    model.bias = [1.0, 0.0]
    model.save(elsewhere)
    with pytest.raises(ModelError, match="already installed"):
        registry.install(conn, elsewhere, models_dir)


# --- evaluation and baselines -----------------------------------------------------------


def test_score_predictions() -> None:
    triples = [("a", "a", "s1"), ("a", "b", "s1"), ("b", "b", "s2"), ("b", "b", "s2")]
    scores = score_predictions(triples)
    assert scores["accuracy"] == 0.75
    assert scores["per_label"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "n": 2}
    assert scores["per_label"]["b"]["precision"] == 0.6667
    assert scores["confusion"] == {"a": {"a": 1, "b": 1}, "b": {"b": 2}}
    assert scores["per_source"]["s1"] == {"n": 2, "accuracy": 0.5}
    assert score_predictions([]) == {"n": 0}
    assert confidence_histogram([0.0, 0.55, 0.99, 1.0]) == [1, 0, 0, 0, 0, 1, 0, 0, 0, 2]


def test_baselines() -> None:
    assert MajorityBaseline(["a", "b", "b"]).predict("x") == "b"
    assert MajorityBaseline(["b", "a"]).predict("x") == "a"  # ties resolve alphabetically
    markers = load_markers(REPO_ROOT / "config")
    assert "پوهنتون" in markers["prs"] and "دانشگاه" in markers["pes"]
    baseline = MarkerBaseline(markers, fallback="pes")
    assert baseline.predict("او به پوهنتون رفت") == "prs"
    assert baseline.predict("او به دانشگاه رفت") == "pes"
    assert baseline.predict("او به خانه رفت") == "pes"  # no marker: fallback
    assert baseline.predict("پوهنتون و دانشگاه") == "pes"  # a tie: fallback


def test_corpus_stats(corpus: sqlite3.Connection, settings: Settings) -> None:
    stats = corpus_stats(corpus, settings, [])
    assert stats["documents"] == 120
    assert {row["label"]: row["docs"] for row in stats["per_label"]} == {"pes": 60, "prs": 60}
    assert stats["token_sample"]["top"][0][0] == "و"
    assert "af-x (holdout)" in format_stats(stats)
