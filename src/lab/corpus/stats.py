"""Corpus statistics: what has been collected, from where, and what was thrown away.

Everything is computed by SQL aggregates or by streaming a bounded sample, so the
report costs the same memory on a small corpus and a large one.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from lab.config import Settings
from lab.ingest.models import Source
from lab.text.normalise import fold
from lab.text.tokenise import tokenise
from lab.timeutil import utc_now

TOP_TOKENS = 30
# Token frequencies come from the most recent documents only.
TOKEN_SAMPLE_DOCS = 2000


def corpus_stats(
    conn: sqlite3.Connection, settings: Settings, sources: list[Source]
) -> dict[str, Any]:
    """Collect the numbers. The result is JSON-ready."""
    totals = conn.execute(
        "SELECT COUNT(*) AS docs, COALESCE(SUM(n_chars), 0) AS chars, "
        "COALESCE(SUM(n_tokens), 0) AS tokens, MIN(doc_date) AS first_day, "
        "MAX(doc_date) AS last_day FROM documents"
    ).fetchone()
    quotas = {source.name: source.quota_chars for source in sources}

    per_source = []
    for row in conn.execute(
        "SELECT s.name, s.label, s.holdout, s.licence, COUNT(d.id) AS docs, "
        "COALESCE(SUM(d.n_chars), 0) AS chars, COALESCE(SUM(d.n_tokens), 0) AS tokens, "
        "MIN(d.doc_date) AS first_day, MAX(d.doc_date) AS last_day "
        "FROM sources s LEFT JOIN documents d ON d.source_id = s.id "
        "GROUP BY s.id ORDER BY s.label, s.name"
    ):
        entry = dict(row)
        entry["quota_chars"] = quotas.get(row["name"], 0)
        per_source.append(entry)

    # What happened to every URL tried: the share that became documents, and why not.
    outcomes: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT s.name, u.status, COUNT(*) AS n FROM urls u "
        "JOIN sources s ON s.id = u.source_id GROUP BY s.id, u.status"
    ):
        outcomes.setdefault(row["name"], {})[row["status"]] = row["n"]
    for entry in per_source:
        entry["urls"] = outcomes.get(entry["name"], {})

    per_label = [
        dict(row)
        for row in conn.execute(
            "SELECT label, COUNT(*) AS docs, SUM(n_chars) AS chars, SUM(n_tokens) AS tokens "
            "FROM documents GROUP BY label ORDER BY label"
        )
    ]
    per_day = [
        dict(row)
        for row in conn.execute(
            "SELECT doc_date AS day, COUNT(*) AS docs FROM documents "
            "GROUP BY doc_date ORDER BY doc_date DESC LIMIT 14"
        )
    ]
    reasons = {
        f"{row['status']}:{row['detail']}": row["n"]
        for row in conn.execute(
            "SELECT status, detail, COUNT(*) AS n FROM urls "
            "WHERE status IN ('rejected', 'blocked', 'error') "
            "GROUP BY status, detail ORDER BY n DESC LIMIT 20"
        )
    }

    counts: Counter[str] = Counter()
    for row in conn.execute(
        "SELECT text FROM documents ORDER BY id DESC LIMIT ?", (TOKEN_SAMPLE_DOCS,)
    ):
        counts.update(tokenise(fold(row["text"])))
    sampled_tokens = sum(counts.values())

    target = settings.corpus.target_chars
    return {
        "generated_at": utc_now(),
        "documents": totals["docs"],
        "chars": totals["chars"],
        "tokens": totals["tokens"],
        "first_day": totals["first_day"],
        "last_day": totals["last_day"],
        "target_chars": target,
        "target_progress": round(totals["chars"] / target, 4) if target else None,
        "per_label": per_label,
        "per_source": per_source,
        "per_day": per_day,
        "not_stored_reasons": reasons,
        "token_sample": {
            "documents": min(totals["docs"], TOKEN_SAMPLE_DOCS),
            "tokens": sampled_tokens,
            "distinct": len(counts),
            "top": counts.most_common(TOP_TOKENS),
        },
    }


def write_report(stats: dict[str, Any], reports_dir: Path) -> Path:
    """Save the statistics as reports/stats-YYYY-MM-DD.json (one file per day)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"stats-{stats['generated_at'][:10]}.json"
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def format_stats(stats: dict[str, Any]) -> str:
    """A terminal-friendly summary."""
    lines = [
        f"documents: {stats['documents']:,}   characters: {stats['chars']:,}   "
        f"tokens: {stats['tokens']:,}   dates: {stats['first_day']} to {stats['last_day']}"
    ]
    if stats["target_progress"] is not None:
        lines.append(
            f"target: {stats['target_chars']:,} characters ({stats['target_progress']:.1%} reached)"
        )
    lines += ["", f"{'label':<10}{'docs':>9}{'chars':>14}{'tokens':>12}"]
    for row in stats["per_label"]:
        lines.append(f"{row['label']:<10}{row['docs']:>9,}{row['chars']:>14,}{row['tokens']:>12,}")
    lines += [
        "",
        f"{'source':<26}{'label':<8}{'docs':>8}{'chars':>13}{'kept':>7}  not stored",
    ]
    for row in stats["per_source"]:
        urls = row["urls"]
        tried = sum(urls.values())
        kept = f"{urls.get('stored', 0) / tried:.0%}" if tried else "-"
        rest = " ".join(f"{k}={v}" for k, v in sorted(urls.items()) if k != "stored")
        name = row["name"] + (" (holdout)" if row["holdout"] else "")
        quota = f" quota {row['chars'] / row['quota_chars']:.0%}" if row["quota_chars"] else ""
        lines.append(
            f"{name:<26}{row['label']:<8}{row['docs']:>8,}{row['chars']:>13,}{kept:>7}  "
            f"{rest}{quota}"
        )
    if stats["not_stored_reasons"]:
        lines += ["", "why URLs were not stored:"]
        lines += [f"  {reason:<34}{n:>7,}" for reason, n in stats["not_stored_reasons"].items()]
    sample = stats["token_sample"]
    if sample["tokens"]:
        top = "  ".join(f"{token} ({count})" for token, count in sample["top"][:15])
        lines += [
            "",
            f"most frequent tokens in the latest {sample['documents']:,} documents "
            f"({sample['distinct']:,} distinct of {sample['tokens']:,}):",
            f"  {top}",
        ]
    return "\n".join(lines)
