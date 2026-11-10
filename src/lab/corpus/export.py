"""Export the corpus as a snapshot file for use elsewhere (for example model training).

Format: JSON Lines, one document per line, gzip-compressed. It is written as a
stream, so the export never holds more than one batch of documents in memory.

Next to the data file sits a manifest: when it was made, which filters applied,
how many documents per source and label, each source's licence note, and the
SHA-256 of the data file. A snapshot without that context is hard to trust later.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.timeutil import utc_now

BATCH = 200


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    manifest_path: Path
    documents: int
    chars: int


def export_corpus(
    conn: sqlite3.Connection,
    out_dir: Path,
    *,
    label: str | None = None,
    model_id: int | None = None,
    model_label: str | None = None,
    min_confidence: float = 0.0,
) -> ExportResult:
    """Write a snapshot of the corpus.

    Filters, all optional:

    - ``label``: keep documents whose *distant* label (from the source) is this.
    - ``model_id`` with ``model_label`` and/or ``min_confidence``: keep documents the
      model labelled ``model_label`` with at least that confidence. This is how the
      classifier is used as a filter: for example "only what the model is at least
      90% sure is Dari". Documents the model has not scored are left out.
    """
    use_model = model_id is not None and (model_label is not None or min_confidence > 0)
    filters = ""
    if label:
        filters += " AND d.label = :label"
    if use_model:
        # Unscored documents have no doc_scores row, so these conditions drop them.
        filters += " AND x.confidence >= :min_confidence"
        if model_label:
            filters += " AND x.label = :model_label"
    # The filters are fixed strings built above; every value is a bound parameter.
    query = (
        "SELECT d.id, s.name AS source, d.label, s.licence, d.url, d.title, d.published_at, "
        "d.fetched_at, x.label AS model_label, x.confidence AS model_confidence, d.text "
        "FROM documents d JOIN sources s ON s.id = d.source_id "
        "LEFT JOIN doc_scores x ON x.document_id = d.id AND x.model_id = :model_id "
        "WHERE d.id > :last_id" + filters + " ORDER BY d.id LIMIT :batch"
    )
    params: dict[str, object] = {
        "label": label,
        "model_id": model_id,
        "model_label": model_label,
        "min_confidence": min_confidence,
        "last_id": 0,
        "batch": BATCH,
    }

    stamp = utc_now().replace("-", "").replace(":", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"corpus-{stamp}.jsonl.gz"
    temporary = path.with_name(path.name + ".tmp")
    per_source: dict[str, int] = {}
    per_label: dict[str, int] = {}
    licences: dict[str, str] = {}
    documents = chars = 0

    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        while True:  # keyset pagination: one batch in memory at a time
            rows = conn.execute(query, params).fetchall()
            if not rows:
                break
            params["last_id"] = rows[-1]["id"]
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                documents += 1
                chars += len(row["text"])
                per_source[row["source"]] = per_source.get(row["source"], 0) + 1
                per_label[row["label"]] = per_label.get(row["label"], 0) + 1
                licences[row["source"]] = row["licence"]
    temporary.replace(path)

    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "file": path.name,
        "sha256": _sha256(path),
        "documents": documents,
        "chars": chars,
        "filters": {
            "label": label,
            "model_id": model_id if use_model else None,
            "model_label": model_label if use_model else None,
            "min_confidence": min_confidence if use_model else None,
        },
        "per_label": dict(sorted(per_label.items())),
        "per_source": dict(sorted(per_source.items())),
        "licences": dict(sorted(licences.items())),
        "note": (
            "Labels are distant: each document carries the label of its source. "
            "Check each source's licence and terms before redistributing."
        ),
    }
    manifest_path = path.with_name(path.name.replace(".jsonl.gz", ".manifest.json"))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ExportResult(path, manifest_path, documents, chars)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
