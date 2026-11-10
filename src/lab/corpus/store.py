"""Corpus storage: every SQL statement that touches sources, URLs and documents.

Keeping the SQL in one module means the rest of the code works with plain Python
values, and the schema can be reviewed next to the queries that use it.

Nothing here loads the corpus into memory. Lookups go through indexes, and
``iter_documents`` pages through the table a few hundred rows at a time.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from lab.corpus.clean import Prepared
from lab.db import transaction
from lab.ingest.models import Source
from lab.text.dedup import MinHasher, signature_from_blob, signature_to_blob, similarity
from lab.text.normalise import fold
from lab.text.tokenise import tokenise
from lab.timeutil import hours_ago, utc_now

# SQLite's default limit on bound parameters is 32766; stay far below it.
_CHUNK = 500
MAX_DUP_CANDIDATES = 50


@dataclass(frozen=True, slots=True)
class NewDocument:
    """A cleaned document on its way into the corpus."""

    url: str
    title: str
    published_at: str | None
    fetched_at: str
    prepared: Prepared


def sync_sources(conn: sqlite3.Connection, sources: Iterable[Source]) -> dict[str, int]:
    """Mirror the configured sources into the database. Returns name -> id."""
    for source in sources:
        conn.execute(
            "INSERT INTO sources (name, kind, label, licence, holdout, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (name) DO UPDATE SET kind = excluded.kind, label = excluded.label, "
            "licence = excluded.licence, holdout = excluded.holdout",
            (
                source.name,
                source.kind,
                source.label,
                source.licence,
                int(source.holdout),
                utc_now(),
            ),
        )
    return {row["name"]: row["id"] for row in conn.execute("SELECT id, name FROM sources")}


def filter_new_urls(
    conn: sqlite3.Connection, urls: Sequence[str], retry_after_hours: int, max_tries: int
) -> list[str]:
    """The URLs worth fetching: never seen, or failed a while ago with tries left.

    This check runs before any page is downloaded. It is the main reason a routine
    run is cheap: almost every URL in a feed is already known.
    """
    retry_before = hours_ago(retry_after_hours)
    skip: set[str] = set()
    for start in range(0, len(urls), _CHUNK):
        chunk = urls[start : start + _CHUNK]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT url, status, tries, last_tried_at FROM urls WHERE url IN ({marks})",  # noqa: S608
            chunk,
        )
        for row in rows:
            retry = (
                row["status"] == "error"
                and row["tries"] < max_tries
                and row["last_tried_at"] <= retry_before
            )
            if not retry:
                skip.add(row["url"])
    return [url for url in urls if url not in skip]


def get_http_cache(conn: sqlite3.Connection, url: str) -> tuple[str | None, str | None]:
    """(etag, last_modified) saved from the last fetch of url, or (None, None)."""
    row = conn.execute(
        "SELECT etag, last_modified FROM http_cache WHERE url = ?", (url,)
    ).fetchone()
    return (row["etag"], row["last_modified"]) if row else (None, None)


def put_http_cache(
    conn: sqlite3.Connection, url: str, etag: str | None, last_modified: str | None
) -> None:
    conn.execute(
        "INSERT INTO http_cache (url, etag, last_modified, checked_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (url) DO UPDATE SET etag = excluded.etag, "
        "last_modified = excluded.last_modified, checked_at = excluded.checked_at",
        (url, etag, last_modified, utc_now()),
    )


def find_duplicate(
    conn: sqlite3.Connection, url: str, prepared: Prepared, threshold: float
) -> tuple[str, int] | None:
    """Look for a stored document with the same or nearly the same text.

    Returns ("duplicate", id) for the same URL or an exact match on the folded text,
    ("near_duplicate", id) when a MinHash signature agrees on at least threshold of
    its positions, or None. All three checks are index lookups.
    """
    row = conn.execute(
        "SELECT id FROM documents WHERE text_hash = ? OR url = ? LIMIT 1",
        (prepared.text_hash, url),
    ).fetchone()
    if row:
        return "duplicate", row["id"]

    # Candidates: documents sharing at least one whole LSH band with this one.
    pairs = ",".join(["(?, ?)"] * len(prepared.bands))
    params: list[int] = []
    for band, band_hash in enumerate(prepared.bands):
        params += [band, band_hash]
    candidates = [
        row["doc_id"]
        for row in conn.execute(
            f"SELECT DISTINCT doc_id FROM doc_bands WHERE (band, hash) IN (VALUES {pairs}) "  # noqa: S608
            f"LIMIT {MAX_DUP_CANDIDATES}",
            params,
        )
    ]
    if not candidates:
        return None
    marks = ",".join("?" * len(candidates))
    best_id, best = 0, 0.0
    for row in conn.execute(
        f"SELECT id, signature FROM documents WHERE id IN ({marks})",  # noqa: S608
        candidates,
    ):
        score = similarity(prepared.signature, signature_from_blob(row["signature"]))
        if score > best:
            best_id, best = row["id"], score
    return ("near_duplicate", best_id) if best >= threshold else None


def insert_document(conn: sqlite3.Connection, source_id: int, label: str, doc: NewDocument) -> int:
    """Insert a document and its LSH band rows. Call inside a transaction."""
    prepared = doc.prepared
    doc_date = (doc.published_at or doc.fetched_at)[:10]
    cursor = conn.execute(
        "INSERT INTO documents (source_id, url, title, text, text_hash, label, published_at, "
        "fetched_at, doc_date, n_chars, n_tokens, signature) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id,
            doc.url,
            doc.title,
            prepared.text,
            prepared.text_hash,
            label,
            doc.published_at,
            doc.fetched_at,
            doc_date,
            prepared.n_chars,
            prepared.n_tokens,
            signature_to_blob(prepared.signature),
        ),
    )
    doc_id = cursor.lastrowid
    assert doc_id is not None  # noqa: S101 - narrowing for the type checker
    conn.executemany(
        "INSERT OR IGNORE INTO doc_bands (band, hash, doc_id) VALUES (?, ?, ?)",
        [(band, band_hash, doc_id) for band, band_hash in enumerate(prepared.bands)],
    )
    return doc_id


def record_url(
    conn: sqlite3.Connection,
    source_id: int,
    url: str,
    status: str,
    *,
    detail: str = "",
    http_status: int | None = None,
    document_id: int | None = None,
) -> None:
    """Remember what happened to a URL, so it is not fetched again."""
    now = utc_now()
    conn.execute(
        "INSERT INTO urls (url, source_id, status, detail, http_status, document_id, "
        "first_seen_at, last_tried_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (url) DO UPDATE SET status = excluded.status, detail = excluded.detail, "
        "http_status = excluded.http_status, document_id = excluded.document_id, "
        "tries = tries + 1, last_tried_at = excluded.last_tried_at",
        (url, source_id, status, detail[:200], http_status, document_id, now, now),
    )


def chars_by_source(conn: sqlite3.Connection) -> dict[str, int]:
    """Stored characters per source name, for quotas."""
    rows = conn.execute(
        "SELECT s.name, COALESCE(SUM(d.n_chars), 0) AS chars "
        "FROM sources s LEFT JOIN documents d ON d.source_id = s.id GROUP BY s.id"
    )
    return {row["name"]: row["chars"] for row in rows}


def iter_documents(
    conn: sqlite3.Connection,
    *,
    columns: str = "d.id, d.text, d.label, d.doc_date, s.name AS source, s.holdout",
    where: str = "",
    params: Sequence[object] = (),
    batch: int = 200,
) -> Iterator[sqlite3.Row]:
    """Yield documents in id order without holding more than one batch in memory.

    Uses keyset pagination ("WHERE id > last seen id LIMIT n") instead of one long
    cursor, so the caller is free to write to the database between batches.
    columns and where are SQL fragments written by this codebase, never user input.
    """
    last_id = 0
    clause = f"AND ({where})" if where else ""
    query = (
        f"SELECT {columns} FROM documents d JOIN sources s ON s.id = d.source_id "  # noqa: S608
        f"WHERE d.id > ? {clause} ORDER BY d.id LIMIT ?"
    )
    while True:
        rows = conn.execute(query, (last_id, *params, batch)).fetchall()
        if not rows:
            return
        yield from rows
        last_id = rows[-1]["id"]


def reindex(conn: sqlite3.Connection, hasher: MinHasher, batch: int = 200) -> int:
    """Recompute every document's MinHash signature and LSH bands. Returns the count.

    Needed after changing the [dedup] settings: signatures made with different
    parameters cannot be compared. Works through the corpus one batch at a time.
    """
    done = 0
    last_id = 0
    conn.execute("DELETE FROM doc_bands")
    while True:
        rows = conn.execute(
            "SELECT id, text FROM documents WHERE id > ? ORDER BY id LIMIT ?", (last_id, batch)
        ).fetchall()
        if not rows:
            return done
        last_id = rows[-1]["id"]
        with transaction(conn):
            for row in rows:
                signature = hasher.signature(tokenise(fold(row["text"])))
                conn.execute(
                    "UPDATE documents SET signature = ? WHERE id = ?",
                    (signature_to_blob(signature), row["id"]),
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO doc_bands (band, hash, doc_id) VALUES (?, ?, ?)",
                    [
                        (band, value, row["id"])
                        for band, value in enumerate(hasher.band_hashes(signature))
                    ],
                )
        done += len(rows)
