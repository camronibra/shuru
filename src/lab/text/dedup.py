"""Near-duplicate detection with MinHash and locality-sensitive hashing (LSH).

The problem: news sites republish each other's stories and wire copy with small
edits. An exact hash misses those, and comparing every new document with every
stored one does not scale.

The method, in four steps:

1. **Shingles.** A document becomes the set of its overlapping word triples
   ("a b c d" -> {"a b c", "b c d"}). Two documents are near-duplicates when their
   sets overlap heavily: Jaccard similarity = |A ∩ B| / |A ∪ B|.
2. **MinHash.** Apply a random hash function to every shingle and keep only the
   smallest value. For two sets, the chance that they share the same minimum equals
   their Jaccard similarity. Repeat with ``num_perm`` different hash functions: the
   list of minimums is the *signature*, and the fraction of positions where two
   signatures agree estimates the similarity. 64 integers stand in for the document.
3. **LSH bands.** Cut the signature into ``bands`` groups. Two documents are
   *candidates* if any whole group matches. Each (band, hash) pair is a row in an
   indexed table, so finding candidates is a handful of index lookups, whatever the
   size of the corpus.
4. **Verify.** Compare full signatures of the few candidates against the threshold.

Everything is deterministic: the same text gives the same signature on any machine
and any Python version, because signatures are stored and compared later.
"""

from __future__ import annotations

import hashlib
import sys
from array import array
from collections.abc import Sequence
from zlib import crc32

# A prime larger than any 32-bit shingle hash, for the hash family h(x) = (a*x + b) mod p.
_PRIME = (1 << 61) - 1
_MASK32 = 0xFFFFFFFF


def _derive(seed: int, role: str, index: int) -> int:
    """A stable pseudo-random integer below _PRIME, from SHA-256 rather than `random`."""
    digest = hashlib.sha256(f"shuru-minhash:{seed}:{role}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _PRIME


class MinHasher:
    """Builds signatures and band hashes with fixed parameters."""

    def __init__(
        self, num_perm: int = 64, bands: int = 16, shingle_size: int = 3, seed: int = 1
    ) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be a multiple of bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle_size = shingle_size
        # One (a, b) pair per hash function; a must not be 0.
        self._perms = [(_derive(seed, "a", i) or 1, _derive(seed, "b", i)) for i in range(num_perm)]

    def shingle_hashes(self, tokens: Sequence[str]) -> set[int]:
        """The set of 32-bit hashes of every run of shingle_size consecutive tokens."""
        k = self.shingle_size
        if len(tokens) < k:
            return {crc32(" ".join(tokens).encode("utf-8"))} if tokens else set()
        return {
            crc32(" ".join(tokens[i : i + k]).encode("utf-8")) for i in range(len(tokens) - k + 1)
        }

    def signature(self, tokens: Sequence[str]) -> array[int]:
        """MinHash signature: num_perm unsigned 32-bit integers."""
        hashes = list(self.shingle_hashes(tokens))
        if not hashes:
            return array("I", [_MASK32] * self.num_perm)
        return array(
            "I",
            [min([(a * h + b) % _PRIME for h in hashes]) & _MASK32 for a, b in self._perms],
        )

    def band_hashes(self, signature: array[int]) -> list[int]:
        """One hash per band. Equal hash in the same band = candidate pair."""
        rows = self.rows
        return [
            crc32(signature[band * rows : (band + 1) * rows].tobytes())
            for band in range(self.bands)
        ]


def similarity(a: array[int], b: array[int]) -> float:
    """Estimated Jaccard similarity: the share of positions where two signatures agree."""
    if len(a) != len(b) or not len(a):
        return 0.0
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def jaccard(a: set[int], b: set[int]) -> float:
    """Exact Jaccard similarity of two sets. Used to check the estimate in tests."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def signature_to_blob(signature: array[int]) -> bytes:
    """Serialise for the database: little-endian unsigned 32-bit integers."""
    if sys.byteorder == "big":  # pragma: no cover
        signature = array("I", signature)
        signature.byteswap()
    return signature.tobytes()


def signature_from_blob(blob: bytes) -> array[int]:
    signature = array("I")
    signature.frombytes(blob)
    if sys.byteorder == "big":  # pragma: no cover
        signature.byteswap()
    return signature
