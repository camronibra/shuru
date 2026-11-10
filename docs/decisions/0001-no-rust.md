# 0001: No Rust extension

**Status:** accepted, 2026-11-09. Replaces the earlier plan to build a Rust text-processing
extension ("textcore", PyO3 + maturin) for MinHash and n-gram hashing, conditional on
measurements.

## Context

The pipeline was designed so that a compiled extension could be added if a step turned out
to be CPU-bound on the Raspberry Pi. The plan was to measure first, and the text processing
was written in pure Python as the reference.

## Measurements

From [bench/RESULTS.md](../../bench/RESULTS.md):

- Ingest CPU per document is about **5 ms** on a PC. The polite delay between requests to
  one site is **2,000 ms**. CPU is 0.25% of crawl time. Even a Pi ten times slower would
  stay under 3%.
- Peak memory for every job is below 200 MB, against a 2 GB machine.
- The heaviest step is MinHash: 4.3 ms per document, 72% of a bulk import's time.

## Decision

Do not build the Rust extension. Remove it from the plan, the build and the docs.

A Rust component would add a second language, a compiler toolchain for ARM, binary wheels
to build and ship, and parity tests. It would make a step faster that the job spends less
than 1% of its time on.

## What would reopen this

- **Processing web archives** (for example Common Crawl), where there is no politeness
  delay and documents arrive by the million. Then MinHash and n-gram hashing become the
  pipeline's speed.
- **Bulk imports on the Pi** that take too long. Measured: 20,000 documents in about two
  minutes on a PC. Before reaching for Rust, the cheaper next steps are vectorising MinHash
  with numpy (one dependency, wheels exist for the Pi) or running bulk imports on the PC.

If it happens, the pure-Python versions in `src/lab/text/` stay as the reference, and the
test `test_signature_is_stable_across_runs_and_machines` pins the exact values a
replacement must produce.
