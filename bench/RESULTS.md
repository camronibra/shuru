# Benchmark results

Produced by `make bench` (`bench/bench_text.py --docs 300`; the run below used 500).
Times are CPU milliseconds per item (median of 5 repeats). The text is synthetic
Persian-script text of realistic length: about 1,540 characters and 366 tokens per
document.

## PC / cloud VM: Intel Xeon Platinum 8573C, x86_64, Python 3.11.18

| step | ms per item | what it is |
|---|---:|---|
| extract | 0.126 | HTML page to article text (selectolax, compiled C) |
| normalise | 0.223 | storage normalisation (`str.translate` and regexes) |
| fold | 0.305 | comparison folding |
| tokenise | 0.096 | regex tokeniser |
| **minhash** | **4.323** | 64-permutation MinHash and 16 LSH bands, pure Python |
| features | 1.472 | hashed character 1–4-grams of a document's first 2,000 chars |
| predict | 1.665 | features, weighted sum and softmax for one document |
| predict_headline | 0.092 | the same for one headline (what the news job does) |
| train_100 | 4.420 | one training epoch over 100 sentences |

- Ingest CPU per document (extract through minhash): **5.07 ms**.
- Polite delay between two requests to the same site: **2,000 ms** (default).
- **CPU is 0.25% of the time a site's crawl takes.** One core could keep up with
  about 390 sites crawled in parallel before CPU, not politeness, set the pace.
- Peak memory of the benchmark process: 65 MB.

### At scale: 20,000 documents, whole pipeline (same machine)

Synthetic documents loaded from six local JSONL files (`kind = "file"`), so there
is **no network delay**: this measures CPU and database work only.

| command | wall time | peak memory |
|---|---:|---:|
| `shuru ingest` (20,000 documents) | 128.6 s | 42 MB |
| `shuru stats` | 1.2 s | 40 MB |
| `shuru train` (80,000 sentences, 5 epochs) | 47.9 s | 177 MB |
| `shuru score` (20,000 documents) | 35.0 s | 47 MB |
| `shuru export` (filtered by model) | 8.8 s | 46 MB |
| `shuru backup` (108 MB database to 43 MB gzip) | 9.2 s | 44 MB |

Where the 128 s of ingest went (from `shuru runs --id 1`): minhash 72%, database
writes 15%, hashing 5%, normalise 4%, the rest under 2% each.

## Raspberry Pi 4 Model B (2 GB)

Not measured yet. To add it, on the Pi:

```bash
make bench                     # paste the output below
```

A Pi 4 core runs this kind of Python several times slower than the machine above.
Even at ten times slower, ingest CPU would be about 50 ms per document against
a 2,000 ms delay per site: still under 3% of crawl time.

## What the numbers mean

1. **Crawling is limited by politeness, not CPU.** A routine crawl spends almost all
   its time waiting between requests, which is the point of the delay.
2. **Memory is far inside the 2 GB budget.** Every job streams through the database,
   so peak memory does not grow with the corpus. Training holds its examples in
   memory: 177 MB for 80,000 sentences. It runs on the PC anyway.
3. **The one CPU-heavy step is MinHash**, and it only matters for bulk imports of
   local datasets (no delay to hide behind). 20,000 documents took 2 minutes here.
4. **No compiled extension (Rust) is needed.** See
   [docs/decisions/0001-no-rust.md](../docs/decisions/0001-no-rust.md) for what would
   change that.
