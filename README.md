# shuru

*A small data pipeline on a mission: stopping machines from misclassifying "low-resource" languages spoken by millions.*

> The pipeline runs end to end and is tested, but it can't read from dataset APIs yet, and the Raspberry Pi
> benchmark is still to do.

Language tools get Dari wrong all the time. Usually they just call it Persian (Farsi).
Dari is spoken by millions of people in Afghanistan, and it's close to Iranian Persian,
but it isn't the same, and treating the two as one means Dari speakers get worse
translation, worse search and worse speech recognition. The same thing happens to plenty
of other languages that sit next to a bigger relative.

The fix starts with data: text that's been cleaned, deduplicated, and labelled with the
variety it's actually written in. shuru is my attempt at building that. It collects text
from sources you choose, cleans it, removes duplicates and stores it as a corpus. It can
also train a small classifier to tell two close varieties apart, use it to filter the
corpus, check it against live headlines it has never seen, and hand you the sentences it
is least sure about so you can label them yourself.

Dari (`prs`) against Iranian Persian (`pes`) is the case I'm working on, but nothing is
hard-coded to them. The sources, labels, marker words and script are all set in config
files.

*shuru* (شروع) means "beginning".

## Trying it

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/camronibra/shuru && cd shuru
make sync     # install dependencies
make demo     # run everything once on made-up data
```

The demo makes eight fake news sites full of random Persian-script words, serves them on
your own machine, and runs every command in turn: crawl, crawl again (nothing new the
second time), train, score, poll headlines, label a few, retrain, export, back up. Nothing
goes out to the internet. If you want to poke around the result afterwards, run
`uv run python scripts/demo.py --keep DIR` instead.

The words are random, so ignore the accuracy numbers the demo prints. It's there to show
the parts working together, not to prove anything about Dari.

## Using your own sources

Sources go in `config/sources.local.toml`. That file is ignored by git, so your list of
sites stays on your machine. `config/sources.toml` explains every option. A source looks
like this:

```toml
[[source]]
name = "some-af-outlet"
kind = "rss"              # or "sitemap", "listing", "file"
label = "prs"
urls = ["https://news.example.af/rss"]
content_selector = "div.article-body"   # optional
licence = "All rights reserved; private research use"
```

Every document from a source gets that source's label. It's worth marking one or two
sources per label with `holdout = true`. The model never trains on those, so they tell you
whether it works on outlets it hasn't seen, which is the number that actually matters.

Then:

```bash
uv run shuru ingest        # fetch anything new
uv run shuru stats         # what got stored, from where, and why the rest didn't
uv run shuru search کابل   # search the corpus

uv run shuru train         # train and compare against two simple baselines
uv run shuru score         # label every document with the model
uv run shuru export --model-label prs --min-confidence 0.9
```

For headlines, put feeds from *different* outlets in `config/news_feeds.local.toml` and run
`uv run shuru news`. `shuru label export` writes the least certain headlines to a
spreadsheet file. Fill in the labels, then run `shuru label apply` and `shuru label report`
to see how the model really does against your own judgement.

`uv run shuru --help` lists everything. Data lives in `~/.local/share/shuru` unless you set
`LAB_ROOT`. Settings, with an explanation for each, are in `config/settings.toml`.

## How it works

The long version is in [docs/architecture.md](docs/architecture.md). The short one:

It collects politely. It reads robots.txt, waits at least two seconds between requests to
the same site, says who it is in the user agent, backs off when a site struggles, and
never downloads a page it already has.

It treats every page as possibly hostile. It only follows links and redirects to the
source's own site, caps download sizes, and parses HTML without ever running it.

Persian-script text comes in many spellings of the same thing. Two kinds of yeh, two kinds
of kaf, three sets of digits, invisible joining characters. shuru evens all of that out
first, so a classifier can't learn "this site uses an Arabic keyboard" and call it
language. Exact and near-duplicate copies are caught too, since news sites run the same
wire stories.

Labels come from the source, not from a person. That's cheap but noisy: outlets quote each
other and carry wire copy, and a model can end up learning one outlet's style instead of
the language. To keep that honest, training and test data are split by source and by date
(never at random), and hand labels are stored separately so you can measure how much the
source labels can be trusted.

The classifier itself is deliberately simple: character n-grams and logistic regression,
written out in plain Python so you can read the whole thing. Every model is compared with
two baselines: always guessing the most common label, and counting known Dari and Persian
words.

Every run is recorded with its counts and timings, so when something is slow you can see
which step it was. Failures can send an alert to your phone through
[ntfy](https://ntfy.sh).

## Running it on a small server

It can also run by itself on a small Linux machine. I'm aiming at a Raspberry Pi 4 with
2 GB of RAM that also runs other things. `sudo ./scripts/bootstrap_server.sh` sets up a
user, installs the code and turns on the systemd timers in `deploy/systemd/`. The jobs run
at low priority with memory and CPU limits, inside a fairly tight sandbox, and are only
reachable over Tailscale.

Training doesn't happen on the Pi. Copy a backup to a proper computer, train there, and
bring the model back with `shuru model install`.

## Speed

From [bench/RESULTS.md](bench/RESULTS.md): cleaning and deduplicating a document takes
about 5 ms of CPU, while the crawler waits at least 2,000 ms between requests to a site. So
crawling is limited by politeness, not by computing power. With 20,000 documents, every job
stayed under 50 MB of memory, and training stayed under 180 MB. That's why I dropped the
plan for a Rust extension: [docs/decisions/0001-no-rust.md](docs/decisions/0001-no-rust.md)
explains it.

## Working on it

```bash
make check    # formatting, linting, type checks and tests
make fmt      # fix formatting
```

The tests don't touch the internet. The crawler tests use fake sites on your own machine.

If you want to read the code, start with `src/lab/cli.py`. Every command is a short
function there that hands off to the module doing the work, and each module begins with a
note on what it does and why.

The first version of this, a single-file scraper called friendlyscraper, is saved under
the git tag `friendlyscraper-prototype`.

## What it can't do (yet)

- Read from dataset APIs. Right now a dataset has to be downloaded and turned into a
  JSON Lines file first.
- Handle sites that only build their pages with JavaScript. Those need an RSS feed or a
  sitemap.
- Act as a general language detector. It only separates the labels you train it on, and
  it's only as good as those labels.
- Read Solar Hijri dates. When a page only gives one of those, the fetch date is used.

## Please be careful with it

Read [docs/data-ethics.md](docs/data-ethics.md) before adding a source. The short version:
follow robots.txt and each site's terms, write down the licence of anything you collect,
don't republish text you don't have the rights to, and don't treat source labels as facts
about anyone's language.

I'm not affiliated with any site you collect from. Only use this where the site's terms
and the law allow it. You're responsible for your own use, including copyright and
dataset licences. It comes as is, with no warranty. See [LICENSE](LICENSE).
