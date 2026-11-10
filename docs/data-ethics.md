# Data ethics and legal care

shuru collects text that other people wrote and published. These are the rules the code
follows, and the ones it cannot follow for you.

## What the code does

- **Asks first.** robots.txt is read for every site and obeyed, including `*` and `$`
  patterns and `Crawl-delay`. If robots.txt cannot be read because of a server error or a
  refusal, the site is not fetched that run.
- **Goes slowly.** At least 2 seconds between two requests to the same site, more if the
  site or your config asks, and one visitor per site even when several sources share it.
  Errors and "slow down" answers (429) make it back off. A site that keeps failing is left
  alone for the rest of the run.
- **Says who it is.** Every request carries the `user_agent` from `settings.toml`. Point
  it at a page that explains what you are doing and how to reach you.
- **Asks for little.** Known URLs are never fetched twice, and feeds are checked with
  conditional requests, so an unchanged feed costs the site almost nothing.
- **Keeps the data out of the repository.** The database, exports and models live in the
  data root, outside the repo. `.gitignore` is a second line of defence.
- **Keeps provenance.** Every document records its URL, source and fetch time. Every
  export carries a manifest with each source's licence note.
- **Stores headlines only** for the news feeds that are scored live.

## What you must do

- **Read each site's terms of use** before adding it as a source. robots.txt allowing a
  path is not a licence to republish what is there.
- **Record the terms** in the source's `licence` field. They travel with every export.
- **Do not redistribute** collected text unless the licence allows it. Training a model
  and publishing the corpus are different questions, with different answers by
  jurisdiction.
- **Datasets** (`kind = "file"`): keep the dataset's licence, attribution requirements and
  version in `licence`. Some, for example on Mozilla Data Collective, are free to use but
  come with specific terms per dataset.
- **Personal data.** News text names people. Do not build anything from the corpus that
  profiles individuals, and delete documents if someone asks you to.

## Being honest about the labels

The labels are claims about outlets, not about sentences ("an Afghan outlet published
this, so it is Dari"). Any model or statistic built on them inherits that assumption. When
you report results:

- say that the labels are distant and what that means (see
  [architecture.md](architecture.md#labels-distant-supervision));
- report the `test_source` score (unseen outlets) next to `test_date`, and the hand-labelled
  score if you have one;
- do not present a classifier trained this way as a judgement of anyone's language or
  identity. Dari and Iranian Persian are close varieties with shared literature and
  speakers who move between them. The tool exists so that Dari text stops being silently
  filed under another language, not to police it.
