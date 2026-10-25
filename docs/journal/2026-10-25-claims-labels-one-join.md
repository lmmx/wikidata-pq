# 2026-10-25: claims_labels joined once a batch of languages, not once a language

E6 (#160, `2026-10-25-claims-labels-direct.md`) has claims_labels' build write each
language's published part files itself. Its next step was the build's join. Each language
is a job, and each job scans every ref and joins them with that language's labels. At a
release that is 618 languages × 20.7M refs a set. This change joins the refs with a batch
of languages' labels in one query, and keeps the per-language write and its checks as
they are.

## Current State

- **Master** is a4d4937 (E6, bench 16.0.0).
- **The published claims_labels files are the same bytes:** names, count, order and
  sha256, for every language. This holds on the bench's quick-tier sample at the first
  run, with every label three times over, and at 16 times the refs and labels.
- **What a batch removes:** each language's own scan of the refs file, its own join's
  build and probe, and its own job dispatch. A batch does one scan, one join and one
  sort, then a slice per language.
- **What stays per language:** `_published_rows` (the order and dedup checks,
  `_check_sort` on the identity), `write_parts` with `_check_written` and the sha256,
  the footer re-plan check, and the manifest entry. This is the same code
  (`_build_rows`) as E6's `build_language`.
- **Measured locally** (indicative: a shared machine, load average 2–28), for the
  claims_labels stage (build, compaction, sort):
  - **the bench's sample:** 7.3–10.2 s and 25–27 CPU-s before, 2.2 s and 7.0 CPU-s
    after;
  - **with its labels three times over:** 7.4–10.4 s and 26–28 CPU-s before, 2.9–3.2 s
    and 9.2–9.6 CPU-s after;
  - **at 16 times the refs and labels:** 103–115 s and 305–315 CPU-s before, 6.7 s and
    24 CPU-s after.
- **Modelled at a release's main set:** about 18 CPU-min saved of the build's 26.5,
  and a lower modelled peak (below).

## Design

### The batch's rows (`_batch_rows`)

- **The query:** each language's labels are scanned from its directories as
  `_language_rows` scans them, and tagged `_k`, the language's place in the batch.
  Tagging by directory, rather than filtering by the `language` column, keeps a
  language's rows exactly the label rows of its own directories.
- **The join:** the batch's labels are concatenated, joined with the refs on `id`
  (inner), sorted by (`_k`, `ref`, `field`) and collected by the streaming engine.
- **Per language:** each language's rows are a zero-copy slice of that frame, found by a
  binary search on `_k`.
- **The same multiset of rows:** an inner join's pairs are (ref row, label row) with
  equal ids. So a language's pairs in the batch's join are exactly its own join's.

### The same order (`_ties_alike`)

- **The order is the same whenever no two rows tie and differ:** if no two of a
  language's rows have the same (`ref`, `field`) and differ, every sort by (`ref`,
  `field`) gives the same sequence, however it breaks ties.
  - This is a release's case: a ref is one id's, and an id has one label a language.
  - It is also the bench's case with its labels three times over: each tie is the same
    row three times.
- **The check:** the rows are sorted, so a tie is two adjacent rows of the same (`ref`,
  `field`), with nulls equal (`eq_missing`). Where there is none, nothing more is done.
  Where there is one, the rows' distinct count must equal their distinct (`ref`,
  `field`) count.
- **Otherwise the language falls back** to its own join (`_language_rows`), as before.
  `_build_rows` then takes it as `build_language` would.

### Scheduling (`_batched_builds`)

- **A batch's labels,** the join's build side and what its job holds, are at most
  `_batch_cap` rows. The cap is a larger key's estimated rows (`KeyPlan.larger`: 3/16 of
  `SORT_IN_MEMORY_BYTES` at 64 bytes a row, 25.2M) over `FINALISE_WORKERS`. That is
  4.2M at a release's settings with 6 workers.
  - So the batches running at once hold at most a larger key's rows of labels between
    them.
  - Batches are packed first-fit decreasing by label rows, into the least-loaded batch
    that fits.
  - There are at least four batches a worker. Where the languages fit in fewer, this
    makes the batches smaller, so a worker holds about what a language's job held on the
    sample. It keeps the bench's peak RSS where it was (below). At a release it changes
    nothing, as there are 63 batches.
- **A language with more labels than a batch's cap runs alone**, as before
  (`build_language`). It runs in a pool of `FINALISE_LARGE_WORKERS`, a worker each, as a
  larger key does (every larger key is among these).
  - These jobs now start only while their estimated peaks fit the memory budget
    (`in_parallel(memory=…)`, `WIKIDATA_FINALISE_MEMORY_GB` or MemAvailable).
  - The estimate is 400 MiB plus 200 bytes a label row. At 16 times the sample, a
    batch's join held 125–212 bytes a label row.
- **A language whose labels aren't all strings** (`_batchable`, from the scans' schema)
  is a job of its own in the run's pool, as before.
- **`cl.BATCHED = False`** keeps today's schedule: a job a language, the larger ones
  apart.
- **The files are the same however the languages are batched.** Only which job builds a
  language changes.

### Restart safety

- **Unchanged:** a language is done once its part files are listed in the sort's
  manifest and are there as listed (`_built`), or once its group file exists.
- **Manifest entries:** the parent appends them as each batch's job returns.
- **A job killed mid-batch** leaves the languages it wrote unlisted. The rerun forms its
  batches from the languages left and rebuilds each one, first removing its part
  directory. This is the same handling as a language cut off mid-write under E6.

## Correctness

- **The bench's quick tier, locally:** a full `bench/pipeline.py` run on master (KEEP=1)
  gave the kept labels and sorted claims. Then claims_labels' build, compaction and sort
  ran on each arm's `src` (E6's scratch script, `cl_run.py`, with a build-time RSS
  sampler). The published files were the same 518 files, by name, order, sha256 and
  size:
  - at the sample's size (199,569 rows, 3.198 MB);
  - with every label three times over (equal to the labels once);
  - at 16 times the refs and labels, each copy's ids suffixed (3,193,104 rows, 6.99 MB).
- **`scripts/test_finalise.py`** (E6's section, extended):
  - the batched files equal a job a language and the group-file path. This holds with
    every language in one batch, with `en` alone over the cap and `de`, `fr` with `xx`
    batched, and with each language alone;
  - with labels three times over, and with a key over the in-memory threshold, as
    before;
  - **a batch killed after writing two languages** (`_build_rows` raising on the third,
    so nothing is listed): the rerun rebuilds all four to the same names and sha256;
  - **an id with two labels in one language** (rows of the same (`ref`, `field`) that
    differ) is refused by the batch and taken from the language's own join. Its rows
    equal the group-file path's, by `ref`. Every other language's files are the same
    bytes.
  - E6's checks all still pass: restarts mid-part, without an entry and with a changed
    part, and the refusal of rows out of order or with a null `ref`.
- **`scripts/test_claims_refs.py`** (the 4 quick-tier chunks): the direct, batched
  build from the refs of the sidecars and from the refs of the sorted claims gives the
  same 518 files. Each language's part files hold its group file's rows in order.

## Measured (local, indicative)

The claims_labels stage, one run each. `cl_run.py` on the kept bench outputs, 4 workers,
the release's plan settings. RSS is the process tree's peak, sampled every 50 ms during
the build.

| input | figure | before (a4d4937) | after |
|---|---|--:|--:|
| sample | wall | 7.32 / 7.55 / 10.15 s | 2.13 / 2.23 s |
| sample | CPU | 25.0 / 26.2 / 26.7 s | 6.8 / 7.0 s |
| sample | build's tree peak / worker peak | 1,061–1,095 / 228–239 MB | 1,096–1,116 / 235–240 MB |
| labels 3× | wall | 7.36 / 7.87 / 10.39 s | 2.87 / 3.19 s |
| labels 3× | CPU | 26.2 / 26.3 / 27.8 s | 9.2 / 9.6 s |
| labels 3× | build's tree peak / worker peak | 1,124 / 1,125, 245–250 MB | 1,162 / 1,167, 251–260 MB |
| 16× refs and labels | wall | 103 / 115 s | 6.7 s |
| 16× refs and labels | CPU | 305 / 315 s | 23.9 s |
| 16× refs and labels | build's tree peak / worker peak | 1,945 / 656 MB | 3,011 / 730 MB |

- **The refs scans:** what is saved scales with languages × refs. On the sample, 18.7
  CPU-s over 518 languages × 364,213 refs is 0.099 µs each. At 16 times, 290 CPU-s over
  518 × 5,827,408 is 0.096 µs. So the cost is linear in the refs, as E6's estimate
  assumed.
- **Batching variants tried:**
  - **One batch a worker:** the sample's tree peak was 1,113–1,129 MB, and with labels
    three times over 1,368 MB, against master's 1,124.
  - **Two batches a worker:** 1,103–1,116 and 1,278 MB.
  - **Four batches a worker:** 1,096 and 1,162–1,167 MB, for about 0.3 CPU-s more. Kept.
- **Join variants for one batch at 16 times:** a quarter of the languages, 2.5M labels,
  in one process.

  | variant | CPU | peak RSS |
  |---|--:|--:|
  | the default build side | 1.7 s | 700–750 MB |
  | labels forced as the build side | 1.6 s | 513 MB |
  | refs forced as the build side | 2.6 s | 1,008 MB |
  | a semi-join of the labels to the refs' ids first | 3.0 s | 948 MB |

  For a whole sample's labels in one batch (9.9M): 4.9 CPU-s and 1,873 MB by default,
  6.1 CPU-s and 1,542 MB with the labels forced as the build side.
  - The default was kept. Forcing the labels as the build side held less at both sizes,
    but cost 25% more CPU at the larger one, and a batch's peak grows with its labels
    either way (about 125–212 bytes a label row by default).
  - With more than about 500 scans concatenated in one query, Polars warns that it
    reached `POLARS_MAX_BLOCKING_THREAD_COUNT` (512). Batches at a release hold about 4
    to 160 languages (by the 4.2M cap), well under that.

## At a release (modelled)

**What the model assumes:**
- the languages' label rows from `docs/dataset_cards_metadata.json` (the 2026-05-07
  build: 621 languages, 642M label rows; a release's main set is about 774M, so about
  1.2 times these);
- the main set's 20,697,204 refs;
- 0.1 µs a language a ref (measured above), 0.43 µs a label row (a batch of 9.9M labels
  at 16 times: 4.87 CPU-s, less its refs) and 50 ms a language;
- 170 bytes held a label row while its job runs (measured: 125–212);
- 6 workers, jobs in submission order, each starting as a worker frees.

**The model's figures:**

| | jobs | CPU | worker-bound wall | peak held by jobs |
|---|--:|--:|--:|--:|
| before: a job a language, in name order, 6 at once | 621 | 26.5 min | 4.5 min | 14.3 GB |
| after: 40 languages over 4.2M labels alone, 2 at once | 40 | 4.2 min | 2.1 min | 11.2 GB |
| after: 63 batches of the other 581, 6 at once | 63 | 4.5 min | 0.9 min | 4.3 GB |
| after, in all | 103 | 8.7 min | 3.0 min | 11.2 GB |

- **CPU:** about 18 CPU-min saved for the main set's build: 518 fewer scans of 20.7M
  refs at 0.1 µs. The scholarly set's build reads the same two sets' labels with its
  own refs. Its refs aren't on record. At 0.5–1.5 times the main set's, it saves
  another 9–27 CPU-min. So a release saves about 27–45 CPU-min in all.
- **Wall:** finalise runs on 4 cores, so the build is about CPU-bound. That is about 6.6
  min before and 2.2 min after per set, at 4 cores' worth.
- **Peak:**
  - **Before:** the peak is set by large languages running together in the 6-worker
    pool: `en` (48.7M labels, about 8.3 GB held) beside `es`, and others.
  - **After, alone:** the 40 largest run at most 2 at once. `en` with `nl` is 12.1 GB,
    and the memory budget now starts them only while their estimates (400 MiB plus 200
    bytes a label) fit. On a 16 GB laptop with 12 GB available, `en` (10.1 GB
    estimated) runs alone.
  - **After, batches:** at most 25.2M labels at once between them, about 4.3 GB. That is
    less than `en`'s job alone before.
  - **So the release's modelled peak falls,** 14.3 → 11.2 GB, or lower under a budget.
    Each worker's floor (about 250–400 MB) is the same in both.

## Negative findings

- **The batched peak exceeds the per-language peak in one regime:** many mid-sized
  languages and no large ones. At 16 times the sample (518 languages of 1–500k labels,
  none alone), the build's tree peak was 3,011 MB against 1,945 MB. Every batch holds
  about 1/16 of all labels, against about 1/518 a job before.
  - The cap bounds this regime at a larger key's rows of labels between the running
    batches.
  - It is not a release's case: a release's peak comes from its large languages, which
    this change runs at most 2 at once, under the budget.
- **Today's join isn't deterministic when ties differ:** when a language has two
  different labels for one id, `_language_rows` gives a different order of the tied
  rows from run to run. In 3 × 20 runs of a 4,000-ref case with one such id, there were 4
  different outputs.
  - This is pre-existing. Polars' sort isn't stable, and the streaming join's order
    varies between runs.
  - It can't happen at a release (an id has one label a language, and the two sets are
    disjoint).
  - The batch path refuses these rows and takes them from the language's own join, so
    it is exactly as before. The test compares that language by its rows.
- **The bench's sample shows the saving only as a few seconds:** about 5 s of wall and
  19 CPU-s of the stage. The figure that settles it at a release is the per-ref-scan
  model, which holds at 1 and 16 times the refs.

## Next steps

- **The scholarly set's refs and claims_labels sizes aren't on record.** A release's
  build logs (`[claims_labels] N refs`) would pin its half of the saving.
- **A batch's peak could be cut further** by forcing the labels as the join's build
  side (`build_side="force_right"`). It held 20–30% less in the one-batch tests (513
  against 700–750 MB, and 1,542 against 1,873 MB), for up to 25% more CPU at the larger
  size. That would allow a larger cap, so fewer batches. It was left for a measured
  follow-up.
