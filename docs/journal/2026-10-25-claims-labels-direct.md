# 2026-10-25: claims_labels written as its published files (E6)

The architecture trace (`2026-10-25-architecture-trace.md`, E6) found a release's
claims_labels written three times. The build writes each language's rows sorted by
(`ref`, `field`) as a group file. Compaction rewrites it with Polars' `unique` and
pyarrow's writer, and checks the rewrite by fingerprint. The sort then hashes it, reads
it, sorts it stably by `ref`, writes the part files and checks them. Yet the build's rows
are already distinct and in `ref` order, so the last two steps change nothing.

#147 plans every published-file decision from the data alone (`plan_key`, `KeyPlan`,
`row_group_rows`, `decoded_bytes`), which makes the sort's boundaries reproducible
anywhere. This change has the build write each language's published part files itself,
with the sort's own writer and checks.

## Current State

- **Master** is 5ef2147.
- **The published claims_labels files are the same bytes:** names, count, order and
  sha256, for every language. This holds on the bench's quick-tier sample, at the first
  run and with every label three times over (as the bench's run on three times the rows
  has them), and on one synthetic 8M-row language.
- **Passes removed for a language written directly:**
  - the group file's write (Polars);
  - compaction's whole rewrite: its read, `unique`, `partition_by`, Arrow conversion, its
    zstd-3 write with content-defined chunking and a page index, and its fingerprint
    check, which reads both files back;
  - the sha256 of the compacted file, twice (compaction's manifest and the sort's
    `check_compacted_copy`);
  - the sort's read, its stable sort and its take of every column.
- **What's left for such a language:** the join and its sort, one conversion to Arrow,
  linear checks of the order, the published write, `_check_written` and its sha256.
- **The bench (16.0.0)** now times claims_labels' build, compaction and sort, after the
  refs stage, apart from finalise's figures. It also keeps the first run's part files for
  parity.
- **Expected value at a release, modelled from local timings:**
  - about 2.9 CPU-µs a row;
  - about 0.5–0.8 billion rows across both sets;
  - so 24–39 CPU-min, or about 4–7 min of finalise's wall over 6 workers;
  - about 1× claims_labels' size (5–8 GB) less scratch disk at its peak, and about twice
    that less written.
- **On the bench the effect is small.** The sample's claims_labels is 199,569 rows in 518
  languages, 3.2 MB, so per-language costs dominate it.

## Design

`claims_labels.build_language` replaces `_write_language` as the build's job (stage
`written`), one language a job.

1. **The rows.** The job builds the language's rows as before (`_language_rows`: the
   refs joined with both sets' labels, sorted by `ref`, `field`).
2. **What the sort would publish** (`_published_rows`):
   - The group file's Arrow schema is derived as Polars writes it: an empty frame
     written to a buffer and read back by pyarrow (all `large_string`). Compaction's
     rewrite and the sort keep that schema.
   - The rows are converted to Arrow, cast to that schema, and combined into one chunk a
     column, as the sort's take leaves them. This matters for the bytes: pq-glue's writer
     is handed the same batches.
   - **Preconditions, else the group file as before:**
     - no null `ref` (the sort puts nulls last; Polars' sort put them first);
     - `ref` non-decreasing by Arrow's comparison;
     - a flat schema;
     - the key planned in memory.
   - **Deduplication:**
     - If (`ref`, `field`) strictly ascends, the rows are distinct: a ref is one id's, and
       an id has one label a language. This is the release's case, and it needs no pass.
     - Otherwise (the bench's run on three times the rows has each label three times),
       the job applies compaction's own `unique(keep="first", maintain_order=True)`. It
       then checks the result against Arrow's hash group-by: as many rows as the input has
       distinct rows, and none twice.
   - **The sort's check, of the identity permutation:** `_check_sort(what, t, arange(n),
     t, "ref")`. A stable sort leaves a sequence where it is exactly when the sequence is
     non-decreasing with nulls last. So this check passing shows the sort's result is
     these rows as they stand.
3. **The plan** (`sort_by_id.plan_rows`, split out of `plan_key`):
   - The job uses the rows and `footer_size`'s estimate. For a flat schema, any files of
     n rows give n values a leaf, at each leaf's nominal width (`_footer_decoded`, from an
     empty file's Parquet schema).
   - The parent passes its in-memory threshold and part size, as the sort plans in its
     parent too. The bench sets both in its own process only.
4. **The write** (`sort_by_id.write_parts`, split out of `sort_in_memory`, which now calls
   it): the sort's code for the part files, its row groups from `row_group_rows(t)`, and
   `_check_written`.
   - The part files' footers are then planned again (`plan_key` of the part files) and
     must give the same plan, so a footer count that differs is caught.
   - The parent appends the language to the sort's manifest (`sources: []`) as each job
     finishes.
5. **Afterwards nothing changes in `main.finalise`.**
   - Compaction and the sort take any fallback languages' group files as before.
   - Compaction of an empty source does nothing. `check_compacted_copy` compares an empty
     copy with an empty manifest.
   - The sort's commit, verification, metadata and local copy take every language from
     the manifest. `replace_local_copy` now creates a key directory that has no local
     files.
6. **Scheduling only:**
   - A language that may be a larger key (its labels' rows or the refs, the fewer, over
     3/16 of the threshold) runs in a pool of `FINALISE_LARGE_WORKERS`, a worker each,
     as the sort runs a larger key.
   - The rest share the run's pool, as the build's jobs did.
   - At a release this is `en` in the main set, by its labels' 48.7M rows, though its
     17.3M claims_labels rows would not make it a larger key.
   - The files are the same either way.

### How each check is kept

| check before | for a language written directly |
|---|---|
| compaction's `_check_deduplicated`: the rewrite's fingerprint against the group file's distinct rows | No rewrite to check. The rows are shown distinct (strict order), or deduplicated by the same `unique` and checked by Arrow's group-by. The written files are checked against them by `_check_written`. |
| `check_compacted_copy`: the sort's input is compaction's files, by size and sha256 | No input file. The sort's rows are the build's in memory, and the files are checked against them by `_check_written`. The files' sha256 is taken as they're written (`_entry`), and checked on a restart (`_built`) and by `verify_table` against the Hub's. |
| `_check_read`: each file read gave its listed rows | Nothing is read. The plan's rows are the rows held. |
| `_check_sort`: the sort is a stable permutation | The same function, on the identity permutation: the rows ascend by `ref`, nulls last. |
| `_check_written`: each row group read back by pyarrow, value for value | The same function, called by the same `write_parts`. |
| the plan from the input's footers | The same plan from the rows, checked against the part files' footers. |
| parity | The bench keeps the first run's claims_labels part files (`final/claims_labels`), so parity covers them (2,842 published keys, against 2,324). |

### Restart safety

- **A language's state:** it is done when its part files are listed in the sort's
  manifest and are there as listed (`_built`: name, size and sha256), or when its group
  file exists (a fallback).
- **What a rerun does:** it redoes every other language, first removing the language's
  part directory, so a part cut short or a `.tmp` left is replaced. The manifest keeps
  the latest entry a key.
- **The build's stages** (`refs`, `written`, `staged`) are as before.
- **Tested** (`scripts/test_finalise.py`):
  - a part cut short mid-write, with a temporary file left and no entry;
  - parts written but no entry;
  - a listed part changed.

  Each rebuilds only that language (7,657 rows: `en`, `de` and `fr`), to the same names
  and sha256.

## Correctness

- **The bench's quick tier, locally:** a full `bench/pipeline.py` run (KEEP=1) on each
  arm: master's `src` (5ef2147) and this branch's, the same harness.
  - **Published claims_labels:** 518 files, the same names, order and sha256.
  - **parity.py compare:** `ok`, over 2,842 published keys, claims_labels' 518 among
    them.
- **The build, compaction and sort alone** (a scratch script, on the kept labels and
  sorted claims, each arm's `src`):
  - the same 518 files by sha256 with the release's threshold;
  - the same with the bench's entity-scaled threshold;
  - the same with every label three times over, at both thresholds, and equal to the
    labels once;
  - the same at 16 times the refs and labels (each copy's ids suffixed): 3,193,104 rows.
- **One synthetic language,** 2M and 8M rows, `en`-like (random Q-ids, 4-word labels):
  the same part file, `ef0608d2…` and `918802995d6b…`.
- **`scripts/test_finalise.py`** (new section): the direct files equal the old path's by
  name, order and sha256 for 4 languages, `en` in 4 part files. Also the same:
  - with labels three times over;
  - with a key over the in-memory threshold, which is written as its group file and
    bucketed by the sort as before, while the others are direct.

  It also covers the restarts above, and refuses rows out of `ref` order and a null
  `ref`.
- **Also run:** `scripts/test_sort.py` and `scripts/test_claims_refs.py` (on the 4
  quick-tier chunks) pass.

## The CI bench (run 40234106173, 3 rounds, 6 runs an arm)

Bench 15.3.0 (renumbered 16.0.0 after, as parity changed: nothing else did), before 3046fdb (master), after 3fccfd7 (this change, rebased on it):

| figure | before | after | change |
|---|--:|--:|--:|
| claims_labels: build, compact, sort | 8.72 s ± 0.40 | 7.59 s ± 0.33 | **−1.13 s (−12.9% ± 0.7%)** |
| CPU, claims_labels | 32.3 s ± 1.9 | 28.7 s ± 1.3 | **−3.7 s (−11.2% ± 2.7%)** |
| claims_labels, 3× the rows | 8.92 s ± 0.35 | 7.97 s ± 0.35 | **−0.95 s (−10.7% ± 2.9%)** |
| CPU, claims_labels, 3× the rows | 33.1 s ± 1.8 | 30.2 s ± 1.5 | **−2.9 s (−8.6% ± 1.1%)** |
| peak RSS, claims_labels: all processes | 1.436 GB | 1.502 GB | **+66 MB (+4.6% ± 1.1%)** |
| peak RSS, claims_labels: one worker | 300 MB | 313 MB | **+13 MB (+4.5% ± 0.9%)** |

- **Parity passed,** over 2,842 published keys, claims_labels' 518 among them.
- **Every other figure** shows no change beyond noise (the release ETA 6h59m00s → 6h55m57s).
- **The peak RSS rise** is per worker, about 13 MB.
  - The build's workers now also write and read back Parquet with pq-glue and pyarrow,
    which before only compaction's and the sort's workers did.
  - It doesn't grow with the rows: on one 8M-row language the process's peak fell, 3,463
    → 2,897 MB (below).

## Measured

**Local timings are indicative only:** the machine is shared, its load average 6–29
while these ran.

**The bench's quick tier,** one run each, claims_labels' stage (wall, CPU with workers):

| | master | this branch |
|---|--:|--:|
| claims_labels, first run | 8.44 s, 29.3 CPU-s | 7.29 s, 25.7 CPU-s |
| claims_labels, 3× the rows | 8.74 s, 30.0 CPU-s | 7.49 s, 26.8 CPU-s |
| peak RSS, all processes / one worker | 1,063 / 221 MB | 1,158 / 238 MB |
| published | 199,569 rows, 3.198 MB, 518 files | the same files |

In the scratch script on the same inputs, master's compaction took 1.78 s and its sort
1.34 s. On this branch they took 0.01 s and 0.39 s: an empty compaction, and the
metadata and local copy.

**One `en`-like language, in one process, past the shared join** (CPU-s):

| rows | before: group write + compaction + sort | after: `_published_rows` + `write_parts` | per row saved |
|--:|--:|--:|--:|
| 2,000,000 | 0.88 + 4.50 + 2.04 = 7.45 | 0.65 + 1.05 = 1.70 | 2.9 µs |
| 8,000,000 | 2.74 + 14.67 + 12.34 = 29.84 | 1.30 + 4.99 = 6.29 | 2.9 µs |

Peak RSS of that process, data generation included: 3,463 → 2,897 MB at 8M rows.

## claims_labels' size

- **On the bench** (measured, each run): 199,569 rows, 3.198 MB, 518 keys and files.
  - The refs are 364,213.
  - `en` is 3,479 rows: the sample's refs mostly name entities outside it, which have no
    labels in it.
- **At a release** (modelled):
  - **The main set:** 328,137,397 rows, 3.19 GB, 618 keys
    (`docs/dataset_cards_metadata.json`, the 2026-05-07 build). `en` is 17.3M rows and
    222 MB. The release 20260928 main set's refs were 20,697,204
    (`2026-10-05-sort-speed.md`).
  - **The scholarly set's claims_labels:** not on record. Its refs (authors, journals,
    cited works) are taken as 0.5–1.5 times the main set's rows.
  - **So both sets:** about 0.5–0.8 billion rows and 4.8–8 GB.
- **At a release, modelled:** 2.9 µs a row × 0.5–0.8 billion rows is 24–39 CPU-min, about
  4–7 min of wall over 6 workers.
  - The largest language bounds it from below: `en`'s chain, 17–35M rows, is about 50–100
    CPU-s saved alone.
  - **Disk:** the old path held about two copies of claims_labels at its peak (group and
    compacted files in compaction, compacted and sorted files in the sort). This one
    holds one. So the peak is about 5–8 GB less, and about twice that less is written.

## Next steps

- **One join across all languages.** The build's per-language join scans every ref: at
  a release about 618 languages × 20.7M refs a set, roughly 20 CPU-min a set (measured at
  16 times the sample: about 300 CPU-s for 518 jobs × 5.8M refs). One join of the refs
  with every language's labels at once is the next cut in claims_labels' build. Queued.

## Negative findings and what's next

- **The bench can't show the size of the saving.** The sample's claims_labels is 518
  languages of about 385 rows each. Its CI change is about a second of wall and a few
  CPU-s.
  - The bench's run on three times the rows triples its rows only before compaction's
    deduplication: the published files are the first run's.
  - So the figure that settles it is the per-row model above, or a release.
- **The build's join is the larger cost left.** Each language's job scans every ref:
  - 518 jobs × 5.8M refs took about 300 CPU-s at 16 times the sample.
  - At a release that is about 618 × 20.7M refs a set, roughly 20 CPU-min a set at the
    same rate.
  - One join of the refs with every language's labels at once would remove it. Not done
    here: it changes the build's query, not its writes.
- **The larger-key estimate is coarse.** A language's labels' rows are 2.3–5.4 times its
  claims_labels rows at a release, and 9.1 times on the sample.
  - With the sample's entity-scaled threshold it takes 16 of the sample's languages as
    larger keys, none of them one, each paying a worker's start. (An earlier estimate,
    three rows a label row, took more, and a scratch run took 38 s against 12 s.)
  - So the bench's claims_labels stage plans with the release's settings (CHANGELOG
    16.0.0).
  - At a release it flags `en` in the main set, by its labels' 48.7M rows. The scholarly
    set's labels are not on record.
