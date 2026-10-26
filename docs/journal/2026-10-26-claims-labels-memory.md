# 2026-10-26: claims_labels' large languages, joined a row group of their labels at a time

The queue's item "claims_labels' join build side (memory only)": claims_labels' jobs were
modelled at 11.2 GB at a release (#164), the largest memory figure of a release, and the
binding one on a 16 GB laptop. This change is #186.

## Current State

- **The change:** a language built alone (`build_language`) is joined with the refs a
  row group of its labels at a time (`_streamed_rows`), and holds about its rows, not its
  labels.
- **The published files are the same bytes:** by SHA-256 in every case tested (below).
- **Modelled at a release** (bench 17.11.0's model, measured locally at 1/16 and 1/8):
  `en`'s job 8.85 → 2.05 GB, `en`'s and `nl`'s at once 11.43 → 5.18 GB (−55%). Through
  1/8 and 1/4, the fractions the bench takes: `en`'s 6.0 → 3.2 GB (−46%). CI's figures are
  in the PR.
- **Unchanged:** the batches (#164's 4.3 GB at a release), the checks, restart safety,
  the global sort by `ref`.

## Where the memory went

- **What is joined:** the set's refs `(field, ref, id)` (the main set's 20.7M at a
  release) with each language's labels `(id, language, value)` from both sets' local
  copies, inner on `id`, sorted by `(ref, field)`.
- **Which jobs:** a language of more labels than a batch holds (`_batch_cap`: 4.2M at a
  release's settings, 40 languages) is a job of its own, FINALISE_LARGE_WORKERS (2) at
  once; the rest are batched. The alone jobs set the peak.
- **What a job held:** about every label of the language. The join's output columns are
  Polars string views, which point into the buffers of the label rows they came from.
  A gather or join copies only the 16-byte views. So the joined rows kept every label
  row group with a match alive until they were turned into Arrow (`_published_rows`).
  Profiled on 1/8 of `en` (6.1M labels, 2.6M refs, 2.2M rows, `/proc` sampling by phase),
  the peak was in the join:

  | join | peak RSS |
  |---|--:|
  | default build side | 1,461 MB |
  | labels forced as the build side | 1,976 MB |
  | refs forced as the build side | 1,547 MB |
  | a semi-join of the labels to the refs' ids first | 1,327 MB |
  | each row group joined with its refs' id range (rows kept as Polars views) | 1,427 MB |
  | the same, each row group's rows copied out to Arrow | 1,141 MB |

  The build side changes little: the labels are held whichever side is built.
- **How it scales:** the peak grows with the language's labels of both sets. `en` at a
  release has the main set's 58.5M (the cards' 48.7M times 1.2) and, assuming one a
  scholarly entity, the scholarly set's 46.4M: about 105M. Its rows are 17.3M.
- **#164's 11.2 GB was a hand model:** 170 B a label row times the main set's labels
  only, for `en` beside `nl`. The bench's new model gives master 11.43 GB for the same
  pair with both sets' labels counted, so the two happen to agree.

## The change (`_streamed_rows`)

- **Refs:** read a field at a time by the streaming engine (`_refs_by_field`). Each field
  keeps its ids, ascending as collect_refs_stage writes them, plus its `ref`s only where
  they are not its ids (the units). The refs are never held whole.
- **Labels:** read a row group at a time, file by file, in the order and from the files
  Polars' glob reads (checked by their rows).
- **Each row group** is joined (Polars, inner on `id`, as `_language_rows`) with the refs
  of its ids' range `[min, max]`, found by binary search in each field's ids.
  - Every ref of an id in the row group is in that range, and every label row is in one
    row group. So the row groups' pairs are exactly the join's.
  - This doesn't rely on the labels being sorted (a test has them reversed). Sorted, the
    ranges are narrow; unsorted, they are wide, slower but still exact.
- **The rows are copied out to Arrow as each row group is joined** (`to_arrow`, large
  strings, compact), so they hold only the labels they took.
- **The sort:** by `(ref, field)` (`pc.sort_indices`), taken a column at a time: each
  column made one array, its parts freed, then taken. So at most the rows and one column
  more are held. Combining the whole table first, as the first version did, held twice
  the rows.
- **The order is the join's** wherever no two rows of the same `(ref, field)` differ. That
  is a release's case: an id has one label a language. Ties of identical rows (the bench's
  labels three times over) are the same rows in any order. Ties that differ are refused,
  as `build_batch` refuses them.
- **Taken by the whole join, as before** (`_language_rows`), where the inputs aren't as
  above:
  - a field's ids out of order;
  - a null in the refs;
  - the refs or labels not strings (including an empty labels file, whose columns have
    no type);
  - ties that differ.
- **The rows go on** to `_build_rows` as an Arrow table. `_published_rows` takes them as
  they are, as `to_arrow` gives a frame of the same rows, and makes a Polars frame only to
  deduplicate. Then the plan, `write_parts`, `_check_written` and the manifest, as before.
- **The memory budget's estimate** for a job built so is 1 GiB plus 200 B ×
  `min(refs, labels)`. The old 400 MiB plus 200 B a label is kept for a job that takes the
  whole join. On a 16 GB laptop with 12 GB available, `en` and `nl` are now estimated at
  5.2 GB each and run together. The old estimate put `en` at about 21 GB, so it ran with
  nothing beside it.
- **`cl.STREAMED = False`** restores the whole join.

## Hypotheses tested

| hypothesis | result |
|---|---|
| build on the labels side / the refs side | no gain: the labels are held either way (table above) |
| pre-filter the labels to the referenced ids (semi-join) | −9% at 1/8, more CPU |
| dictionary or categorical keys | not tried: the keys (`id`) are small inline views; the bytes are the labels' values |
| a merge join over sorted inputs (Arrow, `index_in` on each row group) | the same peak as the row-group join (1,151 MB), slower (Arrow's string hash) |
| the join a row group at a time (`_streamed_rows`) | **kept** |
| reading only the matching rows of a row group (Polars' prefiltered decode, `is_in`) | lower transient (728 → 460 MB at 1/4) but 3.4× the read time: rejected |
| smaller batches of languages | not this: the alone jobs are one language each |

- **The fixed part** is a row group being read and joined: about 0.5–0.9 GB for a
  128 MiB row group, Polars' decode and the allocator's free pages. It is the same at any
  scale.
- **Holding the rows apart from Polars' allocator** (copied into pyarrow's pool, or
  written to an IPC file during the loop) moved the peak by −6% to +10% at 1/8. Not kept.

## The bench (17.11.0, `bench/labelsjob.py`)

- **Why a new step:** the sample's claims_labels stage batches every language (none has
  over 4.2M labels), so the bench never ran an alone job. Its claims_labels RSS rows
  measure batches, which this change doesn't touch.
- **What the step does:** builds `en` at 1/8 and 1/4 of a release's inputs and `nl` at
  1/4, each by the commit's own `build_language` in a process of its own (VmHWM, CPU,
  wall). The inputs:
  - the main set's labels (the cards' × 1.2);
  - the scholarly set's (`en`: one a scholarly entity, assumed; `nl`: none, assumed);
  - the main set's 20.7M refs, and the cards' rows of each language (refs with a label);
  - the labels' values the sample's own `en` labels of each set (main 19 B,
    scholarly 95 B on average);
  - ids random Q-ids, each set sorted by id in row groups of 128 MiB, as the sort writes
    them;
  - refs with a label drawn from both sets' labels. That gives the rows more scholarly
    titles than a release's main-set refs likely name, so it errs against this change.
- **Modelled:**
  - `en` on the line through its two fractions;
  - `nl` with `en`'s fixed part;
  - the two at once, as FINALISE_LARGE_WORKERS runs them at most.
- **The model's limit:** the peak isn't quite linear. At 1/4, measured directly, master
  held 2.48 GB and this change 1.22 GB. The line through 1/16 and 1/8 gives 2.88 and
  1.05 GB there: 16% over master, 14% under this change. So the step takes 1/8 and 1/4
  (its first version took 1/16 and 1/8, the table below). Through 1/8 and 1/4, `en` at a
  release is 6.0 GB before and 3.2 GB after (−46%). The whole's figures compare two
  commits more than they forecast a release.
- **Parity:** each job's files hold a plain eager join's rows, in order, and both arms'
  files are byte-identical (`labels_job_parity_ok`).

Measured locally by the step's first version, at 1/16 and 1/8 (a shared machine, load
3–8: memory is reliable, times are not):

| figure | before (master) | after |
|---|--:|--:|
| peak RSS, `en` at 1/16 | 1,383 MB | 794 MB |
| peak RSS, `en` at 1/8 | 1,881 MB | 878 MB |
| peak RSS, `nl` at 1/8 | 1,097 MB | 1,012 MB |
| CPU, `en` at 1/8 | 11.9 s | 11.2 s |
| wall, `en` at 1/8 | 6.7 s | 10.9 s (under rising load) |
| modelled, `en` at a release | 8.85 GB | 2.05 GB |
| modelled, `nl` at a release | 2.58 GB | 3.13 GB |
| modelled, both at once | 11.43 GB | 5.18 GB |
| modelled, `en`'s CPU at a release | 101.5 s | 89.7 s |

- **`nl` modelled higher after** (3.13 GB): its labels are few beside its refs (27M
  against 20.7M), so the old join held little more than the new one. And its single
  128 MiB row group of short labels (3.4M rows) costs this change more in its fixed part
  than `en`'s. The pair falls regardless, by `en`.

## Correctness

- **`scripts/test_finalise.py`**, its claims_labels sections, all pass. Locally the
  claims-file section at its end needs a release's claims. New checks:
  - each language built alone gives the same files as with its labels joined whole;
  - the same with the labels (out of order) in row groups of 300 rows;
  - each language's rows equal its join's rows, in order;
  - with the labels three times over, the same files batched and alone;
  - the fallbacks (a field's ids out of order, a null ref, labels not strings, ties that
    differ) are taken by the join.
- **The bench step's parity**, locally: the same SHA-256s for both arms at every
  fraction, and each job's rows a plain join's.

## Trade-offs

- **Wall:** a job now joins a row group at a time, so it uses fewer threads than the
  whole join's streaming engine. Its CPU is about the same, but its wall is closer to its
  CPU time. Locally at 1/4 of `en` it was 15.9 s against 15.4 s; at 1/8, under rising
  load, 10.9 against 6.7 s.
  - At a release the alone phase is about 2–3 min a set, outside the release ETA
    (`link.CLAIMS_LABELS_BUILD`).
  - With the jobs this small, running more of them at once (under the memory budget)
    would take the wall back. That is not done here.
- **Memory:** the build's modelled peak falls from ~11.4 GB to ~5.2 GB. It is now set by
  `en` beside `nl`, about as high as the batches (#164: 4.3 GB). On a 16 GB laptop it is
  no longer the release's largest figure.

## Uncertainty

- **The scholarly set's labels by language** aren't on record. `en`'s 46.4M is an
  assumption, and the more it is, the more master held. This change's peak doesn't
  depend on it.
- **The refs' share naming scholarly works** isn't on record. This change's rows grow
  with the labels they take.
- **The line through two fractions** overstates both commits at the whole (above).
