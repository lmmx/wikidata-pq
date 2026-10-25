# 2026-10-25: Each bucket sorted into its part file, no packing pass

The claims key (any key over `SORT_IN_MEMORY_BYTES`) was sorted in three passes:
bucketing, a sort of each bucket to a scratch file (`final-*.parquet`, the Hub's
settings), and packing. Packing ran once every bucket was sorted. It glued consecutive
sorted buckets into the key's part files (`pq_glue.concat`, a worker each) and compared
each part file with its buckets byte for byte (`_check_glued`). At a release that is 667
bucket files (about 45 GB) written to scratch, read back and copied into 86 part files,
the bucket files kept on disk until the table's `done` stage, after its upload.

Scripts are in `2026-10-25-sort-into-parts/`.

## Why the pass could not go while keeping today's part files

Packing's assignment (`_pack`) cuts the sorted buckets where their cumulative *encoded*
size is nearest each multiple of `total / n`, with `n = ceil(total / COMPACT_FILE_BYTES)`.
Every cut and `n` itself depend on `total`, the sum of every bucket's encoded size. That
is known only once the last bucket is encoded. So no bucket can be placed in its part
file before every bucket is sorted, and until then each bucket's bytes must be kept
somewhere, on disk at a release (45 GB). Moving them into part files then is a copy:
today's pass.

Byte identity of the published files was therefore not compatible with removing the
pass, nor with appending buckets to part files as they finish. The user then authorised
decoupling the sort's thresholds from encoded sizes (the planning functions are owned by
sort/decoupled-thresholds). With the part files planned before the sort, the pass can go.

## The change

- **The plan comes first.** `sort_bucketed` takes each part file's buckets (consecutive,
  every bucket once) before any bucket is sorted. Until the planning interface lands,
  `_part_plan` stands in for it: `_pack`'s rule (unchanged) applied to each bucket's
  share of the key's *source* bytes, by its rows counted. So `ceil(source bytes /
  COMPACT_FILE_BYTES)` part files, at most one a bucket, cut nearest even rows.
- **A part file of one bucket is written by that bucket's sort.** `_sort_bucket` writes
  straight to the part file, with its sha256, and checks it against the sorted rows
  (`_check_written`, unchanged). There is no scratch file, glue or glue check: the part
  file *is* the bucket's sorted file. `pq_glue.concat` of one file is that file byte for
  byte (checked), so this is the file packing made.
- **A part file of several buckets is glued as soon as its buckets are sorted.** Each is
  still sorted to a scratch file and checked. The parent glues the part (`_pack_file`:
  `pq_glue.concat` and `_check_glued`, unchanged) on a thread while the other buckets
  sort, or on its own thread once none is left. The glue reads pages just written, and
  the bucket files are then removed. So the key's scratch holds only the buckets of
  part files not yet glued.
- **No packing jobs.** There are no workers started for them, and no pass after the
  sorts.

## Checks kept

| check | where |
|---|---|
| `_check_sort` | every bucket's sort, unchanged |
| `_check_written` | every bucket's file, unchanged: a part file of one bucket is checked itself, row for row |
| `_check_glued` | every part file of several buckets, against its buckets' bytes, unchanged |
| the bucket checks (`_read_bucket`, `bucket_key`) | unchanged |
| a glued part file's rows | the sum of its buckets' rows sorted (new, cheap) |
| `_check_sorted` (order within and across part files) | unchanged, after the last part file |
| `_check_whole` | unchanged in substance: each bucket's rows sorted against those bucketed, and their sum against the sources' |
| `check_compacted_copy` | unchanged (before the sort) |

## Restart safety

- **`files.jsonl`** lists each part file once it is made and checked. Each entry has its
  buckets' names and whether it was glued. A restart reuses a listed part file only if
  its size, its sha256 and its buckets match the plan, and then does not sort its
  buckets again. An entry from before this change has no `glued` field and counts as
  glued, so a part file of one bucket packed by the old code is sorted again, not
  reused.
- **`sorted.jsonl`** lists each bucket sorted to a scratch file, as before. A restart
  stopped after a part file's buckets were all sorted, but before it was glued or listed,
  glues it again from them, sorting nothing.
- **The bucket files are removed only after their part file is listed.** A run stopped
  between the listing and the removal leaves bucket files, which the restart removes.
- **A bucket's fragments are removed once it is sorted, as before.** Until now its
  scratch file was kept until `done`, so a lost part file could be glued again. Now a
  part file's buckets are removed once it is listed. So a bucket that must be sorted
  again after that point has its fragments spilled again (`_spill_again`). That is a
  part file changed after it was listed, or a different plan on a restart. Each source
  file with a fragment of such a bucket is bucketed again (`_bucket_source`), its row
  groups planned exactly as before (checked: rows, row groups, direct buckets and which
  fragments exist). The new entry is appended to `bucketed.jsonl`, with the digests of
  the fragments now written, and the fragments of buckets not to be sorted are removed.
- **Every step is deterministic,** so a restart from any point makes the same files.

`scripts/test_sort.py` covers each case on a sample (4 files, 145k rows, 21 buckets,
part files of one bucket and of several, 2 workers):
- a sort interrupted by a bucket's failure: the parts made before it are kept, 2 of
  them glued while other buckets sorted, and only the failed bucket is sorted again;
- a glued part file not listed: glued again from its sorted buckets, the same bytes,
  nothing sorted again;
- a glued part file not listed once its buckets were removed: its buckets' fragments
  are spilled again and the buckets sorted again, the same bytes, nothing else sorted;
- a part file of one bucket not listed: sorted again, the same bytes;
- a finished sort reused whole;
- the glue check catching a part file that is not its buckets' bytes;
- all of the test's earlier cases.

## Byte identity, given the same plan (local)

`compare.py` sorts test_sort's sample (20,000 rows a slice: 145,388 rows, 21 buckets) with
master's code. It then sorts it again with this branch's code, given master's assignment
(read from master's `files.jsonl`). The run used 2 workers.

| | part files (buckets each) | rows | bytes | sha256 |
|---|---|---|---|---|
| master | 3 (7, 7, 7) | 48,466, 48,466, 48,456 | 3,727,761, 4,086,368, 4,094,330 | |
| this branch, master's plan | 3 (7, 7, 7) | the same | the same | all 3 identical |
| this branch, its own plan (`_part_plan`) | 4 (5, 5, 6, 5) | 34,632, 34,656, 41,487, 34,613 | 2.65–3.36 MB | (other boundaries) |

- **Given the same assignment, the part files are byte for byte master's.**
- **The stand-in plan differs on this sample** because its source files (pyarrow's
  defaults, small row groups) are larger than the sorted output. So `ceil(source /
  COMPACT_FILE_BYTES)` gives one part file more.
- **On the bench the plans agree:**
  - the first run's claims key: 1 part file of 2 buckets;
  - the 3× run: 4 part files of one bucket each. The bench sets the target at about 0.35
    times a source file, so 9 part files are asked for, at most one a bucket.
- **So the bench's published files should be byte-identical to master's.** A part file
  of one bucket is the bucket's own file, which packing copied unchanged.

## What it saves

**On the bench (measured on CI, below):**
- **The 3× run:**
  - no packing step (4 workers, their starts, the copy and the glue check);
  - no scratch copy of the 4 part files, so the disk's peak and the bytes written
    should fall by about the claims key's sorted bytes.
- **The first run:** one packing worker fewer. The glue runs in the parent.

**At a release (modelled, not measured; no release-scale disk run exists):**

| | master | this change |
|---|---|---|
| bucket files on scratch at the peak | all 667, ~45 GB, kept until `done` (past the upload), beside the 86 part files (~45 GB) | the buckets of part files not yet glued: about a part's buckets plus the workers', ~(8 + 6) × 67 MB ≈ 1 GB |
| buckets read back | ~45 GB, after all the sorts, mostly evicted from a 15 GB page cache | the same bytes, read while still cached |
| the glue check's reads | ~90 GB (parts and buckets) | the same, mostly cached |
| packing as a pass | after the sorts, on 6 workers, disk-bound | overlapped with the sorts, but for the last part file |
| packing workers | 86 a set, ~0.6 CPU-s each | none |

- **Scratch disk:** about 44 GB less at the peak, per claims key. That matters on an
  ordinary laptop: the claims' sources, bucket files and part files were ~135 GB at
  once, and are now ~90 GB.
- **Disk reads:** up to ~90–135 GB fewer, at 200–500 MB/s about 3–11 minutes of
  disk-bound wall, and what is left is overlapped with CPU-bound sorting.
  #131's profile put packing's copy and check at "maybe 2–7m", unmeasured.
- **CPU:** the packing workers' starts, about 1 CPU-minute a set, ~15 s of wall on 4
  cores.
- **The bench's model of a release's workers** (bench/release.py,
  `RELEASE_CLAIMS_WORKERS`) still counts 86 packing workers a set. Taking them out is a
  bench change, under 30 s of the modelled finalise ETA.

### What a fuller removal would need

The copy itself stays for part files of several buckets (~8 at a release).
- **Why it stays:** pq-glue's writer collects its whole input before writing
  (`write_parquet` takes the stream's batches first). So one part file can't be written
  from several buckets sorted in turn without holding them all in memory: a part is
  ~500 MiB of Parquet, ~8× a bucket's 2.7–3.3 GB peak.
- **What would remove it:** an appending writer in pq-glue (row groups encoded and
  written as they come, the footer at close), with one job a part file sorting its
  buckets in turn. That would remove the remaining scratch writes, the glue and its
  check.
- **Its cost:** 86 jobs of ~8 buckets each, against 667. The tail and the memory budget's
  granularity coarsen, so it is not clearly a gain over the overlapped glue.
- **Status:** not tried; it needs a pq-glue release.

## Bench (bench 12.0.0, measured)

Run 40213269487: before is master fa959e4, after is 8c10ddb (this branch with master
merged in). Quick tier, 3 rounds, paired. A change is shown with its 95% interval; one
whose interval holds zero is "noise".

| figure | before | after | change |
|---|---|---|---|
| measured end to end, CPU (split + pipeline + finalise) | 2m21s | 2m18s | noise |
| measured end to end, wall | 52.71s | 51.29s | noise |
| **ETA: finalise (modelled)** | 2h46m15s | 2h41m30s | **−4m45s (−2.8% ± 2.6%)** |
| ETA, a release (modelled) | 9h00m27s | 8h48m17s | noise (the split's noise dominates) |
| CPU, finalise (first run) | 1m33s | 1m31s | noise |
| **finalise, 3× the rows, wall** | 51.83s | 49.50s | **−2.33s (−4.7% ± 3.9%)** |
| **disk used, peak rise** | 653 MB | 620 MB | **−33 MB (−5.0% ± 0.2%)** |
| device bytes written | 1.116 GB ± 0.298 | 0.797 GB ± 0.035 | noise |
| peak RSS, finalise 3×, all processes | 9.21 GB | 9.17 GB | noise |
| peak RSS, finalise at a release, modelled | 20.62 GB | 20.70 GB | noise |
| `compact._check_glued`, per byte | 1.6s | 1.5s | −7.0% ± 2.3% |
| the 3× run's packing step | +0.37s wall, +2 CPU-s | gone | |
| the claims key's workers (first run, 3×) | 4, 11 | 3, 7 | |

- **Parity passed.** Every key's published rows, order and schema are the same. The part
  files are the same count (1 and 4), as the plans agree on the bench. Parity
  fingerprints rows, not file bytes. The bytes are master's by construction (a part
  file of one bucket is its sorted file, which `concat` copied unchanged), and were
  checked locally (above).
- **The finalise ETA's −4m45s** is the per-byte part losing the 3× run's packing jobs'
  own CPU: the copy and the byte check of 4 part files, beyond their workers' starts,
  which the model already takes out. Its interval only just excludes zero.
- **It did not replicate.** A second run on the docs-only head 4dbe949 (run 40214230188,
  the same code) gave:
  - the finalise ETA 2h44m37s → 2h42m55s, noise;
  - the 3× run's wall −1.53s (−2.7% ± 2.4%);
  - the disk's peak −32 MB (−5.0% ± 0.1%);
  - parity passed.

  So the 3× wall and the disk's peak hold across both runs. The ETA's gain is real in
  direction but within noise.
- **The disk's peak** is over the whole bench run, not the claims' sort alone. So its
  −33 MB is what is left after the peak elsewhere in the run.
- **What the bench can't show:** a release's 45 GB of bucket files and their re-reads. It
  has no release-scale disk run, and the 3× run's part files are one bucket each.

## Judgement

**Merge candidate.** It meets the bar:
- parity passes;
- the 3× run's wall (−4.7% ± 3.9%, then −2.7% ± 2.4%) and the disk's peak (−5.0%, in both
  runs) improve beyond noise. The finalise ETA did so in one run of two (−2.8% ± 2.6%);
- nothing regresses.

At a release its main value is disk: ~44 GB less scratch at the peak per claims key, and
packing's ~90–135 GB of disk-bound reads turned into page-cache hits overlapped with the
sorts. That is modelled, not measured.

**It depends on the planning interface.** `_part_plan` is a stand-in, to be replaced by
sort/decoupled-thresholds' plan when it lands. sort_buckets takes any plan of
consecutive buckets.
