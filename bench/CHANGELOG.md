# Changelog

The benchmark harness (`bench/`, `.github/workflows/bench.yml`): how wikidata-pq is
measured, not what is measured. Changes to wikidata-pq's code or dependencies (a newer
polars-genson, say) are what the benchmark measures, and never change its version.

Its version is in `bench/pyproject.toml`, and every result records the version that
measured it. Results compare only within a major version and tier: a change to what a tier
measures or how (its workload, the timing, parity) is a new major version; a new tier,
or a change to how results are shown or stored, is a minor or patch version. Each version's entry says exactly what changed.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
harness follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [13.1.0] - 2026-10-25

### Added

- **claims_labels' refs stage** (`claims_labels.collect_refs_stage`), run as a release's
  finalise runs it once the claims are sorted: after each finalise run (the sample's rows,
  and three times them), on its sorted claims (`pipeline.claims_refs`). It is timed apart
  from finalise, between markers of its own (`claims refs`, `claims refs x3`), so it is in
  none of the figures measured before.
  - **New figures:** its wall time and CPU time (this process and its workers) in each run,
    and its peak RSS (the process tree's and the largest worker's) in each.
  - **Its time at a release, modelled** as finalise's part per byte is (`refs_model`): the
    run on three times the rows less the first, a byte of the claims key's source, times
    the release's claims bytes (both sets', claims_model's) over the cores, with the first
    run's wall time less that as a fixed part.
  - **The end to end with it:** the measured split + pipeline + finalise total, and the
    release ETA, each with the stage added (`measured_cpu_refs`, `measured_wall_refs`,
    `eta_refs`).
  - **Why:** the stage reads every sorted claims file again (2 workers, about 7 GiB each at
    a release). lmmx/wikidata-pq#139 takes the refs from sidecars written with the claims
    partition files, and the bench ran no claims_labels, so it could not see that.
- **A commit whose claims partition files have refs sidecars** (`claims_refs.py`): the
  split's sidecars are kept with its parts (`keep_parts`), and with any the pipeline wrote
  (a part left as JSON) are linked into the run's work directory before each refs run.
  Each is linked as many times as its part is in the sorted claims (the parts' cycles; 3
  times that for the run on three times the rows), each copy under a name of its own (a
  sidecar's name is its part's content, and the union reads every file there).
- `wikidata.claims_refs` (`write_refs`, `sidecar_refs`) as stages of phases.py, on their
  thread's clock as the split's are: the sidecars' write appears in the split's table of
  where the CPU goes, out of `write_claims`' own time.

Every figure measured before is measured as it was, hence the minor version.
## [13.0.0] - 2026-10-25

### Changed

- **The split's parity compares a built part's claims partition file by its content**, not
  its bytes: the same rows in the same order, the same schema with its metadata, and the
  same row groups' rows as the file the commit's own claims call
  (`process.normalise_claims_direct`) makes of the files' claims for that part's rows.
  - **Why:** a split that writes the file a row group at a time, so as never to hold a
    part's claims whole, encodes each row group on its own (its own dictionaries and
    pages). 12.0.0's byte check fails such a file while its content is the pipeline's.
  - **What it still catches:** any changed, missing, extra or reordered row, any change of
    schema, metadata or row group boundaries. The published files are checked as before
    (`parity.py`), and their content is unchanged.
  - Everything else is as in 12.0.0.

## [12.0.0] - 2026-10-25

### Changed

- **The pipeline runs on the split's own output**, not on the bench's chunk files. The
  split (split-dump and route-release, timed as before) writes its parts of a dump slice
  made of the tier's files, and the pipeline's chunks are those parts: the main set's then
  the scholarly set's, each set's in its manifest's order, cycled as the files were.
  - **Why:** a split that does part of the pipeline's work (the claims' decode, writing
    each part's claims partition file and leaving the part without its claims column)
    could not be measured on the files. The pipeline of a release reads what its split
    wrote; the bench's now does too.
  - **The workload:** the parts are cut where the split cuts them (10,000 entities a
    chunk, then routed), not where the files end:
    - quick: c0, r50, s30, s90 are 4 split chunks, and 5 parts: c0 and r50 (main), and
      3,000, 10,000 and 3,220 entities of s30 and s90 (scholarly). So 5 chunks, not 4;
      with groups of 4, two groups.
    - full: the 11 files are 6 split chunks and 7 parts (4 main, 3 scholarly), cycled
      6 times: 42 chunks, not 66.
    - The entities are the same, and so the ETA's entity scale. Each chunk's source size
      (for the groups' thresholds) is its part's file and its claims file, as
      `dump.chunk_sizes` counts a release's.
  - **A chunk's set** (for the claims' scales and the tables' figures) is the set the
    split routed its part to, not read from a file's name.
- **The split's parity checks the decoded claims.** Each set's parts must hold the
  sample's rows, every column as the files have it, as before. A part without a claims
  column must have a claims partition file. That file must be, byte for byte, the one the
  commit's own claims call (`process.normalise_claims_direct`) makes of the files' claims
  JSON for that part's rows, in this process, so at the split's thread count. Its cost is
  outside every figure.
- **The split's claims are timed:**
  - `claims_build.write_claims` and `claims_arrays`, and `dump.route_parts`, on their
    threads' clocks, as the split's other stages are;
  - `process.place_built_claims` (the pipeline's step for a built part) counts as the
    pipeline's claims (`CLAIMS_CALLS`).

### Kept

- Every figure and model as in 11.0.0, its `*_9` and `*_10` figures included. 11.0.0's
  ETA model is unchanged, so 12.0.0 has no `*_11` set. What changes is what the pipeline is
  given: a commit whose split writes JSON parts gives the pipeline the same rows as the
  files, in other chunks.

## [11.0.0] - 2026-10-25

### Changed

- **The pipeline's worker starts are a fixed cost**, not scaled by entities, as bench
  9.0.0 did for finalise's workers. `pool.process_chunks` spawns a worker process for
  every `CHUNKS_PER_WORKER` chunks. A spawned worker's start (the interpreter, the spawn
  and the package's imports, `wikidata.main`'s among them) is 0.6-0.7 CPU-s (0.62 ± 0.08
  on run 40208739011). So a
  release's starts go with its chunks. The bench's 4 chunks start 4 workers, and bench
  10.0.0's model scaled their CPU time by entities (3,667×): that is 14,668 starts. A
  release's 23,035 chunks start 2,884 (below).
  - Measured: each chunk worker's CPU time outside its chunks, from the workers
    themselves. That is its CPU time at its exit (`phases.track_process`, as its atexit
    handlers run) less its chunks' (`bench_work`'s, process and partition).
  - Modelled (`pipeline.starts_model`): that time is taken out of the part scaled by
    entities. A release's count is added back at the bench's mean. Everything else is as
    in 10.0.0:

    release pipeline CPU = 10.0.0's − the workers' CPU outside their chunks × entity
    scale + the release's starts × their mean; pipeline ETA = wall × that / CPU

  - A release's starts (`pipeline.pool_starts`): `process_chunks` replayed with chunks
    of equal length, in each set's chunks (`release.RELEASE_CHUNKS`: 12,182 and 10,853,
    measured), at the arm's `CHUNKS_PER_WORKER` (8) and the ETA's cores (4): 1,524 + 1,360
    = 2,884. Chunks of unequal length can make up to `workers − 1` more a set (the last
    workers each take fewer than 8): 2,884-2,890, under 0.3%. On the bench it counts 4,
    as measured. On the full tier it counts 12, where 10-12 were measured (run
    40198052042), the bound being 9-14.
- **The dump slice is written as the dumps write their lines**, not in orjson's form
  (`dumpslice.dump_escaped`, the re-escaper of `scripts/test_reshape.py` with PHP's
  floats):
  - every non-ASCII character as `\uXXXX` (a surrogate pair beyond the BMP);
  - `/` as `\/`;
  - floats as PHP's `json_encode` writes them (`serialize_precision` −1): the shortest
    digits, in exponential form when the decimal point is more than 3 places left of the
    first digit or more than 17 right of it (`2.77778e-5`, `1.0e+20`, where orjson writes
    `0.0000277778` and `1e+20`). An integral float in fixed form, which PHP writes as an
    integer, is left as orjson wrote it, so the row it reads back to is the sample's.

  Of a line's bytes, 13.1% (c0), 9.8% (r50), 15.2% (s30) and 17.4% (s90) are in strings
  that carry escapes (the first 3,000 lines of each), and the lines are 3-4% longer.
  The bench overstated any split change that copies spans verbatim, since orjson's form
  needs no unescaping (#127: 0.41 of master's CPU on the slice, 0.52 on real lines). The
  split's parity is as before: the chunks hold exactly the sample's rows. Checked
  locally on r50 and s30 through `dump.entity_row`, and on CI through the compiled
  reshape.

Bench 10.0.0's figures are kept on the same runs:
- `pipeline_eta_10` and `release_pipeline_cpu_10`: every start scaled by entities;
- `split_eta_10`: the split on 10.0.0's slice (`dumpslice.orjson_path`), run after the
  timed run and outside its figures (`orjson_split_*`);
- `eta_10`: both together.

### Added

- **Measured, the sample end to end**, nothing modelled: the split's (split-dump and
  route-release), the pipeline's and finalise's first run's CPU time (`measured_cpu`) and
  wall time (`measured_wall`), summed. These are the first two rows of the table, with a
  line under the score giving each stage's CPU time. Work moved from one stage to
  another (the claims' decode from the pipeline into the split, say) leaves this total
  where it was.
- **The ETA by stage**, under the ETA: each stage's measured CPU time on the sample, how
  it is scaled to a release, and its rate, the ETA's seconds a CPU-second of the
  sample's. The pipeline's rate is given part by part: its claims stages (by set), its
  workers' starts and the rest. A CPU-second moved from one part to another changes the
  ETA by the difference of their rates. On run 40208739011: the split 327, the pipeline's
  claims 365 (main) and 610 (scholarly), its rest 1,142, finalise's part per byte 330
  (claims) and 917 (the rest).
- Measured: the pipeline's workers (`pipeline_workers`, `pipeline_worker_cpu`, a row of
  the table, `pipeline_worker_startup_cpu`, `pipeline_workers_counted`); each chunk's
  start, end and process CPU time at its start (`t0`, `t`, `pcpu0` in the chunk log).
- Measured, in the raw results only: the pipeline's CPU time by table and set, with
  each table's rows and partition files (`pipeline_tables`, `pipeline.tables_measured`).
  A call's table is the one it names: its own argument, now also its `dst_dir` (the
  pipeline's `partition_parquet` and `place_unsplit` name it only there), or the
  pipeline's claims calls. Otherwise it is the table of the innermost call enclosing it.
- Modelled: `release_pipeline_workers`, `release_pipeline_workers_cpu`,
  `pipeline_workers_cpu_scaled_10` (what 10.0.0 counted for them),
  `release_chunks_per_worker`.

What the pipeline ETA measures has changed (and so the release ETA). So has the split's
workload, the slice's bytes (and so its figures and its ETA). Hence the major version.
Every other measured figure is measured as before.

### Measured with it

Quick tier, PR #130's run 40208739011 (3 rounds, 12 runs: EPYC 9V45 ×6, EPYC 7763 ×4,
EPYC 9V74 ×2; both arms' `src/` the same). Both models were computed by the harness in
each run. Each figure is the mean over the 12 runs, ± its 95% interval:

| | bench 10.0.0's model (same runs) | bench 11.0.0 | change |
|---|---|---|---|
| ETA: pipeline | 4h15m ± 33m | 3h37m ± 29m | −38m ± 4m (−14.9% ± 0.4%) |
| ETA: split | 2h04m ± 16m (orjson's slice) | 2h08m ± 15m (the dumps' form) | +4m ± 3m (+3.6% ± 2.8%) |
| ETA, a release | 8h45m ± 1h10m | 8h11m ± 1h06m | −34m ± 6m (−6.4% ± 0.6%) |
| the pipeline's starts at a release, CPU | 2h31m (14,668 starts) | 30m (2,884 at 0.62 CPU-s) | |

On the slice in the dumps' form, the split takes 6.2% ± 2.8% more CPU time (23.4 s
against 22.1 s) and 7.7% ± 3.0% more wall time. The slice's bz2 is 2.5% larger (79.2 MB
against 77.3 MB), so its ETA, per byte, rises 3.6% ± 2.8%. The orjson slice's split runs
second, after the timed run (as the split from gz does), so the change may carry a
little of what running second does.

The same pipeline model on the PR's first run (40207799475, 12 runs) gave −14.6% ± 0.4%.
Applied offline to #128's run 40205337787, it gave −14.3% ± 0.3%, against the "perhaps
−20%" that 10.0.0 estimated.

Measured, the sample end to end, on run 40208739011 (the mean ± its 95% interval):
- CPU: 125.2 ± 12.8 s (the split 23.4, the pipeline 19.3, finalise 82.5);
- wall: 46.0 ± 5.0 s.

Neither moved beyond noise between the arms. Parity passed, including the split's on
both slices and the split from gz's. No measured figure moved beyond noise between the
identical arms.

### Left alone

- **The other tables' weights** (labels, descriptions, aliases, links: about 4.5 CPU-s
  of the pipeline's 22, 3.4 of them the main set's). Their CPU time goes with their
  partition files far more than their rows. On the full tier (run 40198052042, each of
  the 11 chunk files' CPU time by table against its files and rows, fitted), each table
  costs 0.32-0.72 ms a partition file and 0.8-1.9 µs a row, with 0-0.03 s a chunk. A
  release's partition files a chunk are not on record. With a release's main-set chunks
  taken as the bench's routed ones (`r10`-`r90`: 195-416 files a table) and its
  scholarly ones as the `s` files, and the main set's rows the 20260507 build's (labels
  642M, descriptions 1,551M, aliases 172M, links 98.8M, × 75.4 / 73.8 million entities),
  these tables would take 8,800-10,700 CPU-s at a release, against the 16,200 that
  entities give them: −10% to −14% of the pipeline ETA. With `c0`'s files a chunk (564
  for labels) it is +2,000 CPU-s: +4%. The sign rests on a count not on record, so the
  weights stay. A release's audit sidecars hold its rows and files by table and chunk.
  Committing their totals would settle it.
- **`merge_group` by table** (1.1 CPU-s, about 7% of the pipeline's CPU time, scaled by
  entities). Its glued tables run at once, a thread each, and each glue's work runs on
  pq_glue's own threads. A table's CPU time can't be told apart without running them
  one at a time, which would change what is measured.
- The pipeline's first chunk on a worker costs about 0.02-0.03 CPU-s more than the same
  file's later chunks (full tier, the main set's files, 9-15 first chunks each). That
  is lazy set-up within the job. It stays in the part scaled by entities: about 350
  CPU-s, 0.6% of the pipeline's at a release.

## [10.0.0] - 2026-10-25

### Changed

- The pipeline's and finalise's claims parts are scaled to a release by the claims' own
  rows and bytes, not by entities. The sample's entities carry about 2.8 times a
  release's claims each (73.1 MB of claims partition for 33,220 entities, against about
  96 GB for 121.8 million: below), almost all of it chunk `c0`'s, the dump's first
  10,000 entities, with 62 statements an entity against the release's main set's 10. The
  claims are 57% of the pipeline's CPU time and 85% of finalise's part per byte, so
  scaling them by entities overstated both ETAs.
  - **The pipeline:** each set's claims CPU time (`normalise_claims_direct` and
    `place_unsplit`, every stage they call included) is scaled by the release's claims
    rows in that set over the sample's; every other part of its CPU time by entities, as
    before; and the pipeline ETA is the sample's wall time scaled by its CPU time so
    reckoned over its own:

    pipeline ETA = wall × (Σ_set claims CPU_set × release rows_set / rows_set + (CPU −
    claims CPU) × entity scale) / CPU

  - **finalise:** its part per byte (bench 9.0.0's) is split by table, and its claims
    share is scaled by the release's claims bytes over the sample's claims key; the rest
    by entities, as before; its fixed part and the release's claims workers' starts as in
    9.0.0:

    finalise ETA = 9.0.0's + part per byte × the sample's bytes × claims share ×
    (release claims bytes / sample claims bytes − entity scale) / the cores

  Bench 9.0.0's figures, every part scaled by entities, are kept as `pipeline_eta_9`,
  `finalise_eta_9` and `eta_9`, and shown under the ETA.

### Added

- Measured, in each run: the claims' work by set (`pipeline.claims_measured`): each
  chunk's claims CPU time (its `normalise_claims_direct` and `place_unsplit` calls'
  own CPU time with every call they make, from the phases log), its claims partition's
  rows and bytes (its audit sidecar) and its chunk file's bytes, summed by set
  (`pipeline_claims_{cpu,rows,bytes,source}_{main,scholar}`; the bench data's `s` files
  are the scholarly set's). `pipeline_claims_cpu`, their sum, is a row of the table.
- Measured: finalise's stage CPU time by table (`phases.table_cpu`: a call's table is
  the one it names, or that of the innermost call enclosing it in its process), and the
  claims' share of its part per byte (`finalise_claims_share`).
- Modelled from them (`pipeline.claims_model`): the release's scholarly claims rows
  (`release_claims_rows_scholar`), the release's claims bytes (`release_claims_mb`), the
  three scales (`claims_rows_scale_main`, `claims_rows_scale_scholar`,
  `claims_bytes_scale`) and the pipeline's CPU time at a release (`release_pipeline_cpu`).
- `release.RELEASE_SOURCE_BYTES`: each set's chunk files, measured.

What the pipeline and finalise ETAs measure has changed (and so the release ETA), hence
the major version. Every measured figure is measured as before.

### The release's claims, from the record

Measured figures are 20260928's unless said; the bench's are measured on the bench
data's 11 chunk files (each processed and partitioned alone, as `bench_work` does).

| | main set | scholarly set | source |
|---|---|---|---|
| entities | 75,432,640 | 46,383,002 | measured (`route.done`; docs/journal/2026-10-03-release-20260928-run.md) |
| chunk files (Parquet, zstd 9) | 12,182, 44.7 GB | 10,853, 49.5 GB | measured (`data/manifest.jsonl`, by `release_eta.py`, before any chunk of the main set was processed) |
| claims group files | 45.1 GB | not on record | measured (`release_eta.py`'s compaction progress, docs/guide/monitoring.md) |
| claims, compacted | 44.7 GB in 30 files, 3,769 row groups | not on record | measured (docs/journal/2026-10-05-sort-speed.md), and pinned by its 667 buckets of 64 MiB of source to 44.69-44.76 GB |
| claims, sorted | 86 part files of 500 MiB (part-00 529 MB) | not on record | measured (likewise) |
| claims rows | 780 million, modelled (774,255,243 measured in the philippesaade build of 20260507, 73,769,737 entities: 10.5 an entity) | 999 million, modelled (below) | docs/dataset_cards_metadata.json, bench 7.4.0 |
| claims bytes, modelled | (44.7 GB measured) | 51.8 GB (below) | |
| the other tables | not on record for 20260928; the 20260507 build's main set: labels 642.4M rows (6.49 GB), descriptions 1,550.9M (5.46 GB), aliases 172.0M (1.38 GB), links 98.8M (1.32 GB), claims_labels 328.1M (3.19 GB) | not on record | docs/dataset_cards_metadata.json (that build's claims, 17.7 GB, had fewer fields, so only its rows carry over) |

The two 44.7 GB figures, the main set's chunk files and its compacted claims, are both
right: neither is a copy of the other. The chunk files' total was recorded on 2026-10-03,
two days before the claims were compacted; the claims' is pinned by the bucket count
(`ceil(bytes / 64 MiB)` = 667) and the 86 part files. They agree because the pipeline's
claims partition is about the size of its chunk file: 0.91-1.11 times it in each of the
bench's 11 files (c0 0.91, r10-r90 0.92-1.11, s10-s90 0.98-1.05). No constant changes.

The scholarly set's claims are modelled from its chunk files' bytes (measured), at the
sample's scholarly chunks' (s30 and s90 on the quick tier) claims rows and bytes a byte
of chunk file: 20.2 rows a kB and 1.046 bytes a byte. Over the 5 scholarly chunk files
those are 17.9-20.4 rows a kB and 0.98-1.05 bytes a byte, so 886-1,012 million rows
and 48.4-51.8 GB.

### Where the claims' work is

Quick tier, run 40201381756 (bench 9.0.0, 3 rounds, 12 runs), each stage's own CPU time
by table, the mean over the runs. A call is the claims' if it runs within
`normalise_claims_direct` or `place_unsplit`; `partition_parquet` partitions every table
but the claims (they are written in their partition's layout already, and linked):

| stage | CPU | claims |
|---|--:|--:|
| `normalise_in_schema` | 7.90s | 7.09s (90%) |
| `write_table` | 5.63s | 5.63s (100%) |
| `partition_parquet` | 3.15s | 0 |
| `merge_group` | 0.76s | not apart (the claims are about 72% of its bytes) |
| the rest of the stages | 1.01s | 0.16s |
| outside the stages (the workers' starts, mostly) | 3.95s | |
| all, the pipeline's CPU | 22.4s | 12.9s (57%) |

| finalise, per byte (3× the rows less the first) | claims | descriptions | links | labels | aliases | entities |
|---|--:|--:|--:|--:|--:|--:|
| CPU | 38.2s (85%) | 2.2s | 2.0s | 1.8s | 1.0s | 0.1s |

The claims' CPU time goes with their rows. Each chunk file's claims CPU time on the full
tier (run 40198052042, 8 runs, each file 6 times a run), against its claims rows, fitted
(± a standard error): on the main set's 6 files 0.015 ± 0.034 s a chunk + 9.21 ± 0.13 µs
a row; on the scholarly set's 5, 0.27 ± 0.20 s + 10.5 ± 1.2 µs. Against entities it is 0.07 ms (r50) to 0.60 ms
(c0) an entity, and against the claims' bytes 0.13-0.30 s a MB. So the pipeline's claims
are scaled by rows, set by set; finalise's, whose part per byte is bytes written,
decoded and checked, by bytes.

| scale, sample to release | quick tier |
|---|--:|
| by entities (bench 9.0.0, every part) | 3,667× |
| the main set's claims rows (780M / 665,335) | 1,172× |
| the scholarly set's claims rows (999M / 509,777) | 1,960× |
| the claims key's bytes (96.5 GB / 73.0 MB) | 1,322× |

### Measured with it

Quick tier, PR #128's run 40205337787 (3 rounds, 12 runs: EPYC 7763 ×10, EPYC 9V45 ×2;
both arms' `src/` the same), both models computed by the harness in each run, each figure
the mean over the 12 runs, ± its 95% interval:

| | bench 9.0.0's model (same runs) | bench 10.0.0 | change |
|---|---|---|---|
| ETA: pipeline | 7h20m ± 41m | 5h02m ± 27m | −2h18m ± 13m (−31.3% ± 0.3%) |
| ETA: finalise | 6h06m ± 31m | 2h58m ± 15m | −3h08m ± 15m (−51.5% ± 0.3%) |
| ETA, a release | 16h13m ± 1h28m | 10h46m ± 59m | −5h26m ± 29m (−33.6% ± 0.2%) |

The claims were 55% of the pipeline's CPU time (13.0s of 23.7s) and 81% of finalise's
part per byte (±0.4 points; finalise now has #124's memory budget). Parity passed, and
no measured figure moved beyond noise between the identical arms.

The same, offline, on #119's run 40201381756's 12 runs (bench 9.0.0's harness; this version's model
applied to their phases logs and claims, as the harness now does in each run), each figure
the mean over the 12 runs, ± its 95% interval:

| | bench 9.0.0's model (same runs) | bench 10.0.0 | change |
|---|---|---|---|
| ETA: pipeline | 6h59m ± 39m | 4h41m ± 23m | −2h18m ± 16m (−32.7% ± 1.1%) |
| ETA: finalise | 6h04m ± 47m | 2h49m ± 21m | −3h15m ± 25m (−53.6% ± 0.5%) |
| ETA, a release | 15h54m ± 1h47m | 10h21m ± 1h06m | −5h32m ± 41m (−34.8% ± 0.6%) |

The intervals are the runners'. The weights' own uncertainty is systematic: varying
them, one at a time, moves the change by up to 2 points:

| varied | pipeline | finalise | a release |
|---|--:|--:|--:|
| as shipped | −32.7% | −53.6% | −34.8% |
| the main set's rows 774M or 792M | −32.8%, −32.6% | | −34.8%, −34.7% |
| the scholarly set's rows 886M or 1,012M | −34.5%, −32.5% | | −35.5%, −34.7% |
| the scholarly set's bytes 48.4 GB | | −54.6% | −35.2% |
| finalise's claims by rows, not bytes | | −49.1% | −33.1% |
| the pipeline's claims by bytes, not rows | −34.4% | | −35.5% |

So a release's ETA falls by a third, 33-36% whichever weights are taken, against a 0.6%
interval between runners.

### What it leaves scaled by entities

Each likely overstated too, and not changed here: the figures for a release that would
weight them are not on record, or not measured apart.

- **The other tables' stages** (5.6s of the pipeline's CPU, 15% of finalise's part per
  byte). The sample has about 3 times a release's labels an entity and 15 times its
  sitelinks (mostly c0's), but a release's chunks are small (3.7 MB on the main set) and
  these stages' fixed costs a chunk (0.05-0.12 s a table, from its partition's files)
  count for more there. With the full tier's fits and the 20260507 build's main-set rows,
  they would take about 6,400 CPU-s off the release's 55,000: about −12% of the pipeline
  ETA, resting on an earlier build's rows and no measured scholarly rows.
- **The pipeline's CPU time outside its stages** (3.95s, 18%), mostly its 4 workers'
  starts. A release starts about 2,900 (23,035 chunks, 8 a worker), against the 14,700
  that entity scaling stands for: perhaps −20% of the pipeline ETA, as bench 9.0.0 did for
  finalise's workers.
- **`merge_group`**: its claims glue is not timed apart from the other tables'.
- The pipeline's ETA is its wall time scaled, as before: the sample's wall is mostly c0's
  (the largest chunk) and the group close alone; a release's chunks are many and even.

## [9.0.0] - 2026-10-25

### Changed

- The finalise ETA takes finalise's worker starts as a release has them, not per byte.
  A finalise job that holds much memory runs in a spawned worker of its own
  (`parallel.in_parallel`, `tasks_per_child=1`: claims compaction, bucketing, each bucket
  sort, each part file, each larger in-memory key), and its pool starts another as each
  exits, job left or not. A worker's start (the interpreter and its imports, Polars,
  pyarrow and the package) is about 0.6-0.9 CPU-s. The run on three times the rows starts
  more of them than the first, so 8.0.0's part per byte, scaled to the release's bytes,
  counted their starts thousands of times over (below). Now each run's CPU time less
  every worker's CPU time outside its jobs (from the workers themselves, below) gives
  the part per byte, and the release's claims workers are added back at the mean CPU
  time a claims worker took outside its jobs on the bench:

  finalise ETA = the first run's wall + (part per byte × its bytes × (scale − 1)
  + the release's claims workers × that mean − the first run's claims workers' time
  outside their jobs) / the cores

  Every other key's workers (larger keys, the sort's light pool, the bench's own
  bucketed labels keys) stay in the fixed part, per key as before. Bench 8.0.0's figure
  is kept as `finalise_eta_8`, shown beside it.
- `finalise_cpu_per_mb`, `finalise_fixed` and `finalise_x3_wall_fit` are this model's
  (8.0.0's part per byte as `finalise_cpu_per_mb_8`).

### Added

- Each process of a run notes its start and its exit in the phases log
  (`phases.track_process`, from `phases_site`): its pid and parent's, the time, its CPU
  time so far and, at exit (as its atexit handlers run, before the interpreter's own
  teardown), its peak RSS (VmHWM). Each outermost timed call (a worker's job) also notes
  its process's CPU time at its start (`pcpu0`) and its own CPU time in all (`cpu`).
- From these, each finalise run's workers by kind (`timeline._workers`: its first job's
  function and table, or idle): how many, their jobs, their CPU time outside their jobs
  (`overhead_cpu`: at exit less their jobs'; `startup_cpu`: to the first job) and their
  largest own peak RSS at exit, in the raw results (`timeline`, each run's
  `processes`). The claims key's workers (`claims_workers`, `_x3`) and their mean CPU
  time outside their jobs (`claims_worker_cpu`, a row of the table) are figures.
- Finalise's peak RSS at a release, modelled (two rows, labelled so): the largest
  bucket-sort worker's own peak at exit against its run's largest bucket's rows
  (`bucket_worker_rss_mb`, `_x3`; `sort_bucket_rows_max`, `_x3`), on the line through the
  two runs (runs under 20% apart in rows: in proportion to the larger's), at a release's
  rows a bucket (`release_bucket_rows`), as `release_bucket_worker_rss_mb`; and
  `release_finalise_rss_mb`, the process tree's with `FINALISE_WORKERS`' default (6) of
  them at their peak at once and the pipeline's process's own peak over the bucket sorts
  of three times the rows. What they assume is below.
- Every 30 s each run writes its stage (the last marker in its phases log) and the
  lowest memory available since its last line to the job's log (`ab.Heartbeat`).
- A round job whose results do not come in is named in the summary and the comment
  (`lost.py`, in the finish job, which now reads the run's jobs and their logs, or where
  a log cannot be had their annotations: `actions: read`, `checks: read`): the run it
  was in, its last stage and memory, and how it ended, a runner lost mid-run (or its
  step ended by SIGTERM, exit code 143) taken as out of memory, as far as can be told. Before, the full
  tier's lost runners left the summary as "1 round of 3 (pending)", or "No round has
  both its halves", saying nothing of the jobs lost.

### Fixed

- The finish job's comment step runs when no round came in (it was skipped after the
  summary failed), so the comment says the run failed, and which jobs were lost.

What the finalise ETA measures has changed (and so the release ETA), hence the major
version. Every measured figure is measured as before.

### Measured with it

Quick tier, PR #119's run 40198030334 (3 rounds, 12 runs: EPYC 7763 ×8, EPYC 9V45 ×4;
both arms' `src/` the same), each figure the mean over the 12 runs, ± its 95% interval:

| | bench 8.0.0's model (same runs) | bench 9.0.0 | change |
|---|---|---|---|
| finalise ETA | 6h38m ± 1h01m | 5h52m ± 50m | −46m ± 13m (−11.2% ± 1.9%) |
| finalise's part per byte (CPU-s a MB) | 0.258 ± 0.039 | 0.226 ± 0.032 | −12.1% ± 1.9% |
| the release's claims workers' starts in the ETA | (in the part per byte) | 3m12s ± 22s (1,300 at 0.59 CPU-s, 4 cores) | |

By runner: −19 to −35 min on the EPYC 9V45, −38 to −77 min on the EPYC 7763. The fit
through the two runs gives the run on three times the rows 48.1 s against its 49.9 s
measured (8.0.0's: 3-5% likewise). Every measured figure is as before: the arms differ
beyond noise only in the throughput (−3.7% ± 3.4%) and the peak RSS of the largest
worker on three times the rows (−0.5% ± 0.3%), between identical pipelines.

A worker's CPU time outside its jobs is almost all its start: 0.59 ± 0.07 CPU-s a claims
worker to its first job, 0.0005 after its last. On the first run, finalise's workers'
CPU time outside their jobs is 46 of its 89 CPU-s (101 workers): its fixed part is
mostly worker starts.

The full tier (run 40198052042, 6 workers, 66 chunks; 4 of its 6 jobs came in, 2 lost
their runners in finalise on three times the rows with 0.1-0.2 GB of memory available,
by their heartbeats): finalise ETA −4.5 to −7.4 min (13 claims workers in the first run and
38 on three times the rows, where the first run's 9 buckets already take many).

### What a release starts, against the bench

`parallel.in_parallel` runs a light job in the run's shared pool (its workers kept for
every job of every call; the bench shuts it down at the end of each step, so its workers'
CPU time is counted) and a heavy one (`tasks_per_child=1`) in a pool of its own, a
fresh spawned worker per job. A pool replaces a worker as it exits whether or not a job
is left, so a heavy call starts up to its worker count more, idle (measured locally: 7
jobs on 3 workers started 9 workers, 2 with no job).

The bench's are counted by the workers themselves (the median of run 40198030334's 12
runs; CPU-s a worker outside its jobs, their mean). The release's are measured
(20260928's main-set claims: 30 compacted files, each a group's file, linked; 667
buckets; 86 part files; docs/journal/2026-10-05-sort-speed.md) except where modelled.

| workers | bench, first run | bench, 3× rows | per byte (3× less the first) | release, main set | release, scholarly set (modelled) |
|---|---|---|---|---|---|
| claims compaction (`_write_run`, a worker a job) | 0 | 0 | 0 | 0 | 0 |
| claims bucketing (`_bucket_source`, a worker a job) | 1 (0.69 CPU-s) | 3 (0.58) | 2 | 30 | about 30 (a group's file each, about 30 groups a set) |
| claims bucket sorts (`_sort_bucket`, a worker a job) | 2 (0.51) | 4 (0.58) | 2 | 667 | 410 (the main set's in proportion to entities) |
| claims packing (`_pack_file`, a worker a job) | 1 (0.69) | 4 (0.59) | 3 | 86 | 53 (likewise) |
| other keys' heavy jobs (larger in-memory keys; labels/en and nl's bucket path) | 88 | 87 | −1 | per key | per key |
| shared pools (compaction's, the sort's) | 0, 4 | 4 (0.62), 4 | 4 | 6 a step | likewise |
| idle (a heavy call's replacements, no job) | 5 (0.06) | 12 (0.07) | 7 | at most 6 a call | likewise |

8.0.0's ETA scaled the per-byte column to the release's bytes (× 3,667 / 2 runs' bytes,
about 1,833 times): about 12,800 claims workers and 7,300 starts of compaction's pool
(the first run links every other key's one group file, starting no compaction worker;
the run on three times the rows glues three, in a pool of 4), and 12,800 idle ones, at
the bench's 0.06 CPU-s each. Against them, the release's 1,276 claims workers (783 and
493), its pools' few, and its claims calls' idle workers (2 sets × 3 calls × the cores:
24 at 4).

A release's worker starts by importing its `__mp_main__`, the `finalise-wikidata` console
script, which imports `wikidata.main` (Polars, huggingface_hub and the package); the
bench's imports `pipeline.py`, light, and then the job's module. Measured on the shared
development machine (indicative): `import wikidata.main` 0.71-0.90 CPU-s, `import
wikidata.sort_by_id, wikidata.compact` 0.80-0.85 CPU-s. So a release's job worker starts
at about the bench's cost, and its idle workers at about that too (the bench's idle ones
cost 0.06 CPU-s), so the release's idle workers are added back at a claims worker's mean.

Not modelled, each a few seconds of the ETA: the release's compaction pool (its keys have
many group files; the first run's have one), and the bench's labels/en and nl taking the
bucket path (3 workers each) where a release sorts labels/en as a larger key (1).

### Rows a bucket at a release

667 buckets is measured (64 MiB of source each: 44.7 GB / 667 = 67.0 MB). The rows are
not: 780 million is 7.4.0's model of 20260928's main-set claims, 1.17 million rows a
bucket. The nearest measured count is the philippesaade build of the 2026-05-07 dump's
main set (73,769,737 entities), whose claims are 774,255,243 rows
(docs/dataset_cards_metadata.json): 1.16 million a bucket as it is, 1.19 million at
20260928's 75,432,640 main-set entities. The buckets are equal in rows (by the ids
counted: the bench's largest is its mean + 0.03%), so the largest is taken as the mean
times the bench's own largest over its mean on three times the rows: 1,169,789.

### Finalise's peak RSS at a release, and what the model assumes

| | quick tier (12 runs) | full tier (8 runs) |
|---|---|---|
| a bucket-sort worker, first run, measured (rows a bucket) | 2.79 GB ± 0.02 (587,707) | 4.41-4.94 GB (1,041,444) |
| a bucket-sort worker, 3× rows, measured | 4.28 GB ± 0.06 (881,616) | 4.77-5.40 GB (1,133,154) |
| a bucket-sort worker at 1,169,789 rows, modelled | 5.74 GB ± 0.11 | 4.96-5.61 GB |
| the process tree, 6 workers, modelled | 35.0 GB ± 0.7 | 30.2-34.1 GB |
| the process tree, measured: first run; 3× rows | 6.07 GB; 13.8 GB (4 workers) | 14.4-15.2 GB; 14.8-15.3 GB (6 workers, on 15.6 GB runners) |

- A bucket-sort worker's peak grows with its bucket's rows as between the bench's two
  runs, a release's row as wide, decoded, as the bench's (its Parquet: 57 bytes a row
  modelled, the bench's 62). On the quick tier, the line through the two runs (5.07 KB a
  row) meets zero rows below zero (−0.19 GB): the peak grows faster than the rows there,
  and a line through them overshoots the full tier's measured peak at 1.13 million rows
  (5.55 GB against 4.77-5.40). The full tier's runs are 8% apart in rows, so it takes the
  larger in proportion to its rows. So a release's bucket-sort worker is about 5.0-5.7
  GB, both figures given. Fitted to the bucket's bytes instead (67.0 MB at a release, 55
  MB on the quick tier's three times the rows), it would be about 5.2 GB.
- The bench's bucket sorts read their rows otherwise than a release's: on the quick tier
  0% of the rows are spilled to fragments at the sample's rows and 10% at three times
  them, against a release's 20-30% (modelled), and for another reason. The bench's spill
  is mostly chunk c0's (Q31-Q83751, 620k of the 1,175k claims rows), whose ids of 2 to 6
  characters are out of string order within it; a release's chunks are in string order,
  and its spill comes from a short-id chunk's row groups spanning tens of the 667
  buckets in string order. So the line through the two runs also moves with the share
  spilled (0% to 10%), and a release's worker, reading more of its rows from fragments,
  may peak otherwise than this model says. The model leans on rows a bucket and the
  bench's spill; the worker starts' model leans on neither.
- The tree's is `FINALISE_WORKERS` (6) such workers at their peak at once, and the
  pipeline's process (0.32-0.56 GB over the bucket sorts): an upper bound. On the quick
  tier's three times the rows the sum peaked at about 3.1 of 4 workers' peaks. Summed
  RSS counts the pages processes share once in each, as the measured figures do.
- So a release's finalise at its default 6 workers needs about 30-35 GB in its bucket
  sorts, two or more times an ordinary laptop's 16 GB; at 2 workers, about 11-12 GB.

## [8.0.0] - 2026-10-25

### Changed

- finalise's run on three times the rows sorts the claims key from a release's files for
  its bytes. A release's claims group file is larger than the compaction target
  (`COMPACT_FILE_BYTES`, 500 MiB): 20260928's main-set claims were 44.7 GB in 30
  compacted files, one a group. So each group file is a compacted file of its own,
  linked rather than glued (`compact._link_run`), and each is a bucketing job. Its part
  files are 500 MiB each: 86, about 2.9 a group file. The run on three times the rows
  now sets the target (`WIKIDATA_COMPACT_FILE_BYTES`, read by config.py, and in the
  pipeline's process for compaction's and packing's plans) in that ratio to the claims
  group files' mean size: 73.0 MB × 524 MB / 1.49 GB = 25.7 MB on the quick tier
  (`x3_compact_file_mb`). The first run is as it was.
- Every other key's three times is far under that target (2.6 MB at most on the quick
  tier, `x3_other_key_max_mb`), so every other key is compacted and sorted into the same
  files as before.
- The run on three times the rows also takes the full tier's several groups: each
  group's file of a key is linked as three groups' (chunks first + j n to last + j n).
  Before, every copy was named as if the run had one group, so the full tier failed with
  `FileExistsError` from 7.4.0 on.

### Added

- finalise's peak RSS, in each of its two runs: the process tree's, its processes'
  resident memory summed at each sample (`finalise_peak_rss_mb`, as `peak_sum_rss_mb` is
  the pipeline's), and the largest worker's own peak (`finalise_worker_peak_mb`, VmHWM
  as last sampled), and `_x3` for three times the rows. timeline.py's sampler takes
  them, every 0.5 s. A worker that starts and ends between two samples is not seen; nor
  is its peak after its last sample. The comment shows them in the table.
- Each run's claims key's compacted source files and part files (`sort_sources`,
  `sort_parts`, and `_x3`), shown with the bucket path under the ETA.

### Fixed

- The pipeline's RSS sampler (`tree_rss_mb`) no longer dies when a thread exits between
  being listed and its `children` being read. Its samples stopped there before.
- The dump slice's cache is kept by tier. The full tier took the quick tier's slice (made
  of its 4 chunk files, not the full tier's 11), so its split's parity failed.

Measured with it (run 40191419639, the full tier, 6 finalise workers): finalise peaks at
14.3-15.4 GB summed over its processes, and a bucket-sort worker at 4.3-5.1 GB, on runners
of 15.6 GB. 3 of its 6 jobs lost their runner during finalise. The quick tier (4
workers) peaks at about 13 GB on three times the rows.

What finalise's figures on three times the rows measure has changed (and so its per-byte
part, the finalise ETA and `finalise, 3× the rows`), hence the major version.

### The claims key's shape on three times the rows, against a release's

Bench figures are measured (quick tier, 7.5.0's and this version's runs on three times
the rows). The release's are measured on 20260928's main-set claims
(docs/journal/2026-10-05-sort-speed.md: 30 compacted files, 44.7 GB, 667 buckets, 86
part files), except where marked modelled (7.4.0's model: about 780 million rows).

| | bench 7.5.0, 3× rows | bench 8.0.0, 3× rows | release, main-set claims |
|---|---|---|---|
| compaction target | 500 MiB | 25.7 MB | 500 MiB |
| group files, and their mean size | 3, 73.0 MB | 3, 73.0 MB | 30, 1.49 GB |
| compacted source files (bucketing jobs) | 1, 219 MB (3 glued) | 3 (each linked) | 30 (each linked) |
| compaction's claims worker jobs | 1 (the glue and its byte check) | 0 | 0 |
| buckets (bucket-sort jobs) | 4 | 4 | 667 |
| source files a bucket reads | 1 of 1 | 3 of 3 | about 16 of 30 |
| part files (packing jobs) | 1 | 4 | 86 |
| part files a source file | 1 | 1.33 (2.84 planned, one a bucket at most) | 2.87 |
| buckets a part file | 4 | 1 | 7.8 |
| rows a part file | 3.53M | 881k | about 9.1M (modelled) |
| jobs on 4 workers: bucketing, bucket sorts, packing | 1, 4, 1 | 3, 4, 4 | 30, 667, 86 |

The first run (the sample's rows) stays at 1 source file, 2 buckets and 1 part file.

What it changes in the part per byte (three times the rows less the first run):

- More like a release:
  - The claims key is no longer glued. A release links its group files, so its
    compaction does no glue and no byte check for claims. The bench's per-byte part
    carried one: a 219 MB copy and its check, in a worker of its own.
  - Bucketing runs 3 jobs at once, not 1, and packing 4, not 1. 7.5.0's timeline put
    2.7 and 0.7 idle core-seconds in those two stages per byte, against a release's last
    wave of jobs.
- Less like a release:
  - Each job starts a fresh worker (`tasks_per_child=1`), at about 0.6-0.9 CPU-s for
    its imports (0.9 measured on the shared development machine). The part per byte now
    has 5 more of these (2 bucketing, 3 packing) and 1 fewer (the glue), for 146 MB of
    claims added. Scaled to a release's bytes, that is about 1,500 worker starts for the
    claims key against its 116 (30 + 86). Modelled, it adds about 18-28 minutes to the
    finalise ETA (4 starts × 0.6-0.9 CPU-s × the scale of 3,667 / 2 runs' bytes / 4
    cores). The glue it no longer carries takes off some of that.
  - A part file is one bucket (a release's is 7.8), and each bucket reads every source
    file, as the three are copies of one.
- Not changed: the buckets (7.4.0's figures), and every other key.

## [7.5.0] - 2026-10-25

### Added

- Where finalise's wall time goes, beside its CPU time (`timeline.py`): over finalise's
  two runs, a process of its own samples every 0.5 s the machine's CPU time (busy,
  iowait, idle), the CPU time of the pipeline's process tree and of each of its
  processes, and the run's disk device's reads, writes and busy time
  (`<run>.cores.jsonl`, in the raw results). It is started before the first run and
  reaped after the second, so its CPU time is in none of their figures.
- Each call phases.py times also records its process, its start (`t0`) beside its end
  (`t`), its depth among the timed calls of its thread, and its leading string arguments
  (`of`: the table and key) (`<run>.phases.jsonl`, in the raw results).
- Each run's `timeline`: finalise's stages as the pipeline's process runs them (each
  table's compaction, the copy's check, the sort's write by table, and for a key sorted
  through buckets its bucketing, bucket sorts and packing, the in-memory keys after
  them), each with its wall time, the cores busy and waiting on I/O over it, the
  core-seconds left idle (the cores × its wall less the machine's CPU time over it), the
  device's busy share, and its worker jobs (how many, on how many processes, running at
  once, the longest, and the time the last ran alone). The comment shows the stages on
  three times the rows and per byte (that run less the first), before and after, the
  jobs in a fold.
- sort_by_id.sort_buckets and sort_by_id.pack_key as stages of phases.py, so the
  timeline can tell the bucket path's steps apart; their own CPU time (the parent's,
  between its workers' results) moves out of `write_table`'s in the table of where the
  CPU goes.

Every figure measured before is measured as it was, hence the minor version.

## [7.4.0] - 2026-10-25

### Added

- The claims key's bucket path as each finalise run took it, measured and recorded:
  its buckets (`sort_buckets`), rows a bucket (`sort_bucket_rows`), the share of its
  rows spilled to fragments rather than read from the sources directly
  (`sort_spilled_share`), and rows a spill flush (`sort_flush_rows`: rows spilled over
  the run's `_ipc_stream` calls), and the same with `_x3` for the run on three times the
  rows. The comment gives them under the ETA, for each commit that has them.
- What the bucket path is here, against a release's (below), in the README.

Every figure measured before is measured as it was, hence the minor version.

### The bucket path's shape, and why the bucket size stays 64 MiB

The bench's claims key is sorted through buckets of 64 MiB of source, as a release's:
2 buckets at the sample's rows, 4 at three times them. A finer bucket size was tried,
to bring the flushes to a release's size, and is not taken: it makes the bench less like
a release where most of the sort's cost is.

The bench's figures are measured (the bench's own claims group file, 12 row groups,
1,175,112 rows, 73.0 MB, through the direct-read plan of `_bucket_source`, #93, with its
fragments written; the 16- and 18-bucket shape also in a whole local run). The release's
bucket count is measured (20260928's main-set claims: 44.7 GB in 30 compacted files, 667
buckets; docs/journal/2026-10-05-sort-speed.md); the rest of its column is modelled, and
given as the range of two models: the dump as 8 shard passes over Q1 to Q140M, each chunk
85k Q numbers (the bench data: chunk 0 is Q8 to Q83751, its ids 99.9% in increasing
order; every routed chunk spans 80k to 95k Q numbers), 62% of its entities in the main
set, a property's 14 rows in each, a chunk's rows in row groups of up to 122,880, 667
equal-row buckets in string order, 30 source files, `_direct`'s plan and the flush rule;
rows per entity uniform (9.5), or heavier for low Q (62 below Q100k, as chunk 0 has, to
6.5 at Q140M, as r50 has). Both come to about 780 million rows, 57 bytes a row (the
sample's claims: 62). #93's own model gives 1.75 reads a row and 27% spilled.

| | bench, 64 MiB buckets (kept): sample; 3× rows | bench, 14 more buckets: sample; 3× rows | bench, 4.6 MB buckets: sample; 3× rows | release, main-set claims |
|---|---|---|---|---|
| bucket count | 2; 4 | 16; 18 | 16; 48 | 667 (measured) |
| rows per bucket | 588k; 881k | 73k; 196k | 73k; 73k | 1.17M |
| rows read directly; spilled | 100%, 0%; 90.2%, 9.8% | 69.5%, 30.5%; 58.8%, 41.2% | 69.5%, 30.5%; 7.0%, 93.0% | 70-80%, 20-30% (#93: 73%, 27%) |
| direct reads a row read directly | 1.21; 1.34 | 2.10; 2.25 | 2.10; 2.54 | 1.13 (#93: 1.21) |
| reads a row | 1.21; 1.61 | 2.66; 2.72 | 2.66; 2.99 | 2.30-2.39 (#93: 1.75) |
| rows per spill flush | none spilled; 26.6k | 16.3k; 21.1k | 16.3k; 18.7k | 12.6k-13.2k |
| source files per bucket | 1; 1 | 1; 1 | 1; 1 | about 16 of 30 |
| in the part per byte: spilled, reads a row, direct reads a row read directly | 14.7%, 1.81, 1.42 | 46.6%, 2.75, 2.35 | 124% (more spilled than the rows added), 3.15 | 20-30%, 1.75-2.39, 1.13-1.21 |
| buckets per row in the part per byte | 1 per 1.18M | 1 per 1.18M | 1 per 73k | 1 per 1.17M |

"The part per byte" is the run on three times the rows less the first (2,350,224 rows),
the part the ETA scales to the release; reads a row are `_direct`'s own accounting (each
bucket reading a row group reads it whole, a row group with rows left is read once more,
and a row spilled costs 2 reads).

Why: a release's row group is one chunk's rows, a narrow id range, so it falls into one
or two of 667 buckets and is read directly once or twice. The sample's key is four
chunks' rows, so a bucket finer than a quarter of the key cuts through a chunk, and its
row groups are read by several buckets each: at 16 buckets, 2.1 direct reads a row, a
release about 1.1-1.2. Its spilled rows, chunk 0's (Q8 to Q83751, half the key's rows,
spread over every bucket, as a release's earliest chunks are) and a property's, go from
15% to 47% in the part per byte, against a release's 20-30%. And it adds 14 buckets'
own costs (a worker process each, a file written with the Hub's settings and read back,
about 0.7 CPU-s a bucket, measured on this key through 2, 16 and 48 buckets) to every
run. A fixed bucket size small enough for 16 buckets at the sample's rows (4.6 MB, 48 at
three times them) spills nearly all the rows added and would also count those costs 16
times per byte: about 2.8 hours more in the finalise ETA. Before the direct reads (#93), when every row was spilled, the 14
more buckets did bring the flushes to a release's (84k and 65k rows a flush at 64 MiB,
33k and 31k with 14 more, a release 26k-33k, modelled); the direct reads take most rows
off that path, and what they cost is set by how many buckets a row group is in.

What the kept shape gets wrong:

- Understated: the spill. 15% of the rows in the part per byte against a release's
  20-30%, and in flushes of 26.6k rows against about 13k, so a cost per flush (an IPC
  stream's per-buffer codec calls, the concatenation of a flush's slices before it is
  written) weighs about half as much a row here. (Flushes of about 5k rows, the
  threshold, are those of a bucket gathering thin slices: a wide row group's, or a
  property's few rows; the model's release flushes average about 13k.)
- Understated: the costs per fragment. A release's bucket has rows from about 16 of its
  30 source files; the bench's key has one at either size (three groups glue into one
  219 MB compacted file), so a bucket reads and checks at most one fragment and one
  source.
- Overstated, a little: direct reads, 1.42 a row read directly in the part per byte,
  a release 1.13-1.21.
- Understated: a bucket's sort is of 588k and 881k rows, a release's of 1.17M.

## [7.3.0] - 2026-10-25

### Added

- sort_by_id._read_bucket, a bucket's rows read from its source row groups directly and
  from its fragments, as a stage (the fragments' IPC reads stay in sort_by_id._read_streams).

## [7.2.0] - 2026-10-25

### Added

- The split from the gz dump (`WIKIDATA_DUMP_FORMAT=gz`), for a commit whose split reads
  it: split-dump then route-release on a gz slice of the same lines (dumpslice.py writes
  it beside the bz2 slice, at level 9 as the gz dump is), run by `pipeline.py --split-gz`
  after each timed run, in a process of its own, so no figure of the timed run (its wall,
  CPU, memory, disk and device writes) takes it in. Its wall and CPU time, peak RSS and
  ETA: its CPU time per byte of the slice's gz, scaled to the gz dump
  (RELEASE_DUMP_GZ_BYTES) over the cores. Also each decoder alone on its slice (lbzip2,
  and the gz decoder the commit's split uses), its wall time scaled the same way. The
  comment shows them against the split from bz2, and the ETA and CPU from gz in the
  table when both commits have them.
- Its own parity: its chunks hold exactly the sample's rows, and its manifests are the
  split from bz2's, entry for entry. It is shown with its figures and kept apart from the
  run's parity (`gz_split_parity_ok` beside `parity_ok`, which is as it was); a commit
  failing it fails the job, as a commit failing the run's parity does.
- The runner installs isal (igzip) and pigz.

Every figure measured before is measured as it was, hence the minor version.

## [7.1.0] - 2026-10-24

### Changed

- The split's stages (dump.write_chunk, dump.reshape, dump.entity_row, dump.route_chunk)
  in the table of where the CPU goes are each the CPU time of the thread they run on,
  not of their process: the split can run its chunks in threads side by side, whose
  process times would each count the others'. What polars hands to its own threads in a
  chunk's write is in no stage, in either arm. The split's CPU and wall are as they were.

### Added

- dump.reshape, the compiled reshape, as a stage.

## [7.0.0] - 2026-10-24

### Added

- The split: the release's first stage, split-dump then route-release (dump.split,
  dump.route), run as the CLI runs them on a slice of a dump made of the sample's own
  entities (dumpslice.py: each chunk row turned back into its dump line, the lines
  bzip2-compressed, made once a job and untimed). Each step's wall and CPU time (lbzip2
  and the workers included), its stages in the table of where the CPU goes, and its ETA:
  its CPU time per byte of the slice's bz2, scaled to the release's dump
  (RELEASE_DUMP_BYTES) over the cores, added to the release's ETA. Its peak RSS: each
  step's process and its own (lbzip2, the workers), summed. The split runs before the
  pipeline, outside its wall, CPU and memory figures.
- Parity also requires each run's split to make exactly the sample's chunks: the
  scholarly works (the `s` files) in the scholarly set, the rest in the main one, row for
  row.
- A new stage in the ETA, hence the major version.

## [6.0.0] - 2026-10-24

### Added

- Where the CPU goes: each run times the pipeline's and finalise's stages (phases.py),
  each stage's own CPU time in its process, its worker's native threads included and the
  stages it calls apart, and the comment shows them by stage before and after, a stage's
  change in bold where it is beyond noise: the pipeline over every chunk, and finalise's
  per-byte part (the run on three times the rows less the first). The stages are wrapped
  by an import hook in the harness (in every worker through phases_site/sitecustomize.py),
  so the pipeline's code is unchanged and either commit is timed the same way.
- The timing is in the timed runs (a JSON line a call, about 7,000 calls a run), hence
  the major version.

## [5.1.1] - 2026-10-24

### Changed

- Every figure is still shown, before and after, but a change is given only when it is
  beyond noise, in bold; within noise its cell is blank (no "≈ ..." in every row). The
  ETA headline is one line, its method and its figures by runner CPU folded into a
  details block under it.

## [5.1.0] - 2026-10-24

### Changed

- Each change is the paired ratio, after over before on one runner, averaged over a
  round's two runners and then over the rounds, given as a percentage with its 95%
  confidence interval ("≈ +0.8% ±5.1%"). Before, changes and the noise test were on
  absolute differences, in which a slow runner's difference counts for more than a fast
  one's; the runners now drawn (EPYC 7763, 9V74, 9V45) differ in speed by up to about
  1.7×. The speed in the headline and the results store's are paired ratios too.
- The full-release ETA is also given by runner CPU, so its absolute figures compare
  like with like across pull requests: their mean over the rounds takes in which runners
  were drawn.

## [5.0.0] - 2026-10-24

### Changed

- Parity compares what a release publishes: finalise's sorted part files, each key's rows
  in order (row count and two sums of position-seeded row hashes) and its Arrow schema.
  They are kept from the first finalise run (KEEP=1) under `final/`. A run that publishes
  no files fails parity.
- The staged group files are compared as multisets: each key's rows in any order, and its
  schema. Their order is the intermediate layout, which a change to processing or to the
  group files may change on purpose (rows cut or reordered by id range), and the
  published files' order checks it wherever it shows: each key is stably sorted by id, so
  a change in the order of an id's rows changes them. Keys whose group files hold the
  same rows in another order are listed (`keys_reordered`), not failed.
- The ETA's finalise run on three times the rows triples the sort's in-memory threshold
  too, so every key is sorted the way the first run sorted it. Before, a third of the
  sample's keys moved to another path at three times the rows (82 → 251 sorted as larger
  keys, a fresh worker process each, its imports about 0.5 CPU-s; 8 more through
  buckets), and that cost, a fixed one per key in the release, was counted as cost per
  byte and scaled to the release's bytes. On CI's runners, worker start-up (imports) was
  half of finalise's CPU samples.
- A change to parity and to the ETA, hence the major version.

## [4.0.0] - 2026-10-24

### Changed

- The Full Wikidata ETA covers the whole run again, finalise included, now measured at
  the release's size rather than scaled by entities (3.0.0) or left out (3.1.1):
  - A key that would be over the bucket threshold in the release (claims) is sorted
    through buckets here too: `SORT_IN_MEMORY_BYTES` is scaled by the sample's fraction
    of the release's entities. Bucket and file sizes are not scaled, so their overheads
    stand as in the release. This changes what "wall, with finalise" measures, hence
    the major version.
  - finalise runs a second time, on the group's files three times over (three groups,
    the same keys, three times the rows). Its time at the release's size is a fixed
    part (per key, the release having the sample's keys) plus the CPU time per source
    byte, from the difference of the two runs, scaled to the release's bytes over the
    cores. The comment says how close the line through the two runs is to the second
    run's wall time.
  - The ETA is that plus the pipeline's stages scaled by entities, each shown. The table
    has the three ETAs and the second finalise run's wall time.

## [3.1.1] - 2026-10-24

### Fixed

- The Full Wikidata ETA extrapolates the pipeline alone, as before 3.0.0. 3.0.0 scaled
  finalise by entities too, which on the benchmark's sample its time does not follow:
  the quick tier's finalise is mostly per key (about 2,300 small keys, each a job of its
  own in compaction and the sort), so it put a release at 66 hours. finalise is given on
  a line of its own, as measured, before and after.

## [3.1.0] - 2026-10-24

### Added

- Prebuilt wheels: a commit whose `bench/wheels` lists `<owner/repo> <run id>
  <artifact>` lines has each of those CI artifacts downloaded into its `wheels/`
  before its environment is made, for a `pyproject.toml` that takes a dependency by
  path from there (its hash pinned in `uv.lock`). A measurement of another
  repository's unreleased change (polars-genson's, pq-glue's) then installs the wheel
  that repository's CI built, in place of building it from a git source in every job
  (9 of a round job's 12 minutes, for polars-genson). The job summary lists the wheels.
  What is measured, and how, is unchanged.

## [3.0.1] - 2026-10-24

### Fixed

- `CPU, finalise` counts the workers of the pool finalise's light jobs share
  (`wikidata.parallel`): each finalise step shuts that pool down when it ends, so its
  workers exit and their CPU time is counted. Without that they outlived the
  measurement, and their CPU went uncounted.

## [3.0.0] - 2026-10-24

2.x results do not compare with 3.0.0: the score now includes finalise.

### Added

- finalise's local stages run after the last group closes, as `main.finalise` runs
  them for a release, claims first: each table compacted (`compact_table`, the group
  files already where its download puts them) and sorted (`sort_by_id`'s sourced,
  written and done stages). Rows: `finalise: compact`, `finalise: sort`, `CPU, finalise`,
  and `wall, with finalise`.
- The Hub is never touched: `HF_HUB_OFFLINE=1`, and every Hub entry point of the
  finalise modules (`download`, `snapshot_download`, `HfApi`, `source_table`,
  `commit_table`, `verify_table`) replaced with one that raises. claims_labels, built
  from both sets' sorted labels, and the cards are not run.

### Changed

- The score and the Full Wikidata ETA are the whole run, finalise included (`wall,
  with finalise`); `wall` is the pipeline alone, as before.

## [2.1.1] - 2026-10-24

### Changed

- Work in progress is not benchmarked: a draft pull request, or one whose head commit's
  subject starts with `wip` or holds `[wip]`, runs only a gate job (seconds) and no
  rounds. Marking a draft ready for review runs the benchmark.

## [2.1.0] - 2026-10-24

The measurements are unchanged: 2.0 results compare with 2.1.

### Added

- The sample's size in every result (`workload.entities`, `workload.source_mb`), and in
  the comment's heading: the quick tier is 33,220 entities, 76 MB of source.
- A throughput row, entities per second.
- A full Wikidata ETA: the release 20260928's 121,815,642 entities through the
  benchmarked stages at each commit's rate on the runner, and the change (≈ when the
  wall time's change is noise). A straight extrapolation; it leaves out downloading,
  splitting and routing the dump, the uploads, and finalise.

## [2.0.1] - 2026-10-24

### Changed

- The results bar shows each job's progress: ⬜ until the job launches, its arrow (➡️
  forward, ⬅️ reverse) while it runs, 🟩 once its half is in. The launch bar is
  unchanged: each job's arrow turns 🟦 as it launches.

### Fixed

- Two comments on a new pull request: jobs starting together could each find no
  comment and each make one. A job that makes the comment now deletes any later ones;
  every job writes to the oldest.

## [2.0.0] - 2026-10-24

1.x results do not compare with 2.0.0: every round's order changed.

### Fixed

- A bias towards the commit after. Each round's order was shuffled from a fixed seed,
  and the three parallel rounds' seeds all came out before, then after: the commit after
  always ran second and gained from it. Comparing identical code showed it 1-3% faster in
  wall time with no change in CPU time (−2.8% ± 0.5% on #35).

### Changed

- A round is a pair of jobs on two runners: forward (the commit before first) and
  reverse (the commit after first), one run of each commit apiece. The round's figures
  are the mean of the two halves, so a change is the mean of the forward and reverse
  changes, and the effect of running first or second cancels. Three rounds are six
  parallel jobs; a round counts once both halves are in.
- The workflow no longer reuses a stored measurement of the commit before (from
  `cache/`, on the same CPU model): runners of one model still differ by 2–9% in wall
  time, against about 1% between runs paired in one job. Both commits always run.
- The pull request comment's bars have a pair of cells a round, its forward job (➡️
  until it happens) and its reverse job (⬅️): `🏁 🟩🟩 🟩⬅️ ➡️⬅️` is one round in and
  one half of another; launched jobs are 🟦. Once every round is in, ✅ takes the 🏁's
  place (❌ if a round failed).

## [1.1.1] - 2026-10-24

### Changed

- The pull request comment's status line has two bars, a cell a round. Launches:
  "⏳ 🟦⬜⬜ launching 1/3 rounds" until every round's pipeline runs have started (a
  `launch` write as each job reaches them), then "🚀 🟦🟦🟦 launched 3/3". Results:
  "🏁 🟩⬜⬜ running: 2/3 rounds remaining", then "🏁 🟩🟩🟩 ✅ done: 3 rounds in <the
  longest round, launch to result>" (🟥 for a round that failed).

## [1.1.0] - 2026-10-23

The quick tier measures exactly what 1.0.0 did, so 1.0.0 results compare with it.

### Added

- Tiers. `quick` is the pull request and master check: 4 chunks, 4 workers, one group.
  `full` is the campaign's 66-chunk workload: the 11 files cycled 6 times, 6 workers
  taking 8 chunks each, groups of 22 merged while chunks are still processing. It runs
  from the Actions tab, with a `before` input to set the commit to compare against.
- A progress bar on the pull request comment's status line, a cell a round, filled as
  rounds come in (not as they start): ⏳ ⬜⬜⬜ until the first, then 🏁 🟦⬜⬜, 🏁 🟦🟦⬜,
  and 🏁 🟩🟩🟩 ✅ once done (🟥 for a round that failed).
- The harness version (`bench/pyproject.toml`) and this changelog. Every result and
  record carries the version and the tier.

### Changed

- The results store keeps one append-only ledger, `records.jsonl`, of every tier and
  version; `RECORDS.md` is a view of it, one table per major version and tier. Raw
  results are in `runs/<tier>/<commit>.json`.
- The pull request comment's opening table takes the commit before's figures from the
  store only when a compatible harness version (the same major version) measured them.
- The cache key includes the harness version.

## [1.0.0] - 2026-10-23

Everything recorded before versioning, tagged 1.0.0 in the store.

### Added

- The benchmark in GitHub Actions: a commit against the commit before it, each on its
  own lockfile, on 4 chunks (`c0`, `r50`, `s30`, `s90`), 4 workers, one group.
- 3 rounds as parallel jobs, each round both commits on one runner, so each change is
  paired within a round.
- Output parity by fingerprint (row hashes per partition key, schemas, quarantine,
  sidecars but `file_size`).
- The pull request comment: one per pull request, edited in place; pending as a run
  starts, then the rounds in as they land; figures as mean ± standard deviation, changes
  paired per round, `=` for an unchanged figure and `≈` for a change inside the 95%
  confidence interval of its rounds' changes; times as durations and sizes in the largest
  unit the value before reaches.
- The results store on the `bench-results` branch (records, raw runs, a cache of
  commits' measurements by CPU model), appended to by master pushes.

[3.1.1]: https://github.com/lmmx/wikidata-pq/pull/43
[3.1.0]: https://github.com/lmmx/wikidata-pq/pull/48
[3.0.1]: https://github.com/lmmx/wikidata-pq/pull/42
[3.0.0]: https://github.com/lmmx/wikidata-pq/pull/41
[2.1.1]: https://github.com/lmmx/wikidata-pq/pull/40
[2.1.0]: https://github.com/lmmx/wikidata-pq/pull/38
[2.0.1]: https://github.com/lmmx/wikidata-pq/pull/37
[2.0.0]: https://github.com/lmmx/wikidata-pq/pull/36
[1.1.1]: https://github.com/lmmx/wikidata-pq/pull/35
[1.1.0]: https://github.com/lmmx/wikidata-pq/pull/34
[1.0.0]: https://github.com/lmmx/wikidata-pq/pull/33
