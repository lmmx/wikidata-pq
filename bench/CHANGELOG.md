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

## [17.6.0] - 2026-10-26

### Added

- **The split's census of the slice's order** (`pipeline.split_census`), for a commit whose
  split takes it (`dump.split_census`, docs/journal/2026-10-26-progressive-finalise.md):
  from the split's manifest, the commit's own summary of how far each Q id came below the
  largest id of its dump shard before it: `split_census_shards`, `split_census_q_ids`,
  `split_census_lag` and `split_census_late` (the most below, and how many more than
  10,000 below, past the shards' heads), `split_census_head_lag` and
  `split_census_head_late` (in them, ids below Q10,000), and `split_census_complete`. Read
  after the split's timed steps, in a process of its own. The summary's line after parity
  gives them.
- **The census's CPU time** in the split's workers, `dump.chunk_census`, a stage of the
  split's span in phases.py (and so out of `dump.write_chunk`'s own time), and its share
  of the split's CPU time on the summary's census line.

The slice is unchanged: its four files are one shard's run each (c0 shard 0, r50 shard 3,
s30 shard 2, s90 shard 7). No figure changes meaning.

## [17.5.0] - 2026-10-26

### Added

- **Parity by the coordinate rule** (`parity.compare`, `canonical`): the rule that a
  globe-coordinate's latitude, longitude and precision are always in `__number`
  (src/wikidata/claims_numeric.py, docs/journal/2026-10-26-canonical-coordinates.md)
  changes published claims cells: the quick tier's sample has 80 of them (c0's 79 integer
  precisions of a mainsnak's coordinate, r50's one), which the commit before has in
  `precision__integer`. Each run's fingerprint now also has its claims keys' published and
  group files by the rule (made in Polars, apart from claims_numeric.py), and the cells
  the rule moves in them. Where the exact comparison fails, parity passes only if:
  - the two runs are the same by the rule (every claims value but those cells, and every
    other table, file, quarantine and sidecar, as the exact comparison has them);
  - the commit after keeps the rule (by it, its files are as they are: no cell moves).

  The report then lists the differing keys and the cells apart (`intended`), and the
  summary's parity line says so. Two commits that both keep the rule compare exactly, as
  before, as do two that both lack it. A fingerprint without the rule's (a cached one from
  an earlier harness) compares exactly.

No figure changes meaning: parity is not a figure, and where it passed before it passes
the same way.

## [17.4.0] - 2026-10-26

### Added

- **The main set's first finalise before the scholarly set's uploads end, in the release
  wall with finalise's transfers** (`overlap.release_wall`, `release_wall_link_{7,30,60,120}_modelled`),
  for a commit whose `just overlap-release` runs it there (`config.FINALISE_EARLY`, read
  as the stage reads the commit's other finalise defaults, `finalise_link_bytes["early"]`;
  docs/journal/2026-10-26-finalise-before-uploads-end.md). For such a commit:
  - **the build branch's uploads:** `_uploads_end`'s groups, equal bytes, now a set's
    each in turn (30 a set, the main set's first), and once the split has ended a waiting
    main group pushed before a waiting scholarly one, as the scholarly run then waits its
    turn behind the main run's (`_main_uploads_end`). The link is as busy, so the uploads
    end as before;
  - **finalise's start:** the later of the compute's end and the main set's last push,
    not every push's;
  - **its first run's commits** after the scholarly set's last pushes, the link FIFO
    (`finalise_link`'s `busy`), so `overlap-release` still ends with them.

  A commit without it is modelled as before, to the second. The model keeps 17.2.0's rule
  that a group is pushed once ready: a set's next group's merge in its upload thread (a
  gap on the link of about a minute a group, #173) is in neither, nor each set's own bytes.
  The journal's model has both: there the saving at 7 MB/s is 24 min, against this
  model's 22 min (on master's runs of 26e9259).
- **Finalise's start at each rate** (`release_finalise_start_link_{r}_modelled`), a row of
  the release wall's table.

No figure before these changes meaning.

## [17.3.0] - 2026-10-26

### Added

- **A chunk whose claims misfit claims.avsc** (`misfit.py`, run by `ab.run_once` after
  the overlap and link stages, outside every other figure). The sample's chunks all fit
  `claims.avsc`, which was inferred from them, so no run took the path a release's misfit
  chunk takes: 10 of 153 chunks read from six places in wikidata-20261020-all.json.bz2
  (docs/journal/2026-10-26-claims-avsc-complete.md). The sample's first chunk file (`c0`)
  is turned back into its dump lines (`dumpslice.py`), its first entity with references
  given a reference snak's globe-coordinate (Q618385's P625 from that dump, the shape of
  every misfit found there), then, in one process:
  - **the split's work on the chunk,** `dump.write_chunk`: `misfit_split_cpu`,
    `misfit_split_wall`;
  - **the pipeline's process step on the misfit's part,** `process.process` as a worker
    runs it, on the part as the split routed it: `misfit_process_cpu`,
    `misfit_process_wall`;
  - their sums, `misfit_chunk_cpu` and `misfit_chunk_wall`, and the parts whose claims
    the split left as JSON (`misfit_claims_json`).
- **The misfit chunk's parity** (`misfit_parity_ok`, shown under the parity line): the
  part's claims file must be what genson's inference makes of the part's claims
  (`normalise_claims_direct` with `claims.avsc` set aside), which is what a misfit part is
  published as: in its rows, order, schema (with its metadata) and row groups' rows. A
  part left as JSON was decoded by inference, so it holds by construction. `ab.py` exits
  1 if it fails, as for the gz split's parity; the outputs' parity is unchanged.

No figure before these changes meaning.

## [17.2.0] - 2026-10-26

### Added

- **An emulated link in the stand-in Hub** (`link.py`): the stand-in Hub's transfers
  take time. An upload or download of `b` sample bytes of a set sleeps
  `b × scale[set] / rate / time_scale`:
  - **`scale`, bytes:** a release's chunk files' bytes over the sample's, by set
    (`RELEASE_SOURCE_BYTES` over the tier's chunk files, the scholarly set's those whose
    names start with `s`), as `upload_eta_modelled` scales the uploads. The group files
    and the published files are each about their chunk files' bytes, so both scale by it.
  - **`rate`:** a release's, in bytes a second: an upload rate, or 75 MB/s for a download
    (`DOWNLOAD_RATE`, the middle of the 50-100 MB/s assumed in
    docs/journal/2026-10-25-release-critical-path.md; none is on record).
  - **`time_scale`:** the release seconds a sample second stands for, by stage, taken
    at master on the quick tier as a release's modelled compute in the stage over the
    sample's wall of it without the link (`TIME_SCALE`):
    - the overlap, 1,020: `release_compute_end_modelled` over `overlap_wall` in #165's
      and #163's runs (1,046 and 994);
    - finalise, 408: `finalise_eta + refs_eta + 300 s` over this stage's own
      `finalise_link_0_wall`, in its first CI run (#168's 40252578464, 12 jobs: about
      8,000 s over 19.7 s).
    The full tier's are scaled by the quick tier's chunk files' bytes over its own (no
    full-tier run at bench 17 is on record).
  - **The sample's transfer share is then about a release's.** At 7 MB/s, the overlap's
    pushes take ~17 s of link against its ~14 s of compute on CI, and finalise's ~49 s
    against ~20 s, as a release's 4h56m against ~3h50m and its ~5h33m against ~2h14m.
  - **One link each way,** shared by every process and thread of a stage (a file lock).
    An upload and a download don't wait for each other. Each transfer is logged, so the
    stage reads the link's busy time.
  - **Nothing reaches the Hub.** The stand-in takes the place of huggingface_hub's
    `HfApi`, `snapshot_download` and `hf_hub_download` before wikidata is imported, and
    `HF_HUB_OFFLINE=1` refuses the network. It serves close_group's calls
    (`overlap.FakeHub`'s) and finalise's: `list_repo_tree`, `create_commit` (with
    `run_as_future`), `preupload_lfs_files`, `hf_hub_download`, `snapshot_download`.
- **The overlapped runs over the link** (`overlap.overlap`, after its two modes): the
  split and both sets' runs at once again, their pushes over a 7 MB/s link. Measured:
  `overlap_link_7_wall`, and the link's busy time `overlap_link_7_up_s`. A change to
  when the runs push (a group's merge out of its upload thread, say) shows here.
- **Finalise over the link** (`link.finalise_stage`, after those): `finalise-wikidata`
  (`main.finalise`) as `just release` runs it after `just overlap-release`, the main
  set, the scholarly set and the main set again, each its own process. It runs on a copy
  of the overlapped runs' output, with both sets' groups on the stand-in's build
  branches and the group files deleted locally, as `WIKIDATA_KEEP_GROUP_FILES`' default
  leaves them. Each run downloads its group files from the build branch, compacts, sorts,
  then commits and verifies each table through the stand-in, and builds claims_labels.
  The cards' figures and the cards are left out: they sample entities the bench's sample
  doesn't have. Whatever order the commit's finalise runs its transfers in, the stage runs
  them in; at master, `sort_table` commits each table after its write and before the next,
  with the CPU idle.
  - **Measured, without the link and over a 7 and a 60 MB/s link:**
    - the three runs' wall, `finalise_link_{0,7,60}_wall`, and each run's, `_run1_wall`
      to `_run3_wall`;
    - the link's busy time each way, `_up_s` and `_down_s`;
    - `_hidden`, the share of the link's time under the compute: 1 − (the wall over the
      link − the wall without it) / the link's time;
    - the bytes published, `finalise_link_published_mb`, and by set and table, with the
      group files' bytes, `finalise_link_bytes`, for the model below.
  - **Check:** each run's published files must be the same, byte for byte, or the stage
    fails.
  - **The commit's defaults** are recorded for the model: whether it keeps the group
    files by default (`config.KEEP_GROUP_FILES`), and its finalise's unsent-bytes bound
    (`config.FINALISE_PENDING_UPLOAD_BYTES`, 0 or absent: commits serial, as at master).
    The stage clears `WIKIDATA_KEEP_GROUP_FILES`, `WIKIDATA_FINALISE_PENDING_UPLOAD_GB` and
    `WIKIDATA_FINALISE_CLAIMS_LAST`, so it runs a release's defaults.
- **A release's wall with finalise's transfers, at 7, 30, 60 and 120 MB/s**
  (`release_wall_link_{r}_modelled`, `overlap.finalise_link`), modelled. These are
  `release_wall`'s phases, with:
  - the build branch's uploads at that rate (`release_uploads_end_link_{r}_modelled`);
  - finalise's three runs, a table at a time in finalise's order (claims first), each run
    a process after the one before (`release_finalise_link_{r}_modelled`). For each table:
    - its group files downloaded again at 75 MB/s, unless the commit keeps them by
      default (`release_finalise_download_modelled` in all);
    - its compute, its share of `finalise_eta`: the claims' `finalise_claims_share`, the
      rest the other tables', by each set's and table's group-file bytes; the claims refs
      after the main set's claims;
    - its published upload at that rate, as the commit orders it. At master, after its
      write and before the next table. A commit with a background committer (an
      unsent-bytes bound above 0) has one thread committing each table in order while
      the next is written, the bound kept, claims_labels' build waiting for the set's
      claims and labels, and the run ending with its last commit.
  - claims_labels' build at a release, 5 min (`link.CLAIMS_LABELS_BUILD`, #164's 13-22
    CPU-min on 4 cores), half in each set's last run;
  - bytes from the stage's sample, scaled as the uploads are: the build branch's
    (`release_group_gb_modelled`) and the published (`release_published_gb_modelled`).
  - The comment shows these walls beside the serial ETA, both kept, and the
    measured stage under the overlap's table.

### Changed

- **The split and the runs in a release's wall** (`overlap._phase`): two of 17.1.0's
  assumptions were wrong. In #163's run (40248725557), they made a faster split read as a
  later release: 7h24m33s → 7h50m17s.
  - **The split's CPU time.** `split_eta` is the split's CPU time spread over every core,
    yet 17.1.0 also gave the runs `(cores − split cores) × split_eta` of idle cores
    inside it. That is time the ETA already charges to the split, so a split leaving
    fewer cores idle took the runs' share during it from 63% to 50%. Now:
    - the split's CPU time is `split_eta × cores`;
    - it uses at most its cores on the sample, and goes no faster than the download;
    - the runs (nice 10) take the cores it leaves, plus their CFS share (110 to 1024) of
      the cores it wants;
    - the runs go no faster than the split makes their parts.
  - **The runs' rest after the split** runs on every core: a release's overlapped runs
    have both sets' workers and their closes. 17.1.0 ran it at the sample's pipeline's
    cores, about 1.9 of 4. The compute now ends at the split's and the runs' CPU time over
    the cores whenever the runs have work.
  - **#163 against #164, recomputed** from that run's 12 jobs: the wall 7h04m07s →
    7h04m27s, the compute's end 3h44m17s → 3h40m54s, the uploads' end 4h59m43s →
    4h59m44s, both link-bound at 7 MB/s. 17.1.0's figures are kept beside these, as
    `release_wall_modelled_17_1` and its parts' `_17_1`, reproduced exactly.
- **Nothing measured changes.** The timed run, the gz split, and the overlap's two
  modes and their figures run as before and come first. The new stages run after them.

## [17.1.0] - 2026-10-25

### Added

- **A release's wall, modelled** (`overlap.release_wall`, in each run by `ab.run_once`):
  a release's wall time as its recipes run it now, shown beside the serial ETA, which is
  kept as it was (`eta`, `eta_refs`: the split, the pipeline, finalise and its claims refs
  one after another, no transfer). Every part is modelled from the run's stage ETAs and
  measured figures; nothing in it is measured. In release seconds from the start:
  - **The download** (`just download-split`): the bz2 (`RELEASE_DUMP_BYTES`) at 14.5 MB/s,
    Wikimedia's 3 connections (`RELEASE_DOWNLOAD_RATE`;
    docs/journal/2026-10-25-release-profile.md), about 1h59m. The split follows it (#85)
    and ends at the later of its ETA and the download's end. The bench had no download
    before: the serial ETA leaves it out.
  - **The split and both sets' pipelines** (`just overlap-release`), for a commit whose
    runs follow the split (`overlap_wall` measured): the split, at nice 0, keeps its ETA.
    The runs, at nice 10, take only the cores it leaves idle. The split's cores are its CPU
    over its wall on the sample (`split_cpu_total / split_wall_total`, about 3.4 of 4), and
    every core is idle while it waits for the download. The pipeline uses at most its own
    cores on the sample (`cpu_total / wall`, about 1.9: its ETA is its wall scaled by its
    CPU, so at those cores). Its CPU at a release is its ETA times those cores, and what is
    left when the split ends runs at those cores. A commit whose runs can't follow the
    split runs them after it (`just release` after `just download-split`).
  - **The uploads:** `upload_eta_modelled`'s bytes (a release's, at 7 MB/s) in a
    release's 60 groups (`RELEASE_GROUPS`, 30 a set) of equal bytes. Each group closes
    once the pipeline has done its share of its CPU time and uploads after the one before,
    on one 7 MB/s link both sets share. The runs end with their last group's upload.
  - **Finalise** (`just release`, once `overlap-release` has ended, and so both sets'
    runs and uploads): `finalise_eta` then `refs_eta`, its three runs (main, scholarly,
    main again for claims_labels) one after another, as `main.finalise` refuses a set
    with a group not yet uploaded. Neither this nor the serial ETA has finalise's own
    transfers (compaction's downloads, its and the sort's commits) or claims_labels'
    build.
  - **The critical path:** the uploads when they end after the compute (the split, then
    the pipeline's rest) plus the last group's own upload: by
    `release_upload_backlog_modelled`. Otherwise the compute.
  - **A check on the CPU-share model:** the split's and the pipeline's ETAs summed, times
    the sample's at-once over one-after-another wall (`overlap_wall / overlap_seq_wall`,
    about 0.89): `release_compute_ratio_modelled`. The sample's few parts, numbered late
    in a short split, overlap less than a release's thousands of chunks, so this check
    reads longer.
  - **Figures:** `release_wall_modelled` (in the table, after the ETAs), and
    `release_download_modelled`, `release_split_end_modelled`,
    `release_pipeline_hidden_modelled` (the share of the pipeline's CPU done during the
    split), `release_compute_end_modelled`, `release_compute_ratio_modelled`,
    `release_uploads_end_modelled`, `release_upload_backlog_modelled` and
    `release_finalise_start_modelled`. The comment gives the wall and its critical path
    under the ETA, and its parts by commit under the overlap's table.

### Changed

- **A round's change is the geometric mean of its two halves' ratios**
  (`combine.geometric`), not their arithmetic mean. An arithmetic mean of x and 1/x is
  never below 1, so a noisy figure read high in A/A runs: device bytes written +7.4% over
  45 A/A runs, against +1.6% as a geometric mean of the same jobs
  (docs/journal/2026-10-25-bench-order-bias.md, #161). A ratio at or below zero (a zero
  figure, or one that changes sign) has no log, so its round keeps the arithmetic mean.
  The figures are the halves' means as before. Only the changes and their intervals move.
- **The scholarly claims' base is the sample's scholarly chunk files** (`claims_model`'s
  `scholar_files`: the tier's files whose names start with `s`, cycled as its parts are),
  not its chunks' source.
  - **Why:** the release's 49.5 GB are route-release's chunk files, one-file JSON at
    zstd 9. Since bench 12.0.0 a chunk's source is its part plus the files the split
    built, in another form, so the ratio compared two different things. The bench's chunk
    files are in the release's form. No scholarly claims rows are on record:
    docs/dataset_cards_metadata.json covers a main set only, and `release.py` and the
    journals have none. So bytes in the same form on both sides are the anchor, as bench
    10.0.0 had it.
  - **Effect:** the scholarly claims at a release go from 760 million rows and 39.5 GB
    back to 999 million and 51.9 GB, which 10.0.0 had (999 million, 51.8 GB). The claims
    key at a release grows from 84.2 to 96.6 GB. finalise's claims share and the claims
    refs are scaled by it, and the pipeline's claims build too where the pipeline does it
    (on the quick tier the split builds them all, so the pipeline doesn't move).
  - **Kept:** bench 17.0.0's figures, on the chunks' source, are kept beside
    (`release_claims_rows_scholar_17`, `release_claims_mb_17`,
    `claims_rows_scale_scholar_17`, `claims_bytes_scale_17`, `pipeline_eta_17`,
    `finalise_eta_17`, `eta_17`, `refs_eta_17`, `eta_refs_17`), and the comment gives them
    under the ETA.

Every measured figure is measured as it was, and parity is as it was. Only modelled
figures move (and the changes' averaging), hence the minor version.

### Measured with it

Recomputed from #159's run
[40238218967](https://github.com/lmmx/wikidata-pq/actions/runs/40238218967): quick tier,
bench 17.0.0, 3 rounds, 12 runs (EPYC 7763 ×5 jobs, Xeon 8370C ×1). before is a4d4937
(its runs can't follow the split), after is 0c61bcc (#159: they can). The anchor's change
is applied by claims_model's own terms (finalise's claims share times the change in the
claims' bytes scale; the refs' per byte; no pipeline build), then `release_wall` is
applied. Each figure is the mean over the 6 runs of each commit:

| | before | after |
|---|---|---|
| ETA, a release: 17.0.0 → 17.1.0 | 7h18m → 7h33m | 7h25m → 7h40m |
| ETA: finalise: 17.0.0 → 17.1.0 | 2h07m30s → 2h22m17s | 2h07m50s → 2h22m36s (+14m46s, +11.6%) |
| ETA, with the claims refs: 17.0.0 → 17.1.0 | 7h18m50s → 7h33m39s | 7h25m39s → 7h40m29s |
| modelled: the download's end | 1h58m57s | 1h58m57s |
| modelled: the split's end (its ETA) | 3h35m58s | 3h42m30s |
| modelled: of the pipeline's CPU, done during the split | 0% | 72% |
| modelled: the compute's end | 5h10m55s | 4h11m02s |
| modelled: the same on the measured ratio (check) | – | 4h41m38s |
| modelled: the uploads' end (4h54m–4h56m of uploads) | 8h31m35s | 5h19m12s |
| modelled: uploads past the compute and the last group | +3h15m46s | +1h03m14s |
| **modelled: the release wall** | **10h54m19s** | **7h42m16s** |

- **The critical path:** the uploads in both. For #159's commit, they end 1h03m after
  the compute plus the last group's own 5 min. On the check's compute (4h42m) they are
  still the path, by about 32 min. The serial ETA leaves the uploads out but adds the
  stages one after another. For #159's commit the two come out about the same (the wall
  is +1m47s ± 15m26s over `eta_refs`, run by run). For the commit before, the wall is
  3h21m over it: its uploads start only after the split.
- **The release wall:** the wall falls by 3h12m (−29.4% ±7.4%) with #159, where the
  serial ETA can't show the change: it moved +6m50s, the split's noise.
- **The geometric mean:** of the 59 figures with a change on these runs, the median
  moved 0.02 points. The most were partition CPU (a 0.0–0.1 s figure, −25.0% → −29.3%),
  the claims refs' ETA (+7.5% → +3.7%, ±84%) and the refs' peak RSS at 3× (+14.5% →
  +13.4%). Timings moved 0.1 points or less: finalise CPU −1.9% → −2.0%, split CPU +3.0%
  → +2.9%. Device bytes written moved −0.70% → −0.70%.

## [17.0.0] - 2026-10-25

### Added

- **The split and both sets' pipelines, as a release runs them** (`overlap.py`,
  `pipeline.py --overlap`): after the timed run (`ab.run_once`) and outside its figures,
  on the dump slice the timed run's split reads, each mode in a directory of its own,
  the Hub a stand-in that keeps each repo's build branch on disk (no transfer).
  - **One after another** (`just release` after `just download-split`): the split, then
    the scholarly set's `main.run`, then the main set's, each with the tier's workers.
  - **At once** (`just overlap-release`), for a commit whose runs can follow the split
    (`dump.SetManifest`): the split and both sets' runs, each run taking each chunk once
    the split has numbered it in its set's manifest (`WIKIDATA_FOLLOW_SPLIT=1`), at
    nice 10, with half the workers each.
  - **Groups:** a group a chunk in both modes (`WIKIDATA_GROUP_SOURCE_GB` at 1 MB), so
    the groups close during the split as a release's do (a release's are about 400 of
    its chunks, a set's slice two to four). A commit before the fixed group rule ignores
    it, and its groups are one a set.
  - **Measured:** each mode's wall, from the split's start to the last run's end
    (`overlap_seq_wall`, `overlap_wall`); the peak RSS of every process at once, the
    split's, the runs', their workers' and decoders', summed (`overlap_seq_peak_rss_mb`,
    `overlap_peak_rss_mb`); and the bytes the groups upload (`overlap_seq_upload_mb`,
    `overlap_upload_mb`).
  - **Modelled, apart from them:** those bytes at a release uploaded at 7 MB/s, the one
    upload rate on record (docs/journal/2026-10-05-sort-speed.md): `upload_eta_modelled`.
    Each set's bytes are scaled by its chunk files' bytes, the release's
    (`release.RELEASE_SOURCE_BYTES`, the 20260928 run's) over the sample's (its files
    whose names start with `s` the scholarly set's), not by entities: the sample's `c0`
    has about 8 times a release's mean bytes a main entity, so entities would scale the
    uploads to about 3 times a release's. On the quick tier, locally: 101 MB uploaded,
    about 4.9 h at a release, as the 5 h on record. And, in the comment only, how much
    of it a release's split could hide for a commit whose runs follow it: the least of
    it and the split's ETA, assuming the groups close evenly through the split and
    upload in turn. The bench sees no transfer, so whether a release's uploads end under
    its compute is the model's, not a measurement.

### Changed

- **Major version:** each round now runs the split and both sets' pipelines twice more
  after its timed run, so a round's work, and what runs beside the next round's timed
  run on the runner, are not those of the versions before. Every timed figure is measured as before.
- **The timed run's group is the tier's share of the parts** (`pipeline.py`), not a count
  of them: the tier's `group` over its `chunks`, times the parts the pipeline runs on
  (since 12.0.0, the split's parts, cycled), rounded up. The quick tier's 4 chunks are 5
  parts: its group was 4, and the rule before the fixed one closed every closable part
  once 4 were, so one group of 5; under the fixed group rule (the fewest that reach it) a
  group of 4 leaves a last group of one part, which doubled `finalise: compact` (two
  claims group files, and six at three times the rows) on bench 17.0.0's first run. Now
  a group of 5, one group under either rule, as the workload has it. The full tier's
  group is a third of its parts.
- `pipeline.py` with `KEEP=1` also keeps the sorted part files of the run on three times
  the rows (`final_x3/`).

## [16.0.0] - 2026-10-25

### Changed

- **Parity covers claims_labels' published files** (`<run>.keep/final/claims_labels`, the
  first run's), with the other tables' (below).

### Added

- **claims_labels' build, compaction and sort** (`pipeline.claims_labels_stage`), as a
  release's finalise runs them once both sets' labels are sorted: after each run's refs
  stage (`claims_refs`), on its sorted labels and refs. `build_claims_labels` (its Hub
  calls replaced with ones that do nothing), `compact_table`, then the sort's local stages
  as `finalise` runs them for the other tables. Timed apart from finalise, between markers
  of its own (`claims labels`, `claims labels x3`), so it is in none of the figures
  measured before.
  - **New figures:** its wall time and CPU time (this process and its workers) in each
    run; its peak RSS (the process tree's and the largest worker's) in each, from the
    timeline's samples in it; and its published files' rows, MB, files and keys in each
    (`claims_labels_rows`, `claims_labels_mb`, `claims_labels_files`, `claims_labels_keys`,
    and `_x3`).
  - **Planned with a release's settings:** the in-memory threshold and the part files'
    size are the release's (`SORT_IN_MEMORY_BYTES`, `COMPACT_FILE_BYTES`), not the
    sample's scaled ones, so the sample's keys are planned as a release's are: each in
    memory, none a larger key (a worker of its own), one part file each. At a release the
    main set's largest key, `en`, has 17.3 million rows (`docs/dataset_cards_metadata.json`,
    the 2026-05-07 build), 1.1 GB by the plan's estimate: in memory, and under the larger
    keys' 1.5 GiB. The sample's `en` has 3,479 rows. The scaled threshold gives the same
    plans for the keys' rows, but not for the build's estimate of them made before its
    join (a language's labels' rows): the sample's refs mostly name entities outside it,
    with no labels in it, so its labels have 9.1 times its claims_labels rows in `en`,
    against 2.8 at a release, and 16 of its keys would be taken as larger keys, none of
    them one.
  - **This run's labels stand for both sets':** the other set's directory of labels is
    left empty.
  - **On three times the rows,** the labels are each three times over, so its rows before
    compaction's deduplication are three times the first run's, and its published files
    the same.
  - **Parity:** the first run's claims_labels part files are kept with the other tables'
    (`<run>.keep/final/claims_labels`), so parity compares them too.
  - **Why:** lmmx/wikidata-pq's E6 writes claims_labels as its published files in its
    build, skipping compaction's deduplicating rewrite and the sort's read and re-sort.
    The bench ran no claims_labels build, so it could not see that.
- **The refs stage's figures as before:** the stage's refs (`refs.parquet`) are left for
  the build, which removes them; and the build starts once the timeline has taken its
  first sample after the refs stage (`refs_memory` takes it for a stage with no sample in
  it).

A major version: parity now covers claims_labels' published files, and a change to parity
is a new major version. Every figure measured before is measured as it was. The run's
disk figures (`disk_peak_mb`, `device_written_mb`) take in the stage's few MB of files.

## [15.2.0] - 2026-10-25

### Changed

- **The finalise ETA scales the other tables' per-byte stages by each table's own rows at a
  release**, not by entities (`pipeline.tables_model`). Modelled; every measured figure is
  as it was.
  - **The stages:** every stage of the labels, descriptions, aliases and links but 15.1.0's
    per-file ones: the decode (`_read_key`), the sort and its check (`_sorted`,
    `_check_sort`), the encode (`compact._write_file_parallel`) and the written file's
    check (`_check_written`), each table's CPU time on three times the rows less the first
    (`phases.table_cpu` by table and function: `finalise_{table}_bytes_cpu`).
  - **What the model assumed before:** that CPU time was in finalise's part per byte,
    scaled by entities (3,667×), as if a release's entities carried the sample's rows of
    each table. They don't. The sample's main set (17,000 entities, a third of them chunk
    `c0`'s well-known items) has 34 labels, 10.3 aliases and 24.8 sitelinks an entity,
    against a release's 8.7, 2.3 and 1.3: about 3.9, 4.4 and 18.6 times. Its descriptions
    are about a release's (21.9 an entity against 21.0). The 15.1.0 model gave these
    stages about 37 min of the finalise ETA
    (`docs/journal/2026-10-25-finalise-passes.md` reckoned 35.6 min, "~3× overstated").
  - **What it assumes now:**
    - **A table's part per byte goes with its rows:** its CPU time a row, and its rows'
      width, are the sample's. Each table's part is scaled by its release rows over the
      sample's (both sets', from the audit sidecars: `pipeline_tables`, the rows of
      finalise's group files), over the cores, as before.
    - **The main set's rows** are the 20260507 build's main set's (73.8M entities;
      `docs/dataset_cards_metadata.json`, from the Hub's footers: measured), grown by 2%.
      20260928's main-set tables were 0.6–2.0% larger than 20260507's by their recorded
      row counts (`docs/journal/2026-10-06-entities-dropped.md`), and the upper end is
      taken (`release.RELEASE_MAIN_TABLE_ROWS`). The byte figures on record are an older
      writer's, so rows are the measure that compares.
    - **The scholarly set's rows** are not on record. They are modelled as the sample's
      rows a scholarly entity (16,220 entities: 60.3 descriptions, 2.55 labels, 0.03
      aliases and 0.0015 sitelinks each) times the release's 46,383,002 scholarly
      entities (measured).
    - **The release's rows, so modelled** (`release_{table}_rows`), and their scales
      (`{table}_rows_scale`): labels 774 million (1,250×), descriptions 4,379 million
      (3,244×), aliases 177 million (1,005×), links 101 million (239×).
    - **Still scaled by entities:** the entities table (a row an entity: its scale is
      the entities'); the CPU time in no table's stage and outside the timed stages; each
      key's files' opens inside `_read_key`, which go with its files (about 60 a key at a
      release against the 3× run's two extra) but are not timed apart. claims_labels has
      no group files in finalise (the refs stage builds it after it: `refs_model`).
  - **The ETA**, recomputed from #152's run 40227111915 (bench 15.1.0, 6 runs an arm), on
    its after arm: finalise 2h09m → 1h49m (−19m54s ± 4m17s over the runs, −15.3%), the
    release 7h04m → 6h45m (−4.6%), with the claims refs 7h12m → 6h53m. The other tables'
    per-byte stages go from about 37 min to about 17 min. Descriptions are most of what
    is left (about 12 of the 17 min), and their scholarly rows are the least sure figure
    (below).
  - **The fit of the run on three times the rows** is unchanged (it doesn't depend on the
    release's scale): within 2.2% of its wall time on those runs.
- **The pipeline's claims are scaled by their rows only where the pipeline builds them.**
  - **Since the split builds the claims** (#134, bench 12.0.0), they are in the split's
    CPU time, scaled with it by the dump's bytes. For a part whose claims the split built,
    the pipeline only places its claims file (`process.place_built_claims`: a rename, an
    empty label lookup; `place_unsplit`: a hard link and its sidecar). Those calls are
    per chunk, about 0.03 CPU-s in all on the sample, and 15.1.0 scaled them by the
    claims' rows. The ETA by stage showed them as a row of 0.0s.
  - **Now** `claims_measured` also gives each set's claims build CPU time (its
    `normalise_claims_direct` calls: a part the split left as JSON, a snak to prune),
    measured: `pipeline_claims_build_cpu`. That alone is scaled by the claims' rows.
    The placements are folded into the pipeline's rest, scaled by entities, as the other
    tables' placements are. Where the pipeline builds no claims, the ETA by stage shows
    no claims row, and says so in the rest's.
  - **Nothing is double-counted or lost:** the placements' CPU time is in `cpu_total`
    either way, and moves only from one rate to another (+36 s of the pipeline ETA on
    #152's runs). The split's claims build was in the split's figures already.
- **Bench 15.1.0's ETAs are kept beside** (`pipeline_eta_15_1`, `finalise_eta_15_1`,
  `eta_15_1`) and shown under the ETA, with each table's rows and scale; the ETA by
  stage gives each table's rate.

### Fixed

- **The comment is sent with its body on stdin** (`comment.py`: `gh api --input -`), not as
  one argument (`-f body=...`). A comment of all the rounds is about 125 kB (#152's), and
  this version's notes took it past the kernel's 128 KiB limit on an argument's length.
  The final comment then failed ("Argument list too long"), and so did the `failed` one
  after it, so the comment kept its second round's figures.

### Not changed, noted

- **The scholarly set's claims are modelled from a source whose meaning changed.**
  `claims_model` takes the scholarly set's claims rows and bytes a byte of its chunks'
  source, times the release's scholarly chunk files (49.5 GB, the old one-file JSON form
  at zstd 9). Since bench 12.0.0 a chunk's source is its part plus the files the split
  built, in another form. The modelled scholarly claims went from 999 million rows and
  51.8 GB (10.0.0) to about 760 million and 39.5 GB. This change leaves that alone.

Every figure measured before is measured as it was, hence the minor version.

## [15.1.0] - 2026-10-25

### Changed

- **The finalise ETA scales the other tables' per-file stages by a release's files**, not
  by entities (`pipeline.files_model`). Modelled; every measured figure is as it was.
  - **The stages:** compaction's plan and links (`compact.rewrite_table`: each link's
    footer read and sha256, and each key's plan), the local copy's check
    (`sort_by_id.check_compacted_copy`: each file's sha256) and the sort's plans
    (`sort_by_id.write_table`'s own time: each key's files' footers), for every table
    but the claims.
  - **What the model assumed before:** their CPU time on three times the rows less the
    first was in finalise's part per byte, scaled by entities (about 3,667×) over the
    cores, at about 458 ETA-s a CPU-s. But the run on three times the rows adds two
    group files to each of the other tables' 2,323 keys (a file a key in the first run,
    three in the second), and these stages cost about 0.3-0.7 ms a file, a few KB each.
    So entities scaled them as if a release's key had about 3,700 times the sample's
    files: about 20 min of the finalise ETA (`docs/journal/2026-10-25-finalise-passes.md`).
  - **What it assumes now:**
    - **Their part per byte** is their files' sha256: the other tables' extra bytes
      (twice their source) hashed twice, once by compaction's link and once by the
      check, at the rate the claims' check hashed the claims key's extra bytes (its CPU
      time on three times the rows less the first, over twice the claims key's bytes:
      about 1.3 GB/s). It stays in the part per byte, scaled by entities: about 0.08 of
      their ~2.8 extra CPU-s.
    - **The rest is per file:** over the extra files (twice the first run's), about
      0.59 ms a file (0.27-0.73 by runner, `finalise_file_ms`).
    - **A release's files** are the sample's keys (2,323: a table's keys are its
      languages and sites, about the release's) times a release's groups in both sets,
      30 a set (`release.RELEASE_GROUPS`; GROUP_TARGET_COUNT, and the main set's claims
      key's 30 group files, measured): 139,380 (`release_files`). That is every key in
      every group of both sets, an upper bound. One main set on record
      (`docs/journal/2026-09-29-hub-layout-and-compaction.md`) had 71,718 group files
      for its labels, descriptions, aliases and links: 2,757 keys, 26 a key of 31
      groups. The model's main set is 69,690.
    - **On one core:** these stages run in finalise's process a file at a time (the
      check's hashes on threads, but each a few KB, its loop in the parent), so they are
      not over the cores. The first run's files' time is taken out of the fixed part, and
      the release's is added: about 1m22s (`release_files_eta`).
  - **The ETA:** on #145's runs (bench 14.1.0, run 40223413409, 6 runs an arm), recomputed
    from their artifacts, finalise 2h35m → 2h16m and the release 8h01m → 7h42m (the
    commit after's). The fit of the run on three times the rows moves by the same terms
    (`finalise_x3_wall_fit`): within 2% of its wall time on those runs, against 4%.
  - **Still scaled by entities:** the other tables' decode, sort, encode and check
    (`_read_key`, `_sorted`, `_check_sort`, `_write_file_parallel`, `_check_written`),
    whose bytes go with their rows; and `_read_key`'s file opens, which go with the files
    but are not timed apart from its decode.
- **Bench 15.0.0's ETAs are kept beside** (`finalise_eta_15`, `eta_15`) and shown under the
  ETA, with the per-file stages' figures; the ETA by stage gives their rate (a release's
  files over the first run's, on one core).

### Added

- **The other tables' group files and keys, measured** in the first run
  (`finalise_files`, `finalise_keys`), and the per-file stages' CPU time on three times
  the rows less the first (`finalise_file_stages_cpu`), with their part per byte
  (`finalise_file_hashed_cpu`), modelled.
- `phases.table_cpu(..., by_fn=True)`: each span's CPU time by table and function.

Every figure measured before is measured as it was, hence the minor version.
## [15.0.0] - 2026-10-25

### Changed

- **A part's entities file, where the split writes one, goes to the pipeline as its claims
  file does** (linked into `data/entities/`, and counted in the chunk's source size, as
  `dump.chunk_sizes` counts it), and the split's parity checks it:
  - the part must then have no `entity` column;
  - the file must be the one the commit's own pipeline makes of the files' `entity` JSON
    for its rows (`process.ENTITY_SCHEMA`'s decode, `atomic_sink_parquet`): the same rows,
    order, schema with its metadata, and row groups' rows. Run in the built tables'
    parity process, outside every figure.
  - Everything else is checked as in 14.1.0.
- **The pipeline's claims CPU counts `place_unsplit`'s calls for the claims only.** The
  pipeline places a part's entities file by `place_unsplit` too, as it places the claims
  file; the claims' measure (`claims_measured`, which takes a worker's k-th claims call
  as its k-th chunk) and the profile's tables (`phases._table_of`) now take each call's
  table from its `dst_dir`. Without it, the claims' measure came out empty, and the ETA
  fell back to scaling finalise by entities alone.

## [14.1.0] - 2026-10-25

### Added

- **claims_labels' refs stage** (`claims_labels.collect_refs_stage`), run as a release's
  finalise runs it once the claims are sorted: after each finalise run (the sample's rows,
  and three times them), on its sorted claims (`pipeline.claims_refs`). It is timed apart
  from finalise, between markers of its own (`claims refs`, `claims refs x3`), so it is in
  none of the figures measured before.
  - **New figures:** its wall time and CPU time (this process and its workers) in each run,
    and its peak RSS (the process tree's and the largest worker's) in each: from the
    timeline's samples in it, or, for a stage shorter than the sampler's interval, the first
    sample after it.
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
## [14.0.0] - 2026-10-25

### Changed

- **The pipeline reads the term and sitelink tables the split built.** A split that builds
  a part's labels, descriptions, aliases and links (fusion A) writes each table beside the
  part as one file, its row groups by key, and its index (`{table}/{chunk}.json`; a table
  with no rows has the index alone), and leaves the part without the four columns. The
  bench hands the pipeline those files with each chunk, as it hands the claims partition
  file, and counts their bytes in the chunk's source size, as `dump.chunk_sizes` does.
- **The split's parity checks the built tables.** A part without the labels,
  descriptions, aliases and sitelinks columns must have all four tables' files, each with,
  key by key, the rows the commit's own pipeline makes of the files' column for its rows
  (`process.normalise_map_direct` or `normalise_sitelinks`, then
  `partitioning.prepare_for_partition`), in order, in the same columns and dtypes, and no
  other keys. Run in the claims' parity process, outside every figure. A part with the
  columns must hold them as the files do, as before.
- **`parity.py` compares a sidecar's `path` without its key's directory.** A built table's
  sidecar names its chunk's one file (`results/{table}/{chunk}`), not each key's would-be
  file (`results/{table}/{key}/{chunk}`); both name the same rows. Its `file_size` may
  differ, as before.
  - **Why:** the pipeline's work moves into the split; the published files are unchanged,
    and checked as before.
  - Everything else is as in 13.0.0.

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
