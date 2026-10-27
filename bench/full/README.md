# The compute benchmark

How fast does one machine (the target: 4 cores, 16 GB of memory, a laptop's disk)
process a whole Wikidata dump, when the network is not what limits it?

The campaign reports two headline numbers from now on:

| | what it times | where |
|---|---|---|
| **Compute benchmark** (this) | a real dump, read from local disk, to the published files, every Hub transfer off the timed path | `just bench-full`, on the machine you want to measure; never on a pull request |
| **Release benchmark** | download to publish, the link included | the CI bench (`bench/`, its link model and its history), unchanged |

This tool sits beside the CI bench and shares none of its code or its version: the CI
bench's figures and records keep their meaning.

## Running it

```sh
# 1. Cache the dump, once (timed apart; md5 and sha1 checked against Wikimedia's sums)
just bench-full fetch --release 20261019
# 2. Run it from disk: the compute benchmark
just bench-full run --release 20261019
```

`run` refuses to start without the disk it needs (see Disk). Every option:
`just bench-full run --help`.

**Two input modes,** the same pipeline (the same decoder, the same processing, checks and
output), only the source of the compressed bytes differs:

- `--source local` (the default): the cached dump, read from disk. No network on the
  timed path at all.
- `--source stream`: the dump fetched from dumps.wikimedia.org as the run goes, and
  decoded and processed as the bytes arrive. This is what master already does at a
  release: `download-dump` writes a growing `.part`, and the split follows it
  (`dump._feed`), so download and decode overlap (`just download-split`, and `just
  overlap-release`, whose runs follow the split). The streaming mode is that path,
  reused, not rebuilt: for the whole dump it runs `download-dump` beside `split-dump`,
  exactly as the recipes do. A scaled run's bytes come through splice.py instead (below),
  which writes the growing `.part` as `download-dump` would. Its wall against a local
  run's shows how much of the download the processing hides (`run.py report LOCAL
  STREAM`).

  What makes the `.part` path a correct stream (checked in `dump._feed` and
  `dump.download`):
  - **Backpressure:** at the `.part`'s current end, the follower waits for more bytes
    (`FOLLOW_POLL`) rather than ending. Its writes into the decoder's pipe block while
    the decoder is behind, and the split holds at most `MAX_PENDING_BYTES` of lines. The
    disk is the buffer between the download and the decode.
  - **Completion:** the end is final only once `download-dump` has checked the whole file
    against the published md5 and renamed the `.part` to the dump. A `.part` that fails
    the check is moved aside, and the split stops (`DownloadFailed`) and deletes what it
    split. splice.py renames its `.part` once it has written the stream's end and
    combined CRC, which the decoder checks. A scaled stream is not the published file, so
    it has no md5 to check; `run` compares its sha256 with the cached copy's after the
    run.
  - **Either decoder:** lbzip2 and #197's in-process decoder (`WIKIDATA_BZ2_DECODER=native`)
    are both fed by the same `_feed`, so both modes work with either, and `run` records
    which one the split used.

**Two orchestrations:**

- `--orchestration overlap` (the default): `just overlap-release` and then `just
  release`, without its promotion. The split and both sets' runs at once (the runs at
  nice 10, following the split), the main set's first finalise as its run ends, then
  finalise's runs overlapped (`finalise-release-runs`). This is a release's wall.
- `--orchestration serial`: `just download-split` and then `just release` without its
  promotion. The split, then the scholarly set's run, the main set's, then finalise's
  runs, each alone, so each stage's wall, CPU and bound are its own.

**Scaled runs** for a machine without the disk, or to try a change quickly. Pass the same
option to `fetch` and `run`:

- `--windows K --window-mb W`: K windows of W MB of the dump, evenly spaced across it,
  cut to whole bz2 blocks and whole lines (splice.py). Every shard, early and late ids:
  the closest a sample comes to the dump's mix. (#194's journal took its real-data
  figures from 12 such windows.)
- `--prefix-gb N`: the dump's first N GB. The dump is 8 sha1 shards back to back
  (docs/journal/2026-10-26-progressive-finalise.md), so a prefix is shard 0's head
  only, its early and largest entities: not representative of the whole. The first
  0.1 GB holds 23,533 entities (about 4.2 kB of bz2 each, against the windows' 0.9 kB)
  and no scholarly works. Finalise stops on an empty set (`state.last_chunk`), so `run`
  refuses such a sample after the split: take windows, or a prefix of several GB.

splice.py keeps the dump's own compressed blocks bit for bit (the dump is one lbzip2
stream; blocks are found by their 48-bit magic at any bit alignment). It re-encodes only
each window's first and last few blocks, about 1% of the bytes, to cut the text at line
ends, and it writes a valid stream with the combined CRC, so the decoder checks it all.
A scaled run reports its rates per GB of bz2, and a linear projection to the whole dump
under a heading that says it is modelled. Nothing in a full run is extrapolated.

- `--custom NAME`: any local bz2 dump, as it is (`fetch --custom NAME --file PATH`).
  For example, the CI bench's slice (bench/dumpslice.py) can be run through the same
  pipeline on the same machine, as a control for a scaled real dump.

A scaled run keeps a release's group rule (`WIKIDATA_GROUP_SOURCE_GB`, ~2 GB of source a
group), so at under a few GB of bz2 each set is one group, closed after the split. The
overlapped orchestration then has nothing to overlap but the chunks' processing. To give
a scaled run a release's ~30 groups a set, set `WIKIDATA_GROUP_SOURCE_GB` to its source
over 30.

Environment settings pass through to the pipeline as at a release. Among them are
`WIKIDATA_OVERLAP_WORKERS`, `WIKIDATA_FINALISE_*`, `WIKIDATA_KEEP_GROUP_FILES`,
`WIKIDATA_DUMP_FORMAT`, `WIKIDATA_SPLIT_WRITER_THREADS` (#196) and
`WIKIDATA_BZ2_DECODER` and `WIKIDATA_BZ2_THREADS` (#197's in-process decoder; the
tool records them, and the decoder and threads the split resolves them to). Each is recorded with what
the pipeline's config resolves it to.

## What it records

`run` writes `CACHE/results/NAME/` (or `--out`):

- **`result.json`:** every figure, machine-readable:
  - the dump: its release, bytes, its md5 and sha1 (the whole dump's, published), and a
    scaled dump's ranges, sha256 and JSON bytes;
  - the machine: CPU model, cores, RAM, the disk's device, model and file system, OS,
    kernel, virtualisation, and the load at the start;
  - the git commit and dirtiness, every package's version, the decoders' versions;
  - every `WIKIDATA_*`, `POLARS_*`, `RAYON_*` and `HF_XET_*` setting, and the
    pipeline's resolved config for each set (workers, finalise's knobs, the decoder,
    `WIKIDATA_BZ2_DECODER`);
  - **each stage**, from its process tree's rusage (wrap.py: every descendant it
    waited for):
    - start, end and wall;
    - user and system CPU, and the cores used;
    - the tree's peak RSS (sampled) and its largest process;
    - its bytes read and written to storage, and its page faults;
    - the machine's user, system, iowait and idle shares over its window, and a bound:
      every core, the disk (iowait), or fewer cores (serial parts, or waiting);
  - finalise's runs, each its own wall and CPU;
  - finalise's steps (compaction and sort, by set and table), claims_labels' build,
    and the group closes, from spans in `wikidata.main`;
  - **totals:**
    - wall and CPU;
    - the time to the first processed chunk (the split's first chunk written), and each
      set's first part ready for its run;
    - peak RSS of every process at once, and the least memory available;
    - the peak disk taken (the free space's fall), and the work tree's peak (each inode
      once);
    - the device's bytes read and written, and its busy time;
    - the network's bytes received and sent;
    - the mean load;
  - **streaming:** the bytes received over time (10-second rates, their p10, p50 and
    p90), the download's end against the split's, the first chunk's time against the
    download's end, the chunks and parts done by the download's end, and whether the
    bytes match the cached dump (sha256);
  - **the transfers a release would make,** logged by the stand-in and never timed:
    uploads by set and kind (group pushes, finalise's commits, cards), and the downloads
    (finalise's group files);
  - per GB of bz2, beside the CI bench's model.
- **`summary.md`:** the same, short.
- **The raw records:** `meta.json`, `stage_*.json`, `*finalise_run*.json`,
  `timeline.jsonl` (a sample a second), `spans.jsonl`, `hub_*.jsonl`, `net.jsonl`,
  `run.log`.

## No transfers on the timed path: the stand-in Hub

Every Hub call is answered by `standin.py`, an on-disk stand-in in the shape of the CI
bench's (overlap.FakeHub, link.LinkHub), installed in every process of the run by
`hook/sitecustomize.py`. HF_HUB_OFFLINE=1 and an unroutable HF_ENDPOINT refuse anything
else. No Hub is ever reached.

- **Nothing is copied.** An upload (a group push, a finalise commit) or a download
  (compaction's group files) is a hard link, so the run's bytes are on the disk once,
  and a transfer costs a link. After the run, the linked files are checked unchanged,
  which shows the pipeline wrote into none.
- **The bytes are logged, not timed.** A release's upload bytes and downloads are in
  `transfers_not_timed`. With them, a download bottleneck (streaming's figures) can be
  told apart from an upload one, at any assumed link rate.
- **Its one cost on the timed path** is the sha256 a real Hub reports for each file, for
  the pipeline's checks. It is computed lazily, once an inode, and shared across
  processes. Its CPU time is reported as `standin_overhead`: 5.4 s for the 1.53 GB a
  0.6 GB run uploads (0.4% of its CPU), about 3.5 s a GB uploaded on this machine.

What it leaves out of a release: the transfers themselves, the Hub's commit latency and
dedup queries (the CI bench models them: bench/link.py), and `promote-release`.

## Disk and memory for a full run (16 GB laptop)

- **Disk:** the dump is 103.5 GB. The run's peak beside it is about 2.9× the dump by
  default (`--disk-factor`), ~300 GB, so **~400 GB in all** when streaming, ~300 GB
  free beside a cached dump. The record behind it
  (docs/journal/2026-10-25-release-critical-path.md, "Peak memory and disk") puts a
  release's peak at ~235 GB with the dump, and ~300 GB with the group files kept until
  finalise. The stand-in holds the build branch on this disk, as kept group files are
  held, and finalise's published files sit beside the group files they replace. A
  scaled run measures its own factor (below).
  - `run` refuses with the figures when the disk is short, and `--force` overrides it.
  - Put `--work` and `--cache` on the large disk; a hard link needs them on one file
    system, or the dump is linked by a symlink instead.
- **Memory:** the record's peaks on a 16 GB machine (same journal): the overlap phase
  ~4–5 GB; finalise's bucket sorts 10–12 GB under the memory budget; claims_labels'
  jobs ~11.2 GB, the closest to the limit. Knobs, as at a release:
  - `WIKIDATA_FINALISE_MEMORY_GB` (the bucket sorts' budget);
  - `WIKIDATA_FINALISE_WORKERS`;
  - `WIKIDATA_OVERLAP_WORKERS` (2 a set);
  - `WIKIDATA_FINALISE_CROSS_RUN_GB` and `WIKIDATA_FINALISE_PENDING_UPLOAD_GB`. With
    the stand-in's instant commits, sorted files never wait for a link, so neither
    binds here.
  - No allocator changes.
- **Time (modelled):** the CI bench's model puts a release's compute at ~7 h on 4 EPYC
  cores: the split 3h20m, the pipeline 1h35m, finalise 2h13m. The validation below
  measures the real dump's split at 1.16–1.26× the CI slice's CPU per bz2 byte, so
  allow ~8 h on CI-class cores. On the campaign's 2.8 GHz VM, the scaled runs give an
  upper bound of ~20–23 h. A full run is the measurement this tool exists for.

## Validated on the campaign's machine (scaled)

The campaign's machine has 7 GB of free disk, so no full run was possible. These are
scaled runs of release 20261019 (103.5 GB of bz2) on a 4-vCPU Xeon VM with 16.9 GB of
memory. `results/` has each run's `result.json` and `summary.md`. Every figure here is
measured on that machine unless it says modelled. Each run is checked for other work on
the machine (`other_cpu_s`): a contaminated run was set aside and run again.

**Input: `--windows 24 --window-mb 25`.**
- 0.599 GB of bz2 from 24 evenly spaced windows: 21,231 of the dump's own blocks, 85
  re-encoded ends.
- It holds 11.19 GB of JSON, **18.68× its bz2 bytes**. The CI bench's slice is 13.69×,
  and #194's 12 windows gave 17.46× (CI 15.3–19.4×).

**Two sessions, the same input.** Session 1 ran on master 95fdbf7. Session 2 ran on
e80525d (#196 merged) with this tool's first commit on top, `src/` unchanged. Between
the sessions every stage got 9–23% slower, finalise too, which #196 does not touch, so the
VM drifted. Compare runs within a session; the absolute times are this VM's on that day.

| session | run | wall s | CPU s | first chunk s | peak RSS, every process (machine in use) | peak disk beside the dump | other work, cores |
|---|---|--:|--:|--:|--:|--:|--:|
| 2 | local, serial | 468 | 1,347 | 7 | 12.2 GB (12.8) | 1.33 GB | 0.05 |
| 2 | local, overlap | 482 | 1,427 | 9 | 12.2 GB | 1.39 GB | 0.05 |
| 2 | **stream, overlap** | **518** | 1,504 | **12** | 12.1 GB | 1.37 GB | 0.08 |
| 2 | the CI slice (control), serial | 97 | 211 | 21 | 2.8 GB | 0.19 GB | 0.05 |
| 1 | local, serial | 415 | 1,195 | – | 12.2 GB (12.9) | 1.33 GB | 0.03 |
| 1 | local, overlap | 445 | 1,311 | – | 12.1 GB | 1.33 GB | 0.06 |
| 1 | the CI slice (control), serial | 85 | 185 | – | 2.8 GB | 0.19 GB | 0.06 |

**Per stage, session 2, local, serial:**

| stage | wall s | CPU s | cores | peak RSS (tree) | written | bound |
|---|--:|--:|--:|--:|--:|---|
| split (lbzip2, routing at its end) | 188.7 | 723 | 3.83 | 2.4 GB | 1.01 GB | CPU, every core (iowait 0%) |
| scholarly set's run | 10.8 | 19 | 1.76 | 1.2 GB | 0.35 GB | CPU, serial parts |
| main set's run | 26.8 | 37 | 1.38 | 1.2 GB | 0.49 GB | CPU, serial parts |
| finalise (3 runs, cross-run) | 241.2 | 567 | 2.35 | 12.2 GB | 1.09 GB | CPU, serial parts |

- **Finalise's runs:** main 136 s, scholarly 98 s, then main again 7 s.
- **The claims' sorts:** main 110 s, scholarly 81 s.
- **claims_labels' build:** main 4.6 s wall (15.9 s CPU), scholarly 3.0 s (10.2 s).
- **Group closes:** main 19.9 s, scholarly 5.5 s.
- **The memory peak** is finalise's first run's claims bucket sorts. The run's largest
  process reached 11.3–11.9 GB of RSS (shared pages included), and the machine had
  12.8 GB in use at the peak. That fits 16 GB with ~3 GB to spare, as the record's
  10–12 GB for a release's finalise says.
- **No transfers timed.** The stand-in logged what a release would send: 0.78 GB of
  group pushes and 0.74 GB of finalise commits (2.5 GB a GB of bz2), and 0.78 GB of
  group-file downloads. Its own sha256 took 5.4 s of CPU (0.4% of the run).

**Per GB of bz2, against the CI bench's model (CPU s a GB):**

| stage | real windows (s2 / s1) | the CI slice, same machine (s2 / s1) | windows / slice | the model's, × 4 cores |
|---|--:|--:|--:|--:|
| split | 1,208 / 1,066 | 1,042 / 847 | **1.16 / 1.26** | 464 |
| pipeline | 94 / 86 | 220 / 179 | 0.43 / 0.48 | 220 |
| finalise | 947 / 844 | 1,395 / 1,307 | 0.68 / 0.65 | 308 |

- **#194's finding, tested end to end.** #194 found that the CI bench probably
  understates the release's split by 25–60% (modelled). Through the identical pipeline
  on one machine, the real dump's windows cost **1.16–1.26× the slice's CPU per bz2
  byte** in the split, the low end of that range.
  - The higher JSON per bz2 byte (18.68× against 13.69×: 1.36×) explains it.
  - A JSON byte of real lines costs the same or less end to end: 0.85–0.92× the slice's,
    which also carries its fixed costs on 79 MB. #194's 1.27× was the split's functions
    timed alone.
- **The rest of the gap is the machine.** The slice's split here is 2.2× the model's CPU a
  GB, from this 2.8 GHz VM against CI's EPYC 7763, the slice's fixed costs, and the
  model's own assumptions.
- **The pipeline's and finalise's per-GB rates** at 0.6 GB are dominated by fixed costs
  (one group a set, small keys, process starts). They bound a release's from above.

**Streaming against local (session 2, overlap, the same bytes: sha256 equal):**
- **The download** took 87 s for 0.6 GB from dumps.wikimedia.org, 6.9 MB/s on average
  over 2 connections; 10-second rates p10/p50/p90 were 4.3/7.4/9.1 MB/s.
- **The first chunk** was processed at 12 s, against 9 s from local disk. By the
  download's end, the split had written 24 of 85 chunks, and 24 main and 23 scholarly
  parts were numbered for the runs.
- **The split was never waiting for bytes** after its start. It consumed 2.5 MB/s of bz2,
  and the bytes arrived 2.7× faster.
- **The wall was 518 s against 482 (+36 s, +7.5%).** By `report`'s formula, 58% of the
  download was hidden. The extra is not waiting: it is the scaled stream's splicer (36 s
  of Python CPU on a machine the split already saturates) and run-to-run noise. At a
  release, `download-dump` takes little CPU (its md5 at the end) and runs at ~14.5 MB/s,
  against the split's ~3 MB/s of bz2 on this VM: the download is under the split.
- **Overlap gained nothing at this scale** (482 s against 468 serial). Each set is one
  group (see Scaled runs), and the runs at nice 10 got 0.1–0.2 cores while the split
  held all four.

**Disk, measured:** a peak of 1.33–1.39 GB beside the 0.6 GB dump, **2.23–2.32× its bz2
bytes**. Linear to a release that is ~230–240 GB beside the dump, ~335–345 GB with it
(modelled). The preflight's 2.9× keeps a margin for a release's larger claims sort.

**Modelled, linear in bz2 bytes, this VM:** a whole release's compute would take ~20–23 h
serial (415–468 s × 172.8). This is an upper bound: the scaled runs' fixed costs scale
up with it. With the CI model's split corrected by the measured 1.16–1.26×, the split
alone on CI-class cores is 3h52m–4h12m.

## Uncertainty

- **The validation ran on a shared 4-vCPU VM.** Each run measures the machine's CPU
  besides its own (`other_cpu_s`). The runs kept here had ≤0.08 cores of it; three
  contaminated runs were set aside and run again. Each configuration ran once a session.
  The VM drifted 9–23% between sessions, so the absolute times are indicative, and the
  same-session ratios (windows against the slice, streaming against local) are firmer.
  The disk and memory figures are firmer still.
- A scaled run's fixed costs (process starts, one group a set, small keys) weigh more
  than at a release, so its finalise and pipeline per GB overstate a release's.
  The split's per-GB rate is the least biased.
- The cards (`--cards`, on by default for the whole dump) sample fixed entities (Q42,
  Q5) that a scaled dump may lack, so they are off for scaled runs, as in the CI bench.
  Rendered cards go where a release puts them (docs/releases/, gitignored).
- The bound per stage is a heuristic over machine-wide counters. In the overlapped
  orchestration the stages share their windows.
- The steps' CPU (`steps`, claims_labels) is the process's own and its descendants'
  (reaped, and live ones read from /proc) over the step. A committer thread's work in
  the same process lands in whichever step is running.
- The streaming comparison is one pair of runs. Its 36 s gap is within the two
  sessions' run-to-run spread, plus the splicer's own CPU; a full streaming run would
  use `download-dump`, not the splicer.

## Versions

`run.py`'s VERSION, recorded in every result. 1.0.0: this first version.
