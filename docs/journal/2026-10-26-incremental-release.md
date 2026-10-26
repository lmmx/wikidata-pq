# 2026-10-26: Incremental releases: weekly churn measured, the split's CPU split, and a release that merges what didn't change

Every release reprocesses every entity: the split's reshape and build, the pipeline, and
finalise's global sort. Each weekly dump re-encodes all ~121.8M entities, but only those
edited since the dump before change. If the derivation (code and release schema) is
unchanged, an unchanged dump line gives the same derived rows. So a release could:

1. decode and hash every line;
2. reuse the release before's rows for the unchanged entities;
3. process only the changed and new ones;
4. build each published table as a sorted merge: the previous rows, less the changed and
   gone ids, plus the new rows. The tables are already sorted by string id, so this is a
   merge, not a sort.

This entry measures the churn between consecutive dumps, splits the split's CPU, models
an incremental release's wall, estimates Xet's dedup of the upload, and sets out the
integrity design. The scripts and data are in `2026-10-26-incremental-release/` (see
Reproduce). Measured and modelled figures are kept apart: the churn, the bench's CPU
shares and the hash's cost are **measured**; every release-scale time and every dedup
figure is **modelled**.

## Current State

- **Docs only.** Nothing in `src/`, `reshape/` or `bench/` changed. No Hub call.
  - About 240 MB of byte-range reads from dumps.wikimedia.org (below), all held in memory
    but 8.3 MB for the hash timing, which was deleted after use.
  - The disk keeps only the tables in this folder (2 MB).
- **Weekly churn (measured):** 32 aligned windows in each of 20261005, 20261012 and
  20261019, 46,415 entities compared by id.
  - **Changed or gone in a week: 0.66% (95% CI 0.26–1.26%) and 0.94% (0.43–1.63%) of
    entities, 0.76% and 1.03% of bytes.** Created in a week: 50–72k entities (0.04–0.06%).
  - **Strongly concentrated.** The windows' counts vary 45–62 times more than a uniform
    rate's. The 4 busiest windows of 32 hold 58–67% of the changes:
    - the oldest items (Q0–Q25M) change 1.2–3.9% a week;
    - the scholarly range (Q25M–Q60M) changes 0.16–0.31% a week;
    - **the main set changes 1.1–1.5% a week, the scholarly set 0.02–0.07%.**
  - **The edits are bot sweeps over classes on single days:** Hausa descriptions on 128
    items of one class (Q57733494) in one window on 2026-10-14; census properties
    (P1082, P1538–P1540) on 82 villages (Q532) on 2026-10-07; Assamese descriptions on
    humans (Q5) throughout.
  - **Nothing defeats line hashing:** none of the 139,250 comparisons had a differing line
    without a new `lastrevid`. A line with a new `lastrevid` and unchanged content (an
    edit and its revert) occurred 2 times in 46k in one week and 0 in the other. No
    `modified`, badge or serialisation change touches entities that were not edited.
- **The split's CPU (measured shares, bench 17.9.0, master b110f78, 6 runs):**

  | | share | at a release (modelled) |
  |---|---|---|
  | the bz2 decode (lbzip2) | 54% ± 3% | 6.3 core-h, 1h35m on 4 cores |
  | the claims' Parquet encode (`write_table`) | 22% | 2.6 core-h |
  | reshape and build (`write_chunk`: the scan, ids, arrays) | 13% | 1.6 core-h |
  | refs, statements, census | 1% | 0.1 core-h |
  | not attributed (the main thread's reads, the pipe, routing's own) | 10% | 1.2 core-h |

  **A hashing-only pass (decode + lines + id + xxh3): 6.7 core-h, 1h40m on 4 cores.** It
  is under the dump's own download (1h59m at 14.5 MB/s), and the hash is 3–5% of the
  decode (measured).
- **The release walls (modelled, means of the bench's 6 runs):**

  | | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
  |---|---|---|---|---|---|
  | master (bench 17.9.0's own model) | 10h50m | 7h41m | 6h40m | 6h17m | 6h12m |
  | full, no group files, frontier (F1, ≈ #180's (c)), dedup 55% | 6h11m | 6h07m | 6h06m | 6h06m | 6h05m |
  | **incremental, merge after the split (I0), dedup 55%** | **4h20m** | **3h45m** | **3h44m** | **3h44m** | **3h43m** |
  | **incremental + frontier (I1), dedup 55%** | **3h34m** | **3h28m** | **3h27m** | **3h26m** | **3h26m** |
  | I1, no dedup | 5h20m | 3h35m | 3h28m | 3h27m | 3h26m |
  | I1 at 5× the churn, dedup 35% | 4h01m | 3h34m | 3h32m | 3h31m | 3h31m |

  - **At fast links the release falls by ~2h40m** (−44%: −2h39m to −2h40m against F1 in
    the same simulator). It doesn't reach the decode floor: the merge still re-encodes all
    129 GB of published Parquet (6.5 core-h, 80% of finalise's CPU), so the release is
    the decode plus the encode on 4 cores.
  - **Churn barely matters:** at 5× the churn (4.7% of bytes a week), the walls move
    5 min at fast links.
- **Xet dedup (modelled): about 55% of the upload (44–62% at 256 KiB–1 MiB pages; 35–47%
  at 2–5× the churn).**
  - The parts are written with Parquet content-defined chunking, so an untouched page
    re-encodes to the same bytes.
  - **The scholarly set dedups almost entirely** (claims 92%: 48 of 53 GB).
  - **The main set's claims barely dedup** (2–9% of pages untouched: a page holds ~1,000
    entities, at ~1% churn). Its language and site tables dedup 63–73%.
  - Concentration within a set barely matters. What matters is that the published layout
    already separates the rarely-edited scholarly articles.
  - **This is not specific to an incremental release:** a full release with the same
    writer gets the same dedup. It is unmeasured on the Hub.
- **Integrity:**
  - **A derivation version** forces a full rebuild on any change to derived rows. CI
    guards it: a golden row digest of the bench slice per version.
  - **Unchanged** means both the same `lastrevid` and the same 128-bit line hash.
  - **Order-independent per-region row digests** are kept for the merge, and all of
    today's exact checks run over the merged output.
  - **A sampled re-derivation every release**, and **a full release as an audit** every
    4th.
  - **Disk at peak ~240 GB** (the dump, the release before's 129 GB, the new parts region
    by region). **RAM** ~3–4 GB for the merge.
- **Verdict: big fish.** Against a full release in the same simulator, −2h22m (I0) to
  −2h40m (I1) at every rate of 15 MB/s and above, and −2h37m at 7 MB/s (I1 with dedup;
  −1h35m without). The first PR is
  measurement only: the split records each entity's `lastrevid` and line hash, and
  given the release before's record, the churn census of a whole release (Plan).

## 1. Churn between consecutive dumps (measured)

### Method (`churn.py`, `bside.py`, `tails.py`)

- **The layout used:** each dump is 8 shards, each in page-id order and monotone in id to
  within 2 past its first ~1,000 entities (`2026-10-26-progressive-finalise.md`, Gate 2).
  So the same shard and id range is one contiguous stretch in every dump.
- **32 points**, shard `j mod 8` at the fraction (j + 0.5)/32 of 20261019's shard (its
  boundaries measured by #180), so each shard gets 4 points at different id ranges, and
  the 32 cover Q0.5M–Q137.7M.
- **At each point:**
  - **20261019:** a 1.25 MiB window, its blocks decoded (bz2probe: each rewrapped and
    decoded by libbz2, which checks its CRC), with the shard's ids and lines kept.
  - **20261012 and 20261005:** the same shard's first id found by 2–3 probes of 192 KiB
    each, then one window covering the id range.
  - **The comparison:** by id, on the range common to all three, 5 ids in from each end.
- **Each entity is:**
  - **unchanged:** the same line, byte for byte, after the separating comma;
  - **meta-only:** different lines that are equal once `lastrevid` and `modified` are
    removed;
  - **changed:** its differing top-level fields recorded, the claims' properties too;
  - **new:** in the newer dump only;
  - **gone:** in the older only (deleted or redirected: the dumps carry neither).
- **The new entities** are at every shard's end (page-id order), not in the windows.
  `tails.py` finds the older dump's largest shard-7 id in the newer dump. Shard 7 ends at
  the file's end, so the bytes after it are shard 7's new entities, and those times 8
  are all shards'.
- **The weights:**
  - The windows are equal bz2 bytes of each shard, so every entity is about equally likely
    to be sampled. Σ changed / Σ entities is entity-weighted; with each entity's line
    bytes it is byte-weighted.
  - The intervals are a cluster bootstrap over the 32 windows (2,000 resamples).
- **Bytes read:**

  | read | count | bytes |
  |---|---|---|
  | windows and locating probes (`churn.py`) | 32 points × 3 dumps, 161 probes | 174.4 MB |
  | 20261019's windows again, for each entity's set and fields' bytes (`bside.py`) | 32 | 41.9 MB |
  | the shard ends (`tails.py`) | 2 pairs | 6.9 MB |
  | a trial point and a debugging read | | 8.2 MB |
  | hash timing (`hash_bench.py`, deleted after) | 2 × 4 MiB | 8.4 MB |
  | **all** | | **~240 MB** |

### Status by week

| pair | entities | unchanged line | same `lastrevid` | changed | meta only | gone | changed + gone | 95% CI | by bytes | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261005 → 20261012 | 46,417 | 45,981 | 45,981 | 433 | 2 | 1 | **0.94%** | 0.43–1.63% | 1.03% | 0.34–2.07% |
| 20261012 → 20261019 | 46,416 | 46,108 | 46,108 | 307 | 0 | 1 | **0.66%** | 0.26–1.26% | 0.76% | 0.24–1.49% |
| 20261005 → 20261019 (two weeks) | 46,417 | 45,700 | 45,700 | 713 | 2 | 2 | 1.54% | 0.80–2.44% | 1.67% | 0.65–3.06% |

| pair | ids added | live share at the end | new bz2 bytes | new entities | share |
|---|---|---|---|---|---|
| 20261005 → 20261012 | 60,589 | 0.87 | 24.8 MB | ~50,100 | 0.04% |
| 20261012 → 20261019 | 82,057 | 0.90 | 29.5 MB | ~71,900 | 0.06% |

- **"Same lastrevid" equals "unchanged line" in every pair.** No line differed without a
  new revision. So nothing in the dump changes for entities nobody edited (`modified`,
  sitelink badges, key order or number formatting), and a raw line hash works as is. A
  normalised hash isn't needed.
- **A new `lastrevid` with unchanged content** (meta-only, an edit and its revert) is 2 in
  46k. Its derived rows differ only in the entities table (`lastrevid`, `modified`).
- **None was new within the windows** (no undeletion or un-redirect seen). Gone: 1 in 46k
  a week.
- **Two weeks:**
  - 717 touched over two weeks against 436 + 308 in the two weeks, so 27 entities were
    touched in both.
  - That is 9% of the second week's, 9 times what independent weeks would give. The
    same entities are edited again.
  - Skipping a release costs little: two weeks' churn is 1.5–1.7%.

### Where the edits are

**By id range** (changed + gone; 95% CI over the range's windows):

| id range | windows | entities | 10-05 → 10-12 | 10-12 → 10-19 |
|---|---|---|---|---|
| Q0–Q25M (the oldest items) | 6 | 7,906 | 3.90% (2.13–7.02%) | 1.24% (0.77–2.13%) |
| Q25M–Q60M (mostly scholarly) | 12 | 11,328 | 0.31% (0.10–0.53%) | 0.16% (0.03–0.33%) |
| Q60M–Q100M | 9 | 15,040 | 0.31% (0.06–0.76%) | 0.96% (0.03–2.66%) |
| Q100M–Q142M | 5 | 12,141 | 0.39% (0.06–0.67%) | 0.40% (0.09–0.69%) |

**By shard:** 0.06–2.37% a week. A shard is a hash of the id, so its spread is only that
of its 4 windows' id ranges (`results/churn.txt`).

**By set and table** (the share of each table's entities whose rows change, and of its
bytes; 20261012 → 20261019, then 20261005 → 20261012):

| set | table | entities touched | bytes touched |
|---|---|---|---|
| main | claims | 0.39%, 0.90% | 1.10%, 1.73% |
| main | labels | 0.05%, 0.06% | 0.05%, 0.09% |
| main | descriptions | 0.68%, 0.38% | 1.81%, 0.43% |
| main | aliases | 0.04%, 0.23% | 0.14%, 1.04% |
| main | links | 0.05%, 0.09% | 0.45%, 0.85% |
| main | entities (every edit) | 1.10%, 1.53% | 2.11%, 2.76% |
| scholar | claims | 0.01%, 0.05% | 0.01%, 0.07% |
| scholar | labels, descriptions, aliases, links | ≤ 0.04% | ≤ 0.04% |
| scholar | entities | 0.02%, 0.07% | 0.02%, 0.08% |

**Concentration:**
- **Across windows:** the dispersion of the windows' counts against a binomial's at the
  pooled rate is 61.5 and 44.7 (1 is uniform). The 4 busiest windows hold 58% and 67% of
  the changes.
- **The sweeps** (`results/churn.txt`, "What changed"; the busiest windows):
  - **Q5.71M, 10-05 → 10-12:** 82 villages (Q532) given P1082, P1538, P1539 and P1540
    (population, households, male and female population), all on 2026-10-07. 10% of the
    window changed.
  - **Q63.94M, 10-12 → 10-19:** 128 Hausa descriptions on items of Q57733494, all on
    2026-10-14. 6.2% of the window.
  - **Q11.24M, 10-05 → 10-12:** P407 and P495 on 28 items of Q21198342, on 2026-10-10.
  - **Throughout:** Assamese (`as`) descriptions on humans (Q5).
- **What changed, overall:** in the later week, descriptions in 61% of the changed
  entities and claims in 36%; in the earlier, claims in 59% and descriptions in 25%.
  Labels change in 4–6%, aliases in 3–15%, sitelinks in 4–5%.
- **So:** edits fall on old, popular items and on bot sweeps over a class's (often
  consecutive) ids, on a day. The scholarly articles, 38% of entities and half the
  published bytes, are barely touched.

## 2. The split's CPU at a release

**Measured** (bench 17.9.0, quick tier, master b110f78, its 6 runs, means). The split's
CPU on the slice is 32.5 s (`split_cpu_total`):

| part | how it is measured | slice CPU-s | share |
|---|---|---|---|
| bz2 decode | `bz2_decode_cpu`: lbzip2 alone over the slice's bz2 | 17.5 | **53.8% ± 2.9%** |
| claims' Parquet encode | `partitioning.core.write_table` (process clock, its thread pool) | 7.3 | 22.2% |
| reshape and build | `dump.write_chunk`'s own: the Rust scan (line lexing, ids, the claims built from the line), entities | 4.3 | 13.2% |
| refs sidecar, `statements`, census | phases.py | 0.27 | 0.8% |
| not attributed | the rest: the main thread's line reads and chunking, lbzip2 in the pipe over lbzip2 alone, `route_parts`' own (its row is a clock artifact, −3.8 s) | 3.2 | 9.9% |

- **A cross-check:** the gz split, whose decode (igzip) is 0.59 s, costs 14.1 s. So the
  split's own CPU is 13.5 s, 41% of the bz2 split's, against 46% here.
- **Line lexing and id extraction:**
  - inside the Rust scan, they are part of the 13%;
  - alone, on the decoded text: `bytes.split` on newlines, the split's id regex on each
    line and an xxh3-64 of it cost **3.0–4.7% of the decode's CPU** (`hash_bench.py`;
    shared machine, 3 rounds, ratios only);
  - blake2b would be 7%;
  - xxh3 over the whole text as one buffer is 0.6%, the floor.

**At a release (modelled):** the split's ETA scales by bz2 bytes, `split_eta` × 4 cores.

| | core-h | on 4 cores |
|---|---|---|
| the split today | 11.8 | 2h57m |
| the bz2 decode | 6.3 | 1h35m |
| **a hashing-only pass: decode + lines + id + xxh3** | **6.7** | **1h40m** |
| the rest (encode, reshape and build, writes) | 5.4 | 1h22m |
| the rest at the measured churn (0.93% of bytes) | 0.05 | 1 min |
| the dump's download at 14.5 MB/s | | 1h59m |

- **The decode is not small:** it is over half the split. A release can't go below a
  decode of every byte of the dump, and the incremental split is that pass plus 1% of the
  rest.
- **That pass is under the dump's own download** (1h40m against 1h59m). So the
  incremental split is bound by Wikimedia's 3-connection rate, not by CPU, and it leaves
  ~0.6 of the 4 cores idle.
- **With the gz dump** the trade turns: igzip decodes the slice in 0.59 CPU-s, so a
  release's hashing pass would be ~20 core-min. But the gz is 156.6 GB, 3h00m at 14.5
  MB/s. With nearly all of the CPU then free for the merge during the download, it could
  match I1's walls, not beat them. Not pursued.

## 3. The model (modelled)

`model.py` takes the same record and, for each of its 6 runs:
- **master:** bench/overlap.py's `release_wall` at 7, 15, 30, 60 and 120 MB/s. These are
  the bench's own figures (15 MB/s added the same way), reproduced exactly at the
  bench's 4 rates;
- **the variants below:** a 10 s-step simulator on 4 cores.

The walls are the means of the 6 runs.

**The incremental release, its assumptions:**
- **The download:** the dump at 14.5 MB/s (`RELEASE_DOWNLOAD_RATE`).
- **The split:**
  - **Its CPU a byte:** split CPU × (decode share × 1.05 + (1 − decode share) × churn), at
    most 4 cores, no byte decoded before it is downloaded.
  - **Its inputs:** each run's own decode share; the hash 5% of the decode; churn by bytes
    0.93% (the two weeks' mean, new entities included).
- **The pipeline for the changed rows:** `release_pipeline_cpu` × churn × 1.5 (its parts'
  fixed costs on fewer rows), inside the split.
- **The merge:**
  - **Its CPU:** `mu` × finalise's CPU (`finalise_eta` × 4 = 8.1 core-h).
  - **mu:** finalise's phases (this record) put **80% (76% on the x3 run)** of its listed
    CPU in what a merge keeps. That is the encode (`_write_file_parallel`, the largest),
    the checks (`_check_written`, `_check_sort`, `_check_sorted`, `_check_whole`), the
    sorted files' reads (`_read_key`, `_read_bucket`), `write_table`, `_pack_file` and the
    metadata.
  - **What a merge drops:** compaction, the bucket source and the sorts.
  - **Its range:** mu 0.8 central, 0.6–1.0 in the sensitivity.
- **No group files:** the changed rows (~1% of the parts) stay on local disk (design C).
- **claims_labels:** rebuilt whole once the rest is merged, in 300 s (`CLAIMS_LABELS_BUILD`).
- **The upload:** 128.9 GB × (1 − dedup), each part as it is written, FIFO on one link.
- **The variants:**
  - **I0:** the merge once the split has ended.
  - **I1:** the merge available by #180's closure curve (both sets, regional 128 MiB
    parts), against the split's progress. That needs #180's synchronised sweeps. It takes
    the cores the split leaves, and up to 2 when the link is short of parts, the split
    giving way.
- **F0, F1:** the same simulator with every entity processed (churn 1, no hash, mu 1) and
  no group files, without and with the frontier. F1 is about #180's (c). The
  incremental's own gain is I − F at the same dedup.

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| master (bench 17.9.0) | 10h50m | 7h41m | 6h40m | 6h17m | 6h12m |
| F0: full, design C, finalise after the split, dedup 55% | 6h24m | 6h07m | 6h06m | 6h06m | 6h05m |
| F0, no dedup | 9h06m | 6h26m | 6h08m | 6h06m | 6h06m |
| F1: F0 + the frontier, dedup 55% | 6h11m | 6h07m | 6h06m | 6h06m | 6h05m |
| F1, no dedup | 6h55m | 6h12m | 6h08m | 6h06m | 6h06m |
| **I0, churn 1×, dedup 55%** | **4h20m** | **3h45m** | **3h44m** | **3h44m** | **3h43m** |
| I0, churn 1×, no dedup | 7h08m | 4h24m | 3h46m | 3h44m | 3h44m |
| I0, churn 2×, dedup 47% | 4h42m | 3h47m | 3h44m | 3h44m | 3h43m |
| I0, churn 2×, no dedup | 7h08m | 4h24m | 3h46m | 3h44m | 3h44m |
| I0, churn 5×, dedup 35% | 5h22m | 3h51m | 3h46m | 3h45m | 3h44m |
| I0, churn 5×, no dedup | 7h09m | 4h25m | 3h46m | 3h45m | 3h44m |
| **I1, churn 1×, dedup 55%** | **3h34m** | **3h28m** | **3h27m** | **3h26m** | **3h26m** |
| I1, churn 1×, no dedup | 5h20m | 3h35m | 3h28m | 3h27m | 3h26m |
| I1, churn 2×, dedup 47% | 3h43m | 3h29m | 3h28m | 3h27m | 3h27m |
| I1, churn 2×, no dedup | 5h20m | 3h37m | 3h29m | 3h28m | 3h27m |
| I1, churn 5×, dedup 35% | 4h01m | 3h34m | 3h32m | 3h31m | 3h31m |
| I1, churn 5×, no dedup | 5h20m | 3h40m | 3h33m | 3h32m | 3h31m |

**The incremental's own gain** (I − F, the same simulator and dedup):

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| I0 − F0, dedup 55% | −2h04m | −2h22m | −2h22m | −2h22m | −2h22m |
| I0 − F0, no dedup | −1h58m | −2h02m | −2h22m | −2h22m | −2h22m |
| **I1 − F1, dedup 55%** | **−2h37m** | **−2h39m** | **−2h39m** | **−2h40m** | **−2h39m** |
| I1 − F1, no dedup | −1h35m | −2h37m | −2h40m | −2h39m | −2h40m |

**Sensitivity** (I1, churn 1×, dedup 55%):

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| mu 0.6 | 3h17m | 3h05m | 3h03m | 3h03m | 3h03m |
| **mu 0.8 (central)** | 3h34m | 3h28m | 3h27m | 3h26m | 3h26m |
| mu 1.0 (the merge costs a whole finalise) | 3h56m | 3h52m | 3h51m | 3h50m | 3h50m |
| the merge capped at 1 core while the link waits | 3h47m | 3h28m | 3h27m | 3h26m | 3h26m |
| up to 4 cores | 3h34m | 3h28m | 3h27m | 3h26m | 3h26m |

**Where the time goes:**
- **I0 at 60 MB/s:**
  - the split ends at 2h01m (the download at 1h59m);
  - the merge on 4 cores ends at 3h38m;
  - claims_labels and the last part's upload end at 3h44m.
- **I1** moves part of the merge into the split's idle cores and the link's waits.
- **Its floor is the CPU:** decode 6.7 + merge 6.5 core-h on 4 cores is 3h18m, plus
  claims_labels.
- **The merge is now as large as the decode.** The next structural step is to copy
  untouched pages rather than re-encode them (Plan, stage 6).

**The parts of the brief, as modelled:**
- **The merge's cost** is above: 6.5 core-h against finalise's 8.1.
- **Reading the release before's outputs:**
  - **From local disk** (its local copy, `HUB_COPY_DIR`, kept from the release before):
    129 GB of sequential reads, ~2 min at 1 GB/s. Their decode is in mu: finalise reads
    its compacted files the same way.
  - **Otherwise a download:** 129 GB at 75 MB/s (`link.DOWNLOAD_RATE`) is 29 min, the
    pipeline's own Hub use at release time, as today's finalise downloads its 124 GB of
    group files. It runs beside the dump's 1h59m download, and is hidden whenever the
    downlink carries ~33 MB/s beside the dump's 14.5. Not done here.
- **claims_labels' propagation** is smaller than it looks:
  - **Its rows** are one a (`ref`, `field`) a language, the label of a referenced entity
    (`claims_labels.py`), not one a claim.
  - **A label edit in R** changes only R's rows in that language's key.
  - **A changed claimer** changes only which refs exist: a ref first or no longer
    referenced. The union of refs is recomputed from the sidecars, the changed entities'
    new ones included, in 17–33 s (`refs_eta`).
  - **So claims_labels is rebuilt whole** in its 300 s from the merged labels. Its upload
    dedups as the labels do.
- **The upload:** full size unless Xet dedups (next section). Modelled at 0 and at the
  dedup model's figure.

## 4. How much Xet could dedup (modelled)

`dedup.py`, from the measured churn:
- **Why an untouched page dedups:**
  - The published files are written with Parquet content-defined chunking (`compact.py`:
    `use_content_defined_chunking`; pq-glue `content_defined_chunking`).
  - A column's page boundaries follow its values, and a page none of whose rows changed
    re-encodes to the same bytes wherever it lands, in any part file.
  - Xet chunks those bytes (content-defined, ~64 KiB), so the page's chunks are already in
    the repository.
- **What a page holds** (consecutive entities of its set, in string order):
  - **claims:** 512 KiB over the widest leaf columns' ~45 bytes a statement
    (statement id, a snak or reference hash) times the statements: **~1,040 entities in
    main, ~570 in scholar**;
  - **a language or site key:** ~13–17k rows, its entities by the key's rows an entity
    (from `dataset_cards_metadata.json`);
  - **entities:** 22k.
- **String order mixes digit lengths.** The `Q45…` block holds Q45,000,000–Q45,999,999
  and, a tenth as many, Q4,500,000–Q4,599,999. A page at window w's ids also takes the
  ids at ×10^k, by the set's density there, at the rate of the window nearest in log(id).
- **P(untouched)** = Π (1 − r)^(E × share).
  - **Rate shrinkage:** each window's rate is shrunk towards its set's, so a window where
    none changed isn't taken as never changing.
  - **The Xet edge:** each touched run's edge costs one Xet chunk.
  - **Weighting:** by each window's bytes of the table.
- **Not modelled:**
  - row groups' starts shifting when rows are inserted before them (about 1 page a column
    a row group, ~0.5%);
  - dictionary-encoded column chunks, whose indices shift when an early new value enters
    the dictionary (their low-cardinality columns are small).

  Both lower the figure a little.

**At 512 KiB pages, 20261012 → 20261019:**

| set | table | GB | untouched pages, uniform | windowed | dedup | GB |
|---|---|---|---|---|---|---|
| main | claims | 41.4 | 2% | 9% | 5% | 2.3 |
| main | labels | 6.6 | 75% | 75% | 73% | 4.8 |
| main | descriptions | 3.8 | 65% | 65% | 63% | 2.4 |
| main | aliases | 2.3 | 76% | 76% | 73% | 1.7 |
| main | links | 5.6 | 70% | 70% | 65% | 3.6 |
| main | entities | 0.2 | 0% | 0% | 0% | 0.0 |
| scholar | claims | 52.8 | 94% | 94% | 92% | 48.3 |
| scholar | descriptions | 7.5 | 97% | 97% | 96% | 7.2 |
| scholar | labels | 3.5 | 98% | 98% | 98% | 3.4 |
| scholar | entities, aliases, links | 0.6 | | | | 0.1 |
| **both** | **all but claims_labels** | **124.3** | | | **59%** | **73.9** |

| pages | 10-12 → 10-19 | 10-05 → 10-12 | 2× churn | 5× churn |
|---|---|---|---|---|
| 256 KiB | 62% | 54% | 51% | 41% |
| **512 KiB** | **59%** | **50%** | **47%** | **35%** |
| 1 MiB | 55% | 44% | 41% | 27% |

(the 2× and 5× columns are the two weeks' mean)

- **Central: ~55%.** The model takes 55%, 47% and 35% at 1×, 2× and 5× the churn.
- **Uniform churn does kill page dedup** where churn is ~1% and a page holds ~1,000
  entities: main claims keep 2% of pages untouched under a uniform rate, 9% with the
  measured concentration across id ranges.
- **The concentration that matters is between the sets.** The scholarly set changes
  0.01–0.07% a week, and the published layout already holds it apart.
- **A full release gets the same dedup** if its rows and writer are the same. So the
  F rows above take it too, and dedup is no argument for or against the incremental
  design.
- **It is unmeasured.** Hub-side behaviour (which chunks the Xet client's global dedup
  finds) is not modelled. The Plan measures it with a stand-in.

## 5. Integrity

### The derivation fingerprint: when it forces a full rebuild

- **Rows from the line alone:** an entity's derived rows depend only on its dump line, its
  set (its own P31, `scholarly.py`), the derivation code and the release schema.
- **The exceptions** are rebuilt whole every release:
  - claims_labels (other entities' labels);
  - the dataset cards' figures (aggregates over the output);
  - the release schema itself (below).
- **Two fingerprints:**
  - **The derivation version** (`DERIVATION_VERSION`, by hand) covers what makes a row:
    - the reshape (`reshape/src`) and `dump.py`'s reshaping;
    - `claims_build`, `entities_build`, `process`'s normalisers and the claims rules
      (e.g. #178's rule B for coordinates);
    - `SCHOLARLY_CLASSES`;
    - the versions of polars-genson, polars, pyarrow and orjson (number and string
      decoding);
    - claims.avsc and the release schema's major version.
  - **The layout** (writer, zstd level, CDC, part and row-group sizes) doesn't count:
    every part is re-encoded by the merge anyway.
- **The guard, on CI:**
  - **The golden digest:** the bench records each table's order-independent row digest
    of the slice (count and Σ of a 64-bit hash of id and row bytes, mod 2^64) per
    derivation version.
  - **A PR that changes those rows** without bumping the version fails parity, as an
    unintended output change does today.
  - **A bump** makes the next release full for the tables it names (claims only, for a
    claims rule), or for all.
- **Reuse requires** the release before to have been built at the same derivation
  version, recorded in its manifest. Otherwise the release is full.

### "Unchanged", exactly

- **The test:** the same `lastrevid` *and* the same line hash (xxh3-128 of the line
  after its separating comma).
- **The record:** both are kept by the release before, in its local index (id, lastrevid,
  hash; 121.8M × 32 bytes, ~3.9 GB raw).
- **Why both:**
  - **The hash alone** catches a serialisation change no revision made. None in 139k
    comparisons, but Wikibase could change its output.
  - **`lastrevid` alone** needs no local index: it is in the published entities table.
- **Without the index**, the release runs on `lastrevid` and a larger sample audit
  (below), which catches a serialisation change across the dump at once.
- **A global change found:** if more than 0.1% of entities with the same `lastrevid`
  differ in hash, the dump's serialisation has changed, and the release becomes full.
  The split can switch mid-run: it decodes every line anyway.
- **A hash collision** with an entity's own previous line is 2^-128 an entity.

### Proving the incremental output is the full output

1. **Per-region digests, kept with each release:** for each table, set and region (the
   string id ranges of #180), the row count and the order-independent digest above.
2. **The merge's own check, per region:**
   - **expected:** the release before's digest, minus the digest of the rows it removes
     (the touched and gone ids' old rows, read in the merge), plus the new rows' digest
     (from the build);
   - **got:** the merged output's digest, computed as it is written;
   - **an exact equality**, by count and digest, like `_check_written`, from the merge's
     own sources.
3. **The exact checks stay, run on the merged output:**
   - `_check_sort` (global order, stable);
   - `_check_written` (rows written are the rows planned);
   - each part's boundaries (last id < next first id);
   - `_check_whole` (Σ regions = the release's totals);
   - **the new duplicate check:** an id's rows are contiguous and from one source, the
     release before or the new build, never both;
   - the schema version of every part;
   - parity, on the bench.
4. **Deletions, exact:**
   - **gone ids:** the release before's ids (its entities table, sorted) less the new
     dump's ids (all seen by the split, collected as numbers, sorted at the end, ~1 GB);
   - **per region:** known once the region's frontier passes, with #180's late-row redo
     if a straggler appears.
   - **Set moves** (P31 changes) delete the id from both sets' previous rows.
5. **A sampled re-derivation every release:**
   - 0.1% of the unchanged entities (122k, by a hash of id and release) take the full path
     as well, and their rows must equal the reused ones, byte for byte;
   - it costs ~0.1% of the split's non-decode CPU;
   - it catches any derivation dependency the fingerprint misses (a library change in
     number formatting, say) that touches more than ~0.003% of entities, with 95%
     probability.
6. **A periodic full release** (every 4th, or at any derivation bump):
   - a full build;
   - its per-region digests compared with the incremental chain's;
   - any difference halts promotion and is reported by region and table;
   - the chain restarts from it.
7. **On CI**, a two-release bench slice (Plan stage 2) runs both paths and compares every
   published file's rows: the incremental's parity.

### Restart

- **Immutable inputs:** the release before's local files, verified once against its
  manifest's sha256s (~80 core-s for 129 GB). They are never written.
- **The split's manifest:** as today, plus each chunk's reused and built counts and its
  index lines (id, lastrevid, hash). A rerun skips the chunks already in it, and
  re-decodes from the last durable chunk with #180's sweeps.
- **The regions ledger** (#180's `regions.jsonl`): per region, `merged`, `written`,
  `committed`, `verified`. A region resumes at its last stage, writing each part to a tmp
  and renaming it, as today.
- **An old region's parts** are deleted only once its new parts are `verified`, so a
  rerun always has its inputs.
- **A mid-run fallback to full** (a fingerprint mismatch, a serialisation change)
  restarts the split in full mode. The merge's work isn't kept.

### Disk and memory, a 16 GB laptop (modelled)

| | peak disk | peak RSS |
|---|---|---|
| master (#180) | ~235 GB | 10–12 GB (finalise), 11.2 GB (claims_labels) |
| incremental, I0/I1 | **~240 GB**: the dump 103.5, the release before's 129 (read-only, each region deleted once its successor is verified), the new parts (up to 129 by the end, as the old go), the changed rows' scratch ~1–5, the index ~2–4 | **~3–4 GB** in the merge (a 128 MiB row group a core, in memory ~3×); ~3 GB in the split; claims_labels as today |
| incremental, the dump not kept (deleted shard by shard once its chunks are durable, #180's sweeps) | ~140 GB | as above |
| the periodic audit (a full release beside the incremental one) | ~360 GB | as master |

- **The release before's parts** are the published files themselves, the local copy
  `download-wikidata` makes. Nothing new is published.
- **A laptop** with a few hundred GB holds the incremental release. The audit's two
  copies at once need a larger disk: run it in place of an incremental release, not
  beside one.

### Progressive finalise (#180, #181) and schema evolution (#182)

- **The frontier carries over unchanged.**
  - **When a region can merge:** when the frontier has passed it (all its changed ids
    seen), so its closure is #180's curve.
  - **What carries over:** the speculative closure with a 10,000-id margin, the late spool
    and the redo by merge (`2026-10-26-progressive-finalise.md`, Gate 3).
  - **What it buys:** I1's −17 to −46 min over I0, for the sweeps of #180's PR 2.
- **The census of #181** already records each chunk's shard and lag. The incremental split
  adds its reused and built counts beside them.
- **A widening of the release schema** (#182, auto mode: a new nullable field or variant,
  found in a changed entity):
  - **No row is reprocessed.** The reused rows' values are unchanged and conform to the
    widened schema with the new columns null.
  - **The cost:** the claims parts get the new column, but the merge re-encodes every part
    anyway, so a widening adds nothing.
  - **The result equals a full build's:** the stored release schema only widens, and the
    unchanged lines' shapes were all in it, so the incremental schema (before ∪ changed
    lines' shapes) is the full build's.
  - **Xet** still dedups the other columns' pages.
- **A change in derivation** (a rule like #178's, a decoder change, a renamed or retyped
  column, a major schema version) is a derivation-version bump:
  - **What it forces:** a full re-derivation of the tables it names, for every entity.
  - **A later refinement:** limit it to the entities the rule can touch (#178's: those
    with a globe-coordinate value, found in the release before's claims). Not in the first
    design.
- **frozen and strict modes** change nothing here.

## Verdict: big fish

- **The structure holds:**
  - **the churn is ~0.7–1% of entities a week**, new entities 0.05%;
  - **nothing changes for entities no one edited**, so a plain line hash, or `lastrevid`
    alone, finds the unchanged 99%;
  - **the decode is the split's floor, and is under the download.**
- **An incremental release is ~3h25m–3h45m at every link rate of 15 MB/s and above,
  against master's 6h12m–7h41m:**
  - −2h20m to −2h40m against a full release with design C in the same simulator;
  - about 3h35m at 7 MB/s with dedup, 5h20m without (I1).
- **It is robust to the churn:** 5× the measured churn moves the fast-link walls ≤5 min.
- **What it doesn't do is reach the decode floor.** The merge re-encodes all published
  bytes (6.5 core-h). Copying untouched pages would cut that for the scholarly set's
  ~45% of bytes (stage 6).
- **What would make it small:**
  - **A derivation bump most weeks,** as in this campaign today: each one makes that
    release full. The incremental gain is for steady-state weekly releases.
  - **A merge costlier than modelled** (mu 1: 3h50m).
- **What needs the user:**
  - **an incremental release mode in the recipes;**
  - **keeping the release before's local copy** (129 GB) between releases;
  - **the audit cadence;**
  - **downloading the release before's published files** when there is no local copy.
    That is the pipeline's Hub read at release time (`download-wikidata`), not an
    agent's.

## Plan

1. **PR 1 (measurement only, no output change): the churn index in the split.**
   - **What the split records:** each entity's (id, `lastrevid`, xxh3-128 of the line)
     in the reshape's scan (Rust, the `xxhash-rust` crate), written per chunk beside its
     manifest line.
   - **The census:** given a previous release's index (`WIKIDATA_PREVIOUS_INDEX`), the
     split's summary gives:
     - unchanged, same-revision-different-line, meta-only, changed, new and gone;
     - by #180's regions and by set.
   - **The bench:** CI measures its CPU (expected ≤1% of the split: the hash is 3–5% of
     the decode in Python, less in the scan), and the published files are unchanged
     (parity).
   - **What it gives:** the next real release records its whole index, and the one after
     it a census of 121.8M entities against this entry's 46k.
2. **PR 2: a two-release bench slice and the incremental's parity.**
   - **The slice:** 20261019's windows for the bench slice's ids, by the byte-range reads
     of this entry's `churn.py`.
   - **The comparison:** a full build of each, and the published rows of full(B) against
     incremental(A → B), compared file by file; the golden row digests per derivation
     version.
3. **PR 3: the merge finalise** (`WIKIDATA_INCREMENTAL=1`, off):
   - a release before's sorted local copy merged by region, less the touched and gone ids,
     plus the changed rows;
   - the per-region digests, the duplicate check and every exact check on the output.
   - Benched against the full finalise on PR 2's slice: its CPU against mu.
4. **PR 4: the incremental split:**
   - the unchanged lines' rows not built;
   - the touched-id set and the gone ids;
   - the sampled re-derivation;
   - the fallback to full;
   - the restart cases.
5. **PR 5: recipes and gates:**
   - `release --incremental`;
   - the derivation version in the manifest;
   - promotion gated on every region `verified` and on the sample audit;
   - the audit cadence (every 4th release full).
6. **Later:** with #180's sweeps (its PR 2), merges during the split (I1). Then
   page-level copying of untouched pages in pq-glue's writer: that cuts the merge for the
   scholarly set. And a stand-in Xet chunker in `bench/link.py`, to measure the dedup
   between the two-release slice's parts.

## Reproduce

```
cd docs/journal/2026-10-26-incremental-release
python3 -I churn.py --newest 20261019 --older 20261012 20261005 \
    --boundaries ../2026-10-26-progressive-finalise/layout-20261019/boundaries.json \
    --points 32 --bytes 1310720 --out raw                          # ~174 MB of range reads
python3 -I bside.py --newest 20261019 --windows raw/windows.jsonl --bytes 1310720 \
    --out raw/bside.csv.gz                                          # ~42 MB
python3 -I tails.py --pairs 20261005:20261012 20261012:20261019 --out raw/tails.json   # ~7 MB
python3 -I analyse.py raw results                                   # results/churn.txt, churn.json
python3 -I dedup.py raw results ../../dataset_cards_metadata.json   # results/dedup.txt, dedup.json
git show origin/bench-results:runs/quick/b110f78160a96ddaa771ae7a0cedaf36c07c5968.json > rec179.json
python3 -I model.py rec179.json ../../../bench results              # results/model.txt
# the hash's cost beside the decode: a bz2 file of whole blocks (bzip2recover's) and its text
uv run --no-project --with xxhash python -I hash_bench.py all.bz2 all.json
```

- **Dependencies:** all stdlib but `hash_bench.py` (xxhash). `bz2probe.py` is #180's,
  copied.
- **raw/:**
  - `entities.csv.gz`: one row an entity and pair, its status and what changed;
  - `bside.csv.gz`: each entity's set and fields' bytes;
  - `windows.jsonl`: each point's offsets, ranges, probes and bytes;
  - `tails.json`: the shard ends;
  - `churn.log`: the run's log.
