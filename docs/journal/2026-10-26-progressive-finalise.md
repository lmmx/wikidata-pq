# 2026-10-26: Progressive finalise: the dump's real layout, a frontier, and regions closed during the split

Every published table but claims_labels is sorted by string id and cut into parts in
string order. Today's release is phased (split, group files to the build branch, finalise
downloading them back, sorting and uploading), on the premise that no published part is
final before the whole dump is read. This entry tests a structural alternative:

- string order is a known permutation of numeric order, so if the dump is read in rising
  numeric order, each string-prefix region becomes final once the numeric frontier passes
  its largest possible id (Q2… at Q30M, Q9… at Q100M, Q10… at Q110M, and so on; only the
  regions holding the last ids, Q14…, wait for the end);
- the dump is 8 shards, each in rising order, so reading them as 8 synchronised sweeps
  makes the frontier (the minimum over shards) rise monotonically through the split;
- closed regions could then be finalised and committed to the build branch during the
  split, and the group files' round trip could go.

The scripts and data are in `2026-10-26-progressive-finalise/` (see Reproduce). Measured
and modelled figures are kept apart: every release-scale time, byte and disk figure is
**modelled**; the layout, the lag and the decode timings are **measured**.

## Current State

- **Docs only.** Nothing in `src/`, `reshape/` or `bench/` changed. No Hub call. About
  224 MB of byte-range reads from dumps.wikimedia.org (below); nothing kept on disk but
  the tables in this folder.
- **Gate 1 (remote blocks can be inspected): passed.** Byte-range reads, a bit-level
  search for the bz2 block magic, each block rewrapped as its own stream and decoded by
  libbz2 (which checks each block's CRC). The block offsets and the decoded text are
  identical to `bzip2recover`'s on the same window.
- **Gate 2 (the real layout): passed, and coherent.** `wikidata-20261019-all.json.bz2`
  (103,488,697,311 bytes) is **8 shards back to back**, each a sweep over the whole id range
  in page-id order. Each sampled id is in the shard `sha1(id) mod 8` names, all 23,645 of
  them, which is exactly Wikibase's exporter rule. The bz2 is one stream (the shards'
  gzip output re-compressed by `gzip -dc | lbzip2`); the gz is many gzip members, one a
  batch a shard. Ids within a shard are monotone to within 2 (max lag 2 in 77,529
  contiguous ids), except the first ~1,000 entities of a shard (ids below Q10k, lag up
  to 1,968). No id among 108,546 came more than 10,000 behind its shard's running maximum.
- **Gate 3 (closing a region safely, and the cost): a design with no ordering
  dependence.** A region closes speculatively (frontier minus a 10,000-id margin). The
  split records every id's region. An exact per-region inventory at the end catches any
  late id, and its part is rebuilt from its own output plus the late rows, with no group
  files and no download. Publication stays gated at `promote-release`. The measured lag
  predicts **no** redo; at the 95% upper bound on the straggler rate (none seen in
  ~10^5), a redo would be at most ~96 GB of parts (as much as finalise does once more).
- **The closure curve (modelled from a byte-weighted sample):** of both sets' 128.9 GB
  of published bytes, **32% (24–40%) is final at half the split, 59% (54–65%) at 80%,
  68% (63–72%) at 90%** (regional 128 MiB parts). The rest is the language and site
  tables' small keys (one file a key, so their one part spans every region), the Q14…
  regions and claims_labels. With every key cut into at least 4 files (design B,
  authorised once), it is 36%, 66% and 82%.
- **The release walls (modelled), against master's 12h01m / 8h53m / 7h40m / 7h07m / 7h04m
  at 7 / 15 / 30 / 60 / 120 MB/s:**
  - **(b)** synchronised split + regional finalise, round trip kept: 10h56m / 7h31m / 6h44m / 6h43m / 6h42m;
  - **(c)** (b) without the group files: **7h21m / 6h41m / 6h32m / 6h31m / 6h30m**;
  - **(d)** (c) + L1b's preupload: **7h17m / 6h36m / 6h32m / 6h31m / 6h30m**;
  - (c) with every key in at least 4 files: 6h50m / 6h33m / 6h30m / 6h29m / 6h28m.

  Against this simulator's own master (a_sim, which reads 10–30 min shorter than (a)),
  (c) saves **−4h31m, −1h42m, −40, −13 and −12 min**. **Of that, the frontier itself
  contributes −2h53m, −1h06m, −37, −19 and −18 min** over design C alone (no group files,
  finalise after the split: c0).
- **Disk and memory (modelled), a 16 GB laptop:** peak RSS stays about today's (10–12 GB),
  if finalise takes at most 2 bucket sorts beside the split. Peak disk falls from ~235 GB
  (master, L3 off) to ~130 GB. If closed parts are kept on disk until the final check, so
  that a redo needs no download, it is ~225 GB.
- **Verdict: big fish**, at every rate under ~30 MB/s, and a real one above it (−12 to
  −33 min). It depends on design C (no group files, authorised as a prototype) and on
  design B's one boundary change. **The first PR is measurement only:** the split records
  each chunk's shard and frontier lag for the whole dump, at the next real split (Plan).

## Gate 1: inspecting remote bz2 blocks

`bz2probe.py` reads a byte range and finds bz2's 48-bit block magic (0x314159265359)
at any bit alignment: blocks inside a stream are bit-aligned, never byte-aligned. It then
copies the bits from the magic on behind a `BZh9` header, as `bzip2recover` does, and
decodes with Python's `bz2` (libbz2), which checks each block's CRC. A probe at an
offset decodes its first ~1.8 MB of text and keeps the ids of the whole lines.

**Cross-checked against `bzip2recover` (measured):** on a 320 KB window at byte 50 GB,
`bzip2recover` found its blocks at the same bit offsets: block 1 runs from bit 343,304,
which is 343,256 + the 48-bit magic, and so on. The probe's 3,600,001 decoded bytes are
byte-identical to the concatenated decode of `bzip2recover`'s block files, 444 ids each.
lbzip2 also writes some tiny blocks (135 bits) between the large ones. Both tools see
them.

**Bytes transferred (measured):**

| read | count | bytes each | total |
|---|---|---|---|
| survey: even offsets (`survey.py`) | 101 | 320 KB | 33.1 MB |
| survey: bisection of the 7 resets, to 4 MB | 49 | 320 KB | 16.1 MB |
| survey: one read across each reset (streams, end marks) | 7 | ~4.4 MB | 32.8 MB |
| entity records: even offsets (`windows.py`) | 101 | 320 KB | 33.1 MB |
| entity records: long windows, lag census | 9 | 8 MB | 75.5 MB |
| `bzip2recover` cross-check | 1 | 320 KB | 0.3 MB |
| decode timing (8 shards × 4 MB) | 8 | 4 MB | 33.6 MB |
| **all** | | | **~224 MB** |

The brief's later revision asked for a sparse start at 16–32 offsets. This survey had
already started at 101 even offsets (33 MB), and every read stayed in memory, except the
`bzip2recover` and decode-timing windows, which were deleted after use.

## Gate 2: the dump's real layout

### What the exporter does (code read)

**Wikibase** (mediawiki/extensions/Wikibase, master at 45d89eb1d5, 2026-10-26):
- `repo/maintenance/DumpEntities.php:63-64, 178-179, 230`: `--sharding-factor` and `--shard`
  set `DumpGenerator::setShardingFilter`.
- `repo/includes/Dumpers/DumpGenerator.php:169-181`, `idMatchesShard`: an entity is in shard
  `hexdec(substr(sha1($id->getSerialization()), 0, 8)) % shardingFactor`, a hash of its id.
- `repo/includes/Store/Sql/SqlEntityIdPager.php:116-136`: the ids come from `page`, `page_id >
  position`, the entity namespaces, **`ORDER BY page_id ASC`**, redirects excluded
  (`NO_REDIRECTS`, a left join on `redirect` with `rd_from IS NULL`;
  `DumpEntities::getRedirectMode`).

**Wikimedia's dump job.** The last readable version is operations/puppet's
`modules/snapshot/files/systemdjobs/wikibase/dumpwikibasejson.sh` and
`wikibasedumps-shared.sh` at f9651f06bb. Commit 1c0b124c50 (2025-08-05) removed them when
the dumps moved to Airflow:

- `dumpwikibasejson.sh:147-185`: the shards run at once, one background subshell a shard.
  Each calls `dumpJson.php --shard $i --sharding-factor $shards --snippet 2
  --page-metadata --entity-type item --entity-type property`, in batches.
- `wikibasedumps-shared.sh:68-79`: batch b of every shard covers the same page ids,
  `b·pagesPerBatch·shards + 1` to `(b+1)·pagesPerBatch·shards`, so each shard walks the
  whole page range.
- `:174`: each batch is written as its own `gzip -9` file.
- `:199-228`: the gz dump is concatenated from `[` gzipped, then for shard 0..N-1 each
  batch file in order (`sort -V`), with `,` gzipped between them, then `]`. So it is a
  multi-member gzip, one member a batch.
- `:246`: the bz2 is `gzip -dc` of the gz into `lbzip2 -n $((shards/2))`: **one bz2 stream**,
  blocks cut by size, not at shards.

**Today's job.** airflow-dags (`dumps/dags/mediawiki_wikibase_dumps.py`, main at 079ffda9,
2026-09-16) runs `/usr/local/bin/dumpwikibasejson.sh` in a Kubernetes pod with `--project
wikidata --dump all` (and a new `--output-dir`). I didn't find that container's copy of
the script. The layout measured below matches the puppet version exactly.

**Stability across releases.** Wikimedia documents no ordering guarantee for the JSON
dump. What the code does is stable:
- the shard rule and `ORDER BY page_id` have been in Wikibase for years (DumpGenerator's
  last change was 2024-02-12);
- the bench data (release 20261012) shows the same 8-shard pattern as 20261019: chunk 0
  holds 10,000 entities over Q8–Q83,751, about 1 in 8; routed chunk 10 is at Q93M and
  chunk 50 at Q140M.

The sharding factor is configuration (`shards` in `wikidump.conf.other`) and could change.
**The design below doesn't depend on any of this for correctness** (Gate 3), only for
how often it redoes work.

### What the file holds (measured)

**8 shards** (`layout-20261019/boundaries.json`, each reset bracketed to 4 MB by bisection):

| shard | starts (GB) | ends (GB) | size (GB) | first id | last id |
|---|---|---|---|---|---|
| 0 | 0 | 12.993–12.997 | 12.99–13.00 | Q31 | Q141,739,876 |
| 1 | 12.993–12.997 | 25.957–25.961 | 12.96–12.97 | Q435 | Q141,778,578 |
| 2 | 25.957–25.961 | 38.909–38.913 | 12.95–12.96 | Q1210 | Q141,745,761 |
| 3 | 38.909–38.913 | 51.825–51.829 | 12.91–12.92 | Q667 | Q141,741,578 |
| 4 | 51.825–51.829 | 64.741–64.745 | 12.91–12.92 | Q700 | Q141,794,888 |
| 5 | 64.741–64.745 | 77.653–77.657 | 12.91–12.92 | Q1233 | Q141,704,425 |
| 6 | 77.653–77.657 | 90.577–90.581 | 12.92–12.93 | Q29 | Q141,730,485 |
| 7 | 90.577–90.581 | 103.489 | 12.91 | Q415 | Q141,792,360 |

- **The hash rule, checked:** for all 23,645 ids of the survey, `int(sha1(id)[:8], 16) % 8`
  equals the shard its byte offset is in. There are no exceptions.
- **One bz2 stream.** The only stream header is at byte 0. There is no end-of-stream mark
  across any of the 7 resets (a 4.4 MB read across each), and the one end mark and its
  CRC are the file's last 80 bits. **A shard is not its own bz2 stream**, and its boundary falls
  inside a block.
- **The gz is a multi-member gzip.** Bytes 0–21 are a member holding `[\n` (ISIZE 2), and
  the next member starts at byte 22, as the script writes it. Its members weren't mapped
  further: gz is not the default (#97).
- **The shards agree.** Each shard's running maximum against the fraction of it read:

  | shard | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
  |---|---|---|---|---|---|---|---|---|---|
  | 0 | Q9.8M | Q27.7M | Q36.9M | Q43.3M | Q53.2M | Q66.8M | Q81.8M | Q94.2M | Q117.1M |
  | 1 | Q10.2M | Q28.4M | Q37.2M | Q43.7M | Q53.3M | Q67.8M | Q81.9M | Q96.1M | Q116.9M |
  | 2 | Q10.2M | Q28.1M | Q37.0M | Q43.5M | Q53.3M | Q67.0M | Q82.0M | Q94.7M | Q117.2M |
  | 3 | Q10.1M | Q28.3M | Q37.2M | Q43.8M | Q53.4M | Q67.9M | Q81.9M | Q96.2M | Q117.0M |
  | 4 | Q10.2M | Q27.9M | Q37.0M | Q43.5M | Q53.4M | Q67.1M | Q82.0M | Q94.5M | Q117.2M |
  | 5 | Q10.2M | Q28.4M | Q37.2M | Q43.8M | Q53.5M | Q68.0M | Q82.0M | Q97.2M | Q117.3M |
  | 6 | Q10.0M | Q27.8M | Q37.0M | Q43.5M | Q53.4M | Q67.0M | Q81.9M | Q94.3M | Q117.1M |
  | 7 | Q10.5M | Q28.4M | Q37.2M | Q43.7M | Q53.5M | Q68.0M | Q82.0M | Q97.2M | Q117.4M |
  | **min (the frontier F)** | Q9.8M | Q27.7M | Q36.9M | Q43.3M | Q53.2M | Q66.8M | Q81.8M | Q94.2M | Q116.9M |

  The shards are the same hash-random eighth of one page-id order, so reading them to
  the same byte fraction reads them to within ~2% of the same id.
- **The raw tables:**
  - `layout-20261019/offsets.csv`: one row a probe, with offset, block bit, entities,
    first and last id, min, max, types, non-Q ids, descents, lag and streams;
  - `ids.csv.gz`: every id decoded, with its probe and index;
  - `sample.csv.gz`: the even-offset windows' 23,645 entities, with their fields' bytes,
    counts and set;
  - `lag.csv`: the long windows.

### Order within a shard (measured)

The long windows are 8 MB of bz2 each: one inside each shard at the fraction
(k + 0.5)/8, and one at shard 0's head. Lag is an id's distance below its shard's
running maximum.

| window | shard | ids | lag > 0 | lag > 2 | lag > 1,000 | max lag |
|---|---|---|---|---|---|---|
| Q3.86M–Q3.94M | 0 | 8,585 | 0 | 0 | 0 | 0 |
| Q27.0M–Q27.1M | 1 | 5,673 | 0 | 0 | 0 | 0 |
| Q37.6M | 2 | 3,983 | 0 | 0 | 0 | 0 |
| Q46.9M | 3 | 6,296 | 0 | 0 | 0 | 0 |
| Q60.9M–Q61.1M | 4 | 13,874 | 0 | 0 | 0 | 0 |
| Q80.5M–Q80.6M | 5 | 11,457 | 101 | 0 | 0 | 2 |
| Q96.4M–Q96.6M | 6 | 12,541 | 31 | 0 | 0 | 2 |
| Q126.0M–Q126.1M | 7 | 15,120 | 19 | 0 | 0 | 1 |
| **shard 0's head** (Q23–Q8,259) | 0 | 965 | 213 | 212 | 107 | **1,968** |

- **Within a shard, the order is monotone to within 2**, the trace of concurrent creation,
  past the first ~1,000 entities of a shard. Those are Wikidata's first items, in their
  2012 page order, all below Q10k.
- **Across windows:** none of the 108,546 ids decoded came more than 10,000 behind the
  running maximum of the earlier windows in its shard. The windows are about 1 GB apart.
- **No duplicates:** no id appears twice within a window.
- **What could still come late:** an entity whose page row is much newer than its id.
  For example, an item restored with a new page id; whether MediaWiki did that for
  restorations before about 2016 isn't settled here. None was seen. With zero in ~10^5
  ids, the 95% upper bound on the rate is ~3×10^-5, so at most ~3,700 of 121.8M. That
  bound, not the expected value, is what Gate 3 prices.

### Entities other than Q (measured)

- **Properties** are in the same stream at their creation's page-id position: 24 seen,
  P3213 to P14113, scattered through all shards. The highest, P14113, sits at 98% of
  shard 4.
  - In string order, `P…` sorts before `Q…`, so the property region is the first of every
    table.
  - It closes only at the end, but it is tiny.
- **Lexemes are not in `all.json`.** The job passes `--entity-type item --entity-type
  property`, and lexemes have their own dump. None was seen in 108,570 entities.
- **Redirects and deletions:**
  - redirects are excluded by the pager;
  - deleted entities have no page, and leave gaps: about 1 id in 13 is present in a
    shard against the 1 in 8 a full range would give;
  - neither shows in the stream.

### Prior evidence in the repository

- `bench/CHANGELOG.md` 7.4.0 took the dump as "8 shard passes over Q1 to Q140M", from the
  bench data's chunk ranges. `2026-10-25-claims-presorted.md` lists r10 at Q93.3M and
  r50 at Q140.1M. Both are confirmed, now with the mechanism.
- The split's manifests record each chunk's `first` and `last` (`write_chunk`). A release's
  manifest would give the whole lag census for free, but none is in the repository (the
  Plan's first PR records it).

## The split today, and what synchronised sweeps need (code read, one measurement)

**Today** (`src/wikidata/dump.py`):
- **The download** (`_fetch`): 3 kept-alive connections fetch 32 MB pieces, appended in
  order. `.part` is always a prefix of the dump.
- **The read** (`_dump_lines`): one `lbzip2 -dc` on every core, fed by `_feed` as the
  `.part` grows. Lines come out serially.
- **The chunks:** `split` cuts them into chunks of `CHUNK_ENTITIES` in dump order, built in
  a thread pool (`write_chunk`).
- **The numbering:** `_Numbering` numbers each set's parts in chunk order, and the runs
  follow (`WIKIDATA_FOLLOW_SPLIT`).
- **A restart** re-reads and re-decodes the dump from byte 0, skipping chunks already in
  the manifest.

**For N synchronised sweeps, at the same total CPU:**
1. **The download:** the 8 shards' byte ranges grow together. `_fetch` would take each
   next piece from the shard furthest behind, into a preallocated file with one
   downloaded prefix a shard, recorded in a ledger. The whole file's md5 is checked at
   the end, as now. Its pieces and connections are unchanged (3, Wikimedia's cap), so its
   rate is too.
   - **The boundaries:** found at the start by ~30 small probes, as `survey.py` does, or
     simply cut at N even byte offsets. A cut needn't be at a shard's edge for
     correctness. A sweep that starts a few MB before an edge reads a short tail of high
     ids first, which only lowers the frontier for a moment.
2. **The decode, one stream a sweep:**
   - Sweep k takes the blocks whose magic falls in its byte range, found by a bit-level
     scan.
   - Each block is rewrapped as a single-block stream (`BZh9` + block + end mark + the
     block's CRC as the stream CRC), as `bzip2recover` writes them.
   - It decodes them with `lbzip2 -dc -n1`, or with libbz2 in the Rust extension.
   - **The seams:** the first partial line of sweep k joins the last partial line of sweep
     k−1.
   - **The integrity lbzip2 checks today is kept:**
     - each block's CRC, checked by its decode;
     - the stream CRC, recomputed from the block CRCs in order
       (`c = rotl1(c) ^ block_crc`) and compared with the stream's trailer;
     - the md5.
3. **The order:** the split takes its next line from the sweep with the lowest running
   maximum. That keeps the 8 sweeps in step by id, not by bytes, and the frontier
   `F = min_k max_k − margin` as high as it can be.
4. **The chunks:**
   - **One chunk, one sweep:** a chunk is cut from one sweep's lines, so it stays a narrow
     numeric range. Each chunk then falls in one or two string regions, and that is what
     lets a closed region's chunk files be deleted.
   - **A chunk's number:** (sweep, n), stable across restarts.
   - **The extra memory:** 8 batches in flight instead of 1, about 8 × 100 MB of lines.
5. **A restart:** from each sweep's last durable chunk, its block bit offset and the
   lines consumed within it. No re-decode from byte 0.

**Decoding the same blocks as one pipeline or as 8 sweeps** (`decode_bench.py`, 8 × 4 MB
windows, one in each shard, 33.1 MB of bz2). **Measured on the shared 4-core machine**
under other agents' load (load average 4.9–6.6), so indicative only. CI is the authority:

| mode | wall (3 rounds) | CPU (3 rounds) | CPU-s a GB of bz2 |
|---|---|---|---|
| one pipeline, `lbzip2 -n4` (today) | 6.72, 6.31, 6.72 s | 17.8, 16.2, 15.6 s | 472–538 |
| 8 sweeps, `lbzip2 -n1` each, all at once | 7.49, 7.10, 6.70 s | 18.7, 17.3, 17.1 s | 517–564 |
| 8 sweeps, 4 at a time | 5.85, 5.55, 5.25 s | 15.4, 15.0, 14.8 s | 446–464 |

- **CPU within ±10% across modes:** splitting the decode by sweep costs no measurable CPU.
  lbzip2 decodes the rewrapped block streams correctly (`lbzip2 -t` passes, and the 695.7
  MB decoded match).
- **The absolute CPU a GB** is 2.5× CI's 160–210. That is the contended machine. Only
  the ratios are evidence.

## The closure curves (modelled)

`closure.py`:
- **The frontier:** it takes the frontier `F(f)` above, with a 10,000-id margin.
- **A part's closure:** it closes a part, a fixed string range `[a, b)`, once `F` passes
  the largest id ≤ Q141.8M whose string is in `[a, b)` (`max_number`, checked against
  brute force).
- **The bytes:** they come from the even-offset sample, a byte-weighted sample of the
  dump. Each table's JSON field's bytes stand for its published bytes, by set, scaled to
  bench 17.4.0's published GB by table (as `tables_of` scales them).
- **The parts:** cut at equal bytes in string order.
  - **claims:** one key.
  - **The language and site tables:** each key's bytes follow
    `docs/dataset_cards_metadata.json`'s per-key shares (a main-set build, used for both
    sets), each key spread over ids as its table is.
  - **entities:** one key.

**When each string region closes** (the fraction f of the synchronised split):

| region | Q15–Q19 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Q12 | Q13 | Q140 | Q141 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| closes at F ≥ | Q20M | Q30M | Q40M | Q50M | Q60M | Q70M | Q80M | Q90M | Q100M | Q110M | Q120M | Q130M | Q140M | Q141M | end |
| f | 0.16 | 0.22 | 0.36 | 0.47 | 0.56 | 0.62 | 0.69 | 0.77 | 0.83 | 0.87 | 0.91 | 0.96 | 0.99 | 1.00 | 1 |

**The share of published bytes final at f**, regional 128 MiB parts (the 128.9 GB of
published files; claims_labels, 4.4 GB, at the end):

| set | table | GB | parts | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | claims | 41.4 | 309 | 0% | 16% | 23% | 26% | 31% | 41% | 54% | 73% | 86% |
| main | labels | 6.6 | 626 | 0% | 0% | 0% | 0% | 0% | 2% | 2% | 2% | 10% |
| main | descriptions | 3.8 | 595 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 4% |
| main | links | 5.6 | 962 | 0% | 0% | 0% | 2% | 5% | 5% | 5% | 5% | 13% |
| scholar | claims | 52.8 | 393 | 0% | 3% | 18% | 36% | 53% | 62% | 75% | 85% | 92% |
| scholar | descriptions | 7.5 | 600 | 0% | 0% | 0% | 0% | 2% | 4% | 4% | 4% | 10% |
| **main**, all | | 62.7 | | 0% | 11% | 15% | 17% | 21% | 28% | 37% | 49% | 60% |
| **scholar**, all | | 66.2 | | 0% | 2% | 15% | 29% | 42% | 50% | 60% | 69% | 76% |
| **both**, all | | 128.9 | | **0%** | **6%** | **15%** | **23%** | **32%** | **39%** | **49%** | **59%** | **68%** |

- **The sampling error:** the 101 windows, resampled 200 times
  (`results/closure_bootstrap.txt`), give both sets' share as 32% (24–40%) at 50%, 59%
  (54–65%) at 80%, 68% (63–72%) at 90%, 72% (69–75%) at 95% and 75% (73–75%) at 99%.
- **Claims are 98–100% final by f = 0.99.** The **end tail (~25%, ~32 GB) is almost all
  the language and site tables' small keys.** A key of one file spans every region, so
  its file waits for the end. claims_labels (4.4 GB) is the rest.
- **The layouts compared** (all tables, both sets, at 50 / 80 / 90%):
  - **today's 500 MiB parts:** 30 / 58 / 65%;
  - **regional 32 MiB:** 34 / 63 / 74%;
  - **regional 128 MiB with every key in at least 4 files:** **36 / 66 / 82%**. That layout
    has about 2,300–3,800 files a table a set. Each table's repo stays under the Hub's
    100k-file guidance, and each key's folder holds 4+ files.
- **Fine parts barely help claims.** A claims part in the `Q1…` string range holds
  9-digit ids, so it waits for them whatever its size. The curve is the data's, not the
  layout's.

The full tables are in `results/closure.txt`.

## The release walls (modelled)

`model.py`, from master's bench record (bench 17.4.0, 296adf4, 6 runs, medians):
- **(a):** the L1b entry's model of master, L1a + L2 as built;
- **(b)–(d):** a fluid simulator of the same inputs, the assumptions below;
- **a_sim:** (a) re-run through that simulator, so that its simplifications cancel in a
  difference.

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| (a) master, L1a + L2 | 12h01m | 8h53m | 7h40m | 7h07m | 7h04m |
| (a) master, L3 on (group files kept) | 11h41m | 8h32m | 7h19m | 6h43m | 6h36m |
| a_sim: the simulator, nothing closed early, round trip kept | 11h52m | 8h23m | 7h13m | 6h44m | 6h42m |
| **(b)** synchronised split + regional finalise, round trip kept | 10h56m | 7h31m | 6h44m | 6h43m | 6h42m |
| **(c)** (b) without the group files | **7h21m** | **6h41m** | **6h32m** | **6h31m** | **6h30m** |
| **(d)** (c) + L1b preupload | **7h17m** | **6h36m** | **6h32m** | **6h31m** | **6h30m** |
| c0: no group files, finalise after the split (design C alone) | 10h14m | 7h47m | 7h09m | 6h50m | 6h48m |
| d0: c0 + L1b | 9h21m | 6h52m | 6h50m | 6h48m | 6h48m |

Against a_sim (the same simulator):

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| (b) | −56 min | −52 min | −29 min | −2 min | 0 |
| **(c)** | **−4h31m** | **−1h42m** | **−40 min** | **−13 min** | **−12 min** |
| **(d)** | **−4h35m** | **−1h47m** | **−40 min** | **−13 min** | **−12 min** |
| c0 | −1h38m | −36 min | −4 min | +6 min | +6 min |
| d0 | −2h31m | −1h31m | −23 min | +4 min | +6 min |
| **the frontier's own share** (c − c0) | **−2h53m** | **−1h06m** | **−37 min** | **−19 min** | **−18 min** |
| the frontier's own share with L1b (d − d0) | −2h04m | −16 min | −18 min | −17 min | −18 min |

**Where it comes from:**
- **At 7 MB/s,** (c) takes the build branch's 124 GB off the link (4h56m of it). The
  frontier then lets the published 129 GB start going up ~1h after the split starts,
  rather than after it ends.
- **At ≥60 MB/s,** the release is CPU-bound in every variant. (c) gains by:
  - running ~8 core-h of finalise inside the split, on the 0.46 core the split leaves
    idle and the 2 it is given;
  - dropping half of the runs' CPU (`rho`).
- **(b)** keeps the round trip's 253 GB on the link, so at 7 MB/s it can't go below the
  link's 10h03m.

**(c)'s sensitivity** (walls):

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| central: rho 0.5, eta 1.05, cap 2 cores, regional 128 MiB | 7h21m | 6h41m | 6h32m | 6h31m | 6h30m |
| rho 0 (the runs' CPU gone with the group files) | 6h54m | 6h14m | 6h05m | 6h04m | 6h03m |
| rho 1 (all of the runs' CPU kept) | 7h48m | 7h08m | 7h00m | 6h58m | 6h58m |
| eta 1.2 (regions cost 20% more finalise CPU) | 7h37m | 6h57m | 6h52m | 6h50m | 6h50m |
| finalise capped at 1 core during the split | 9h24m | 7h22m | 6h43m | 6h24m | 6h22m |
| finalise at nice 10 (0.5 core) during the split | 9h48m | 7h21m | 6h43m | 6h24m | 6h22m |
| today's part boundaries (500 MiB, keys whole) | 7h27m | 6h43m | 6h33m | 6h32m | 6h31m |
| regional 32 MiB | 7h06m | 6h36m | 6h31m | 6h30m | 6h29m |
| regional 128 MiB, every key in at least 4 files | 6h50m | 6h33m | 6h30m | 6h29m | 6h28m |
| the same, with (d)'s preupload | 6h48m | 6h33m | 6h30m | 6h29m | 6h28m |
| the closure 10% of the split later | 7h37m | 6h43m | 6h34m | 6h33m | 6h32m |
| the closure 20% of the split later | 8h05m | 6h54m | 6h36m | 6h35m | 6h34m |

- **Finalise's share of the cores during the split decides the slow-link gain.** Capped
  at 1 core, or run at nice 10, the frontier's closed regions wait. The default must
  give finalise real cores, about 2. Memory allows that (below).
- **The rest moves it by tens of minutes,** the curve's sampling error included.

**The model's assumptions:**
- **The split:** its CPU is bench 17.4.0's split ETA on 4 cores. It uses at most its
  sample's 3.54 cores. It runs no faster than the download (103.5 GB at 14.5 MB/s, the
  shards fetched in step), and its time is proportional to bz2 bytes read.
- **The sweeps** cost no more CPU than one pipeline (the measured ratios).
- **The runs:**
  - in a_sim and (b), at nice 10 in what the split leaves;
  - in (c) and (d), absorbed into the split (design A), with `rho` (central 0.5) of
    their CPU kept for reading ~23k chunk files rather than 60 group files.
- **finalise:**
  - **its CPU:** `tables_of`'s by table and set × 4 cores, × `eta` (1.05) for the costs a
    region adds;
  - **its order:** tables in finalise's order (main claims first), closed bytes first;
  - **its cores:** at most 2 during the split, all 4 after;
  - **its pending bound:** it waits while 64 GB is pending upload
    (`FINALISE_PENDING_UPLOAD_GB`).
- **The link:**
  - one FIFO link at the rate, carrying the group pushes too in a_sim and (b);
  - in (c), a region's parts are committed to the build branch when the region is
    written (`create_commit`, as today);
  - in (d), every part goes up as it is written.
- **claims_labels:** 300 s once everything else is done, then its 4.4 GB.
- **(b):** a region waits for the group files holding its rows to be pushed. They are laid
  out by region, so each region's slices are downloaded once, and the 75 MB/s download
  is not on the path.
- **Not modelled:** verify calls, commit latency, the claims refs stage (17 s), the
  three finalise processes' starts, and the md5. a_sim is 10–30 min under (a).

## Disk and memory, a 16 GB laptop (modelled)

| | peak RSS | peak disk |
|---|---|---|
| (a) master | overlap ~4–5 GB; finalise 10–12 GB (3 bucket sorts under the budget); claims_labels 11.2 GB (#167) | ~235 GB; ~300 GB with L3 (#167) |
| (b) | as (a), plus ≤2 bucket sorts (2 × 3.3 GB) beside the split: ~10–11 GB in the split | ~160–220 GB in the simulator (the dump, chunk files until pushed, pending uploads); take (a)'s ~235 GB |
| (c), (d) | in the split: worker 2.8 GB + 8 sweeps' in-flight lines ~0.8 GB + decoders ~0.1 GB + ≤2 bucket sorts 6.6 GB ≈ **10.5 GB**; after it, as (a): 10–12 GB; claims_labels 11.2 GB | **~130 GB**: the dump 103.5, open regions' chunk files ≤ ~25, pending parts, a region's scratch. **~225 GB** if the closed parts are also kept until the final check, so that a redo needs no download |

- **What lets disk fall** is that each chunk is one sweep's narrow id range, so it lies
  in one or two string regions. Its files can go once those regions are written,
  rather than at the release's end.
- **The dump** could also be released shard by shard after the final check (not
  counted).
- **The cap of 2 bucket sorts during the split** is set by memory. Three, beside the
  split's 3.6 GB, would pass 13.5 GB.

## Gate 3: closing a region safely, and what a redo costs

**Correctness never depends on the dump's order.** The order only sets how often work is
redone.

1. **Regions:** the release's string-id ranges, fixed before the split. The part
   boundaries are planned per region, and every key's parts are whole regions. Design B's
   one-time change of published boundaries covers this.
2. **Closing (speculative):** region r closes when `F = min_k (sweep k's running max) −
   M ≥ max_number(r)`. An exhausted sweep counts as +∞. `M = 10,000` is 5,000 times the
   measured lag past a shard's head and 5 times the head's. It delays every closure by
   under 0.01% of the split.
3. **Recording:** the split knows every id's region as it builds its chunk.
   - **For every chunk,** it records each region's rows and an order-independent digest by
     table: the count, and the sum mod 2^64 of a 64-bit hash of (id, row bytes). This is
     recorded in its manifest line, so it is durable with the chunk.
   - **A row whose region is already closed** is a late row. It is also appended to that
     region's late spool before the chunk's manifest line, and counted.
4. **The final exact check** (after the split, before promotion), for every region and
   table:
   - **the inventory:** the rows routed equal the rows written plus the late rows, by
     count and by digest. That is the per-region `_check_written`, with the split as its
     source;
   - **the totals:** Σ over regions equals the split's totals. That is `_check_whole`'s
     role;
   - **each part's order:** `_check_sort`, stable;
   - **the boundaries:** consecutive parts in boundary order (last id < first id), so the
     global order is the parts' concatenation;
   - **parity,** where the bench runs it;
   - **duplicates:** every region's `entities` rows are unique by id. This check is new:
     a dump that repeated an entity, or a stale version of one, halts with both rows
     named and is never silently de-duplicated;
   - **the schema version** of every part (below).
5. **A redo, from the region's own output:**
   - **The rebuild:** each part hit by late rows is rebuilt as a stable sorted merge of
     its own verified rows and the late rows. Equal ids across them would be a duplicate,
     which halts. **No group files and no re-download are needed** if the part is still
     on disk.
   - **Keeping parts on disk:** the default keeps every closed part until the final
     check (the ~225 GB row above).
   - **Without the disk:** fetch only the hit parts from the build branch, 134 MB each,
     about 2 s at 75 MB/s.
   - **The rebuilt part** passes the same checks, and its inventory is now exact.
6. **Replace, never append:** a rebuilt region is committed to the build branch in one
   `create_commit`, deleting its old paths and adding its new ones. The paths are
   deterministic per (table, key, region part), as `commit_table` pairs adds and deletes
   today.
7. **Publication:** stays gated at `promote-release`. It requires every region
   `verified` at the release's final schema version, and the final check's record.

**What a redo costs, in bytes** (`redo.py`). There are 720 parts of 134 MB closed before
the split's end. For each, the CPU is the claims sort's whole cost a published byte,
0.30 core-s a MB: an upper bound for a merge. The upload is the whole part, assuming no
deduplication.

| late ids | parts hit | bytes invalidated | rebuild CPU | local I/O | upload at 7 MB/s | at 30 MB/s |
|---|---|---|---|---|---|---|
| 0 (the measured lag, M = 10k) | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1.0 | 0.13 GB | 0.01 core-h | 0.3 GB | 0 min | 0 min |
| 10 | 9.9 | 1.3 GB | 0.11 core-h | 2.7 GB | 3 min | 1 min |
| 100 | 93 | 12.5 GB | 1.0 core-h | 25 GB | 30 min | 7 min |
| 1,000 | 541 | 72.6 GB | 6.0 core-h | 145 GB | 2h53m | 40 min |
| 3,700 (the 95% bound) | 716 | 96.1 GB | 7.9 core-h | 192 GB | 3h49m | 53 min |

- **A late row invalidates one 134 MB part, not a region or a table.** One late row costs
  ~40 core-s and one part's upload.
- **The worst case** is that every closed part is hit. That is a second finalise of the
  closed bytes: the release is no longer than one that never closed anything, plus the
  rebuild. Integrity is never at risk.
- **Xet's content-defined chunking** (the parts are written with it, `compact.py`)
  should send far less than a whole part when a few rows are inserted. That is
  unmeasured, so the table takes none.
- **If late rows turn out common** at a real release (the first PR measures it), the
  split can raise M as it goes, closing later. The curve's sensitivity above prices
  that.

**The alternative, an id-only index pass then frontier-ordered processing:**
- **What it adds:** one more full decode of the dump. The decode is 53% of the split's CPU
  (#167: 1h52m on 4 cores), and the index waits for the download (1h59m).
- **What it buys:** an exact last-occurrence per region with no speculation.
- **What it costs:** +1h52m of CPU at fast links. At 7 MB/s it delays the first closures
  by ~2h.
- **It gives nothing the exact check doesn't:** the measured lag makes speculation
  almost free. **Rejected.**
  - On gz, an id pass is ~16 CPU-min with igzip, but gz is 1.5 times the bytes and isn't
    the default.

## Two kinds of completeness: rows and schema

- **Row completeness** is the inventory above, per region and exact.
- **Schema completeness:**
  - **The problem:** a region closed under release schema version v is written in v. If a
    later chunk widens the schema (auto mode, `2026-10-26-schema-integrity.md`: a new
    nullable field, or a promoted sibling), the closed parts lack the new columns.
  - **Each part records its version.** Before promotion, every part below the final
    version is reconciled: rewritten with the new columns as all-null.
  - **Values never change in a widening, so no row moves and no boundary moves.**
- **The cost of a reconciliation:**
  - **The rewrite:** it can copy each existing column chunk's bytes and add all-null
    column chunks and a new footer, which is I/O-bound (pq-glue's `concat` copies row
    groups this way, but adding columns would need a new writer path). A decode and
    re-encode costs about the parts' share of finalise's encode.
  - **How much:** every part closed before the widening, at worst all ~96 GB closed
    early: ~200 GB of local I/O, a few core-h at most.
  - **The upload:** up to 96 GB (3.8 h at 7 MB/s). With Xet's chunk deduplication of the
    unchanged column chunks it could be far less, but that is unmeasured.
- **How often:**
  - **The schema:** the published claims schema is stored and complete
    (`claims_schema`, `claims.avsc` completed against it, #172).
  - **When it widens:** only when a dump carries a shape no earlier release had, such as
    a new datatype or value field. Wikidata adds those rarely, a few a year at most, and
    the release that first carries one widens once.
  - **The other tables'** schemas are fixed.
  - **So:** expect it in a minority of releases. When it happens, its cost is bounded by
    the closed bytes, and it delays promotion, never correctness.
- **To make it cheaper,** the split can widen early: a chunk's new field is known when
  the chunk is built, and closures at that moment are still few. A widening found at
  frontier f re-touches only the share closed by f, 0–68%, by the curve above.

## Restart, sketched

**The ledger** (append-only JSONL in the state dir, fsync'd, as the manifests are):
- **`sweeps.jsonl`:** each sweep's byte range. Then, as each chunk becomes durable: its
  (sweep, n), the block bit offset and the lines consumed, the sweep's running max, and
  the chunk's per-region inventory and late rows.
- **`regions.jsonl`:** per region, with each table and key as the stage needs:
  - `closed` (F at closing), `written` (files, sha256, rows, digest, schema version),
    `committed` (the commit oid), `verified`;
  - and, for a redo: `reopened` (late rows), `rebuilt`, `recommitted`, `reverified`;
  - and `reconciled` (the schema version).

**Resuming:**
- **The split** resumes each sweep from its last durable chunk, with no re-decode. F is
  recomputed from the sweeps' recorded maxima, so a closed region stays closed: F only
  rises.
- **A region** resumes at its last stage:
  - `closed` but not `written`: written again (each part to a tmp, then renamed, as today);
  - `written` but not `committed`: committed;
  - `committed` but not `verified`: verified.
- **A redo** resumes the same way. Its inputs, the retained part and the late spool, are
  immutable until its recommit is verified, and only then are the old local parts
  deleted.
- **A redo interrupted mid-commit** is either committed or not: one `create_commit`.
  `verify_table`'s comparison, run per region, tells which.
- **A late row arriving while its region's redo is running:** the late spool keeps
  growing, so the final check finds the region's inventory short and redoes it again.
  Redos converge, because the split ends.
- **The download** resumes each shard's range from its recorded prefix.

## Order and part boundaries across regions

- **Within a part:** the order is the stable sort by string id, as today.
- **Across parts:** regions are disjoint string ranges in order, so concatenating parts
  in boundary order gives the global stable order.
- **Equal ids:** rows sharing an id come from one entity line in one chunk, so their
  relative order is the chunk's, as today's stable sort keeps it. The same id in two
  chunks is a duplicate (Gate 3).
- **The boundaries** are planned per region before the split, from the previous
  release's byte statistics or the start-of-split probes:
  - each region is cut to ~128 MiB a claims part, so its rows fit one bucket sort;
  - every key's file boundaries are unions of whole regions;
  - with at least 4 files a key for the language and site tables, so that their small
    keys close too.
- **They are published boundaries,** changing once and documented (#97, design B).
- **Regions too small or too large** don't break anything: a region's sort uses
  `plan_key`'s bucket path when its rows exceed the in-memory threshold, as a key does
  today.

## Verdict: big fish

- **The dump has the structure the hypothesis needs:** 8 hash-sharded sweeps in page-id
  order, monotone to within 2. It is documented by the exporter's code, and it reads the
  same in the two releases on hand.
- **A synchronised split can close regions as it goes,** at the same decode CPU.
- **With the group files bypassed (design C),** a release at 7 MB/s comes to about
  **7h20m against 12h01m**. That is **−4h31m** against the same simulator's master,
  −2h53m of it the frontier's own over C alone. It is −1h42m at 15 MB/s, −40 min at 30,
  and −12 to −13 min at 60–120 MB/s.
- **Disk falls** from ~235 to ~130–225 GB, and memory stays at today's ~11–12 GB peak.
- **The integrity cost:** a per-region exact inventory and a duplicate check, both new
  and both cheap. A rare redo invalidates one 134 MB part a late row.
- **What would make it small:**
  - **Late rows by the thousand** at a real release, but even then the result only
    degrades towards today's finalise, never to a wrong release;
  - **Finalise starved of cores during the split.** The default must give it about 2.
- **What it needs from the user:** nothing new. Designs A, B and C are authorised,
  "overlapping the stages" is authorised, and no new Hub API is used: per-region
  `create_commit`s go to the build branch. (d)'s preupload is L1b's open decision, and
  adds 4 min here.

## Plan

**PR 1 (measurement only, no change to any output): the frontier census in the split.**
- **What the split records:** in each chunk's manifest line, the shard of its first id
  (`sha1(id)[:8] mod 8`), the chunk's max lag below its shard's running maximum, and the
  number of ids more than M behind it. From `write_chunk`'s lines, this is a few
  microseconds a chunk in the Rust reshape's scan, or in Python on the ids it already
  has.
- **What `split_manifest` adds:** a summary, of the shards seen, the max lag, and the late
  ids beyond M = 10,000.
- **What the bench adds:** its slice is made from the sample's own entities, so the bench
  can't show a real lag. Its `dumpslice.py` would write the slice in shard order
  (sha1 mod 8, then id), so the census runs on it, and the split's figures show no
  change.
- **What it gives:** the first real release's split then records the whole dump's lag
  census, ~121.8M ids against the ~10^5 sampled here, at no cost. It settles the redo
  risk before anything depends on it.

**PR 2: the sweep reader,** behind `WIKIDATA_SPLIT_SWEEPS=8`, off by default:
- the download in step by shard;
- the block scanner and rewrap;
- a decoder per sweep (`lbzip2 -n1`, or libbz2 in reshape);
- the seams;
- the stream CRC recomputed from the block CRCs;
- each sweep's restart offsets.

The published output is unchanged: the chunks differ, the sorted tables don't, and
parity checks it. The bench's slice is bz2-compressed as the dump job does it (a single
lbzip2 stream of 8 shard blocks), and CI measures the split's CPU and wall.

**PR 3: regions and the regional finalise for the claims** (design C's prototype):
- the region plan;
- the split's per-region inventory and late spool;
- finalise of a closed region from the split's own claims files during the split, at
  most 2 bucket sorts;
- per-region commits to the build branch;
- the final exact check, the redo by merge, and the restart cases above;
- the duplicate check on `entities`.

It is benched on bench/link.py's stand-in Hub and its emulated link, with the release
wall's model extended by this entry's closure curve.

**PR 4: the other tables and design B's boundary change:** every key in at least 4
region-aligned files, documented once.

**PR 5: the recipes:**
- `overlap-release` and `release` merged into one run;
- the group files' push dropped;
- `promote-release` gated on every region `verified` and `reconciled`.

## Reproduce

```
cd docs/journal/2026-10-26-progressive-finalise
python3 -I survey.py 20261019 --n 100 --out layout-20261019        # ~82 MB of range reads
python3 -I windows.py 20261019 --boundaries layout-20261019/boundaries.json \
    --uniform 101 --uniform-bytes 327680 --long 9 --long-bytes 8388608 --out /tmp/entities.csv
#   (sample.csv.gz is its uniform rows; lag.csv its long windows)
python3 -I closure.py                      # closure.json, results/closure.txt
python3 -I closure.py --bootstrap 200      # results/closure_bootstrap.txt
git show origin/bench-results:runs/quick/296adf44ea33d5a65912af4e2f8d32829dc3b328.json > rec174.json
python3 -I model.py rec174.json            # results/model.txt
python3 -I redo.py                         # results/redo.txt
python3 -I decode_bench.py DIR             # DIR: w0..w7/s.bz2 (bzip2recover's blocks), all.bz2
```

All of them are stdlib only. `survey.py` writes `ids.csv` (gzipped here).

## PR 1: the census of the dump's order, in the split

**Measurement only.** The split records the evidence that sets the redo risk (Gate 3) for
every id of every real release, not a sample. No file the split writes changes but its
own manifest, `manifest.split.jsonl`. Nothing else in `src/` changes, and neither do
`reshape/`, `hub.py` or the bench's slice.

**What each chunk's manifest line records** (`shards`, one object for each dump shard the
chunk holds ids of):
- `n`: the chunk's ids in that shard. `other`: its ids that are not Q ids
  (properties). They are outside the lag, since `P…` is its own region.
- `first` and `max`: the chunk's first and largest Q id in the shard.
- `before` and `after`: the shard's running maximum Q id before and after the chunk.
- Past the shard's head (Q ids ≥ Q10,000):
  - `lagged`: how many came below the running maximum;
  - `lag`: the most below;
  - `late`: how many came more than 10,000 below (M, Gate 3's margin).
- In the head (below Q10,000): the same as `head_lagged`, `head_lag` and `head_late`.

**The shard rule** is `dump.shard_of`: Wikibase's `DumpGenerator::idMatchesShard`,
`hexdec(substr(sha1($id->getSerialization()), 0, 8)) % 8`, the sha1 of the whole id
(`Q42`, `P31`). Its first 4 bytes mod 8 are its 4th byte mod 8. The test checks it against
the literal formula, and against every one of the survey's 30,059 ids
(`layout-20261019/sample.csv.gz`), each in the shard its byte offset is in.

**Where the running state lives, so that it is exact:**
- **The running maximum carries across chunks in dump order, but chunks finish in any
  order.** A chunk-local summary can't give the straggler count exactly: whether an id is
  more than M behind depends on the running maximum before its chunk.
- **So the census is cut in two:**
  - **in the worker** (`chunk_census`): each id's shard, and, per shard, its Q ids'
    numbers and the largest of them before each, all in dump order;
  - **in the split's main thread** (`_Census`): a chunk's line is completed with the
    running maxima and written once every chunk before it has its line.
- **The split's manifest is now written in chunk order.** That doesn't change numbering:
  `_Numbering` already advanced only over the unbroken run of chunks from the first.
- **Restart:** the running maxima are the lines' own (`after`), so a rerun folds them
  from the manifest and resumes. A chunk that finished but wasn't yet written when the
  split was killed is split again, as one still in a worker is today.
- **A manifest begun before the census** leaves the lines after it lower bounds.
  `split_census` says so (`complete: false`).

**The summary:** `dump.split_census()` reads the split's manifest and gives, by shard and
over all shards:
- the chunks, ids, Q ids and other ids;
- the most a Q id came below its running maximum, and how many came more than 10,000
  below, past the head and in it;
- the first 100 chunks holding a straggler past the head (`late_chunks`);
- the shard's largest Q id.

The split prints it as it ends.

**Its cost** (measured, shared 4-core machine, indicative only): 0.61 µs an id in the
workers (the sha1, the numbers and the prefix maxima) and 0.10 µs in the main thread. At
121.8M ids, that is about 74 s + 12 s of CPU over a release's split.

**On CI** (bench 17.8.0, quick tier, 3 paired rounds, #181 at 03d30c7):
- **The census itself:** 0.02 s of the split's workers' CPU, **0.06%** of the split's
  (`dump.chunk_census`, phases.py).
- **The split:** CPU 33.9 → 35.0 s, wall 9.26 → 9.53 s, no change beyond noise.
- **End to end:** CPU 1m59s → 2m02s, wall 44.83 → 45.72 s, and the release ETA 6h27m50s →
  6h35m18s, all no change beyond noise.
- **Peak RSS of the split:** 2.812 → 2.855 GB, no change beyond noise.
- **Parity:** passed, as did split parity (bz2, gz and orjson) and the misfit chunk's.
- **The earlier run** (2de6712, before #179 was merged in) read the split 4.8% ± 4.0%
  faster, and this one 3% slower within noise. The census cannot speed the split, so
  that is the bench's noise.

**The bench slice's census** (measured: each file's ids in the slice's order, as the
census takes them):
- **Its shards:** the quick tier's four files are each one shard's run: c0 shard 0 (chunk
  0, Q31–Q83,751), r50 shard 3, s30 shard 2, s90 shard 7.
- **Heads:** shard 0's head lags at most **1,968**, as this entry's long window of
  20261019's shard 0 found.
- **Past the heads:** at most **1,006**. Q49,907 comes after Q50,913, at c0's row 6,012,
  an early straggler the long windows didn't sample. r50 lags at most 3; s30 and s90
  don't lag. None is more than 10,000 below.
- **Lower bounds:** r50 and s30 are routed subsets of their chunks, and a subset never
  lags more than its chunk, so their figures are lower bounds for the dump's chunks 50
  and 30.
- **The margin:** nothing contradicts M = 10,000. The past-the-head bound to use in
  closure is 1,006, not 2.

**Tests** (`scripts/test_split_census.py`):
- the shard rule, against the literal formula and the survey's 30,059 ids;
- a synthetic 8-shard dump, its chunks finishing in random orders, every line against a
  brute-force pass, and a restart from a partial manifest;
- a split of a slice of the bench's files with a planted straggler:
  - exact, and in chunk order;
  - killed with SIGKILL and rerun to the same manifest and files;
  - all 172 files byte-identical to master's split, its manifest the same less `shards`.

## PR 2: the sweep reader, behind `WIKIDATA_SPLIT_SWEEPS`

**Only the order the split reads its lines in changes**, and only with the flag on
(`WIKIDATA_SPLIT_SWEEPS=8`, the dump's sharding factor; off by default). The chunks are
cut from the merged lines as they were cut from the serial ones, so their boundaries and
numbering follow the merged order. Nothing downstream of the split changes, and neither do
`hub.py` or the allocator. The code is `src/wikidata/sweeps.py`, `reshape/src/bz2.rs` and
a few lines of `dump.split`.

**The cuts** (`sweeps.plan`, found once a dump and kept in the split's state as
`sweeps.json`, beside its manifest, so a rerun takes the same ones):
- **Where:** sweep k starts at the first line of the k-th shard present. Sweep 0 starts at
  the stream's first block.
- **How it is found, on the local file:**
  - **The blocks:** bz2 blocks start at any bit. `reshape`'s `bz2_blocks` finds each
    block's 48-bit magic at all 8 alignments (memchr's memmem on the 5 bytes every
    alignment covers whole). It checks each hit with what a real block has after the
    magic: its CRC, a randomised bit of 0 and a BWT origin below 900,000.
  - **Decoding:** each block is decoded alone by libbz2 (its CRC checked), rewrapped as a
    stream of its own, as `bzip2recover` writes them.
  - **A probe:** at a byte offset, the shard (Wikibase's rule mod N) of the first line
    that starts in the first block after it.
  - **The bracketing:** 4 × N probes at even offsets bracket each change of shard, as
    #180's survey did. Bisection narrows it to 128 kB. The blocks there are decoded in
    order, and the first line of the new shard gives its block and its offset in the
    block's text.
- **What it costs:** on the bench's 8-shard dump (below), 46 probes and about 2.5 s. At
  a release, about 140 probes, each a 2 MB local read and one block's decode: ~15 s, once.
- **Robustness:** a dump whose probes are not in shard order is refused, with the advice
  to unset the flag. Two shards whose first lines fall in one block are read as one
  sweep. The cuts are exact line boundaries, so every line is read exactly once
  whatever the layout. The layout only sets how good the frontier is.

**The decoders** (`lbzip2 -dc -n1`, one a segment, as many at once as the cores):
- **Segments:** sweep k reads its byte range, from its first block to the next sweep's,
  in segments of about 512 kB of bz2 (~6.5 MB of text at the dump's ratio). Each segment
  is built in memory and decoded whole by an `lbzip2` of its own.
- **Slots, the cores' number:** at most `DECODERS` (the cores) segments are decoded at
  once over all the sweeps, as many threads as today's one `lbzip2 -n4` takes. A sweep
  holds a slot only while its segment decodes. It queues the text in 1 MB pieces (at
  most 4) after letting the slot go, so a sweep waiting for the merge never holds one,
  and nothing can deadlock (see Memory, below).
- **Why the slots:** the first version ran 8 decoders at once, one a sweep, as #180's
  decode_bench had. On the bench's 8-shard dump (81 MB of bz2) that took **1.7 times the
  CPU** of the same 8 decoders run 4 at a time: 56 s against 33 s, alone, on the shared
  machine. One `lbzip2 -n4` took 34 s. CI measured the same thing on that version: the
  reader +66% CPU and the split +39% (this PR's first head, `740801b`, its stage then numbered 17.10.0). #180's +8% for "all at once"
  doesn't hold for whole segments decoding side by side.
- **No bit shifting:** a segment's stream is built from four parts:
  - a stream header;
  - a tiny dummy block, whose length in bits puts the segment's first block at the bit
    of its byte where it lies in the file;
  - the file's bytes as they are;
  - an end-of-stream mark with the segment's stream CRC, folded from its blocks' CRCs
    (found as the sweep reads them).

  It is handed to the decoder as an in-memory file (`memfd`), and the text is read whole.
  Python's `communicate()` cost ~2.5 s a GB of text in its 32 kB reads. The dummy's few
  bytes of text are dropped.
- **Threads:** one a sweep, reading the file (`pread`, scanned in Rust with the GIL
  released), building its segments and running their decoders. The merge runs in the
  split's main thread.

**Memory, and the CPU it cost to save it.**
- **What CI measured:** `6350998` peaked at 3214 MB in the sweeps' split, against byte
  order's 2540.
- **The accounting method:** the split ran in-process on the bench's 8-shard dump with a
  sampler every 0.2 s. It read:
  - the process's RSS, and its children's (the decoders);
  - glibc's heap, read-only (`mallinfo2`): in use, free but held in arenas;
  - the sweeps' queued text and the merge's unconsumed lines;
  - the chunks pending in the worker pool, by their lines' bytes.
- **The CPU accounting:** the reader alone, with this process's CPU, its main thread's and
  its children's (the decoders) apart.
- **Conditions:** the shared machine, at load 1–6 (RSS barely moves with load). Ranges over
  2–4 runs each.

| MB, or CPU-s | byte order | `6350998` (2 MB segments, whole) | streamed text + chunk cap (`f2c6718`, `a427c64`) | **final:** 512 kB segments, whole, queued in 1 MB pieces + chunk cap |
|---|---|---|---|---|
| split's peak RSS (the process) | 1711–2301 | 3062–3197 | 2105–2149 | **1670–2096** |
| the decoders, children (at most) | 41–58 | 46–49 | 123–135 | **32** |
| the sweeps' text queued (at most) | 0 | 262–341 | 28–31 | **27–28** |
| chunks pending in the pool (at most) | 410–716 | 1034 | 512–548 | **292–486** |
| the largest chunk's lines | 325 | 459 | 256 | **256** |
| reader alone: the decoders' CPU | 30–39 (`lbzip2 -n4`) | 30–39 | **58–61** | **36–39** |
| reader alone: this process's CPU (its merge thread) | | 3.5 (1.3) | 1.9–2.2 (1.1–1.3) | 2.5–2.6 (1.1–1.2) |

- **Whole segments held per sweep** (`6350998`). With 2 MB segments, each ~26 MB of text,
  a sweep held 2 queued, 1 in its thread waiting to be put, and the merge's current one,
  whose consumed lines stayed referenced. That is ~100 MB a sweep × 8.
- **Streaming the text cost CPU** (`f2c6718`, `a427c64`). A decoder's text was passed on a
  megabyte at a time, and its sweep let its slot go while its queue was full. Memory came
  down, but the decoders' CPU doubled. An `lbzip2` whose output isn't read keeps decoding
  into its own buffers, so 8 decoded at once again. CI on `a427c64`: the split's CPU
  +27.8%, the reader's +42%.
- **The final design:** segments of 512 kB of bz2 (~6.5 MB of text), each decoded whole
  while holding a slot, as in `6350998`. Then, with the slot let go, the text is queued in
  1 MB pieces (at most 4 queued). A sweep holds at most one segment's text beside its
  queue, ~10 MB. The cost is 4× the decoder spawns: ~160 on the bench dump, ~200k at a
  release, each a few ms.
- **The chunks.** The merge puts every shard's head together, and the heads hold the
  dump's largest entities (its earliest items, as in the bench's c0). So the first chunk of
  10,000 was 459 MB against byte order's largest, 325, and the workers' memory scales with
  chunk bytes. The real dump will do the same.
  - **The fix:** with more than one sweep, a chunk also ends once its lines reach 256 MiB
    (`dump.SWEEPS_CHUNK_BYTES`). That is below byte order's largest chunk at a release
    (20260928's first, 498 MB of JSON), so it binds only on the sweeps' head chunks.
  - **What doesn't change:** byte order's chunks and one sweep's, and the line order. The
    cuts are deterministic, so a restart is unchanged.
- **The plan's probe buffers** are freed, and showed in no sample.

**What CI measured** (bench 17.10.0 on `6350998`, 17.13.0 after it; 3 paired rounds each; sweeps against byte order on the 8-shard dump):

| head | reader CPU | reader wall | split CPU | split wall | split peak RSS (sweeps vs byte order) | timed path, flag off |
|---|---|---|---|---|---|---|
| `6350998` | +22.0% ±7.1% | +53% | 37.2 vs 34.5 s (+7.9% ±1.8%) | 11.90 vs 9.73 s (+22.2%) | 3214 vs 2540 MB | 0.99× ±0.01× |
| `a427c64` (streamed) | 30.8 vs 21.4 s (+42.1% ±18.7%) | +61.9% | 45.8 vs 35.9 s (+27.8% ±2.4%) | 13.41 vs 9.91 s (+35.5% ±3.3%) | 2394 vs 2475 MB | 1.00× ±0.01× |
| `8b62b08` (final) | 26.4 vs 21.5 s (+22.7% ±7.6%) | 8.75 vs 5.65 s (+54.7% ±7.3%) | 41.0 vs 37.7 s (+8.8%, no change beyond noise) | 12.54 vs 10.61 s (+18.3% ±17.8%) | 2221 vs 2476 MB | 1.02× ±0.02× |

- **Parity, every head:** passed. Flag off, the split is byte-identical to master's. Flag on, the published files are byte-identical (101.7 MB) and every entity is in exactly one part.
- **The final head's costs:** the split's CPU is within its noise and its peak RSS is under byte order's. The reader alone still costs +23% CPU and +55% wall, from the plan (~1.4 s a dump) and the per-segment decoder spawns.
- **The frontier:** at most 1,530 below the largest id read, against 43,551 in byte order. 1,530 is an observed lag on the tested data, not a guaranteed bound. PR 3's correctness must come from the exact per-region inventory check and redo, never from a fixed margin.
- **Status:** the flag stays off by default. This is an enabling change for PR 3, not a performance win on its own.

**The seams:** sweep k's text is its blocks' text from its first line on. The first bytes
of sweep k's first block, up to that line, end sweep k−1. That block is decoded once more,
alone, at the start (7 blocks a dump), so the line straddling each cut is joined exactly.
Within a sweep, lines straddle blocks and segments as the decoder's text runs on.

**The merge:** the next line always comes from the sweep whose running maximum Q id is
least (the lowest-numbered sweep if several are equal). A property's line doesn't raise
the maximum. The merge takes a run of lines from one sweep until its maximum passes the
next sweep's, so the heap is touched once a run, not once a line. The order depends only
on the sweeps' lines, never on timing.

**Integrity**, as today's one `lbzip2` checks it:
- **each block's CRC,** checked by its decoder;
- **each segment's stream CRC,** checked by its decoder, so the blocks a decoder read are
  exactly the ones the scan found, in order;
- **the dump's stream CRC:** once every sweep has ended, it is recomputed from every
  block's CRC in file order. Each sweep's fold is combined with `rotl(a, n) ^ b`, since
  the fold is linear. It is compared with the dump's own CRC, in its last 80 bits;
- **the md5:** download-dump's, as before.

A corrupt block, a wrong stream CRC, a dropped block and a spurious block magic each halt
the split (tested, the first three). The block check passes compressed data by chance
about once in 3 × 10^4 dumps. Such a hit is still caught, by its segment's or the
dump's CRC.

**Restart:**
- **What is recorded:** each chunk's manifest line holds every sweep's position after the
  chunk (`sweeps`, on the split's existing restart state):
  - the segment its next line is in: its first block's bit, and the sweep's CRCs folded
    before it, with their count;
  - the bytes of that segment's text before the line;
  - the sweep's running maximum;
  - or, once the sweep has ended, its CRCs folded.
- **The rerun:** it resumes each sweep from that segment (at most 512 kB of bz2 decoded
  again a sweep, never from byte 0) and the merge from the running maxima, from the
  manifest's last line, which the census writes in chunk order. So the lines that follow,
  and the chunks, are the same.
- **What refuses to mix:** a split begun with the flag is refused without it (its
  `sweeps.json`), and one begun without it is refused with it (lines without positions).

**The download: scoped out of PR 2.** The sweeps read a whole, downloaded dump.
- **What it refuses:** with the flag on and the dump not yet downloaded (only its `.part`,
  or nothing), the split refuses with the reason, rather than following the download as it
  does in byte order (`_feed`). download-dump writes the dump from its start, as one
  prefix, so a sweep beyond the first would wait for nearly the whole download.
- **`WIKIDATA_FOLLOW_SPLIT` itself isn't refused.** It makes each set's run follow the
  split's numbered parts, which the sweeps keep: the runs follow a sweeps' split as they
  follow a serial one. So `just overlap-release` on a downloaded dump works with the flag.
  Its download-split step finds the dump complete and splits it. What can't run is the
  split following the download, and that is what is refused.
- **What an interleaved download needs** (a later PR, ~150 lines of `dump.py`, no new
  dependency):
  1. **The cuts before the download:** the same probes by HTTP byte range, as #180's
     survey ran them. About 140 reads of ~300 kB, ~40 MB, before the first piece. A
     block of the real dump is ~75 kB of bz2, so 300 kB holds a block and the next's
     magic.
  2. **A preallocated `.part`** (sparse, the dump's size), and a ledger beside it
     (`.part.jsonl`, fsync'd): each sweep's downloaded prefix of its own byte range.
  3. **`_fetch` taking each next piece** from the sweep furthest behind, by the fraction
     of its range downloaded. The pieces stay 32 MB, the connections 3 (Wikimedia's cap),
     so its rate is today's. The pieces in flight are written at their offsets, and each
     sweep's prefix advances as they land in order.
  4. **The feeders following their sweep's prefix,** as `_feed` follows the `.part` today:
     a feeder at its sweep's downloaded end waits, and stops if the download fails its
     md5 (the `.part` moved aside).
  5. **The md5** of the whole file at the end, as now. The stream CRC check at the split's
     end is independent of it.
  6. **A restart** resumes each sweep's range from the ledger.
