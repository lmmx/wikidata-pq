# 2026-10-25: Keys sorted in memory read from their group files, not glued

Hypothesis 1 of docs/journal/2026-10-25-release-profile-bench10.md (#131). The change is
lmmx/wikidata-pq#135, built on #147 (the sort planned from the data alone,
docs/journal/2026-10-25-decoupled-thresholds.md).

## Current State

- **The change.** Compaction glued every key of every table but the claims into compacted
  files (`_write_run`) and checked the glue byte for byte (`_check_glued`). For a key the
  sort reads whole, those files were only its input. Now such a key's group files are
  linked one by one, as a key of one group file already was.
- **Why the published files are unchanged.** Under #147 the sort plans every decision
  from footer counts (`plan_key`), so the plan is the same from group files as from
  glued files. Compaction asks the same rule of the group files
  (`sort_by_id.sorted_in_memory`).
- **The published files are byte-identical**, locally on the bench's group files (below).
  Parity passed on CI.
- **The bench**, against #147's head:
  - disk −78 MB (−12.8%);
  - finalise ETA −2m22s, within noise on that run (±30–35m);
  - finalise CPU unchanged.
- **Pending:** #147's merge. Then master is merged in and the PR re-benched.

## The first version, and why it was dropped

Before #147, three of the sort's decisions read encoded sizes: the part count, the
in-memory threshold, and `_row_group_rows`' sample from the largest file.

- **How #135 first reproduced them:** with the glued files' exact sizes, from
  `pq_glue.concat_size` (lmmx/pq-glue#12: `concat`'s own glue counted into a byte count,
  column chunks as zeros of their lengths), and the sample from the largest planned
  file's first group file.
- **The evidence:** all 2,323 part files of the bench's 3× layout, 2,380 with small
  compacted files and 2,380 with 64 KiB row groups were byte-identical. Every planned size
  equalled the glued file's.
- **Why it went:** the user then authorised decoupling the thresholds from encoded sizes.
  With #147, none of that plumbing is needed: no `glued` plan, no sample, no pq-glue
  release. pq-glue#12 is closed unmerged.
- **Where it is kept:** as #135's earlier head, ec153ce, in the PR's history.

## The replacement for `_check_glued`

There is no copy to go wrong. Each failure mode is covered thus:

- **The sort's input is exactly the group files' bytes.**
  - A link is its group file. Where linking fails, the copy is checked by sha256
    (`_link_run`, unchanged).
  - Its sha256 and footer rows go into compaction's manifest.
  - `check_compacted_copy` (unchanged) checks the sort's copy against them: the exact
    files per key, each file's size and sha256.
- **The sort read them all, nothing lost or duplicated:** `_check_read`, in `_read_key`.
  Each file's rows as read must equal those compaction listed for it. It runs for every
  key sorted in memory.
- **Unchanged:** #147's plan-rows check, `_check_sort`, `_check_written` and parity.

## Evidence (local; exact for bytes, indicative for time)

- **The data:** the bench's quick-tier group files, laid out as its 3× run lays them:
  each key's group file linked as groups i, i + 4 and i + 8.
- **The tables:** every table but the claims, at the bench's scaled 3× threshold.
- **The arms:** #147's head (2724e29) against this branch.

| settings | part files by sha256, names, count and order |
|---|---|
| the bench's (500 MiB compacted files) | 2,323 / 2,323 identical |
| 300 kB compacted files | 2,415 / 2,415 identical (85 keys of several part files) |

`scripts/test_finalise.py` covers:
- the links being the group files;
- a restart mid-key, which relinks only the missing file and gives the same entry;
- the same part files by name and sha256 as from glued files of the same rows;
- `_check_read` catching a short read and an unlisted file.

`scripts/test_sort.py` passes.

## The bench

**Isolated: this change against #147's head.** Run 40219197286, dispatched, before
2724e29, after c6c7986, 6 runs.

| figure | before | after | change |
|---|--:|--:|---|
| finalise ETA | 2h40m10s ± 30m04s | 2h37m48s ± 35m46s | −2m22s, within noise |
| CPU, finalise | 1m24s ± 12.8s | 1m24s ± 11.7s | none |
| disk used, peak rise | 609 MB | 531 MB | **−78 MB (−12.8%)** |
| peak RSS, finalise, all processes | 3.845 GB | 3.791 GB | within noise |
| peak RSS, finalise at a release, modelled | 20.17 GB | 20.21 GB | within noise |
| release ETA | 8h44m26s | 8h43m25s | within noise |
| parity | | | passed |

Finalise's part per byte (the 3× run less the first, CPU), by stage:

| stage | before | after |
|---|--:|--:|
| `compact._check_glued` | 1.5s | 0.0s |
| `compact._write_run` | 1.0s | 0.0s |
| `compact._key_schema` | 0.5s | 0.0s |
| `compact.rewrite_table` (links, in the parent) | 0.5s | 1.5s |
| `sort_by_id.check_compacted_copy` | 0.1s | 1.1s |
| `sort_by_id._read_key` | 0.9s | 1.7s |

- **The timed stages:** 43.4s → 43.6s. What compaction saves, three files a key cost in
  the sort.
- **The CPU outside the stages:** +53s → +48s. That is the ~2,300 glue jobs a run no
  longer dispatched to the light pool.
- **An earlier bench of the first version, against master 44f4e0e:** run 40211115627,
  finalise ETA −2m51s (−1.5% ± 1.1%) and disk −38 MB, with the same per-stage pattern.
- **Against master** (the PR's own run, 79438b1, with #147 included): CPU, finalise
  −5.2s (−6.4% ± 2.9%) and disk −78 MB. That is #147 and this change together.

## What a release would see (a reading, not a model)

A release has about 30 group files a key, and the bench's 3× run has 3. Both sides of the
trade go mostly per file or per column chunk. Entity scaling overstates both.

- **What goes:**
  - the glue's copy of the other tables' bytes (at least ~15 GB written and read back);
  - the check's ~60M column chunks compared in Python (about 8m, #131's cross-check);
  - a schema read a group file.
- **What comes:**
  - ~69k more file opens in `_read_key` (about 1–2m);
  - ~69k more files hashed (the same bytes as before);
  - a combine of each column a key in `_sorted`. pyarrow reads a file's row groups into
    one chunk, and the files read apart give one chunk each.
- **So the net** is likely a few to ~10 minutes, mostly disk-bound wall the bench does
  not model.
- **The scratch disk** no longer holds a second copy of those ≥15 GB.

## Negative findings

- **The bench's per-byte stages net out at about zero.** Its saving is in job dispatch, a
  per-key cost at a release, which the model scales by entities.
- **Threads for the links** (FINALISE_WORKERS at once, tried in the first version) were
  slower on the bench's small files: 6.1–7.7s of compaction wall against 4.8–5.1s, from
  GIL contention. Not kept.
- **The first version's 3× wall** rose by +1.0s (+2.0% ± 1.2%) against master. Against
  #147 the 3× wall moved within noise (46.75s → 47.04s).

## Missing

- #147's merge, then a re-bench on master.
- A release-scale finalise run, which would size the disk-bound wall saved and the extra
  opens and combines at ~30 group files a key.
