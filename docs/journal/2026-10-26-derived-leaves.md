# 2026-10-26: The claims' derived leaves left out of the group files, put back by finalise

#192 (`2026-10-26-group-file-bytes.md`) found that every `numeric-id` and `entity-type`
in the claims (a main snak's, each qualifier's and each reference snak's datavalue) is
exactly a function of the datavalue's `id` on the sample: 6.3% of the claims' column
bytes. A release uploads ~121 GB of group files to its build branch, and at the ~7 MB/s
uplink (#187) those uploads are on the critical path ("the uploads, by 43m").

**The hypothesis:** leave those leaves out of the claims group files (the build branch's
intermediate files, read once, by finalise) wherever they are derivable, and put them back
exactly in finalise before the published files are written. The published files (the
data contract) stay byte for byte as they are.

The scripts and their outputs are in `2026-10-26-derived-leaves/`.

## Current State

- **Open, in the PR.** The claims partition files, and so the claims group files glued
  from them, are written without the six derived leaves wherever every value of them is
  the rule's of its id (`src/wikidata/claims_derived.py`); finalise puts them back as it
  reads them. The published files are master's, byte for byte (test_schema_evolution's
  fits scenario against master's code; test_claims_derived's against
  `WIKIDATA_CLAIMS_DERIVED=keep`).
- **Measured on the sample:** the claims partition files −6.33% (71.18 MB → 66.68 MB);
  the bench slice's claims group files in the tests −6.2% to −7.1%.
- **Measured CPU on the sample** (1,175,112 statements, best of 3, the shared machine):
  - the split's claims write +0.31 CPU-s (+6.7%: the check and drop 0.38 CPU-s, the
    write of fewer leaves 0.07 less);
  - finalise's read +0.15 CPU-s by pyarrow (+5.4%) and +0.24 by pq-glue (+11%): the
    rebuild 0.38 CPU-s, the read of fewer leaves a little less.
- **Modelled at a release** (the sample's shares scaled to ~90 GB of claims group files
  and ~1.8 billion statements, ×1,530):
  - **bytes:** ~5.7 GB fewer to upload to the build branch: **~−13.6 min at 7 MB/s**,
    −3.2 min at 30 MB/s, and finalise's download ~−1.3 min at 75 MB/s;
  - **CPU:** the split ~+470 CPU-s, finalise ~+230 to +370 CPU-s: ~+2 min and ~+1 to
    +1.5 min of wall on 4 cores where each is the path (at 7 MB/s neither is: the
    uploads are, by 43 min).
- **The real data:** the rule holds on 9,039,413 entity values read from the 20261019
  `all` dump and the 20261021 lexemes dump (70 windows, 440 MB by byte range), with no
  exception (below).
- **The bench** (CI, paired, 3 rounds): pending on the PR.

## The census (`census.py`)

The quick tier's claims partition files, as the split writes them (`bench/pipeline.split_stage`
on c0, r50, s30, s90: 5 files, 1,175,112 statements), every datavalue's `id` shape against
its `entity-type` and whether its `numeric-id` is the id's digits:

| site | id shape | entity-type | numeric-id | values |
|---|---|---|---|---|
| main snak | `Q…` | item | the digits | 545,452 |
| | `P…` | property | the digits | 206 |
| | `L…` | lexeme | the digits | 16 |
| | `L…-F…` | form | null | 5 |
| | `E…` | entity-schema | null | 9 |
| | null | null | null | 629,424 |
| qualifiers | `Q…` | item | the digits | 100,176 |
| | `P…` | property | the digits | 39 |
| | `L…` | lexeme | the digits | 7 |
| | `L…-S…` | sense | null | 105 |
| | null | null | null | 275,262 |
| references | `Q…` | item | the digits | 777,548 |
| | `P…` | property | the digits | 128 |
| | null | null | null | 1,242,426 |

- No exception: 1,423,691 entity values and 2,147,112 null ids (a datavalue of another
  type, or none: a somevalue or novalue snak), every one as the rule gives it.
- No `M…` (mediainfo) id: Wikidata's dumps have none. The rule gives it no type, so a
  file with one would keep its leaves (below), not lose them.
- `claims.avsc` and the stored claims schema have no enum for `entity-type` (a string),
  so the rule is the census's, and is checked on every file rather than trusted.

## The design

- **The rule** (`claims_derived.derive`): `entity-type` is `Q` item, `P` property, `L`
  lexeme, `E` entity-schema for an id of that letter and 1 to 18 ASCII digits; `L…-F…`
  form and `L…-S…` sense; null otherwise. `numeric-id` is the digits of a `Q`, `P` or `L`
  id of that shape, null otherwise. No regex for the letter-and-digits ids (nearly all):
  slices, `ascii_is_decimal` and a cast; a regex only on the rest.
- **Checked at write time, not assumed** (`matching`, `reduce`). Each leaf of each of the
  three sites is dropped only if every value Parquet would store for it (under every
  non-null parent) is the rule's of its sibling id, nulls included. The comparison runs
  first on the arrays as laid out, with no copy (equal there, equal as stored); only where
  that differs, on the values as stored (parent validity merged down). A leaf with any
  other value is kept in that file, and only that leaf.
- **Recorded in the file** (`KEY`, `wikidata.derived_leaves` in the Arrow schema's
  metadata: the Parquet paths of the leaves left out). A file with none left out is
  untagged, the bytes written before; `WIKIDATA_CLAIMS_DERIVED=keep` writes every file so.
- **Where the files are written:** a claims partition file is written by the split a row
  group at a time (`claims_build.RowGroups`, each row group a file glued at the end) or by
  the pipeline whole (`claims_refs.write_with_refs`). Each row group is reduced as it is
  written; a row group that allows fewer leaves than one before it has those written again
  with the leaf put back (`RowGroups._one_form`), as pq-glue glues only files of one
  schema and metadata.
- **Where forms meet:**
  - **a group's partition files** (`push.core._harmonise`, which already conformed a
    group's files of earlier schema versions): files of other leaves dropped are written
    again in the group's form (those every file is without), a row group at a time
    (`ReleaseSchema.upgrade_file`: put back, conformed, dropped again);
  - **finalise's compaction** glues each form apart (`compact._runs`, as it did each
    schema version), each run checked against its version's schema less its leaves;
  - **finalise's sort** reads each file as what it is (`sort_by_id._as_read`): its
    leaves put back (`restore`) in its version's schema, then conformed to the frozen
    one. So a release's build branch can hold group files of either form, as across a
    restart on an upgraded code, and publishes the same files.
- **Schemas, explicitly** (`check_form`, `ReleaseSchema.check`): a file naming leaves must
  be exactly its version's schema less them, and they must be derived leaves of it. A
  file less a leaf it does not name, naming one it has, or naming one not derived is
  refused, as is a reduced table given to `conform_table` (which would make the leaves
  null).
- **The checks stay independent.** The sort's checks (`_check_sort`, `_check_written`,
  `_check_read`, the bucket checks) compare the written files with the rows read and put
  back, as before with the rows read. `check_compacted_copy` is on the files' bytes. The
  plan is the same: `footer_size` counts a dropped leaf as its sibling `id`'s values
  (the same levels) at its width, so the buckets, parts and row groups are those of the
  full files (checked equal on the sample). Parity (bench 17.14.0) compares the published
  files exactly, and the claims group files with the derived leaves left out of both
  runs' (in pyarrow, apart from `claims_derived`).

## The real data (`census_dump.py`)

70 windows of 6 MiB read by byte range from dumps.wikimedia.org (no Hub), held in memory,
their whole bz2 blocks decoded (#180's `bz2probe`), every entity parsed and nothing kept:
64 spread evenly through `wikidata-20261019-all.json.bz2` (103.5 GB; 505,365 items and 52
properties) and 6 through `wikidata-20261021-lexemes.json.bz2` (158,614 lexemes, their
forms' and senses' own statements too). 440.4 MB read.

| id shape | entity-type | values | numeric-id |
|---|---|---|---|
| `Q…` | item | 8,962,223 | the digits, every one |
| `L…` | lexeme | 51,895 | the digits, every one |
| `L…-S…` | sense | 20,533 | none, every one |
| `L…-F…` | form | 4,327 | none, every one |
| `P…` | property | 435 | the digits, every one |
| **all** | | **9,039,413** | |

- **No exception:** every entity value's entity-type and numeric-id is the rule's of its
  id, at every site (main snaks 3,622,706, qualifiers 519,606, references 4,776,727, and
  the lexemes' forms' and senses' statements 120,374).
- **No `E…` or `M…` value in these windows** (the sample has 9 entity schemas); the rule
  gives `E…` entity-schema with no numeric-id, as the sample shows, and `M…` (MediaInfo,
  Commons' type) none, so a file with one keeps that site's leaves (tested above).
- **No id, no entity value:** 13,509,092 datavalues of other types (string, time,
  quantity, monolingualtext, globecoordinate), 3,691 somevalue and 3,065 novalue snaks:
  their leaves null, as the rule gives a null id.
- `results/census_dump.json` has every count by site and the reads.

## Bytes and CPU (`cpu.py`)

`cpu.py` on the sample's claims partition files (the split's, as above), each rewritten
by `write_table` (the split's writer) whole, best of 3 (CPU-s, the process's own):

| file | statements | bytes, full → without | write full | reduce + write | read full (pyarrow / pq-glue) | read without + restore |
|---|---|---|---|---|---|---|
| main c0 | 619,789 | 43,419,005 → 40,863,604 | 2.40 | 2.53 | 1.29 / 1.06 | 1.36 / 1.16 |
| main 1 | 45,546 | 2,321,570 → 2,198,331 | 0.15 | 0.15 | 0.09 / 0.06 | 0.11 / 0.07 |
| scholar 0 | 82,852 | 4,232,658 → 3,945,093 | 0.34 | 0.40 | 0.22 / 0.16 | 0.26 / 0.18 |
| scholar 1 | 322,868 | 16,022,065 → 14,860,112 | 1.26 | 1.37 | 0.96 / 0.64 | 0.94 / 0.70 |
| scholar 2 | 104,057 | 5,185,325 → 4,809,852 | 0.41 | 0.41 | 0.24 / 0.19 | 0.28 / 0.23 |
| **all** | **1,175,112** | **71,180,623 → 66,676,992 (−6.33%)** | **4.56** | **4.86** | **2.79 / 2.11** | **2.94 / 2.35** |

- **The check and drop** (`reduce`): 0.38 CPU-s, about 0.1 µs a datavalue. The first
  version, which compared the leaves as stored (parent validity merged down, nested
  columns combined) and derived with one regex, took 2.87 CPU-s: comparing the arrays as
  laid out first (no copy) and the regex-free rule made it 7.5 times cheaper.
- **The rebuild** (`restore`): 0.38 CPU-s. In finalise's bucketed sort each row is put back
  once: a row group read directly by a bucket after its filter, a spilled row after its
  take.
- **What the CPU is against:** the split's claims write and finalise's claims reads are
  each a few percent of their stage's CPU, so the stages' totals move by less (the bench
  below).

## Tests

- **`scripts/test_claims_derived.py`** (new; on bench-data's r50 and s30, the stand-in
  Hub of test_schema_evolution.py):
  - **keep / drop:** the same files published, byte for byte; every file uploaded the
    same but the claims group files, each without its six leaves;
  - **mixed:** each claims row group or file written with every leaf, without the
    numeric-ids alone, or without all six, by its first id: a file's row groups of
    several forms written in one, a group's partition files of several forms glued in
    one, group files of several forms on the build branch; keep's files published;
  - **mixed, killed** at a file's row groups written again, mid merge (a file written in
    the group's form) and in the sort's rebuild, each run again until complete: keep's
    files published;
  - **a new entity type mid-release:** an `M5` mediainfo value injected in one entity: its
    group's claims file keeps the main snak's two leaves, the others drop all six; keep's
    files published;
  - **units:** the rule on ids of every form (and junk: `Q`, `Q01`, 19 digits, `M5`,
    lowercase, whitespace, a full-width digit); a round trip through pyarrow and pq-glue;
    `footer_size` the same as the full file's; a changed value (a qualifier's numeric-id,
    a reference's entity-type), a MediaInfo value and an entity value without an id each
    keep their site's leaf; a value under a null datavalue (not stored) does not; a
    schema less a leaf it does not name, naming one it has, or naming one not derived is
    refused, as are reduced rows given to `conform_table` or rows that lost their
    metadata given to `restore`; a file whose third row group allows a leaf fewer.
- **`scripts/test_schema_evolution.py`** (every scenario, the leaves dropped by default):
  fits publishes master's files byte for byte (and uploads them but for the claims group
  files); evolved, native, overlapped (each kill point, run again), strict, incompatible
  and promote pass. Fields added mid-release, a reference snak datavalue's among them,
  are put back in each file's own version and conformed to the frozen one. (Its
  `upgrade_file` kill hook now passes keyword arguments on.)
- **`scripts/test_merge.py`, `test_claims_build.py`, `test_group_restart.py`:** pass.
- **`scripts/test_split_claims.py`** (r50, s30): the split's claims files the pipeline's
  in rows, order, schema and row groups, and its after_claims and in_move restarts,
  pass. Its after_claims hook took the claims file's path as `write_built`'s last
  argument, which is its schema since #182: fixed. Its after_numbered step finds no
  scholarly manifest midway on these two chunks (a routing step this does not touch).

## The bench

Pending: the PR's paired A/B (bench 17.14.0, quick tier, 3 rounds).
