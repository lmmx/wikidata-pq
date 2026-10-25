# 2026-10-25: The split's own CPU, profiled, and the claims built from the dump line

After fusion A and E3, the split was the release's largest stage: 3h10m of a modelled
~7h27m (bench 15.2.0).
- **bz2 decoding:** about 1h47m, which lbzip2 keeps at its floor.
- **The rest, ours:** about 1h25m at a release, ~17 of the slice's 35–39 split CPU-s. The
  gz split's CPU (16–17 s) is nearly all of it.

This entry profiles that rest, ranks what is in it, and records the two changes made:
- lmmx/wikidata-pq#162: in-scope cuts;
- the structural PR, stacked on #162: the claims built from the dump line in the
  reshape's scan.

## Current State

- **#162** (`perf/split-claims-nulls`):
  - the claims decode's nulls written in runs;
  - the map key check without a hash set;
  - `statements`' schema derived once for each schema seen.

  Its outputs are byte for byte master's. CI: `partitioning.claims.statements`
  **−91.1% ±2.7%**, CPU of the split from gz **−3.8% ±0.5%**. The bz2 split's CPU is
  within noise (±3.2 s).
- **The structural PR** (`perf/split-claims-from-line`, based on #162, so CI measures it
  against #162):
  - the reshape builds each line's claims from the line itself, writing no claims text
    and decoding none;
  - each row's set (`is_scholarly`) comes from the same scan;
  - any chunk it does not mirror exactly takes the text path from its start.

  On the bench slice its files are byte for byte master's, the claims partition files
  included.
- **Measured locally:** about −1.6 CPU-s a slice against #162, and about −2.3 against
  master, on a shared, loaded VM. CI's paired figures are the authority.
- **CI, lmmx/wikidata-pq#163** (363c23b against #162's 80395b5, 3 rounds, quick tier):

| | before (#162) | after | |
|---|---|---|---|
| CPU, split from gz | 16.8 s ± 1.2 | 15.6 s ± 1.0 | **−1.1 s (−6.8% ±2.0%)** |
| ETA: split from gz | 1h33m30s | 1h27m11s | **−6m19s** |
| CPU, split (bz2) | 37.3 s ± 1.0 | 36.0 s ± 2.2 | −1.3 s, within noise |
| ETA: split | 3h23m05s | 3h15m49s | −7m16s, within noise |
| wall, split | 10.42 s | 9.81 s | within noise |
| peak RSS, split | 2.684 GB | 2.815 GB | +131 MB (+4.9%) |
| ETA, a release | 6h59m47s | 6h52m14s | within noise |
| parity, split (bz2 and gz) | passed | passed | |

  - **Peak RSS:** the rise is the chunk's claims held as arrays (about 1.45× their text)
    where the text was held before.
  - **Phase rows:** `claims_build.claims_arrays` goes to 0. The reshape and build now count
    in `dump.write_chunk` (+2.7 s). `dump.route_parts` reads −6.6 s, an artifact: the
    split's stages take their thread's clock, `write_table` (called inside it) its
    process's.
  - **The span's own CPU** (the split and routing of the slice): 7.4 → 6.5 s.
- **CI, #162** (80395b5 against master 3046fdb): `partitioning.claims.statements` 1.2 →
  0.1 s (**−91.1% ±2.7%**), `claims_build.claims_arrays` 3.1 → 2.9 s, CPU of the split from
  gz 16.2 → 15.5 s (**−3.8% ±0.5%**); the bz2 split within noise.

## The profile

**Method:**
- `dump.write_chunk` on the bench slice's dump lines (bench 11+'s escaping, from the
  slice's `.json.bz2`): c0, its first 10,000 entities (463 MB), then the other 23,220
  (573 MB).
- Each stage's own thread and process CPU, taken with wrappers
  (`time.thread_time`/`process_time`, nested stages subtracted).
- `perf record` of the timed section only, attached by pid after warm-up, at 1999 Hz, plus
  DWARF call graphs for the callers.
- `perf stat`'s hardware counters are not available on this VM.

**The split's in-worker CPU, master, per slice (≈13.4 CPU-s):**

| stage | CPU-s a slice | share | notes |
|---|---|---|---|
| genson's claims Parquet encode (`write_table`, its thread pool) | 5.2 | 39% | the build branch's encode, a floor the contract sets; not ours |
| `claims_batches`: the claims text decoded into Arrow | 3.6 | 27% | `Col::null` 4.6% and `Bits::push` 1.7% of all samples |
| the reshape (`entity_rows`) | 2.0 | 15% | about 60% of it is the claims text writer (`claims::snak` alone 8% of all samples) |
| the fused term and link builders (`fused_partitions`) | 1.0 | 7% | zstd and `Writer::strings` most of it |
| `statements` | 0.65 | 5% | 0.43 of it a Polars plan run on no rows for the schema, 8 ms a batch |
| `route_parts`' own: `is_scholarly`'s regex | 0.43 | 3% | `regex_automata` find_fwd |
| the refs sidecar, entities, `check_ids`, the glue, Python | ~0.5 | 4% | |

**Also:**
- Kernel time was 9% of the timed section's samples. Most of it is page faults and
  zeroing, from buffers allocated fresh for each chunk and batch: the claims text column
  alone is ~350 MB of c0. Allocator changes are out of bounds, but not materialising the
  text removes its share.
- **Per-part fixed cost:** a 10-entity chunk costs ~40 ms of CPU.
  - ~9 ms of it was `statements`' schema, removed by #162;
  - ~11 ms is the genson writer on tiny row groups;
  - ~4 ms is `fused_partitions`; ~3 ms is the entities write.
  - At a release's 23,035 parts this is ~920 CPU-s (~4 min of wall on 4 cores). The bench
    models ~6,500 parts, so it sees about a quarter of that.

**Ranked, by what a cut could take of the ~17 CPU-s (bench terms):**
1. **The claims parsed twice:** the reshape writes the claims as text, and
   `claims_batches` lexes that text again. The writer, the text column, the second lex,
   the regex pass and the per-batch Python together are about a third of the in-worker
   CPU. That cut is the structural PR.
2. **The decode's null writes:** about 6% of the in-worker CPU (#162).
3. **`statements`' per-batch schema:** about 3% (#162).
4. **Per-part fixed costs:** ~25 ms a part of the remaining ~30 ms is genson's writer and
   the builders' file writes. That is ~2–3 min at a release, little of which the bench
   sees.
5. **genson's encode:** 39%, but fixed by the contract and the writer (zstd levels and a
   single-threaded write are already rejected).

## The structural change: claims from the line

**Before:**
- the reshape (`lib.rs` `entity_row`, `claims.rs`) wrote each line's claims as orjson's
  text, reshaped;
- `route_parts` ran `is_scholarly`'s regex over the whole claims column;
- `claims_build.write_claims` cut each part into 16 MB batches of that text, and
  `claims_batches` decoded each batch into genson's arrays.

**Now** (`claims_arrow.rs` `RawBuild`, `lib.rs` `entity_rows_built`):
- At the `claims` key, the reshape's scan hands the line to the claims decoder, which
  builds the arrays from the line's own bytes and returns where the value ends. The scan
  goes on from there.
- It is the same decoder (`P`), in a `raw` mode:
  - **Numbers:** each is checked by the reshape's grammar (`scan_number`, factored out of
    `Line::number`). One the reshape would rewrite is rewritten by its `write_number`, and
    genson's `Num::read` reads the rewritten text. So floats keep genson's
    (not-nearest-double) values.
  - **The reshape's changes of shape, made as the line is read:**
    - a map (the claims, qualifiers, a reference's snaks) given as `[]` or `null` is
      `{}`;
    - a snak's `datavalue {value, type}` becomes its `datavalue` and `datavalue_type`
      fields, missing ones null;
    - a reference without snaks gets `"snaks":{}`.

    Each is decoded from the literal text the reshape would write (`P::lit`), so the plan's
    rules (`empty_as_null`, nullability) apply exactly as on the text.
  - **Strings:** decoded from the dump's escapes (`\uXXXX`, surrogate pairs, `\/`) to the
    same UTF-8. `hex4` is now strict: `from_str_radix` took a leading `+`. The reshaped
    text never held one, so the text path is unchanged.
- **The routing:** for each mainsnak the scan records whether its reshaped text would
  match `P31_SNAK`.
  - In that text the snak's other fields come first, then `"datavalue":` and the value.
    So the regex matches when:
    - those fields hold no brace;
    - one of them is `property` = `P31`;
    - the value is an object whose text begins `{"entity-type":"item","numeric-id":\d+,"id":"(Q\d+)"`.
  - Matches can't overlap: a match holds a `{` only at its own two braces. Starts can only
    be real mainsnak keys: strings are escaped, and an escaped key is a misfit.
  - A row is scholarly if any match's id is in `SCHOLARLY_CLASSES`. An id with non-ASCII
    digits, which `\d` would take, is in no class.
  - A P31 snak that could match, whose value the reshape would rewrite (an escape, a
    rewritten number), is a misfit.
- **The sets:** each row is built in the set of the row before. If its own set is the
  other, it is moved there (`Col::split_tail`, `append`).
  - Each set's arrays are cut into batches of ~16 MB of the lines' claims
    (`BATCH_TEXT_BYTES`), each its own arrays.
  - Measured: one part-long batch cost the Parquet writer's calling thread 1.2 s more on
    c0, as if each row group were the whole batch, and ~1 CPU-s more in all.
  - In these batches, the claims files come out byte for byte the text path's.
- **The fallback:** anything the raw path does not mirror exactly ends it for the chunk,
  which is reshaped and built as before (`entity_rows`, the regex, `claims_batches`):
  - a line the scan leaves to `entity_row`;
  - any misfit, the decoder's own or one of the raw path's (a datavalue member other than
    value and type, a map given as another non-object, a statement without a mainsnak);
  - the uncertain P31 above.

  So a chunk's rows, parts and files are always one path's. The text path's per-part
  misfit rule (one part JSON, the other built) still holds within a fallen-back chunk.
- **dump.py** (approved for this change): only `_reshape` and `route_parts`.
  - `_reshape` asks `claims_build.reshape_built` first. `reshape()` keeps the text path,
    for `test_reshape.py`.
  - `route_parts` takes the sets from the flags, and writes each part's claims with
    `claims_build.write_built`: `statements`, `RowGroups`, the refs sidecar and
    `check_ids`, as `write_claims` writes them.

  `split()`, `_move_routed`, `_set_manifest`, `_write_manifest`, `EXPECTED_FIELDS` and
  `MAX_PENDING_BYTES` are untouched.

**Memory:**
- A chunk now holds its claims as arrays (~1.45× their text) rather than as text plus one
  16 MB batch at a time.
- The claims text column (~0.75× the lines) is gone.
- Per chunk in flight that is about +0.1–0.2 GB against the text path. CI's peak RSS will
  say whether it shows.

**Checks:**
- **`scripts/test_claims_from_line.py` (new).** For every chunk, either the line path
  leaves it to the text path, or all of these hold:
  - every other column is the text path's, byte for byte;
  - each row's set is the regex's;
  - each set's claims equal the text path's arrays, and the text path builds them all.

  It ran on:
  - all 33,220 slice lines, in chunks of 2,000, in the split's batches and in 64 KiB
    batches: all built, 16,220 scholarly rows, none left to the text path;
  - 6,000 of them shuffled, so the sets interleave as they do in a release's chunks (the
    bench's chunks hold one set each): 2,873 scholarly;
  - 72 edges, mostly the routing's: escaped ids and properties, braces in the snak's
    fields, members out of order, values past 18 digits, other digits, P31 in qualifiers
    and references, the reshape's map, reference and datavalue shapes;
  - 20,000 mutants.

  All passed: 2,720 mutants built, the rest left to the text path.
- **`scripts/test_split_claims.py`, end to end** (3,000 lines of c0 and 3,000 scholarly,
  plus its misfit entity): passes. Its crash after a claims file now covers `write_built`.
- **The other tests:** `scripts/test_claims_build.py`, `test_reshape.py` and
  `test_claims_refs.py` pass.
- **Byte comparison:** `write_chunk` on c0 and the rest gives every file of master's split,
  byte for byte.

**Measured locally (stage CPU, process time, the mean of 2 runs; the VM shared and loaded):**

| | #162 | from the line |
|---|---|---|
| c0: reshape + `claims_batches` + `route_parts` | 0.97 + 1.22 + 0.22 s | `entity_rows_built` 1.49 s |
| c0: `write_chunk`, process CPU | 5.77 s | 4.83 s |
| rest: reshape + `claims_batches` + `route_parts` | 1.00 + 1.40 + 0.20 s | 1.88 s |
| rest: `write_chunk`, process CPU | 5.57 s | 4.92 s |
| `write_chunk` wall, c0 / rest (one chunk alone) | 4.4 / 4.3 s | 3.8 / 3.7 s |

- **Against #162:** about −1.6 CPU-s a slice. At the bench's 327 s of ETA a CPU-second,
  that is ~9 min of the split's ETA, and ~10% of the gz split's CPU.
- **Against master** (with #162): about −2.3 CPU-s, ~12 min.

## Rejected or not built

- **Struct-level null deferral alone** (each struct's nulls owed to its children, the
  leaves still written one by one): −3% of `claims_batches`. The leaves' own nulls (a
  record's absent fields, a promoted scalar's siblings) are where the time is. #162 defers
  every column's.
- **One batch a part from the line:** byte-identical rows, but the genson writer's calling
  thread took 1.2 s more on c0 (with total CPU +~1 s). Row groups that are slices of one
  part-long batch cost as if each held the whole batch. Batches of ~16 MB, as the text
  path's, are byte for byte its files.
- **`is_scholarly` as an exact byte matcher over the text** (memmem for `"mainsnak":{`):
  about −0.35 CPU-s a slice. Subsumed by the scan's flags, which need no text.
- **`plan_of` cached across calls:** a plan of ~200 nodes, rebuilt 23–30 times a chunk:
  well under a millisecond each.
- **Not tried, out of bounds:**
  - the page-fault share (~5%), which is the allocator's;
  - genson's encode (39%): zstd levels and a single-threaded write are already rejected.

## Next

1. **The per-part fixed costs at a release's part count** (~4 min of wall). The bench
   models a quarter of them, so a release-shaped measure of 23,035 small parts' write costs
   would size it. genson's writer on tiny row groups (~11 ms) is most of the rest.
2. **`fused_partitions`' zstd and string writes** (7%). It already compresses each page
   once, so the gain left is small.
3. **Real-dump fallbacks:** the bench slice's lines are synthesised from chunk files. On
   a release's own lines, the rate of chunks the raw path leaves to the text path is
   unknown: a field claims.avsc lacks, for one, sends a chunk there whole. A count from a
   real dump's first few GB (the split logs nothing for it yet) would show whether the
   saving holds at a release.

## The fallback rate on a real dump

The bench's lines are synthesised from chunk files, so the rate at which a release's
chunks fall back to the text path was measured on the real dump.

**What was run:**
- The first 400 MB of `latest-all.json.bz2` (2026-10-21, 103.5 GB), streamed through
  lbzip2. Nothing decompressed was stored.
- The stream was cut into the split's chunks of 10,000 entities: 142,282 lines, 3,985 MB
  of JSON, Q1 to about Q150k.
- Each chunk went through `claims_build.reshape_built`. For each chunk that fell back,
  each line was run alone to find the reason, and the text path's decode was run on that
  line's claims.
- Every chunk also went through `test_claims_from_line.check`, in pieces of 2,000 lines.

**What came out:**

| | |
|---|---|
| chunks built from the line | 14 of 15 |
| chunks left to the text path | 1 (chunk 7), for 1 line: Q618385, "a field claims.avsc lacks" |
| lines left to `entity_row` | 0 |
| P31 values the reshape would rewrite | 0 |
| the text path's decode of Q618385's claims | the same misfit: its part's claims go as JSON on either path |
| line path against text path, on 140,281 rows (5 scholarly) | 0 differences: columns, sets and claims arrays all equal |

**Reading it:**
- By chunks the rate is 1 in 15 (6.7%); by lines it is 1 in 142,282. At that line rate a
  chunk of 10,000 entities falls back with p ≈ 7%.
- The sample is the dump's first, richest items, so the rate further on is likely lower.
  Only the dump's start can be read without downloading it all.
- **The cause is `claims.avsc`, not the raw mode.** The line holds one of the fields the
  avsc lacks: `claims.avsc` was inferred from the bench's 11 chunks, and its phantom
  fields include:
  - a reference snak's `latitude`, `longitude`, `globe`, `altitude`, `upperBound`,
    `lowerBound` and `precision__number`;
  - the `latitude__integer` / `longitude__integer` of a mainsnak or qualifier.

  The text path misfits on the same line. So the raw mode can't cover it (no decoder of
  the avsc can), and the chunk's output is the same either way.
- **The cost:**
  - a fallback chunk costs what master costs, plus the raw pass up to the misfit line: on
    average half a pass, ~0.7 CPU-s for a chunk this size;
  - at 7% of chunks that keeps about 90% of the saving.

**The common cause's fix:** `claims.avsc` holding those fields, re-inferred from more
data (`scripts/avro_schemas.py`). It would lower both paths' misfits:
- the raw path would build those chunks;
- the text path would build those parts, which today go as JSON to genson's inference in
  the pipeline, a much costlier route.

The published schema (`claims_schema`) already has these fields. That change is outside
this PR.
