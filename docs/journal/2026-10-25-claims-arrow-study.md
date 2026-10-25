# 2026-10-25: The claims built as Arrow in the split (a design study, with a prototype)

Continues lmmx/polars-genson#319 (`docs/journal/2026-10-25-claims-call-profile.md`, genson
1.5.0). That profile found no cut inside genson. It said the structural cut was upstream:
the split parses each dump line with sonic-rs and writes its claims back out as JSON text,
and genson then reads that text and lexes it again.

## Current State

- Nothing changed in `src/`, `reshape/` or `bench/`. This is a study. The prototype and
  scripts are in `2026-10-25-claims-arrow-study/`.
- **The prototype** is a columnar builder in Rust, fed from reshape's parsed line
  (sonic-rs). It appends each entity's claims to buffers laid out as
  `process.claims_schema`.
  - **Exact on the bench's data:** its arrays equal the claims call's output on 6 whole
    chunks and 4 slices of 2,000–3,000 entities (c0, r10–r90, s10–s90). Its claims
    partition files are byte-identical to the pipeline's.
  - **What exactness needed:** three non-obvious rules (see Semantics).
- **Measured:** the builder costs 1.2–1.4 times reshape's claims text writing. It
  replaces genson's whole claims decode.
- **Recommendation:** go, once three things are settled (see Recommendation). The
  biggest is that the CI bench can't measure this until its pipeline runs on the split's
  own output.

## 1. Is the claims schema fully known at split time?

**The output schema is fixed.** `process.claims_schema` is written out in process.py, and
every chunk's claims are conformed to it (`schema=claims_schema, unnest=True`).

**The decode schema is stored, but inferred from data.** `schemas/claims.avsc` was
inferred by `scripts/avro_schemas.py` from the bench's 11 chunks.
- genson decodes a chunk strictly against it only when every row fits.
- Otherwise the chunk is inferred, checked against `claims_schema` (the run halts on a
  misfit) and conformed.
- The choice is per chunk: one row that doesn't fit sends the whole chunk to inference.

**It can be fixed ahead of time without changing the published schema,** on one
condition. A builder must give exactly genson's strict decode against `claims.avsc`, and
must treat anything that doesn't fit as a misfit: the chunk then takes today's path.
- That path writes the JSON column, and genson infers, checks and conforms the chunk.
- Data that fits gives genson's own output. Data that doesn't goes through genson as it
  does now.
- The published schema is `claims_schema` either way.

The options a builder must reproduce, from `normalise_claims_direct` and
`CLAIMS_INFERENCE_OPTIONS`:
- `map_encoding="kv"`, `map_threshold=0`: `claims`, `qualifiers` and `snaks` become lists
  of `{key, value}`.
- `empty_as_null` (genson's default): an empty list or map becomes null.
- `force_scalar_promotion` and `wrap_scalars`: a bare string, integer or number in place
  of a promoted record goes to its `name__type` field (`datavalue__string`,
  `precision__integer` / `__number`, `latitude__number`, `longitude__number`).
- `prune=QUARANTINE_FIELDS` and `extract_invariants=LABEL_INVARIANTS`:
  - an official dump has none of these fields;
  - the builder treats any of them as a misfit, so such a chunk goes through genson's
    pruning and extraction as now.
- The fields `claims_schema` has and `claims.avsc` lacks (`latitude__integer`,
  `longitude__integer`): always null on the strict path.

## 2. What moves, and what vanishes

### The split, per slice (Rust, one thread, the best of 7)

The split's lines are the bench's chunks turned back into dump lines
(`bench/dumpslice.entity_line`), so they are what `split_stage` reshapes.

| slice | parse only | reshape now | without claims | claims as text | claims built |
|---|--:|--:|--:|--:|--:|
| c0, 2,000 entities | 0.15 s | 0.62 s | 0.26 s | 0.36 s | 0.50 s |
| s30, 3,000 | 0.09 | 0.39 | 0.12 | 0.27 | 0.36 |
| s90, 3,000 | 0.09 | 0.37 | 0.12 | 0.25 | 0.32 |
| r10, 3,000 | 0.03 | 0.12 | 0.04 | 0.08 | 0.11 |

- **Claims as text:** writing the claims as text is 58–69% of reshape's CPU.
- **Claims built:** the builder costs 1.2–1.4 times the text writing. It holds no lex: the
  sonic-rs parse is already paid for.

### The claims path, end to end, per slice

All CPU seconds, user and sys, the median of 5 (`e2e_slice.py`). The rows don't depend on
the design:
- **claims text write and builder:** from the table above;
- **JSON claims zstd:** the chunk write with the claims column less the write without it;
- **claims call:** `normalise_in_schema` as `normalise_claims_direct` calls it;
- **IPC write and read:** the typed claims as an Arrow IPC file (zstd), read back into
  genson's batches;
- **write_table:** the claims partition write, the same in every design.

| slice | claims text write | builder | JSON claims zstd | claims call | IPC write | IPC read | write_table |
|---|--:|--:|--:|--:|--:|--:|--:|
| c0_2000 | 0.36 | 0.50 | 0.43 | 0.93 | 0.37 | 0.40 | 0.75 |
| s30_3000 | 0.27 | 0.36 | 0.20 | 0.75 | 0.20 | 0.32 | 0.40 |
| s90_3000 | 0.25 | 0.32 | 0.23 | 0.78 | 0.28 | 0.21 | 0.40 |
| r10_3000 | 0.08 | 0.11 | 0.10 | 0.22 | 0.11 | 0.10 | 0.18 |

Three designs, the claims path's CPU each (user only in brackets):

| slice | now | A: typed claims beside the chunk (IPC) | B: the split writes the claims partition |
|---|--:|--:|--:|
| c0_2000 | 2.48 (2.08) | 2.03, −18% (1.75, −16%) | 1.25, −49% (1.22, −41%) |
| s30_3000 | 1.62 (1.40) | 1.29, −21% (1.12, −20%) | 0.76, −53% (0.75, −47%) |
| s90_3000 | 1.66 (1.29) | 1.20, −28% (1.07, −16%) | 0.72, −57% (0.70, −46%) |
| r10_3000 | 0.57 (0.53) | 0.50, −13% (0.48, −10%) | 0.29, −49% (0.28, −47%) |

How each total is made:
- **now:** text write + JSON zstd + claims call + write_table;
- **A:** builder + IPC write + IPC read + write_table;
- **B:** builder + write_table.

**What vanishes in A** (reshape builds the claims and writes them as an Arrow IPC file
beside the chunk; the pipeline reads it and runs `statements` and `write_table` as now):
- the claims text write and its zstd;
- the pipeline's Parquet read of the JSON;
- genson's lex, walk, appends, `extract_invariants` scan and `conform`.

What comes in its place: the builder, and the IPC file's write and read.

**What vanishes in B** (the split also writes the claims partition file, so the pipeline's
claims step only checks and places it): everything A removes, and the IPC round trip too.
`write_table` moves into the split.

**Why not genson's own decoder in the split?** It would decode the claims text reshape
has just written, so it keeps the lex. It drops only the Parquet read (about 12% of the
call) and swaps the JSON column's zstd for an IPC write, then adds the IPC read. On
c0_2000 that makes the path 2.70 CPU-s, against 2.48 now (+9%): a loss.

**Why not typed claims inside the chunk Parquet?** Measured on the whole of c0 and
rejected: nested Parquet is the costly write the campaign already knows.
- **The write:** 5.8 s with pyarrow and 3.6 s with genson's writer, against 1.0 s for the
  JSON column; Polars took 21.7 s.
- **The size:** 54–66 MB, against 40.5 MB for the JSON column.
- **The read:** 1.2 s user, against 0.27 s.

(`pipeline_side.json`, `ipc_side.json`.)

### The pipeline worker, whole chunks, fresh processes

Design A against now (`pipe_rss.py`, 3 runs each):

| chunk | now: user, wall, peak RSS | A: user, wall, peak RSS |
|---|---|---|
| c0 | 4.84–4.98 s, 1.6–2.6 s, 1.45 GB | 2.85–3.12 s, 0.95–1.01 s, 1.40 GB |
| s90 | 2.63–2.82 s, 1.5–1.9 s, 1.10 GB | 1.51–1.60 s, 1.0–1.2 s, 1.10 GB |

### The release, modelled (not measured)

**Assumptions:**
- bench 10.0.0's figures (#128, "after"): the split 32.9 CPU-s, the pipeline's claims
  stages 13.6 CPU-s (`normalise_in_schema` 8.3, `write_table` 6.0), reshape 5.3 CPU-s, and
  ETAs of split 3h03m, pipeline 5h03m, finalise 3h06m;
- this VM's ratios hold on CI;
- the claims text write is 58% of reshape (3.1 CPU-s);
- the JSON claims' zstd is about 2.5 CPU-s of the split's chunk writes (the release
  profile's local 2.8–4.5 CPU-s, 85% of it claims).

**Design A:**
- the pipeline's claims call 7.5 → about 2.6 CPU-s (IPC read);
- the split +0.9 CPU-s (builder −3.1 +4.0; IPC write in place of the JSON zstd);
- the sample: about −4 of 57 CPU-s (−7%);
- the ETA: pipeline about −38m, split +5m, net about −33m.

**Design B:**
- the pipeline's claims stages 13.6 → about 0.5 CPU-s;
- the split +4.4 CPU-s (+0.9 builder, −2.5 JSON zstd, +6.0 `write_table`);
- the sample: about −8.7 of 57 CPU-s (−15%);
- the ETA: pipeline about −1h42m, split +25m, net about −1h17m of 11h13m.

**The part from the scalings alone:**
- the split is scaled by dump bytes, 1,339× (103.5 GB of bz2 over the 77.3 MB slice), over
  4 cores: about 335 s of ETA a CPU-s;
- the pipeline's claims are scaled by claims rows: 1,172× main, 1,960× scholarly;
  - over 4 cores, with the pipeline's busy factor (wall × 4 / CPU, about 1.2), that is
    about 474 s a CPU-s for a half-and-half mix.
- So each CPU-s moved from the pipeline to the split reads about 139 s shorter, whatever
  it costs: about 17 s shorter for main-set claims alone, 253 s for scholarly.
- **In B:** moving `write_table` (6.0 CPU-s) accounts for about −14m of the −1h17m.
- **In A:** the move is about 0.9 CPU-s, about −2m.

**Wall time at a release:**
- From bz2, the split is CPU-bound (lbzip2's decode is 1h42m of its 3h03m), so what moves
  into it adds wall time.
- From gz via Wikimedia (about 3h to download, split ETA 58m), the split waits on the
  download, and work moved into it is nearly free in wall time.

## 3. Semantics: what has to be identical

**Shown identical** (`identity.py`):
- the claims arrays, against the call's output with `claims.avsc`;
- `statements(...)`;
- the claims partition file's bytes (`write_table`).

They were checked on r30, r50, r70, r90, s10 and s50 whole, and on slices of c0, s30, s90
and r10. Getting there needed three rules, each found by a mismatch:

1. **genson's float parse is not the nearest double.** The typed decoder parses numbers
   through `serde_json::Number` without `float_roundtrip`.
   - On c0's first 2,000 entities, 4 of 1,302 latitudes, 19 of 1,302 longitudes and 7 of
     1,202 precisions came out a unit in the last place from the nearest double to the
     text. For example, `-3.8209166666666667` is decoded as `-3.8209166666666663`.
   - The published claims hold these values. A builder must parse floats the same way to
     stay identical (the prototype calls `serde_json`).
   - Correct rounding would change published content: a separate decision.
2. **Promotion follows `claims.avsc`, not `claims_schema`.**
   - An integer latitude goes to `latitude__number`, as genson's strict path puts it
     (the avsc has no `latitude__integer`). The inferred path did the same on c0's slice.
   - Untested: a chunk whose latitudes are all integers. Inference could give
     `latitude__integer` there, so that chunk's output depends on its path.
3. **The partition file's bytes depend on how the claims are batched.**
   - Equal arrays in other batches give a different file: the same table combined into
     one chunk changed the bytes.
   - genson's batches are `rows.div_ceil(rayon threads × 4)` rows (polars-genson-py
     `parquet_io.rs`), each its own arrays at offset 0. Built in those batches, the file
     was byte-identical.
   - So **today's published claims files already depend on the worker's core count**, a
     4-core runner against a 20-core release machine. Byte identity is per machine shape.

**Rules a production builder needs that the prototype doesn't implement:**
- **Strict-fit failures:** a null where the avsc's record field is not nullable
  (`mainsnak.snaktype`, `property`, `datatype`) must be a misfit. The prototype appends a
  null.
- **Misfit is per chunk:** one row that doesn't fit sends the whole chunk to today's path.
  The split holds the chunk's lines, so it can reshape them again as text.
- **`dump.entity_row`'s fallback lines** (the lines the compiled reshape leaves to
  Python) carry their claims as JSON. Their chunk takes today's path, or its cells are
  decoded by genson.

**Checks:**
- `check_ids` and the sort's checks are untouched.
- The bench's parity (published keys) would hold.
- **The bench's split parity would not:** it compares the chunks to the sample's 7 JSON
  columns, and a built chunk has no claims JSON.
- **The scholarly routing would change form:** `is_scholarly` is a regex over the
  reshaped claims text. It would become the same test over the parsed line, its result
  tested for equality with the regex on every bench chunk.
- **Restart safety:**
  - a chunk's manifest line may be written only once all its files are in place, the
    claims file renamed into place last;
  - in B, `route`'s `_move_routed` moves the claims files too.

## 4. Memory and disk

**Disk:**
- **A:** the typed claims as IPC (zstd) are 1.27× the JSON column's bytes on c0, 1.49× on
  r10 and 1.67–1.76× on s30 and s90. A higher zstd level barely helps: level 9 gives
  0.95× level 1's bytes at 6 times its CPU.
  - Over a release's chunk files (94.2 GB, about 80 GB of it claims), that is about
    +40–50 GB, held until each chunk is processed. Modelled.
- **B:** the chunk files lose their claims column (about −80 GB). The claims partition
  files, 0.91–1.11 times the chunks, exist from the split onwards rather than as the
  pipeline goes, so the peak is about what it is today.

**The split worker's peak RSS** (the prototype, c0's first 2,000 entities: lines 136 MB):
- parse only: 145 MB;
- reshape now: 270 MB;
- built, the whole slice held: 542 MB;
- built in batches of 125 or 1,000 entities: 368 MB.
- The prototype's batch export copies each batch, so a real writer would be lower.
- Arrow holds 1.44 times the claims text's bytes: 157 MB against 109 MB.
- **In B**, `write_table` takes a whole table: the split would hold a part's claims whole,
  about 1.4 GB on a c0-like chunk.
  - That is per split worker, and a release machine runs `cpu − 2` of them.
  - So B needs either a streaming claims writer, or the split's in-flight claims bounded,
    as `MAX_PENDING_BYTES` bounds its lines.

**The pipeline worker:** in A, about as now (1.40 against 1.45 GB on c0; 1.10 GB on s90).
In B its claims step holds almost nothing.

## 5. Implementation plan

**Where the builder lives:**
- **In genson-core** (one place for the semantics) needs two things first:
  - **The allocator:** genson-core sets mimalloc as the global allocator in its library
    (`genson_rs/mod.rs`, unless `alloc-stats`). Linking it into reshape would change
    reshape's allocator, which the rules don't allow. A default feature gating it would
    leave polars-genson as it is and let reshape keep the system allocator.
  - **A typed-plan entry point over a parsed input:** `NormalisePlan`/`ArrowPlan` walks
    genson's own tape, so its walk would have to be generic over the input.
- **In reshape** (the prototype's form, about 300 lines): genson's strict-decode rules for
  this one schema, duplicated. They are guarded by a test that the arrays equal genson's on
  the bench's chunks; the bench's parity would catch drift too.

**PRs, in order** (after the reshape agent's raw-span PR):
1. **bench/** (its owner): run the pipeline on the split's own output, and make the
   split's parity compare the decoded claims. Without it, CI can't see the change: today
   the pipeline reads `bench-data`'s JSON chunks.
   - Also add the figure asked for: split + pipeline + finalise CPU and wall on the sample.
2. **genson** (if the builder lives there): gate the global allocator; expose the typed
   plan over a parsed input. Benched through the CI wheel (`bench/wheels`, as in #121).
3. **reshape/src/lib.rs:**
   - the builder, appended in genson's batches and handed over as an Arrow C stream;
   - the misfit signal, per chunk;
   - the P31 test, for routing.
4. **dump.py:**
   - write the built claims (A: an IPC file beside each part; B: the claims partition
     file), or the JSON column on a misfit;
   - route by the parsed P31 test;
   - write the manifest only once a chunk's files are all in place; move the claims in
     `_move_routed`.
5. **process.py:**
   - claims from the built file where there is one (A: read, `statements`,
     `write_table`; B: check ids, place);
   - write the empty lookup the official dump gives;
   - the JSON path unchanged otherwise.

**Tests:**
- arrays equal to genson's on every bench chunk;
- byte-identical partition files at the same thread count;
- misfit cases (an unknown field, a type mismatch, a null at a non-nullable field, a
  pruned or extracted field) take today's path;
- routing equal to the regex;
- `scripts/test_reshape.py` and the bench's parity.

## Recommendation

**Go, once three things are settled:**
1. **The bench:** it must run the pipeline on the split's output, or nothing here can be
   measured on CI.
2. **Which design:**
   - **A** keeps the stage boundary: −13 to −28% of the claims path locally, +40–50 GB of
     chunk disk at a release.
   - **B** removes about twice as much: −49 to −57%. But it writes the claims partition
     in the split, so the split needs bounded claims memory, and it moves a pipeline
     stage into the split.
3. **Where the builder lives:** genson-core, with its allocator gated, or reshape, with
   the identity test.

**The user's deferred item:**
- **A** doesn't touch "the pipeline following the split": nothing in the pipeline starts
  before the split ends.
- **B** doesn't overlap the stages either, but it moves the claims stage of the pipeline
  into the split's pass. That is close enough that the user should decide it.

## Scripts

In `2026-10-25-claims-arrow-study/`:
- **The prototype:** `main.rs` and `proto.Cargo.toml` (its `Cargo.toml`). Put lib.rs's code up to `fn run`, without
  its pyo3 and rayon imports and its `//!` lines, in `src/reshape_inc.rs`:
  `awk '/^fn run\(/{exit} {print}' reshape/src/lib.rs | grep -v -E '^use (pyo3|rayon)|^//!'`.
  Then drop its trailing doc comment.
  - Modes: `claims-proto LINES [REPS]`.
  - `SUMS=1` prints leaf checksums; `EXPORT=FILE BATCH=N` writes the built buffers.
- **Inputs:** `make_slices.py` writes the slices, as chunk rows and as dump lines.
- **Comparisons with genson's output:** `leaf_sums.py` and `identity.py` (with
  `pipeline_side_lib.py`, the claims call as `normalise_claims_direct` makes it).
- **Timings:**
  - `e2e_slice.py`, `pipeline_side.py`, `ipc_side.py`, `pipe_rss.py`;
  - `detcheck.py`: the partition write's bytes against the claims' batching.
- **Results:** the `.json`, `.jsonl` and `.txt` files.
