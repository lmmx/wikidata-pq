# 2026-10-26: A globe-coordinate's numbers in one place, on every path (rule B)

Before this change, a whole-number globe-coordinate value (`47`, not `47.0`) landed in a
published column chosen by the path its part took, not by the value. #172 left this as
the known divergence (`scripts/test_claims_schema.py`'s DIVERGES). The user chose rule B:

> A globe-coordinate's `latitude`, `longitude` and `precision` always go to the canonical
> `__number` (Float64) column, whether the JSON wrote `47` or `47.0`, in every path.

- **The schema stays as it is.** `__integer` beside each field remains, and is now always
  null for a globe-coordinate.
- **The scope is globe-coordinates only.** A time's precision (an integer enum, 0 to 14)
  stays in `precision__integer`. Nothing else in claims changes.

## Current state

- **Done** (`src/wikidata/claims_numeric.py`, one call at each place the claims are
  written):
  - the rule applied on every path;
  - its tests (`scripts/test_claims_numeric.py`; `scripts/test_claims_schema.py`'s
    DIVERGES no longer diverge once canonical);
  - bench 17.5.0, whose parity passes the intended change and lists it apart;
  - real-dump frequencies, from six ranges of the 20261019 dump.
- **No polars-genson release is needed.** Nothing changed in genson or in the Rust builder.
- **The cost** is about 2 to 150 ms a chunk, measured locally. On CI no figure moved beyond noise.

## Where the values were placed, path by path

Each placement below was read from the code and checked on data (the bench's chunks and
11 real parts, below).

| path | latitude, longitude: `47` | precision: `1` | null precision |
|---|---|---|---|
| **The split's build** (claims_build.claims_arrays / reshape_built, reshape/src/claims_arrow.rs `promoted`) | `__number`. claims.avsc has no `__integer` there (`has_int_field` false), so an integer goes to the float field | `__integer`. claims.avsc's precision record has both fields, as a time's precision is an integer | null record |
| **genson's strict decode** against claims.avsc (process.normalise_claims_direct, `normalise.rs promoted_scalar_matches`) | `__number`, by the same rule (the build mirrors it) | `__integer` | null record |
| **genson's inference** (the fallback: a part that misfits claims.avsc, and every part of a release before claims.avsc) | `__integer` where the part's inferred record has an integer field, else `__number`. The record has one when any property's coordinates at that place (mainsnak, qualifier, reference) are all integers in the part. Integers and floats in one property's node widen to a number (`unification.rs`), but different properties' nodes unify into a record of both. | `__integer` wherever a time is at that place in the part (nearly always), else `__number` | a record of nulls where every other precision there is a number, else a null record |

- **No genson option changes this.** `force_scalar_promotion` (mainsnak, datavalue,
  precision, latitude, longitude) is what makes each a record of promoted fields. With it,
  inference names an integer field `__integer` and a float one `__number`. Without it, a
  field of both kinds widens to one `number`, but that changes the published schema.
- **`claims_build.py`** compiles the decode plan from claims.avsc and the target schema;
  the decision itself is the Rust builder's. The build and genson strict are equal
  (`scripts/test_claims_build.py`), and stay equal: the rule is applied after both.

## Time's precision is `precision__integer` on every path

- **From the code:** a time's precision is a JSON integer.
  - On the build and strict paths it goes to `__integer`, because the record has an
    integer field.
  - On inference, every time property's precision node is an integer, so the unified
    record has `precision__integer` and the integer goes there.
  - A float time precision would go to `__number` on every path, but none exists.
- **From the data:**
  - **The dump:** all 7,824,856 time precisions in the six ranges below are JSON
    integers (values 0 to 11; 86% are 11). None is a float, and no datavalue type other
    than time and globe-coordinate has a `precision`.
  - **The bench's 11 chunk files:** 888,938 time precisions, all integers.
  - **On the arrays** of the bench's c0, r50, r90, s30 and s90, and of 11 real parts by
    all three paths: every time precision is in `__integer`; `precision__number` is
    null for each.
- **Nothing else puts an integer in `precision__integer` chunk-dependently**, but the
  globe-coordinate's own integer precision: under inference, it went to `__number` in a
  part with no time at that place. This change moves it to `__number` on every path.

## The rule (`claims_numeric.canonical_coordinates`)

On the (`id`, `claims`) table each path hands over, before `statements` lays it out:

- **At** each statement's mainsnak, its qualifiers' snaks and its references' snaks;
- **for** each snak whose `datavalue_type` is `globecoordinate`:
  - each of `latitude`, `longitude`, `precision` holding an integer: the integer goes to
    `__number` as a Float64, and `__integer` becomes null;
  - a record of nulls (inference's null precision): a null record, as the build gives it.

It is applied at three places:
- `process.normalise_claims_direct` (genson, strict or inference): one line;
- `claims_build.write_claims`: the split's text path;
- `claims_build.write_built`: the split's raw path.

So the Rust builder and genson strict still decode exactly as before, and are still
compared exactly (`test_claims_build.py`, `test_claims_from_line.py`). The rule is then
one function on their equal output.

- **Cheap:** the arrays are rebuilt only along the path to a moved cell (`from_buffers`
  with the parents' buffers). A chunk with nothing to move gets its own table back, the
  same object.
- **Measured locally:**
  - c0 (620k statements): 20 to 150 ms. The first call on fresh arrays is the dearest.
  - r50: 3 ms; s30: 4 ms; s90: 5 ms.
  - Against the build's 8 s for c0, that is under 2%.

## Exactness

- **An integer is moved only if a Float64 holds it exactly.** The cast is Arrow's
  checked `int64 → float64`, which raises outside ±2^53, and `NotExact` then halts the
  split's worker or the pipeline's process step. Nothing is rounded.
  - 2^53 itself moves exactly.
  - An integer with a float already beside it also halts. No path writes one.
- **Tested** at each place for latitude, longitude and precision (2^53 + 1):
  - a precision halts on every path, as it is held as an integer on each;
  - a latitude or longitude halts on the inference path.
- **What this doesn't cover.** On the build and strict paths, an integer latitude or
  longitude never reaches this code as an integer. claims.avsc has only a float field
  there, so genson and the builder decode the token straight to a double (serde's
  `as_f64`, as before this change). That is exact for |v| ≤ 2^53. Wikibase bounds
  latitude to ±90 and longitude to ±360; the largest whole number in the sample is 336.
  - An absurd integer past 2^53 there would be rounded, as it always has been.
  - No path can tell it from a float token of the same value.
  - Halting there would need the Rust builder and genson to treat such a token as a
    misfit. That is a genson change, left out of this narrow fix.

## Tests

- **`scripts/test_claims_numeric.py`** (new). 31 cases, each alone in its part (the
  thinnest part, where inference has an integer field) and beside a release's context:
  - values whole, fractional, negative, negative fractional, zeros, float zeros (−0.0),
    a null precision, the extremes (−90, 180, 360), an integer beside a fraction, and
    2^53;
  - at a mainsnak, a qualifier and a reference, each beside a time.

  For each case:
  - the build, genson strict and genson's inference, each made canonical, give the same
    Arrow arrays (schema and values);
  - each globe-coordinate's numbers are in `__number`, a whole number exactly
    `float(v)`, and its `__integer` is null;
  - each time's precision is `{__integer: v, __number: null}`;
  - between each path's arrays before and after the rule, only those cells differ, with
    the same number;
  - the split's claims file (`write_claims`, `write_built`) is `normalise_claims_direct`'s,
    with claims.avsc and with the fallback;
  - the bench's Polars rule (below) agrees on each path.

  It also tests the exactness halts above, and that a slice gives its slice of the
  whole's.
- **`scripts/test_claims_schema.py`:** each DIVERGES case still parts in genson's raw
  arrays, as expected, and is now the same once canonical. The MISFITS are the same raw
  and canonical.
- **Unchanged and passing:**
  - `test_claims_build.py` (edges, misfits, c0, r50, s30);
  - `test_claims_from_line.py` (edges, 20,000 mutants);
  - `test_split_claims.py` (3,000 lines each of c0 and s30, interrupted runs);
  - `test_claims_refs.py` (r50).

## How often it happens (real data)

**What was read:** byte ranges of `wikidata-20261019-all.json.bz2` (103.49 GB) from
dumps.wikimedia.org, not the Hub. These are the same offsets as #172's ranges, 200 MB
each. Each was resynced with `bzip2recover`, streamed through lbzip2, cut into chunks of
10,000 lines, then deleted. Every snak was scanned (mainsnak, qualifier and reference),
and the JSON token's type was read (orjson keeps `47` and `47.0` apart).

| range | lines | chunks | coordinates | integer precisions | integer lat/lon cells | null precisions | chunks with an integer precision |
|---|---|---|---|---|---|---|---|
| head (0) | 56,603 | 6 | 18,800 | 125 | 420 | 621 | 6 |
| p10 | 282,307 | 29 | 5,299 | 9 | 4 | 0 | 5 |
| p25 | 218,700 | 22 | 30,626 | 97 | 319 | 389 | 3 |
| p50 | 214,116 | 22 | 30,078 | 114 | 342 | 365 | 5 |
| p75 | 131,276 | 14 | 25,643 | 102 | 329 | 467 | 7 |
| tail (99.8%) | 449,919 | 45 | 60,536 | 2 | 65 | 0 | 1 |
| **all** | **1,352,921** | **138** | **170,982** | **449 (0.26%)** | **1,479** | **1,842 (1.1%)** | **27 (20%)** |

- **Where the coordinates are:**
  - mainsnaks: 164,493;
  - qualifiers: 6,476, of them 1 integer precision and 12 integer lat/lon cells;
  - references: 13, all floats (the #172 misfit shape).
- **The integer precisions' values:** 1 (425), 10 (16) and 0 (8). The largest whole
  numbers are latitude 90 and longitude 336.
- **What changes on the build and strict paths** (every release chunk that fits
  claims.avsc, nearly all since #172): only the integer precisions, 0.26% of
  coordinates, from `precision__integer` to `precision__number`.
  - About 1 in 3,000 lines, in 20% of chunks.
  - Integer latitudes and longitudes were in `__number` there already.
  - Modelled for a release (~121.8M lines, assuming the sample's rate): about 40,000
    cells of about 15 million coordinates. That falls to ~30,000 without the
    coordinate-dense head range.
- **What changes on the inference path** (a misfit part, or a release made before
  claims.avsc): also its integer latitudes and longitudes, where the part's inferred
  record had an integer field.
  - One chunk of each range was run through all three paths: 11 parts, main and
    scholarly.
  - One part had this: p50's chunk 1, main set, put 3 latitudes and 3 longitudes in
    `__integer` under inference, and in `__number` on the build and strict paths. So
    #172's note that "inference does the same, since every real part also has float
    coordinates there" doesn't always hold: a property whose coordinates are all
    integers in the part (one other than P625, say) is enough.
  - None of the 11 parts had a record of nulls.
  - After the rule, all three paths gave the same arrays for all 11 parts. The bench's
    Polars rule agreed on each.

| real part (chunk 1 of the range) | rows | cells moved: build | strict | inference |
|---|---|---|---|---|
| head, main | 10,000 | 15 precisions | 15 | 15 |
| p50, main | 9,152 | 1 precision | 1 | 1 precision, 3 latitudes, 3 longitudes |
| p10, p25, p75, tail (main and scholarly), p50 scholarly | 352 to 9,648 | 0 | 0 | 0 |

## The bench (17.5.0): parity split into the intended and the rest

- **The sample does hold whole-number coordinates,** so parity against the commit
  before shows intended differences:
  - c0: 79 integer precisions of a mainsnak coordinate, 103 integer latitudes, 103
    integer longitudes and 212 null precisions;
  - r50: 1 integer precision;
  - r90: 1 integer latitude and 1 longitude (r90 is in the full tier only);
  - none in a qualifier or reference.

  Each integer latitude and longitude was in `__number` already. So the moved cells are
  **80 precisions** (c0's 79 and r50's 1), from `precision__integer` to
  `precision__number`, in the quick tier's published claims. The full tier has the same
  cells, in each copy of its cycled files.
- **The parity** (`bench/parity.py`):
  - Each run's fingerprint now also holds its claims keys' published and group files
    fingerprinted by the rule, and the cells the rule would move. The rule is made apart
    in Polars, from the statement-level columns.
  - Where the exact comparison fails, parity passes only if three things hold:
    - the runs are the same by the rule: every claims value but those cells, and every
      other table, file, quarantine file and sidecar, compared exactly;
    - the commit after keeps the rule: by it, its files are as they are, and no cell
      moves;
    - nothing outside the claims differs.
  - The report lists the differing keys and the cells under `intended`, and the summary
    says so.
  - After this merges, both arms keep the rule, and parity is the exact comparison again.
- **Checked locally on fake runs:**
  - before against after passes, `intended: 79 mainsnak precisions` (c0);
  - after against after, and before against before, pass exactly;
  - after with one other cell changed (a rank) fails;
  - the rule undone (after against before) fails (`after_keeps_it: false`).
- **The rule's Polars form on multi-file, multi-row-group scans** (the bench's layout,
  streaming engine): the commit before's files by the rule hash the same as the commit
  after's, whose hashes the rule leaves as they are.
  - Polars 2.0 panics on a `when` over struct columns of several chunks in an in-memory
    frame. So the rule uses no struct literals, and runs on the streaming scans the
    fingerprint makes.
- **The misfit stage** (17.3.0) is unchanged and needs nothing new. Its parity compares
  the split's built file with `normalise_claims_direct` under inference. Both now go
  through the rule, and c0 has the moved precisions, so it exercises the rule on the
  fallback path.
- **The CI bench** (#178, head cdbdaf0, quick tier, 3 rounds, bench 17.5.0):
  - **Parity passed by the coordinate rule.** `intended`: `claims/all`, its published and
    group files, `mainsnak precision: an integer` 80 cells before, 0 after. Every other
    one of the 2,842 published keys and 2,324 group keys was the same, as were the
    quarantine and the sidecars, but the claims' `file_size`, which may differ.
  - **The misfit chunk's parity passed.**
  - **No figure changed beyond noise:**
    - speed 0.99× ± 0.02×;
    - CPU, end to end: 2m01s → 2m01s;
    - wall, end to end: 45.13s → 45.30s;
    - wall with finalise: 35.37s → 35.57s;
    - the split's CPU: 34.3s → 34.0s;
    - the pipeline's CPU: 5.1s → 5.1s;
    - Full Wikidata ETA: 6h30m40s → 6h29m09s;
    - release wall, modelled: 7h02m47s → 7h03m10s.
  - As an output correction with no cost to show, it merges on parity and noise, not on
    a speed-up.

## A past release's published data

- **Against a release made with master's code before this change:** the claims differ
  only in those cells:
  - every globe-coordinate's integer precision (`precision__integer` → `precision__number`,
    the same number);
  - in a part that went to the fallback (a misfit, ~1 in 15 chunks before #172, ~0
    after): its integer latitudes and longitudes in `__integer`, and any record of
    nulls at a precision.
- **Every other value, field, type and row order is the same:**
  - shown by the rule's parity on the bench;
  - shown on the 11 real parts by the three-path comparison above.
- **A release made before claims.avsc** (all by inference) differs in the same cells,
  by the same rule. Its other differences come from changes since, not from this one.

## What was weighed

- **Rust and genson, or one Arrow step after them.** Placing the values in the Rust
  builder would need it to know the snak's `datavalue_type` while it decodes the value,
  but `datavalue_type` follows `datavalue` in the dump's member order. It would also
  break the builder's exact equality with genson strict. Changing genson's strict decode
  or inference would need a release. One step on both outputs is identical by
  construction, keeps both decoders' checks exact, and needs neither.
- **The record of nulls → null part of the rule.** It goes slightly past the stated
  rule, which is about whole numbers, but it is the same divergence family (a
  globe-coordinate's number, placed by the part's own types). DIVERGES lists it, and
  the build's null is the canonical form. No real part had one.
- **Not covered:** the philippesaade source (no `WIKIDATA_RELEASE`). It has no
  `datavalue_type`, no claims.avsc and no build, so the rule leaves it alone.

## Missing

- **A whole-dump count.** Six ranges were read; the head range is unusually rich in
  coordinates.
