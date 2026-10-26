# 2026-10-26: claims.avsc completed against claims_schema

`src/wikidata/schemas/claims.avsc` was inferred from the bench's 11 chunks. A release's
chunk whose claims hold a member it lacks is a misfit: the split's raw path gives up on
the chunk, the text path writes that part's claims as JSON, and the pipeline decodes them
with genson's inference (the fallback). The bench's chunks never misfit (the schema came
from them), so the cost was never measured. This change adds to claims.avsc every field
of the published schema (`claims_schema`) it lacked, but two, and adds a bench stage
(bench 17.3.0, `bench/misfit.py`) that runs a misfit chunk.

## Current State

- **Done:**
  - `claims.avsc` completed (`scripts/avro_schemas.py --complete`): 11 fields added, all
    nullable, none changed;
  - `scripts/test_claims_schema.py`: the completion, and each misfit shape's build
    against the fallback's arrays;
  - bench 17.3.0: the misfit chunk stage, its figures and its parity.
- **The hypothesis holds.** About 1 chunk in 15 misfits on a real dump: 10 of 153 chunks
  (6.5%; Wilson 95% interval 3.6% to 11.6%), from six places in the dump. Every misfit
  was the same shape: **a reference snak's globe-coordinate.** With the completed schema,
  none of the 153 misfits, and every output is the same.
- **The cost is real, but not large.** Measured locally:
  - a misfit chunk costs 1.1 to 7.9 CPU-s in the split and the process step. Completed, it
    costs 0.6 to 2.9 CPU-s: 45% to 64% less (0.5 to 5.1 CPU-s a chunk);
  - modelled at a release: about 790 misfit chunks, so about 2,000 CPU-s (33 CPU-min)
    less, between about 1,100 and 3,600 by the rate's interval.

  On 4 cores that is about 8 minutes of CPU-bound wall. The pipeline overlaps the split
  at a release, so less of it reaches the release wall.
- **One known divergence, now at references too** (see "What was weighed"). No real-dump
  part hit it.

## The misfits

**What the 7% figure rested on:** the split profile's real-dump check
(`2026-10-25-split-profile.md`, #163). It read the dump's first 400 MB: 15 chunks, 1 of
which misfit (Q618385, on a line of 142,282). At that line rate, a chunk of 10,000 lines
misfits with p ≈ 7%. That was one sample from the dump's start.

**What was read here:** byte ranges of `latest-all.json.bz2` (wikidata-20261020, 103.5 GB)
from dumps.wikimedia.org, not the Hub. Each range was resynced to its bz2 blocks
(`bzip2recover`) and streamed through lbzip2. Only the misfit chunks' lines were kept
(zstd), then deleted. The lines were cut into the split's chunks of 10,000. Each chunk went
through `dump.write_chunk` with master's claims.avsc and with the completed one, and each
misfit line was found and classified.

| range (offset in the dump) | chunks | lines | misfit chunks | misfit lines |
|---|---|---|---|---|
| head (0, 300 MB) | 10 | 94,927 | 1 | 1 (Q618385) |
| p10 (10%, 200 MB) | 29 | 282,307 | 0 | 0 |
| p25 (25%, 200 MB) | 22 | 218,632 | 2 | 3 (Q140626491; Q222, Q16641) |
| p50 (50%, 200 MB) | 22 | 213,900 | 3 | 4 (Q140352773; Q140455088; Q141758228, Q4098) |
| p75 (75%, 200 MB) | 14 | 131,132 | 0 | 0 |
| tail (from 99.8%, 250 MB) | 56 | 552,142 | 4 | 5 (Q139040617; Q139323461; Q139824733; Q140316313, Q140317847) |
| **all** | **153** | **1,493,040** | **10 (6.5%)** | **13 (1 in 115k lines)** |

The p25, p75 and tail rows come from the same harness run earlier on this branch (the
WIP's `results.jsonl`). The head and p50 rows were run again for this entry, and agree; p10 is new.

- **Every misfit is the split's own "a field claims.avsc lacks"**, on both paths (the raw
  path and the text path). Nothing else misfit in 1.49 million lines: no null where
  claims.avsc has none, no number shape, no pruned field.
- **Every misfit line inspected** (Q618385, Q140352773, Q140455088, Q141758228, Q4098)
  holds a reference snak whose value is a globe-coordinate:
  - P625 in a reference of P276 or P159, say;
  - its members: `latitude`, `longitude` (floats), `altitude` (null), `precision`
    (float), `globe`;
  - the bench's chunks had no reference coordinate, so claims.avsc's reference datavalue
    lacked those five members.

  The other 8 misfit lines are built by the completed schema, with output identical to
  the fallback's, but their shape wasn't printed.
- **Integer coordinates are no misfit, and stay so.** A mainsnak's integer latitude or
  longitude (52, not 52.0) is common: 34 of 3,871 coordinates in head's chunk 7, and 93 of
  4,437 in p50's chunk 18 (1-2%). Genson's strict decode gives each to `latitude__number`,
  since its record has no `__integer` field. Inference does the same, since every real
  part also has float coordinates there.
- **The scholarly parts never misfit** in this sample. All 10 misfit parts were main-set
  parts.

**The published schema's fields claims.avsc lacked** (`scripts/test_claims_schema.py`
lists them, `ADDED`):

- `mainsnak.hash`: never in a dump (Wikibase writes no mainsnak hash). It is in
  `claims_schema`.
- **A reference snak's members:**
  - `hash`: never in a dump either;
  - `datavalue.precision.precision__number`;
  - `upperBound`, `lowerBound`: a quantity with bounds;
  - `latitude`, `longitude`, `altitude`, `globe`: a globe-coordinate (the real misfits).

## The schema change

- **`scripts/avro_schemas.py`:** `complete` (from the WIP commit 0075984, reviewed here)
  walks `claims_schema` against the stored Avro schema. It adds each missing field as
  nullable at every level, with records named as genson names them. A field is never
  changed or reordered, so whatever fitted before decodes exactly as before.
  `--complete` rewrites claims.avsc in place, and an inference run (`main`) completes
  what it infers.
- **Left out on purpose:** `latitude__integer` and `longitude__integer`, at each of the
  three snak places. Genson's strict decode gives an integer to `__number` only where
  its record has no integer field. With an `__integer` field there, every integer
  coordinate above (1-2% of a chunk's coordinates) would move to `__integer`. But
  inference puts them in `__number` in every real part, so the published output would
  change for nearly every chunk.
- **claims.avsc is its own completion** (the test checks it). Every field of
  `claims_schema` is in it but those 6.

## Proof that the output is unchanged

- **On the real dump** (`avsc-tools/sample2.py`, not committed: scratch). For each of the
  four misfit chunks rerun here (head 7; p50 2, 3 and 18), and the WIP run's ten,
  master's fallback was run on the part's JSON claims. That is
  `process.normalise_claims_direct`, genson against master's claims.avsc, which falls
  back to inference.
  - **Against the completed split's built claims file:** the same rows, order, schema
    (with its metadata) and row groups' rows. The refs sidecars are the same too.
  - **The same part through genson strict against the completed claims.avsc:** the
    same.
  - **The misfit part's chunk file**, but its claims column, is the same.
  - **Every other file of the chunk** (the scholarly part, terms, entities and their
    indexes) is the same, byte for byte.
  - **The 57 fitting chunks** of the three rerun ranges (head, p10, p50) write every file byte for byte
    as master does.
- **`scripts/test_claims_schema.py`:** 8 misfit shapes, each in a part with a release's
  context (a reference's stated-in item and retrieved time; a mainsnak's float
  coordinate). They are:
  - a reference's coordinate, as on the dump: alone, beside a time, with a null
    precision, with an integer precision, and an integer latitude beside a float one;
  - a reference's quantity with bounds;
  - a reference snak's hash;
  - a mainsnak's hash.

  Each case must pass three checks:
  - it misfits the schema before the completion (the `ADDED` fields removed);
  - the split's build equals genson strict against claims.avsc;
  - the build equals genson's inference, the fallback: Arrow schema and values.
- **`scripts/test_claims_build.py`** (edges, misfits, r50) and
  **`scripts/test_split_claims.py`** (3,000 lines each of c0 and s30, plus the misfit
  entity) pass. The latter's misfit entity is now a null snaktype, since the mainsnak
  hash it used fits.
  - **On the 0075984 run's failure in `after_numbered`:** that run gave it whole chunk
    files, c0's 10,000 lines first. The first two chunks of 2,000 then held no scholarly
    entity, so that set had no manifest yet, which the check reads. That is the input,
    not the change. With 3,000 lines a file, as the test is meant to be run, it passes.

## The cost of a misfit chunk

**Measured locally** (`avsc-tools/time2.py`, scratch; the shared 4-core VM, so
indicative). The setup:
- the split's work on the chunk (`dump.write_chunk`) and the process step on its misfit
  part (`process.process`);
- CPU of the process (its threads with it), the median of 4 alternating runs per arm.

| chunk | JSON | master: split + process | completed: split + process | saved |
|---|---|---|---|---|
| head 7 | 254 MB | 1.96 + 5.98 = 7.93 CPU-s | 2.87 + 0.01 = 2.89 | **5.05 CPU-s (64%)**, wall 3.94 → 1.94 s |
| p50 18 | 251 MB | 1.89 + 4.94 = 6.84 | 2.80 + 0.02 = 2.82 | **4.02 (59%)**, wall 3.77 → 1.89 s |
| p50 2 | 48 MB | 0.46 + 0.77 = 1.22 | 0.61 + 0.02 = 0.63 | **0.60 (49%)** |
| p50 3 | 43 MB | 0.54 + 0.60 = 1.14 | 0.60 + 0.02 = 0.62 | **0.52 (45%)** |
| bench stage: c0 + a reference's coordinate | 3.5 + 14.8 = 18.3 | 5.9 + 0.0 = 5.9 | **12.4 (68%)**, wall 7.9 → 3.9 s |

Where the time goes:
- **The process step does the work.** On master it decodes the part's JSON with genson,
  a strict attempt and then inference. Completed, it places the split's file.
- **The split's share goes up.** On master, the split leaves the misfit part's claims
  unbuilt and writes them as JSON. Completed, it builds them: 0.1 to 2.4 CPU-s more.
- **Net,** about 0.017 CPU-s a MB of the chunk's JSON.

**Modelled for a release** (assumes the sample's rate holds across the dump, and the mean
measured saving per misfit chunk):
- 6.5% × ~12,182 split chunks ≈ **790 misfit chunks** (440 to 1,410 by the interval);
- × 2.55 CPU-s (the four real chunks' mean saving) ≈ **2,000 CPU-s ≈ 34 CPU-min**
  (1,100 to 3,600). By bytes it comes out the same: 0.017 CPU-s/MB × 144 MB (the sampled
  chunks' mean) × 790 ≈ 1,950 CPU-s.
- **Wall:** at most ~8 min on 4 cores. The pipeline's share runs beside the split at a
  release, at nice 10 (`just overlap-release`), so how much reaches the release wall
  depends on the critical path. The bench's `release_wall_modelled` assumes every chunk
  fits, so it doesn't include this.
- **What the model leaves out:**
  - **Correlated sample:** the sampled chunks are contiguous runs, not independent
    draws, so the interval is optimistic;
  - **Rate along the dump:** the misfit rate may vary along it (the tail, the newest
    items, had 4 of 56);
  - **Size skew:** misfit chunks may be heavier than the mean.

## The bench (17.3.0)

- **The sample has no misfit.** The quick and full tiers' chunks all fit claims.avsc, by
  construction, so no existing figure can move with this change.
- **`misfit.py`:**
  - turns c0 back into dump lines (`dumpslice.py`);
  - gives its first entity with references (Q31) a reference globe-coordinate: Q618385's
    own P625 from the dump;
  - runs `write_chunk` and the misfit part's `process` in one process.
- **Figures:** `misfit_split_cpu`, `misfit_process_cpu`, `misfit_chunk_cpu`, their walls,
  and `misfit_claims_json`.
- **Parity:** the part's claims file must be what genson's inference makes of it. `ab.py`
  exits 1 if not.
- **Run after the overlap and link stages,** outside every other figure. It adds about 8
  s an arm-run on master and 4 s completed, plus the parity's inference (~15 CPU-s) on a
  built arm. That is minor (CHANGELOG: no figure changes meaning).

## What was weighed

- **The fallback is inference, and inference is data-dependent.** Its types are the
  part's own, so "the fallback's output" holds only for the part as it is. claims.avsc
  is fixed. The two part only where a part shows a place with fewer variants than a real
  chunk does (the test's `DIVERGES`, each also asserted for a mainsnak on master):
  - **an integer latitude or longitude where the part has no float one:** inference
    gives `__integer`, claims.avsc `__number`;
  - **a null precision where every snak there has a precision:** inference gives a
    record of nulls, claims.avsc a null.

  For a mainsnak or qualifier, this has been the rule for every fitting chunk since
  claims.avsc was introduced, and no real chunk is that thin there. For a reference it is
  new. A misfit part has one to three reference coordinates, and 1-2% of coordinates are
  integer-valued. So a few of a release's ~790 misfit parts (very roughly 10) could hold
  a reference coordinate's integer in `latitude__number` (47.0), where the fallback put it
  in `latitude__integer` (47). That is:
  - the same number;
  - the same schema;
  - where every mainsnak's integer coordinate already is.

  None of the 10 real misfit parts did (every reference coordinate there was a float).
  Matching inference here would need the reshape and genson's strict decode to treat an
  integer at a float-only field as a misfit when the part has no float there, which a
  fixed schema can't express.
- **Adding `__integer` to the reference record instead** would match a lone integer
  reference coordinate, but break a part with an integer beside a float one (inference:
  `__number`). It would also make the reference record unlike the mainsnak's.
- **Nullability of the non-null fields** (snaktype, property, datatype, a reference's
  hash, snaks-order): left as inferred. Nothing in 1.49 million lines misfit on them, and
  `test_split_claims.py` uses a null snaktype as its misfit.
- **Memory and disk:** unchanged in kind. A misfit part's claims are built in the split's
  batches (`claims_build.BUDGET`) like every other part's. Its chunk file no longer
  carries a JSON claims column.

## Missing

- **The other 8 misfit lines' shapes** (built and identical, but not printed).
- **A rate from the whole dump,** rather than six ranges.
- **CI's figures:** the bench's before → after for the misfit stage, on this PR.
