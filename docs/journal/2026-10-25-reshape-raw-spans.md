# 2026-10-25: The split's reshape copies each line's bytes in runs, with no DOM

The bz2 split is the release's front: 2h30m–2h49m on 4 cores against the download's
~1h57m, CPU-bound. Of its CPU on the bench's 77.3 MB slice (1,036.8 MB of JSON), the
compiled reshape (`dump.reshape`, reshape/src/lib.rs) was 5.4 s, about 30 min of the
split's ETA at a release (docs/journal/2026-10-25-release-profile.md).

**Before:** a sonic-rs `Value` DOM per line, then the row re-serialised as orjson writes
it, then each column copied into a `bytes` for Python.

## What differs between a dump line and its row

Measured with orjson re-dumping each string and number token of the line:

| lines | lines already in orjson's form | bytes in tokens orjson rewrites |
|---|---|---|
| the dump's first 2,000 entities (143 MB, real lines) | 0% | 13.3% |
| 2,214 lines of the bench slice (66.6 MB) | 100% | 0% |

- **Real dump lines:** every line differs, but only in two places.
  - **Strings with escapes:** the dumps write every non-ASCII character as `\uXXXX` and
    `/` as `\/`, and orjson writes UTF-8 and plain `/`. These were 345,136 strings: 1.73M
    BMP escapes, 3,920 surrogate halves and 494,631 `\/`.
  - **Floats with an exponent:** 502 of 463,017 numbers (e.g. `2.77778e-5` →
    `0.0000277778`).
  - **Verbatim:** everything else, about 87% of the bytes (keys, structure, unescaped
    strings, integers, literals).
- **The bench slice:** bench/dumpslice.py writes its lines with orjson, so they are
  already in orjson's form. **The bench overstates this change's gain against a release**
  (see the ratios below).
- **The reshape's changes of shape add rewrites:**
  - terms → their values;
  - a snak's datavalue moved to the snak's end as `datavalue`/`datavalue_type`;
  - `"snaks":{}` added to a reference without snaks;
  - `[]` maps → `{}`.
- **Nothing in the split re-parses the rows:**
  - the chunk write compresses the strings as they are;
  - `is_scholarly` is a regex over the claims text (0.48 s a slice).

## The change

- **One scan per line:** reshape/src/lib.rs scans each line once and validates it as
  orjson.loads does. It needs no DOM and no sonic-rs.
  - **The run model:** the row so far is `out` followed by the line's bytes from `run` to
    the scan's position. A writer flushes the run only where the row departs from the
    line. Whole sitelinks objects, and statements outside their snaks, are copied in
    long runs.
  - **Rewritten spans:**
    - a string holding a `\` is transcoded (orjson's escapes);
    - a number other than an integer of at most 18 digits goes through the earlier
      `write_number` (ryu, `-0` → `0`).
  - **A datavalue's value and type** are kept as spans of the line, or in scratch only
    if rewritten.
  - **String ends:** found with an inline SSE2 probe of the first 64 bytes, then
    memchr2.
- **Zero-copy hand-off:** the column buffers go to Python through the buffer protocol (a
  `Buffer` pyclass) rather than copied into `bytes`. pyarrow and Polars keep them alive,
  zero-copy. Their unused room is let go first (`shrink_to_fit`).
- **Left to `dump.entity_row`:** a line the scan does not expect, which entity_row then
  reshapes or rejects:
  - a byte below 0x20, i.e. whitespace between tokens or a raw control character (the
    dumps have neither);
  - invalid UTF-8;
  - an escape in a key, the id or a term's language;
  - nesting past 128 levels;
  - a repeated key in any object;
  - JSON grammar errors;
  - the wrong types entity_row would reject or reshape in place.
- **Repeated keys:** orjson keeps the last value, in the first key's place.
  - **Before, wrong output:** the DOM reshape wrote both. Of 20,000 seeded mutant lines
    that entity_row accepts, it wrote 317 differently.
  - **Now detected:** each key gets a fingerprint (length and first 8 bytes). An object
    with up to 8 keys compares its keys linearly; a larger object sorts its keys once it
    closes.
  - **Cost:** about 0.2 CPU-ms per MB, ~12% of the new reshape, for exactness on input
    Wikibase can't write.

## Correctness

- **`scripts/test_reshape.py` passes.** It is extended with:
  - 104 scan edges: whitespace, escapes in keys, ids and languages, every escape orjson
    refuses, raw control characters and invalid UTF-8, the number grammar and 64-bit
    edges, literals, trailing content, repeated keys at every level, nesting at 120, 200
    and 300, a datavalue first, between, last, alone and not an object, and every
    qualifier, reference, term, alias and sitelink shape;
  - 20,000 seeded mutants of the edges: 2,987 taken by the fast path, 17,013 left to
    entity_row, entity_row raising on 16,142;
  - each file given, also re-escaped as the dumps escape;
  - the threaded path, against one thread.
- **Run on:**
  - the dump's first 2,407 entities (150 MB);
  - 8,502 entities from the bench slice's middle;
  - each also in the dump's escaping.

  0 rows differ, and 0 lines are left to entity_row.
- **A local split** of those 10,909 entities (2 chunks, routed) with master's crate and
  with this one: every chunk file is **byte-identical**, the manifests are equal, and
  every row equals entity_row's.

## Measured locally

Interleaved, fresh processes, 4–5 rounds each, so indicative only. CPU-ms per MB of JSON:

| lines | master | this | ratio |
|---|---|---|---|
| bench slice (orjson's form) | 4.74–5.85 (median 4.91) | 1.98–2.05 (2.00) | 0.41 |
| bench slice, escaped as the dumps escape | 4.59–5.07 (4.72) | 2.40–2.58 (2.49) | 0.53 |
| the dump's first entities (real lines) | 4.55–4.96 (4.80) | 2.39–2.62 (2.48) | 0.52 |

**At the bench slice's 1,036.8 MB:** 5.1 → 2.1 CPU-s. The master figure matches the
release profile's 5.4 s.

**At a release:** real lines run at 0.52 of master's CPU. The reshape's ~30 min of the
split's ETA would become about 15–16 min, and the split ~2h15m–2h34m. The bench, at
0.41, will show more.

### The split of the reshape's CPU between the claims and the rest of the row

Measured by reshaping each entity's claims alone (with its type and id), and the rest
alone. Medians, CPU-ms per 100 MB cut:

| lines | crate | claims | the rest | claims' share |
|---|---|---|---|---|
| bench slice (orjson's form; claims 84% of the bytes) | master | 424 | 80 | 84% |
| | this | 163 | 37 | 81% |
| the dump's first entities (dump escaping; claims 77% of the bytes) | master | ~597 | ~185 | 76% |
| | this | 257 | 140 | 65% |

**At the bench slice's scale:** the claims are about 1.7 of this reshape's 2.1 CPU-s.
The claims writer here is only the general run model plus the snak's datavalue move.
Nothing is specialised to the claims (see polars-genson#319: the claims may be built
straight into Arrow later).

## Rejected variants

All A/B locally against the version before each change.

| variant | result |
|---|---|
| a scalar look at a string's first 16 bytes before memchr2 | slower: 2.56 against 2.25 CPU-s a slice |
| in `transcode`, no memchr when the next byte is already `\` | within noise (445–456 against 429–480 MB/s) |
| `#[inline(always)]` on the key bookkeeping | no gain (1.98–2.23 against 1.94–2.16 CPU-ms/MB) |
| repeated keys compared as slices (no fingerprints) | +28% CPU, against +12% fingerprinted |
| `is_scholarly` from the scan's parse | not built, see below |

**`is_scholarly` from the scan's parse:**
- **What it could save:** at most 0.48 s a slice (~2% of the split's CPU).
- **Why it wasn't built:**
  - the routing is a regex over the claims text, whose results on adversarial text
    (e.g. a string holding `"mainsnak":{`) a structural test would not match exactly;
  - routing feeds the bench's parity.

The PyBytes copy's cost was also measured, on master's DOM path: handing the buffers over
without a copy took it from 5.39 to 4.25–4.59 CPU-s a slice.

## Missing

The CI bench's before → after for the split (ETA, CPU, wall, decoding alone, peak RSS)
is on the PR.
