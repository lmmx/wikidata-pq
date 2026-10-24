# 2026-10-24: What each step costs on CI's runners, and sitelinks against a known schema

Continues docs/journal/2026-10-24-sort-writer.md.

## Current State

- Each chunk's sitelinks are normalised against a fixed Avro schema (`SITELINKS_AVRO`,
  a map of nullable `SITELINK_FIELDS` records), passed as polars-genson's
  `normalise_json(avro_schema=...)` (lmmx/polars-genson#304), not one inferred from the
  chunk. The decode to `SITELINK_SCHEMA` is unchanged.
- The bucket path's scratch files are written by `_write_file_parallel(scratch=True)`
  (zstd 1, no content-defined chunking) (#53). #45 had passed `scratch` to a writer
  without it, so a key sorted through buckets (claims at full size) raised TypeError.
  The bench never takes that path; `scripts/test_sort.py` does, and passes again.

## Why

This sandbox overstates memory-heavy steps (its first touch of fresh memory is slow),
so step costs were measured on CI's runners. #49 timed the sort's phases; #55 timed
each chunk's process and partition steps; neither is to merge.

**The pipeline's critical path is chunk 0** (bench 3.1.1, quick tier). Chunks 1–3 end
by about 3.6 s, chunk 0 at 7.7 s, and then the group's close (2.5 s of merging, 3.0 s
in all) runs alone. Chunk 0's steps, wall/CPU, mean of 6 runs:

| step | wall | CPU |
|---|---|---|
| sitelinks | 3.43 s | 3.19 s |
| claims (genson, layout, write) | 2.64 s | 8.81 s |
| claims, genson's call | 1.72 s | 5.82 s |
| partition_parquet | 1.06 s | 2.46 s |
| genson, the other maps | 1.04 s | 0.79 s |

Sitelinks ran while all four workers were busy, so its CPU was its wall time.
Standalone, 2.57 of its 3.17 CPU-s were schema inference, for a column whose decoded
dtype is fixed. Across the bench's 11 chunk files, sitelinks infer four different
schemas: nullable fields (c0), plain strings (r*), all-empty `badges` as `array<null>`
(most s*), and all-empty maps as `map<null>` values (s70). The widest of them, with
`empty_as_null`, gives the same decoded rows on every chunk (frames and schemas equal,
all 11).

**The sort's claims key on CI** (#49): read 0.8–1.0 s, take 0.5–0.6 s, write 2.1–2.7
s, exact-order check 4.3–5.3 s. The check is about 35% of the sort and 17% of the run
with finalise. It already uses 3.6 cores of 4 (locally), so overlapping its source
side with the sort would gain little. It could only be made cheaper by checking less
(hashing the sort's own read of the sources, or sampling), which is a decision about
the check's independence, not made here.

## Measured, and not kept

- **Run slices for the sort** (#54, closed): the claims key's 1,175,112 rows sort into
  1,939 runs of consecutive rows (an entity's statements stay together), so the
  sorted table can be zero-copy slices, each row group's combined once. Locally the
  take went from 8.7–10.2 s to 0.14 s (2.8 s with the combine), but on CI the sort
  went 14.46 s → 14.24 s (≈): CI's take was only 0.6 s.
- **Polars in place of pyarrow for `_read`**: the same schema and values, but in fresh
  processes the two read the claims key equally fast (2.6–3.0 s), Polars 0.8 GB higher
  at its peak.
- **The group merge's per-key overhead**: one aggregation in place of a data frame a
  language changed nothing measurable (1.59–1.90 s against 1.55–1.93 s). Single
  threaded, the merge is the glue (0.57 ms a key) and `mkdir` of each language's
  staging directory (0.37 ms a key): filesystem work.

## Correctness

- `normalise_json(avro_schema=...)`: lmmx/polars-genson#304's tests compare it with
  the inferred route on every decode route; the full Python suite passes.
- Sitelinks against `SITELINKS_AVRO`: equal frames on all 11 chunk files; the bench's
  parity covers the links table's 827 keys.

## Missing

- The bench figure for sitelinks against `SITELINKS_AVRO`.
- The other maps (labels, descriptions, aliases) infer a schema and check it against
  their stored one: that check halts a run whose data has changed shape. Skipping their
  inference would skip the check, so they are left as they are.
