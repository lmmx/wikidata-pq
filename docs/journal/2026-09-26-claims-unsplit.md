# Claims unsplit: no per-language claims subsets

Follows `2026-09-26-claims-label-maps-and-language-rule.md`, which chose rule E (a claim goes into language L if its property or its entity has a label in L, and a monolingual text claim also into its own language).

## Measurements

Timings are from `scripts/debugging/profile_chunk.py` on the user's 20-core machine, single runs, with rule E partitioning.

- `chunk_1037` (29.7 MB source): 20.2 s total. `partition: claims` took 9.4 s of it, and claims normalisation took 1.1 s.
- `chunk_3` (790 MB source, 10,000 entities): 366 s total, of which:
  - `partition: claims` 270.7 s (74%);
  - claims normalisation (`normalise_from_parquet` + lookup) 27.8 s;
  - the other five tables' partitions 2.7 s combined.
- Rule E turns the 10,000 entities of `chunk_3` into 48,226,043 (claim, language) rows.
- The live run completed chunks 0–6 at 229–370 s per chunk (mean 313 s), which projects to ~27 days for 7,449 chunks.
- With rule E, most claims fall in most language subsets, because core properties such as P31 have labels in almost every language. A language's claims subset is close to the whole claims table, with that language's labels joined in.

## Decision

- Claims are not split by language. Each claim is one row (`id`, `property`, `datavalue`, `datatype`, `rank`, `references`, `qualifiers`), with no language or label columns.
- Labels stay split by language:
  - `claims_labels` holds the property, value and unit labels, as `field, ref, language, label`;
  - `labels` holds the entity's own labels.
- Users join claims to these tables in the languages they want. Rule E, or any other rule, becomes a join done by the user.
- Claims rows get a constant partition key (`UNSPLIT_COL` = `partition`, `UNSPLIT_KEY` = `all`). The key is not written to the files. Claims therefore take the same partition, merge, audit and upload path as the other tables, as one file per group under `all/`.

## After the change

- `chunk_1037` claims partition, on the 4 GB container: 17,134 rows (one per claim, equal to the `claims_base` count), 0.3 MB, 0.4–0.6 s.
- An `explode("claims")` of the processed `chunk_1037` claims (3 MB in memory, 16,283 rows) peaks at 1.5–1.7 GB RSS and takes 1.4 s. This holds eager or lazy, with or without `empty_as_null`, and at 1 or 4 threads.
  - The same step came first in rule E's `prepare_claims`.
  - `chunk_3`'s claims partition was killed at the container's 4 GB limit.

## Missing

- A `chunk_3` timing of the unsplit claims partition on the 20-core machine.
- An explanation of the explode's memory use on the nested claims type.
- A dataset card snippet showing the claims-to-labels join.
