# Changelog

The project has no version numbers, so changes are listed by date, newest first. Each
entry gives what changed for someone running the pipeline or using the datasets. Commits
and the journal (`docs/journal/`) have the details.

## 2026-10-05

- **Chunk ranges in file names have one width per run.** Group and compacted file names
  (`chunks-{first}-{last}.parquet`) pad both numbers to the digits of the run's last chunk
  index, at least 4: release 20260928's main set, with chunks up to 12181, gets
  `chunks-00000-00149`. Names had been padded to 4 digits, so they grew to 5 past chunk
  9999 and no longer sorted in chunk order, and compaction stopped on the first 5-digit
  name. Compaction and the sort read group files at any width, in chunk-number order.
- `scripts/release_eta.py` covers every step of `just release`, not only processing: it
  marks each step done, in progress (with its progress and ETA) or to do.

## 2026-10-04

- **Chunks are processed in parallel.** `WIKIDATA_WORKERS` chunks (default 6) run at once,
  each in its own process, and a group uploads in a background thread while the next
  chunks run. A group still covers consecutive chunks: it closes only over the partitioned
  chunks below every chunk still running or queued. See [Run loop and state](reference/run.md).
- **Less fixed time per chunk.** Each chunk reads its own state file instead of every
  chunk's (13,000 files, about 0.4 s a read, six reads per chunk).
- `scripts/bench_workers.py` measures throughput by worker count on a release's own chunks.
- `scripts/test_pool.py` tests the worker pool.

## 2026-10-03

- **`just release {release} {previous}`** runs both sets' processing, finalise and
  promotion in order, resumable throughout.
- **Finalise for two sets.** Claims are compacted and sorted first, their refs collected
  and their local copy deleted. claims_labels waits until both sets' labels are sorted, and
  `finalise.done` gates promotion. Scholarly dataset cards were added.
- The entities table keeps a property's `datatype`.
- A map empty in every row of a chunk (inferred with `Null` values) is accepted and cast to
  the stored schema. Requires polars-genson 0.9.5, then 0.9.6, which fixes forced maps
  losing their value type in parallel inference.
- `scripts/release_eta.py`: progress and time left across both sets of a release.

## 2026-10-02

- **The scholarly set.** `route-release` moves each release's scholarly works (43 "instance
  of" classes) into their own set, published as `wikidata-scholar-{table}`.
- `scripts/p31_survey.py`: which classes the philippesaade copy left out.

## 2026-10-01

- **Releases from the official dumps.** `WIKIDATA_RELEASE` selects a dump by date:
  `download-dump`, `split-dump`, per-release working directories, and a build branch per
  repo with `promote-release` to `main`.
- **Every field of the dump is kept.** Snak types and hashes, statement ids, types and
  qualifier order, reference hashes and snak order, sitelink badges, datavalue types, and
  a seventh table, `entities`.
- A release's claims_labels is built at finalise from its sorted claims and labels.
- Dataset cards for releases.
- Requests to dumps.wikimedia.org carry a descriptive User-Agent.
- Outside the pipeline: demos for embedding views, the external-identifier sparse
  autoencoder (`sae/`) and its browser app (`space/`).

## 2026-09-30

- **Sort by id.** Each key's rows are sorted by id across its files, in string order, into
  `part-{i}-of-{n}.parquet`.
- `download-wikidata`: a local copy of the repos.
- **Dataset cards rendered from the data**, with a subset per key, and pushed only when
  changed. Their figures are computed from the local copy and refused when stale.
- Demo scripts in `demos/`.

## 2026-09-29

- **Compaction.** Each key's group files are rewritten into files of about 500 MB on the
  Hub.

## 2026-09-27

- **Quarantine.** Snaks on deleted properties are pruned from claims with polars-genson's
  `prune` (about 3× faster than before) and kept in `quarantine/`.
- A main snak collapsed to a bare property id is handled.
- Downloads retry transient Hub errors for about three hours.
- Parquet outputs are written atomically.
- Dataset cards for the six tables.

## 2026-09-26

- **Grouped push with post-check.** Partitioned chunks are merged and uploaded in groups,
  verified by sha256, and the run continues through every chunk.
- **claims_labels**: the label maps repeated in every claim are moved to their own table,
  split by language.
- **Claims are not split by language**: one row per statement, in one folder `all`.
- Each chunk is processed and partitioned in its own process.
- Each chunk's claims are conformed to one stored claims schema.
- Local files are deleted once processed and partitioned.
- Sitelinks are read as a map whatever the number of sites.
- Every table keeps the entity id, and id counts are checked.

## 2026-09-25

- polars-genson's typed output for claims and the map tables.

## 2026-01-14 to 2026-01-17

- The partition step is part of the pipeline, with state updates for process and partition.
- Schema handling for maps; polymorphic `datavalue` and `precision` fields; integer
  latitudes and longitudes.

## 2025-10-08

- JSON columns are normalised with polars-genson.

## 2025-08-05 to 2025-08-12

- First version: per-chunk state tracking, download with size checks, processing into
  labels, descriptions, aliases, links and claims, partitions with the source file names
  kept, and prefetch of upcoming chunks in a background thread.
