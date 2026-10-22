# 2026-10-21: The chunk read takes only the columns process() uses

Continues docs/journal/2026-10-21-schema-check.md. Scripts and logs are in
`2026-10-21-chunk-read-columns/`.

## Why

`process()` read every column of a chunk into a frame (`pl.read_parquet(pq_path)`), but
uses the frame for only three things: `id` (the id count), `sitelinks`, and in a release
`entity`. genson reads labels, descriptions, aliases and claims from the file itself. So
most of the read was decoded for nothing.

In chunk 0 of release 20261012 (10,000 entities, Q31 to Q83751, remade by
`make_chunk0.py` as `dump.write_chunk` and `dump.route_chunk` make it):

| column | JSON |
|---|---|
| claims | 374.8 MB |
| sitelinks | 28.7 MB |
| labels | 13.2 MB |
| descriptions | 12.6 MB |
| aliases | 4.3 MB |
| entity | 1.1 MB |
| id | 0.1 MB |

The three columns used are 30 MB of the 435 MB.

**Profile:** `process()` at feat/genson-schema-check on polars-genson 1.2.2 and Polars
2.0.0, in an environment from uv.lock with only polars-genson changed. It was timed by
polars-genson's `profile_process.py` (`profile_keep.py` here also keeps the last run's
output). Over warm runs, the chunk read was the second-largest step:

| step | wall | CPU |
|---|---|---|
| claims, genson | 3.07 s | 9.8 s |
| `polars.read_parquet[process]`, the chunk read | 2.90 s (0.48–3.38) | 2.9 s |
| sitelinks | 0.61 s | |

The read added 464 MB to the process (`logs/steps_base_then_projected.txt`).

**The read alone,** in fresh processes (`read_probe.py`; `logs/read_probe.txt`, 8 runs
each):

| read | wall | page faults |
|---|---|---|
| all columns | 0.41–0.54 s | 118k |
| the three columns | 0.05 s | 9.8k |

In a process that has already done other work, the full read took up to 3.4 s.

## Change

`process()` reads `["id", "sitelinks"]`, and `entity` too when `RELEASE` is set.
`entity` exists only in a release's chunks and is used only then.

Separately, pyproject.toml now declares `httpx`, which `pull/download.py` imports. It was
installed only as a dependency of the locked huggingface-hub (1.32.0). An unlocked
install now resolves huggingface-hub 2.2.0, which uses `httpx2`, and `process` failed to
import. uv.lock changes only by the declaration. It was locked with its own options
(`--exclude-newer "3 days" --exclude-newer-package "polars-genson=0 days"`), and
`uv lock --check` passes with them.

## Measured

**Setup** (`ab.sh`; `logs/ab.txt`):
- unchanged `process()` against the three-column read (`logs/projection.diff`);
- 3 processes each, alternating, 4 runs per process on chunk 0;
- run 0 is the first in a fresh process. That is how production runs a chunk:
  `pool.py` spawns a process per chunk.

| | unchanged | three columns |
|---|---|---|
| run 0 wall | 4.68, 5.47, 7.26 s | 3.97, 4.02, 4.31 s |
| run 0 CPU | 13.0–21.4 s | 11.9–12.9 s |
| warm runs wall, median (range) | 6.81 s (4.65–7.64) | 4.26 s (3.88–4.78) |
| process peak RSS | 3,708 MB | 2,164 MB |

- **Outputs:** labels, descriptions, aliases, entities, links, claims and claims_labels
  are byte-identical (`cmp`) between the two, and with the committed code.
- **Committed code:** one more process of 4 runs, with a peak RSS of 2,153 MB.
- **What remains** (three-column read, warm runs):
  - claims take 2.63 s: write 0.98, decode 0.61, infer 0.60, extract 0.28, read 0.10,
    conform 0.05 (`logs/claims_stages_projected.txt`);
  - sitelinks take 0.57 s;
  - the links table's `sink_parquet` takes 0.23 s;
  - labels, descriptions and aliases take 0.20, 0.13 and 0.06 s.

## Checks

- **flake8** (the Justfile's settings) reports E303 in claims_labels.py and E305 in
  dump.py, files the change doesn't touch.
- **ty** reports 120 diagnostics before and after the change.

## Missing

- **Chunks:** only chunk 0 was run, a chunk of unusually large entities.
- **Modes:** only release mode. In the source-repo mode the read takes `id` and
  `sitelinks`, the columns that mode's frame is used for.
- **Throughput:** a sweep over `WIKIDATA_WORKERS` and threads per worker has not been
  run. `pool.py` describes one chunk's work as mostly single-threaded, but the claims
  step now keeps about 3.5 cores busy.
