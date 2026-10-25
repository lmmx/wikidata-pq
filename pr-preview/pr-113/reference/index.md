# Reference

One page per part of the package (`src/wikidata/`), in the order the data passes through them. Each page says what the code does, the decisions behind it, and the files it reads and writes. Each ends with the files it describes and their checksums at the time of writing (see [About these docs](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/about/index.md)).

## Code map

| Module                                                 | Page                                                                                                      | Role                                              |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `config.py`                                            | [Configuration](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/config/index.md)           | environment variables, paths, tables, constants   |
| `dump.py`, `scholarly.py`                              | [Dump and routing](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/dump/index.md)          | download, split and route an official dump        |
| `main.py` (`run`), `pool.py`, `state.py`, `initial.py` | [Run loop and state](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/run/index.md)         | the chunk loop, worker processes, per-chunk state |
| `process.py`                                           | [Processing](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/process/index.md)             | a chunk's JSON to seven typed tables              |
| `partitioning/`                                        | [Partitioning](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/partitioning/index.md)      | each table split by language, with audit sidecars |
| `push/`                                                | [Push](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/push/index.md)                      | groups of chunks merged, uploaded and verified    |
| `main.py` (`finalise`)                                 | [Finalise](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/finalise/index.md)              | the order of the finalise stages                  |
| `compact.py`                                           | [Compaction](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/compact/index.md)             | group files rewritten into ~500 MB files          |
| `sort_by_id.py`                                        | [Sort by id](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/sort/index.md)                | each key's rows sorted by id across its files     |
| `claims_labels.py`                                     | [claims_labels](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/claims-labels/index.md)    | a release's labels of what its claims refer to    |
| `cards.py`, `card_stats.py`                            | [Dataset cards](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/cards/index.md)            | README.md of each repo, from the data             |
| `hub.py`                                               | [Hub branches and promotion](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/hub/index.md) | the build branch, and promotion to `main`         |
| `pull/`                                                | [Pull](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/pull/index.md)                      | downloading the philippesaade copy's chunks       |
| `scripts/`                                             | [Scripts](https://lmmx.github.io/wikidata-pq/pr-preview/pr-113/reference/scripts/index.md)                | tools outside the package                         |

## Data flow of a release

```
flowchart TD
  dump["dump/wikidata-R-all.json.bz2"] -->|split-dump| chunks["data/chunk_N.parquet<br/>(10,000 entities, JSON columns)"]
  chunks -->|route-release| main["releases/R/data"]
  chunks -->|route-release| sch["releases/R-scholar/data"]
  main --> proc
  sch --> proc
  proc["process: results/{table}/chunk_N.parquet"] --> part["partition: results/{table}/{key}/chunk_N.parquet<br/>audit/{table}/chunk_N.parquet"]
  part --> merge["merge: staging/{table}/{key}/chunks-A-B.parquet"]
  merge -->|upload, verify| branch["Hub: build-R branch<br/>{key}/chunks-A-B.parquet"]
  branch -->|compact| compacted["{key}/chunks-A-B.parquet, ~500 MB"]
  compacted -->|sort| sorted["{key}/part-i-of-n.parquet"]
  sorted -->|promote| mainbranch["Hub: main, tag R"]
```

## Conventions throughout the code

- **Resumable stages with ledgers.** Each multi-step operation appends a line per finished stage to a JSONL ledger in `state/` (chunks, groups, compaction, sort, claims_labels). On a restart it continues after the last recorded stage. Each stage can be repeated without harm, so a crash between doing the work and recording it is safe.
- **Atomic writes.** Files are written to a `.tmp` path and renamed into place, so a file at its final path is always complete.
- **Halt rather than lose data.** Row counts, id counts, schemas, sizes and hashes are checked at each step, and any difference stops the run.
- **Delete what the next step no longer needs** (`CLEAN_UP_LOCAL`), so disk use stays bounded by a few chunks and one group.
- **One code path for two sources.** `WIKIDATA_RELEASE` switches between an official dump and the philippesaade copy. The differences are confined to the config, the pull step, the schemas and the finalise order.
