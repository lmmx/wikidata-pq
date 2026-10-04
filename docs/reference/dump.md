# Dump and routing

`dump.py` turns an official dump into the chunk files the pipeline processes;
`scholarly.py` decides which entities go into the scholarly set. Download, split and route
run on the main set's directory only, and refuse `WIKIDATA_SCHOLAR`.

## Download

`latest_release()` reads the directory listing of dumps.wikimedia.org and returns the
newest date whose directory holds `wikidata-{date}-all.json.bz2` (the newest directories
can lack it while the dump is still being written).

`download()` fetches the release's md5 sums, then the bz2 into `DUMP_DIR` with HTTP range
requests. It appends to a `.part` file and resumes from its size after a network error. The
finished file is checked against its md5 before being renamed into place. Requests carry a
descriptive User-Agent (`USER_AGENT`): Wikimedia answers 403 to urllib's default.

## Split

`split()` streams the dump through `lbzip2 -dc` (all cores), groups the entity lines into
batches of `CHUNK_ENTITIES` (10,000, the size of the philippesaade files), and hands each
batch to a process pool that writes `chunk_{N}.parquet` (zstd level 9, about the size of the
bz2). At most `MAX_PENDING_BYTES` (4 GiB) of decompressed lines wait for the workers, which
bounds memory. Each finished chunk appends its manifest line; a rerun skips the chunks
already in the manifest, and `split.done` marks the end.

Each entity becomes a row with the columns the process step reads (`entity_row`):

| Column | Content |
|---|---|
| `id` | the entity id |
| `labels`, `descriptions` | JSON `{lang: value}` (the dump's `{lang: {language, value}}`) |
| `aliases` | JSON `{lang: [value, ...]}` |
| `sitelinks` | JSON, as in the dump (`site`, `title`, `badges`) |
| `claims` | JSON, each snak's `datavalue` replaced by its `value`, with the value's type as `datavalue_type` |
| `entity` | JSON of the entity's own fields: `type`, `datatype` (properties), `ns`, `title`, `pageid`, `lastrevid`, `modified` |

Everything else in the dump is kept as it is: snak types and hashes, statement ids and
types, `qualifiers-order`, references with their hash and `snaks-order`, sitelink badges,
and item values' `entity-type` and `numeric-id`. The term reshaping relies on each term's
`language` repeating its key; an entry where it does not is counted in the manifest's
`mismatched_terms`. The dump writes an empty map as `[]`, which `_map` reads as `{}`.

The manifest also lists the field names found in each chunk's entities and sitelinks.
`split_manifest()`, which every later step reads, halts if any of them is missing from
`EXPECTED_FIELDS`. A field there that `process.py` does not read would otherwise be dropped
without notice.

## Scholarly works

An entity is scholarly if any of its "instance of" (P31) main-snak values is in
`SCHOLARLY_CLASSES`: 43 classes. 37 of them are classes that philippesaade's copy of
2026-05-07 had left out almost entirely (at least 99% of their entities missing), such as
scholarly article, doctoral thesis, erratum and preprint. The other 6 were partly missing and
are scholarly works too: scientific publication, geological map, scholarly conference
abstract, technical report, book review and scholarly chapter. `scripts/p31_survey.py`
produced the figures, and the journal entry of 2026-10-01 records them.

`is_scholarly` finds the P31 values with a regular expression on the claims JSON
(`P31_SNAK`) rather than decoding it, which keeps routing fast. The expression is anchored
on `"mainsnak"`, so a qualifier's P31 does not count, and allows no `{` before the
`datavalue`, so the match stays inside one main snak. This depends on the split writing
`datavalue` after the snak's other keys.

## Route

`route()` runs 8 workers (`ROUTE_WORKERS`), each reading one split chunk and writing its
scholarly and other rows to `routed/chunk_{N}.parquet` in the scholarly and main data
directories. An empty part is not written. A chunk's result is appended to `route.jsonl`
(and fsynced) before the split chunk is deleted, so an interrupted route resumes from the
log. The split chunks are gone once routed, and a rerun after `route.done` does nothing, so
routing with other `SCHOLARLY_CLASSES` means splitting the dump again into empty data
directories.

Once every chunk is routed, each set's non-empty parts are renumbered from 0 in split
order: groups are ranges of consecutive chunk numbers, so the numbers must have no gaps.
The split's manifest is kept as `manifest.split.jsonl`, each set gets a manifest of its
own parts (with `split_chunk`, the number it came from), the parts move to their new
names, and both sets get `split.done` and `route.done`. `split_manifest()` refuses to run
without `route.done`.

## The pull step for a release

A release's chunks are already local, so its pull step is `check_chunk()`: the chunk file
must exist with the manifest's size, and the chunk's state moves to `PULL`.
`chunk_sizes()` gives the manifest's sizes to the group sizing.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/dump.py` | `e80bbf872bee70d05bea89eecff0cd99c83bb6f9ce97da63a7cae9b0810d4467` |
    | `src/wikidata/scholarly.py` | `29a2d326b616c6334b65b0121ca8299964a3b1de057bf4ee71e2bb51c1aef9f8` |
