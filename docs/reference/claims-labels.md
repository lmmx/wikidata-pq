# claims_labels

claims_labels holds the labels of everything a statement refers to, so that the claims
table can hold only ids. Users join it to the claims in the languages they want. Each row
is `field, ref, language, label`:

| `field` | `ref` | Labels of |
|---|---|---|
| `property-labels` | a property id | the snak's property |
| `labels` | an item or property id | the snak's entity value |
| `unit-labels` | the unit's URI (`http://www.wikidata.org/entity/Q…`) | a quantity's unit |

The table is split by language and sorted by `ref`.

## From the philippesaade copy

That copy embedded these labels in every snak. [Processing](process.md#claims) moves them
out chunk by chunk, and the groups and compaction deduplicate them.

## For a release

An official dump's claims carry only ids, so `claims_labels.py` builds the table at
finalise, from the set's sorted claims and labels. Its stages are recorded in
`state/claims_labels_build.jsonl`:

1. **`refs`** (`collect_refs_stage`), run right after the claims are sorted: every distinct
   `(field, ref, id)` referenced by a snak in the main snaks, qualifiers and references of
   the local copy of the claims, written to `compact/claims_labels_build/refs.parquet`.
   Each claims file is a job, `FINALISE_WORKERS` at once, its refs written to
   `compact/claims_labels_build/refs/` as it finishes and kept on a restart.
   `id` is the entity whose labels name the ref; for a unit, the URI with its prefix
   removed. Units that are not entity URIs (the quantity `1`) are left out. Once the refs
   are collected, the claims' local copy can be deleted.
2. **`written`** (`build_claims_labels`), once both sets' labels are sorted: for each
   language, the refs joined to that language's labels from **both sets' local copies**
   (`labels_dirs`). Each language's rows are written as one group,
   `{lang}/chunks-0-{last chunk}.parquet`, padded as group names are (`chunk_range_name`).
   Each language is a job, `FINALISE_WORKERS` at once; a language whose file is written
   is kept on a restart.
3. **`staged`**: the repo and its build branch created, and the group files moved into
   compaction's source directory, `compact/src/claims_labels`. Compaction (which downloads
   nothing for it) and the sort then treat them like any other table's groups; the sort's
   commit is the table's only upload.

Both sets' labels are read because the sets refer to each other: an article's authors and
journal are in the main set, and an item's "described by source" is often in the scholarly
set.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/claims_labels.py` | `7b748f7c7fbd659af82cb09430515f9fe5843fdc7fe1b72edeffcfc1e849d863` |
    | `src/wikidata/process.py` | `f37d81c2f6ac4d8429e5aae9b51a3d6c0e14ff9080ee492292374f2f3495e308` |
