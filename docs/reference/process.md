# Processing

`process.py` turns one chunk's JSON columns into typed tables, written to
`results/{table}/chunk_{N}.parquet`. polars-genson does the JSON work. It infers one schema
across a column's rows and normalises every row to it, so rows of different shapes become
the same typed columns.

## Order and resumption

`process()` handles the tables in this order: labels, descriptions, aliases, entities (a
release only), links, claims (which also yields the chunk's claims_labels). A table whose
file already exists is read back instead of recomputed, so a chunk interrupted halfway
redoes only its remaining tables. Leftover `.tmp` files from an interrupted attempt are
removed first. Every table is written with `atomic_sink_parquet` (a `.tmp` file, then a
rename).

After each table, `check_ids` compares its number of distinct ids with the source chunk's
and raises on any difference. Every entity has a row in every table, if only with an empty
map. Once all tables are written, the chunk moves to `PROCESS` and its source file is
deleted.

## Labels, descriptions, aliases

`normalise_map_direct` calls `normalise_from_parquet` on the column with `map_threshold=0`
(every object is a map, keyed by language) and `typed=True`. polars-genson writes typed
Parquet directly, with the inferred schema (as Avro) in the file's metadata. The result is
a list of `{key, value}` structs per entity: `value` a string, or a list of strings for
aliases.

The inferred schema is compared with the expected one (`_map_schema`). The result is then
cast to the expected schema, so that a column whose map is empty in every row of the chunk
(inferred with `Null` values) gets string values like every other chunk.

## Entities (a release)

The `entity` column's JSON is decoded with `ENTITY_SCHEMA`: `type`, `datatype` (a
property's value type, null for an item), `ns`, `title`, `pageid`, `lastrevid`,
`modified`. One row per entity.

## Links

`normalise_sitelinks` uses `normalise_json` with `force_field_types={"sitelinks": "map"}`.
The column is always a map of site codes, but genson would treat an object with few
distinct keys as a record (the default `map_threshold` is 20). Each value decodes to
`SITELINK_SCHEMA`: `site`, `title`, and for a release `badges`, a list of strings.
`max_builders=100` bounds the number of schema builders genson merges in parallel.

## Claims

`normalise_claims_direct` calls `normalise_from_parquet` on the claims with
`CLAIMS_INFERENCE_OPTIONS`:

- `map_threshold=0` and `unify_maps`: claims, qualifiers and reference snaks are maps
  keyed by property, and their values unify into one snak record.
- `force_field_types={"mainsnak": "record"}`: a main snak is always a record.
- `force_scalar_promotion` for `mainsnak`, `datavalue`, `precision`, `latitude`,
  `longitude`: fields that are a scalar in some rows and an object in others are
  promoted to a record holding the scalar (`datavalue__string`,
  `precision__integer` and so on), rather than falling back to a string.
- `no_unify={"qualifiers"}`.

Two of its outputs are written beside the claims:

- **Label maps** (`extract_invariants=LABEL_INVARIANTS`). The philippesaade copy repeats,
  inside every snak, the full multilingual labels of its property (`property-labels`), its
  item value (`labels`) and its unit (`unit-labels`). genson moves each distinct map to a
  lookup file, keyed by the sibling field (`property`, `id`, `unit`). `lookup_to_long`
  turns the lookup into one row per label: `field, ref, language, label`. This is the
  chunk's claims_labels table. An official dump's claims carry no labels, so for a release
  the lookup is empty and claims_labels is built at finalise instead (see
  [claims_labels](claims-labels.md)).
- **Quarantine** (`prune=QUARANTINE_FIELDS`). Snaks on properties since deleted from
  Wikidata could not be rendered in the philippesaade copy. Their `datavalue` is left as
  `{value, error}`, or the snak collapses to the bare property id. genson removes any record
  holding one of these fields (a main snak takes its statement with it) and writes it to a
  separate file. `quarantine_rows` turns that file into one row per snak (entity,
  statement property, part of the statement, property, snak JSON) in
  `quarantine/chunk_{N}.parquet`, kept locally and never uploaded.

The claims are then read back conformed to `claims_schema`, the schema every chunk must
have: a field the chunk lacks is inserted as null, so every chunk, group and file on the
Hub has the same schema. A field the stored schema lacks is dropped by this read, but it
also makes the inferred schema differ, and the schema check halts the run on it.

`claims_schema` differs by source. For a release, each statement has `id`, `type`,
`qualifiers-order` and references with `hash` and `snaks-order`. Each snak has `snaktype`,
`hash` and `datavalue_type`, and an item value has `entity-type` and `numeric-id`.
`DV_SCHEMA` holds every datavalue field across datatypes: entity id, string, time,
quantity, coordinates, monolingual text.

## Schema checks

The inferred schema of each map table and of the claims is compared with the stored one
using `DeepDiff`. `is_acceptable_diff` allows only differences where the chunk's schema is
narrower than the stored one:

- a field missing from the chunk (`dictionary_item_removed`);
- a type of `Null` where the stored type is anything (genson's type for a field never seen
  with a value);
- a record with a subset of the stored record's fields.

Any other difference, such as a new field or a different type, raises `Schema mismatch`.
A new field means the dump has data the schema does not hold yet. A different type for
known data has, during development, meant a bug in polars-genson's inference, which was
fixed there (journal, 2026-10-01 and 2026-10-03).

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/process.py` | `f37d81c2f6ac4d8429e5aae9b51a3d6c0e14ff9080ee492292374f2f3495e308` |
