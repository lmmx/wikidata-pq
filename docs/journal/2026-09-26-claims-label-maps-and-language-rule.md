# Claims review: embedded label maps, data loss, language rule

Findings from a review session on 2026-09-25/26 that measured one real source file.
Measurements come from `chunk_1237.parquet` (105 MB, 10,000 entities, 625 MB of claims JSON, 54,295 claims) with polars-genson 0.8.0 `typed=True` on a 4-core / 15 GB machine. Each figure is from a single run.
`chunk_1237` is category-heavy: 94% of entities have an English label, the median entity has 3 labels, and most descriptions are bot-generated. The measurement scripts (`measure.py`, `bench.py`, `part_*.py`, `rules.py`, `options.py`, and the `hoist/` Rust prototype) stayed in that session's scratchpad and are not in either repo.

## Current State

### Embedded label maps in `claims`

- Every `labels`, `property-labels` and `unit-labels` map inside the claims JSON is the full multilingual label set of the referenced entity — on `chunk_1237` those maps total 610.9 MB of the 625.3 MB claims JSON (97.7%), across 154,783 maps.
- The 154,783 label maps on `chunk_1237` belong to 14,257 distinct referenced ids, and no id has two different maps — the map is a function of the sibling `id` / `property` / `unit` value.
- Claims JSON on `chunk_1237` with the label maps removed is 13 MB, and one copy of each distinct map is 19 MB.
- `normalise_from_parquet(typed=True)` on the full `chunk_1237` claims takes 26.3 s (read 3.13 s, infer 10.65 s, parse+normalise+decode 9.18 s, write 3.30 s) at 5.75 GB peak RSS, writing 309 MB, plus 4.45 s to read the output back.
- `normalise_from_parquet(typed=True)` on `chunk_1237` claims with the label maps removed takes 1.22 s (infer 0.69 s, parse+normalise+decode 0.23 s, write 0.21 s) at 0.39 GB peak RSS, writing 3.4 MB, plus 0.18 s to read back — the inferred schema equals the full schema minus the three label-map fields.
- A zero-copy Rust prototype using serde_json `RawValue` removes the label maps from the 625 MB `chunk_1237` claims in 0.68–0.75 s on 4 threads, keying each map on its sibling id and keeping the first copy per id.
- `pl.read_parquet(columns=["claims"])` reads the `chunk_1237` claims column in 0.24 s, against 3.13 s for `read_string_column` in genson-core.

### Partition step cost

- `prepare_claims` explodes `property-labels` to one row per language while each row keeps the full claim, including `datavalue` with its label list and `qualifiers` / `references` with their embedded label maps (src/wikidata/partitioning/claims.py).
- The current partition of the first 100 entities of `chunk_1237` (1,041 claims) produces 115,348 rows and 84 MB on disk, taking 10.5 s at 4.6 GB RSS; 500 entities and the whole file were killed at 15 GB.
- Partitioning `chunk_1237` claims with label maps removed but one full claim row per language produces 5.39M rows and 97 MB on disk in 63 s at 7.6 GB, and loading those rows ran out of memory.
- Joining one row per claim (54,295 rows, 1.2 MB) to a per-language label table built from the extracted label maps reproduces the current layout's 5,389,254 label rows in 0.64 s and 13.1 MB.

### Data loss in the current pipeline

- `normalise_from_parquet` writes only the normalised output column and `df.genson.normalise_json` returns only the normalised column — the entity `id` is absent from the `labels`, `descriptions`, `aliases`, `links` and `claims` outputs (src/wikidata/process.py:42-101, 186-227).
- The `n_ids` checks that would compare entity counts across tables are commented out in `process()` (src/wikidata/process.py).
- `read_string_column` skips null rows in both the `Utf8` and `LargeUtf8` branches (polars-genson genson-core/src/parquet.rs:79-102) — `normalise_from_parquet` output can have fewer rows than its input and cannot be re-aligned with the input's `id` column by position.
- `transform_quantity` filters with `unit-labels.list.len() > 0` and its negation — `unit-labels` is null for dimensionless quantities (unit `"1"`) under `empty_as_null`, the comparison is null, and both filters drop the row, so population (P1082) and other unitless quantities are lost (src/wikidata/partitioning/claims.py:99, 136).
- On the first 100 entities of `chunk_1237`, 1 of 14 quantity claims is lost through the null `unit-labels` filter.

### Which language subsets a claim reaches

- The current partition writes a claim to language L when both the property and the value (and, for quantities with units, the unit) have a label in L; plain values need only a property label, and monolingual text uses the text's own language (src/wikidata/partitioning/claims.py).
- On `chunk_1237`, 35,566 claims have an item value; the share kept by the current rule is en 99%, fr 90%, de 89%, ja 85%, cy 70%, sw 40%, while the property has a label for en/fr/de/ja 100%, cy 98%, sw 65%.
- Candidate rules measured on `chunk_1237` (rows = claim × language pairs written; "unnamed" = rows where the claim's own entity has no label in that language):

| Rule | Definition | Rows | Claims in no subset | Unnamed-entity rows | Null property label | Null value label | en | fr | ja | sw |
|---|---|---|---|---|---|---|---|---|---|---|
| A | current rule as coded | 5,363,036 | 421 | 5.1M | 0 | 0 | 53,198 | 49,634 | 46,711 | 21,880 |
| A2 | A with the quantity filter fixed | 5,389,254 | 124 | 5.1M | 0 | 0 | 53,495 | 49,931 | 47,003 | 21,966 |
| B | property has a label in L | 8,099,942 | 0 | 7.8M | 0 | 2.6M | 54,262 | 54,097 | 52,916 | 31,306 |
| C | entity has a label in L | 408,378 | 0 | 0 | 114k | 53k | 49,811 | 13,637 | 3,633 | 1,517 |
| C2 | entity has a label, description or alias in L | 2,580,079 | 0 | 2.2M | 709k | 672k | 52,720 | 32,641 | 20,435 | 17,416 |
| D | C ∪ A2 | 5,539,333 | 0 | 5.1M | 114k | 53k | 54,087 | 50,583 | 47,301 | 22,856 |
| E | C ∪ B | 8,214,226 | 0 | 7.8M | 114k | 2.6M | 54,294 | 54,151 | 53,010 | 31,826 |
| F | every language | 30,405,200 | 0 | 30.0M | 22.3M | 14.8M | all | all | all | all |

- Entities with a label on `chunk_1237`: en 9,379, fr 1,614, ja 549, sw 402 of 10,000.
- Rule A2's 5,389,254 rows on `chunk_1237` match the row count of the current partition code with the quantity filter fixed.
- Rule D contains every row of rule A2 and adds 150,079 rows (+2.8%) on `chunk_1237` — the added rows are claims in languages where the entity has a label but the property or value lacks one, and the 124 claims that A2 places in no subset account for only a small share of them.
- On `chunk_1237`, 8,521 of 10,000 entities gain languages only through descriptions — the C2 row count is 6.3× the C row count.
- 135 entities on `chunk_1237` carry a `mul` label, 2 of them carry only `mul`, and 1,979 of the 14,257 referenced ids carry a `mul` label.
- Rules C, C2, D and E move monolingual-text claims into the entity's languages in addition to the text's own language, while rule A files them only under the text's language.

## Missing

- A label-map extraction option in polars-genson that returns the claims without the embedded maps plus a `(ref_id, language, label)` side table.
- A way for `normalise_from_parquet` to carry input columns such as `id` into its output, and to keep null input rows as null output rows.
- A `fill_null(0)` on the `unit-labels` length in `transform_quantity`.
- Label-map extraction in `process.py` and a join-based `prepare_claims` that builds each language subset from the extracted label table.
- A `subject_id` and `subject_label` column on each claims row.
- Label fallback along MediaWiki language fallback chains with a `*_label_lang` column.
- A choice of language rule among A2, C, D and E, and of how `mul` labels and monolingual text are treated.
- A per-chunk audit comparing entity ids and claim counts from the source file to every output table.
- Measurements of the label-map share and the rule table on a chunk of ordinary items rather than categories.
