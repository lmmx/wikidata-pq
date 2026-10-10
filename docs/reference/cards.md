# Dataset cards

Each repo's `README.md` is rendered from a template and two JSON files of figures, so every
number and example in a card comes from the published data. `card_stats.py` computes the
figures; `cards.py` renders and pushes the cards.

## Inputs

| File | Written by | Holds |
|---|---|---|
| `docs/dataset_cards/{dump,scholar}/{table}.md` (or `docs/dataset_cards/{table}.md`) | by hand | the card's text, with placeholders |
| `dataset_cards_metadata.json` | compaction, then the sort | files, bytes and rows per key of each table |
| `dataset_cards_stats.json` | `card_stats.update_stats` | sample rows and language coverage, with a digest of their inputs |

For a release, the two JSON files are in `docs/releases/{set}/`.

## Figures

`compute_stats` reads the local copy (`hub/{table}`) after checking it against the
metadata (the same keys, files and row counts), and computes:

- **Sample rows**: the rows of a few fixed ids in a few keys (`SAMPLES`). Douglas Adams
  (Q42) and human (Q5) for the main set; the 1953 paper on the structure of DNA
  (Q1895685) and "Attention Is All You Need" (Q30249683) for the scholarly set.
- **Coverage**, for the tables split by language: how many item and property ids have a
  row in any key, in `en`, in `mul`, and in `mul` but not `en`. One flag per id number
  keeps memory at a few bytes per id however many keys a table has.

Without the copy, or with one that differs, it stops with `local copy differs from the
metadata ...: run download-wikidata`. A table whose figures are current is skipped and
needs no copy.

Each table's figures carry `inputs_sha256`, a digest of its metadata entry and of what is
computed (its `SAMPLES` and `STATS_COLUMN` entries, `PREFIXES`). `current()` compares the
digest with the present inputs, and `update_stats` recomputes only stale tables.

## Rendering

`render_card` fills the template's placeholders:

| Placeholder | Filled with |
|---|---|
| `{{configs}}` | the front matter's subsets: one per key, plus `all` |
| `{{default}}` | the default subset: `en`, `enwiki` for links, `all` for claims and entities |
| `{{key_examples}}` | the largest keys, and the largest with a hyphen |
| `{{sample}}` | the sample rows |
| `{{sizes}}` | totals, the ten largest keys, and every key in a collapsed list |
| `{{languages}}` | the `en` and `mul` coverage, with a note on Wikidata's `mul` code |
| `{{release}}` | the release's date |

Rendering refuses:

- figures whose digest is not current;
- a placeholder it does not know, or one left without a value;
- an example in the text that names a subset the repo does not have (by the
  `wikidata-…/{key}/*` paths and `load_dataset(…, "key")` calls).

A table with no metadata yet (a repo's first upload) gets only the `all` subset and empty
placeholders. That is the card the first group upload adds.

`write_cards` renders every table's card to the rendered cards directory, and
`push_card` uploads a card only if it differs from the repo's `README.md`. `render-cards`
(`just cards`) renders without pushing. `finalise` always pushes the cards that changed;
it has no option to skip them.

??? info "Documented against"
    Commit `b8ac85a` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/cards.py` | `517ae2f35bfd45f4dd396d2385326fa42c84b671bab502191796a5d6af091577` |
    | `src/wikidata/card_stats.py` | `d7774100eb85214541696cf146600541c0a04d822d1ed45a08e4a5edb136053a` |
