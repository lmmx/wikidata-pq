# 2026-10-01: Finding items in the Space

## Current State

- space/data.js `search` reads the row groups of `names.parquet` whose lowercased-label range can hold the query, at most 40 matches, ranked by exact name, then items with a code, then Wikipedias; a query of several words also matches items whose label starts with the first words and whose description has the rest ("transformer album" finds Q631153 and Q7834117).
- transformer (Q85810444) has one external ID (Google Knowledge Graph) and no code, and machine learning model (Q115215420) has none; `names.parquet` now holds every labelled item with an external ID (53,578,070), items without a code greyed in the list.
- space/index.html adds, above the names, up to 6 classes whose label has every word of the query (from `classes.parquet`, by how many items are directly of them) and an "Open Q…" match for a Q id — "machine learning model" lists the type Q115215420 first.
- An item without a code opens on its label and description (from Wikidata when not in `names`), its Wikidata links, and for a type the items under it.
- space/data.js `description` finds an item's description in `names` by its lowercased label, for items reached by a link.
- Sunday roast's neighbours of kind dish showed Q118819064 and Q118823515 by Q id: their only labels are Italian; the publish now falls back to other languages, and space/index.html asks Wikidata for a label in any language for a shown item with none.
- space/index.html shows an item's own statements with item values (up to 14 properties, 6 values each), fetched live from Wikidata and labelled as unused by the model, in a "Linked on Wikidata" panel above "Catalogue families".
