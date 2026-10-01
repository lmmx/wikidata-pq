# 2026-10-01: The Space's layout

## Current State

- The first item page was one table of features, parent chains and neighbours in a white box; a redesign split it into two panels, with features in four bands by level (Broad, General, Specific, Niche) and neighbours grouped by the feature linking them most.
- Four art directions for the item page (Concordance, Catalogue card, Art deco, Proximity map) were published as an artifact on the French Bulldog's data; the first version drew nothing (a global `const top` clashing with `window.top`); the Concordance (one row per neighbour, one column per feature of the item) was kept, the others not, and three ways of fitting it into the page showed that layouts hiding information missed the aim.
- space/index.html keeps its dense look and takes from the Concordance only the head of the neighbour list: the item's features as columns, broad to niche then by weight, with level brackets, names set vertically (linked to the features' pages) and the item's weight on each as a bar.
- The item page puts "Similar items" first and wider (1.25 : 1), beside a stack of "Linked on Wikidata" (hidden when empty) over "Catalogue families"; families are one line each (name, catalogues, item count, weight bar), the band heading giving the level, and those under 40% of the item's top weight fold under "lighter ones".
- Headings and notes name the item rather than lean on "it" ("Similar items", "Catalogue families", "the item's own statements").
- A feature's page shows its parent chain, level, catalogues with decoder-weight bars, narrower features, examples and its 40 strongest items; the home page shows example items, "How this works" on the nested levels, and the 64 broadest features.
- The design was checked by rendering in jsdom 29.1.1 against the Hub files, not by looking at it in a browser.
