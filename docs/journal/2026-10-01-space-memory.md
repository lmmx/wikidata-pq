# 2026-10-01: The Space's memory, and v1 running a tab out of it

## Current State

- With v1 as the Space's default run, the Kalman filter (Q846780) stayed at "Ranking the members and fetching their names" and the browser console showed `Uncaught out of memory` (Firefox's message when a tab's heap is exhausted); v0 went back to being the default (sae/runs.json lists v1 first, and the Space opens the last run listed) while this was fixed, with v1 left at `?run=v1`.
- One neighbour search for the Kalman filter downloaded about 11 MB but left 508 MB on the heap on v1 (559,088 candidates over 13 features, 707,407 postings) and 416 MB on v0 (418,872 candidates, 526,829 postings), measured in Node 20 with `--expose-gc` after a forced collection.
- The heap held one object per posting (`{ id: "Q…", unit, kinds }`, each with its own kinds array), cached for the page's life with no limit, and one object per candidate, built twice (an accumulator, then a copy with `similarity`), each with a `shared` array.
- Each item view's candidates stayed reachable after leaving it: its `pin` listener on `window` was removed only when a later pin event fired on another view.
- Opening Kalman filter, Emmy Noether, red fox, caffeine, Casablanca and Kalman filter again on v1 in jsdom with that code aborted Node with a V8 heap out-of-memory.
- space/data.js `postings(feature)` now returns columns in typed arrays (`ids` Uint32, `units` Float32, `kinds` flat Uint32 with `kindAt` offsets, `kindsOf(i)`), about 17 bytes a posting, read with hyparquet's `parquetRead` `onChunk` (no object per row); the cache keeps the most recently used features up to 6,000,000 postings in all (`KEEP_POSTINGS`, about 100 MB).
- `neighbours` accumulates in typed arrays (dot product, a bit per seed feature, and the list and row of each candidate's kinds) and returns one small `Candidate` object per candidate, whose `id`, `kinds` and `shared` are getters; the rankings match the old code's on the Kalman filter, Emmy Noether and red fox, to four decimals.
- After one search the heap holds 76 MB (Kalman filter, v1), 52 MB (Kalman filter, v0), 58 MB (Emmy Noether, v1) and 46 MB (red fox, v1), against 446-508 MB before; the search takes 1.8-2.3 s against 4.2-4.5 s; decoding a search's row groups still peaks at about 170-420 MB above the start, briefly.
- space/index.html removes an item view's `pin` listener on the next `hashchange`; space/data.js keeps at most the latest 64 MB of each file's fetched byte ranges (`KEEP_BYTES`) besides its footer, where it had kept every range fetched.
- Walking nine item views on v1 in jsdom (Emmy Noether, red fox, caffeine, Hilbert space, Tetris, Casablanca, Kalman filter, Emmy Noether, red fox) holds the Node heap at 305-359 MB, jsdom and the page's other caches included, with no errors.
- v1's postings had 23 rows with a null `id` and v0's 6: properties with external IDs (P31, P2037, P14004, ...) are coded, and `qnumber` makes their ids null; on v1 several share the GitHub topic feature, so for the Kalman filter they added up to one candidate at similarity 2.87, above every real one. space/data.js skips postings without an id, and sae/publish.py leaves ids not starting with Q out of the postings.

## Missing

- The deployed Space has the old space/data.js until `space/upload.sh`; v1 goes back to being the default by moving it last in sae/runs.json and uploading that.
- The published postings still hold the null-id rows until a run is published again; the page skips them.
- The memory figures are Node's; the browser's own figures have not been measured.
