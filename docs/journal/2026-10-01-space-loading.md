# 2026-10-01: Loading the Space's data in the browser

## Current State

- The first space/index.html loaded DuckDB-WASM 1.33.1-dev57.0 and defined views over the Hub files of `items`, `names` and `postings` before reading `features.parquet` — on the deployed Space it stayed at "Loading the features" on an iPad and a desktop browser.
- The Parquet footers then measured items.parquet 0.91 MB (1,585 row groups), postings.parquet 2.01 MB (4,909), names.parquet 0.80 MB (1,470), features.parquet under 0.01 MB.
- space/data.js reads the Hub files with hyparquet 1.31.2 and hyparquet-compressors 1.1.2 (zstd): each file's footer once, then the row groups whose footer min/max statistics can hold the key looked up, read in parallel, with 64-bit integers converted to numbers; space/index.html imports both from jsDelivr (`+esm`).
- Reading row groups one after another took 11.1 s for the Kalman filter's neighbours and 20.8 s for 30 labels, against 5.4 s and 2.3 s in parallel (Node 20, undici through the container's proxy).
- The item view with each step awaited in turn took: features 1.5 s, classes 1.5 s, the item 1.9 s, its description 1.7 s, the French Bulldog's neighbours over 16 features 6.7 s (107 requests, 14.4 MB), the labels of 40 neighbours 1.3 s.
- space/index.html starts the description, the classes and the neighbours together once the item is found, renders the features before they arrive, fetches the footers of `names`, `items` and `postings` while `features` loads, and starts reading the postings of the item's features when the item is found.
- Reading only each feature's first 20,000, 40,000 or 60,000 postings by weight kept 1, 0 and 4 of the Kalman filter's true top 10 neighbours, and 1, 7 and 8 of the French Bulldog's, and saved 1-2 s; postings are read whole.
- space/index.html shows a spinner and a bar counting the item's features whose postings have arrived (space/data.js `neighbours` takes `onRead(done, total)`), then "Ranking the members and fetching their names"; for French Bulldog the count reached 1 of 16 at 3.7 s, 7 at 5.2 s, 16 at 9.9 s.
- On an iPad the page stayed at "Loading the model's features" with no error shown; space/index.html now runs a classic script first that defines `Array.prototype.at` (and on typed arrays) where missing (Safari before iOS 15.4; hyparquet 1.31.2 calls `.at(-1)` in src/assemble.js, and the page called it twice), reports a failed script fetch or unhandled error in the status line, notes a wait past 30 s, and names each startup step ("Loading the page's code", "Finding the model runs", "Loading the model's features").
- The spinner shows for the first load, the neighbours, a feature's members and statement checks; "Searching" and "Looking it up" are plain text.

## Missing

- The page has not run on an iPad since the `Array.prototype.at` fallback and the error reporting.
