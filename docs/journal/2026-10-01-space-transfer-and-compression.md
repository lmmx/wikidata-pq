# 2026-10-01: Fewer requests and fewer bytes for the Space

## Current State

### Requests

- A range request through the Hub's `resolve` URL took 0.51-0.88 s, of which 0.15 s was the 302 to the CDN, against 0.34-0.40 s sent to the CDN URL directly (curl, three runs); the CDN URL carries `Expires=` about an hour ahead.
- space/data.js opens a file with one request for its last 1 MiB (`Range: bytes=-1048576`), which gives the CDN URL (the response's `url`), the file's length (`content-range`) and its footer, and sends later ranges to the CDN URL, through the Hub again a minute before it expires or when the CDN refuses.
- space/data.js fetches a run of adjacent row groups in one request when the columns read are 80% or more of their bytes (the postings, the names), and leaves column-by-column reads to hyparquet otherwise — merging the `id` and `label` reads of 40 scattered items fetched 19.8 MB against 5.9 MB.
- Benchmark from Node against the v0 files (features, French Bulldog's item and neighbours, 40 labels, the Kalman filter's item and neighbours): 14.6 s, 81 requests, all redirected, before; 12.5 s, 56 requests, 3 redirected, after (neighbours 30 → 16 requests, labels 1.30 → 1.05 s).
- space/index.html draws the neighbours without waiting for the walk of the item's members ("under this type" joins the filter row when it ends), and fetches the Wikidata link labels in parallel batches.

### Compression

- On 10 row groups of each v0 file (pyarrow 25, zstd): names 5.14 MB as published, 4.65 MB at level 19, 4.21 MB (82%) at level 19 with the sorted `key` as DELTA_BYTE_ARRAY; items 6.92, 6.68, 6.27 MB (91%) with `id` as DELTA_BYTE_ARRAY; postings 2.22, 2.11 MB (95%).
- In the postings, `unit` (float32, 0.032 to 1.0) took 1.48 of 2.11 MB; as a 16-bit integer (`round(unit × 65535)`, BYTE_STREAM_SPLIT) it takes 0.91 MB and the postings 1.55 MB (73%), with a largest rounding error of 7.7e-6.
- hyparquet 1.31.2 reads DELTA_BYTE_ARRAY only in version 2 data pages (src/datapage.js), and reads 16-bit BYTE_STREAM_SPLIT, DELTA_BINARY_PACKED and list columns there.
- Writing names at level 19 took 4.9 s for 200,000 rows against 0.4 s at level 9.
- A v0 publish at zstd level 19 ran long enough to be stopped (the 200,000-row timings scale to about 22 minutes for names, 8 for items and 2 for postings, single-threaded).
- sae/publish.py writes every table through one function (`write`): pyarrow, zstd level 9 (`--zstd-level`, 19 for a further 3-10%), version 2 data pages, dictionaries for the columns without an encoding, `items.id` and `names.key` as DELTA_BYTE_ARRAY, and `postings.unit16` as BYTE_STREAM_SPLIT; space/data.js reads `unit16 / 65535`, a float `unit`, or `weight / norm` — checked on a fake publish served locally.

## Missing

- v0 has not been republished with the version 2, delta-key, 16-bit layout.
