# Benchmark results

The record of wikidata-pq's benchmark (`bench/README.md` on master): one entry per push
to master that changes the pipeline, the new commit against the previous one. The
Benchmark workflow appends each entry; nothing here is edited or removed. This branch is
data only and is never merged.

- `RECORDS.md`: one row per entry, oldest first.
- `records.jsonl`: the same entries as JSON, one per line, with CPU, memory and package
  versions.
- `runs/<commit>.json`: each entry's raw results (`bench/ab.py`'s `results.json`).
