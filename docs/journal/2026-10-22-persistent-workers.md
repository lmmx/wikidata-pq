# 2026-10-22: Worker processes that take several chunks

**The change:** `pool.process_chunks` runs chunks in worker processes that each take up to
`chunks_per_worker` chunks (`CHUNKS_PER_WORKER`, default 8, or
`WIKIDATA_CHUNKS_PER_WORKER`). A worker is then replaced. Before, every chunk had a
spawned process of its own; `WIKIDATA_CHUNKS_PER_WORKER=1` still does that.

**The effect:** over 22 and 66 chunks with 6 workers, the pool takes 7–9% less time, the
output files are byte-identical, and the workers hold about 1.3 GB more memory between
them. Scripts and logs are in `2026-10-22-persistent-workers/`.

## Why

**What a fresh process per chunk costs:**
- **Imports:** each chunk's process paid the imports, about 0.4 s
  (polars-genson's `docs/journal/2026-10-22-write-path-comparison.md`).
- **Page faults:** each started from no memory of its own. On the 4-core VM measured here,
  memory a process frees goes back to the host, so a fresh process faults in all it
  uses, and that system CPU dominated the run-to-run spread.
- **One process for many chunks:** running the 11 sample chunks one after another in a
  single process saved about 3.5–5 s of `process()` time in all, with no RSS growth.

**Why chunks had a process each:** `pool.py` gave every chunk its own process so that "the
memory it used goes back to the OS when it exits (freed native memory otherwise stays in
the allocator and RSS climbs chunk after chunk)". That is what the long run below checks.

## Change

`pool.py`:
- **Workers:** `workers` spawned processes run `_serve`. Each receives chunk indices on a
  pipe, runs `work(c)`, and replies None, or the traceback (printed to stderr too) when it
  fails.
- **Replacement:** a worker that has taken `chunks_per_worker` chunks is told to exit when
  its last chunk is done, and the next chunk starts a new worker.
- **What stays as it was:**
  - chunks start in order;
  - `start(c)` and `done(c)` run in the parent;
  - groups close as before;
  - a failed chunk stops new chunks starting, the running ones finish, and the run raises.
- **Failures:**
  - A chunk that raises takes its worker with it: the worker exits after replying, and no
    chunk follows it there.
  - A worker that dies mid-chunk fails that chunk with the signal named ("its worker died,
    killed by signal 9"), as a process per chunk did.

## Correctness

**Tests** (`scripts/test_pool.py`): the existing tests pass, and run again with workers
taking several chunks. New ones check:
- workers are reused, up to `chunks_per_worker` chunks each;
- a chunk that raises stops the run, and its worker takes no later chunk;
- a SIGKILLed worker fails its chunk with "killed by signal 9", with 1 or 8 chunks per
  worker.

**Real data** (`logs/outputs_identical.txt`): every file a 22-chunk run writes, 14,330
under `results/` and `audit/`, is byte-identical between `per=1` and `per=8` in all three
rounds.

**Lint:**
- flake8 is clean on the changed files.
- `ty check` on them reports only the two diagnostics in `main.py` that master has. The two
  in master's `pool.py` (`-p.exitcode` on an `int | None`) are gone.

## Measured

**Setup** (`pool_bench.py`):
- **Pool:** `pool.process_chunks` as `run` drives it, with the real `process_and_partition`
  per chunk, 6 workers, chunks in index order, no upload (groups never close).
- **Chunks:** chunk i is a link to the (i mod 11)-th of the 11 sample chunks of release
  20261012, so a run repeats the c0, r10–r90 and s10–s90 mix.
- **Machine:** 4-core VM, normal settings: nothing run between runs, mimalloc and its
  settings untouched, polars-genson 1.2.2.
- **Memory:** the workers' summed RSS, sampled every 0.2 s.

`logs/runs.txt`:

| chunks | per worker | wall (s) | summed RSS mean | summed RSS peak |
|---|---|---|---|---|
| 22 | 1 | 80.3, 71.1, 70.9 (median 71.1) | 6.3–6.5 GB | 9.6–10.9 GB |
| 22 | 8 | 66.3, 66.1, 69.1 (median 66.3) | 7.0–7.4 GB | 10.6–11.9 GB |
| 22 | master's pool | 72.8, 72.9 | 6.3–6.5 GB | 9.4–10.1 GB |
| 66 | 1 | 228.6, 216.5, 217.8 (median 217.8) | 7.4–7.5 GB | 11.0–11.9 GB |
| 66 | 8 | 199.9, 200.1 | 8.7–8.9 GB | 12.3–12.7 GB |
| 66 | never replaced | 199.6, 196.7 | 8.6–8.8 GB | 12.7–13.7 GB |

**Time:**
- 8 chunks per worker saves 7% on 22 chunks and 8% on 66 (about 0.27 s a chunk), against
  a process per chunk.
- The chunks' system CPU falls: 255–288 s against 240–247 s on 66 chunks.

**Memory:**
- Summed RSS is about 1.3 GB higher on average and 0.8 GB at the peak.
- A worker keeps the memory of its last chunk until the next one reuses it. After c0 with
  its partitioning, that is 2.5–2.9 GB.

**No growth** (`logs/rss_by_worker.txt`): with workers never replaced, a worker took up to
16 chunks. Its RSS after a given chunk kind is the same in every one of the six cycles:

| chunk | RSS after it, cycles 1–6 |
|---|---|
| c0 | 2.5–2.9 GB |
| s90 | 2.1–2.2 GB |
| s30 | 1.7–1.8 GB |

Replacing workers every 8 chunks costs nothing measurable, and bounds anything a longer
life might keep.

## Limits

- **One machine:** a 4-core VM with 6 workers, where page faults are expensive. On a
  machine with cheap faults the saving is smaller, about the imports alone.
- **Memory headroom:** a worker holds up to its recent chunks' peak, not its current
  one's. Where 6 workers have little memory to spare, set `WIKIDATA_CHUNKS_PER_WORKER=1`.
