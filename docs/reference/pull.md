# Pull

`pull/` downloads the philippesaade copy's chunks from the Hub. It is used only without
`WIKIDATA_RELEASE`. A release's chunks are already local, and its pull step is
`dump.check_chunk` ([Dump and routing](dump.md#the-pull-step-for-a-release)).

## One chunk (`core.py`)

`pull_chunk` takes the chunk's files at `INIT` or `PULL`:

1. Compares each file's local size with the size in the source repo's file listing
   (`size_verification.py`, cached per run).
2. Moves a file already present at the right size to `PULL` without downloading it.
3. Moves each remaining file to `PULL` before its download starts, so an interrupted
   download is retried on the next run.
4. Downloads them with one `snapshot_download` call, `allow_patterns` listing the files,
   into `data/huggingface_hub/{repo}/data/`.
5. Checks every downloaded file's size and raises on a mismatch.

`download.py` retries transient Hub errors (timeouts, dropped connections, gateway errors)
with delays from 30 s up to 30 min, about three hours in all, so an unattended run survives
a Hub outage. Other errors fail at once.

## Prefetch (`prefetch.py`)

While a chunk is processed, `prefetch_worker` runs in a background thread and downloads the
chunks ahead of it:

- up to `PREFETCH_MAX_AHEAD` (60) chunks ahead;
- while the local source files total under `PREFETCH_BUDGET_GB` (60 GB);
- only while the disk has over `PREFETCH_MIN_FREE_GB` (100 GB) free;
- skipping chunks whose files are all present.

A new prefetch pass is queued only once the previous one has finished. A prefetch error is
printed and does not stop the run; the chunk is pulled again in its turn.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/pull/__init__.py` | `dc128037adf9a49406dd164a41775e08c1ab31a1b634eb722cb1f7e3b5690310` |
    | `src/wikidata/pull/core.py` | `810b4100024331502da9baca1a57ff037ba588a014b5b4e2d5623c7f0e4e9c64` |
    | `src/wikidata/pull/download.py` | `e590c97c2bcadd74372257f71a97ff6aa02e986043e5c0a7a84a5c8a2e044481` |
    | `src/wikidata/pull/prefetch.py` | `a69cf089d01bd8dcd9c39935879e1819fcd96abf08b538cf4ea39aa3a9a334df` |
    | `src/wikidata/pull/size_verification.py` | `5af4e38f843577136f96dbce800c4a1173f85e89fd5baa5db0975b572e0feedf` |
