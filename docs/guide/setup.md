# Setup

## Software

- Python 3.13 or later, with the project installed: `uv sync` in the repository. This
  installs the commands the Justfile calls (`process-wikidata`, `finalise-wikidata`,
  `download-dump`, `split-dump`, `promote-release` and others, listed under
  `[project.scripts]` in `pyproject.toml`).
- [just](https://github.com/casey/just), to run the recipes in the `Justfile`.
- `lbzip2`, which `split-dump` uses to decompress the dump on every core
  (`apt install lbzip2`). There is no fallback to another decompressor: without it,
  `split-dump` exits with `lbzip2 decompresses the dump on all cores: apt install lbzip2`.
- [polars-genson](https://github.com/lmmx/polars-genson), installed as a dependency. It
  infers the schema of each chunk's JSON and normalises the JSON to typed columns. The
  pipeline depends on its exact behaviour: a schema it infers differently halts the run
  (see [Processing](../reference/process.md#schema-checks)), and the fix belongs in
  polars-genson, not in a workaround here.

## Hugging Face credentials

The pipeline creates and writes dataset repos under the account in `HF_USER`
(`permutans`, in `src/wikidata/config.py`). Log in with a token that can write to it:

```sh
hf auth login
```

`huggingface_hub` reads the token from its cache or from `HF_TOKEN`. New repos are public
unless `HF_REPO_PRIVATE` is set to `True` in the config.

## Disk

Every working file of a release is under `releases/{release}/` (the main set) and
`releases/{release}-scholar/` (the scholarly set). For the release 20260928:

| Stage | Held on disk |
|---|---|
| `download-dump` | the bz2 dump, 103 GB |
| `split-dump` | the dump and its chunks together, about 210 GB; delete the bz2 once `split.done` exists |
| after `route-release` | the chunks of both sets, 49.5 GB (scholarly) and 44.7 GB (main) |
| `run-release` | the chunks not yet processed, plus at most one upload group's partitions and its staged copy (each up to `GROUP_MAX_GB`, 25 GB) |
| `finalise-release` | one table's local copy at a time; the claims sort holds the copy, its id-range buckets and the sorted files, about three times the claims |

Source chunks are deleted once processed, partitions once merged into a group, and a
group's staged files once verified on the Hub (`CLEAN_UP_LOCAL`, on by default).

The [philippesaade build](source-copy.md) holds at most the prefetch budget of source
chunks (60 GB), plus one group's partitions and staged copy, and its prefetch pauses below
100 GB free. Its finalise reads the local copy in `hub/`, 35.5 GB.

## Memory and cores

Each chunk is processed in its own process. A single chunk's process can reach several GB
(the [worker benchmark](tuning.md#how-many-workers) measured 16 to 20 GB for all of a
trial's processes together, at 2 to 8 workers). The defaults (6 workers) were chosen on a
machine with 20 cores and 125 GB of memory.

## Limits

- Every run builds every table, in every language, from every chunk of the set. There is
  no option to limit languages, tables or chunks: filter the published tables instead.
- A release is always routed into two sets, and the main set's claims_labels is built
  from the scholarly set's labels as well as its own
  ([Finalise](../reference/finalise.md)), so the scholarly set cannot be skipped.
- One run per working directory. Nothing locks it, and two runs of the same set would
  process the same chunks and close the same groups.
- `just download` (`download-wikidata`) fetches every table's repo of the set in full. For
  some languages only, use `snapshot_download` with `allow_patterns`, as in the README.
