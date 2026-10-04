# Build a release

A release is one of Wikidata's JSON dumps, named by its date (for example `20260928`), from
[dumps.wikimedia.org/wikidatawiki/entities](https://dumps.wikimedia.org/wikidatawiki/entities/).
Building it takes six commands. Each is safe to run again after an interruption: it picks
up from the progress recorded on disk.

```sh
just latest-dump                  # prints the newest release with a full JSON dump
just download-dump 20260928       # wikidata-20260928-all.json.bz2, checked against its md5
just split-dump 20260928          # into chunks of 10,000 entities
rm releases/20260928/dump/*.bz2   # once releases/20260928/data/split.done exists
just route-release 20260928       # move the scholarly works into their own set
just release 20260928 20260507    # process, finalise and publish both sets
```

The second argument to `just release` is the tag given to the files currently on each main
set repo's `main` branch before the new release replaces them. `20260507` names the build
from the philippesaade copy, which was the dump of 2026-05-07.

## 1. Download

`download-dump` fetches the bz2 dump into `releases/{release}/dump/`, resuming a partial
download (`.part`) after an interruption, and checks the result against the release's
`md5sums.txt`. `WIKIDATA_DUMPS_URL` points it at a mirror's `.../wikidatawiki/entities`
directory; the md5 sums always come from dumps.wikimedia.org.

## 2. Split

`split-dump` decompresses the dump with `lbzip2` and cuts it into
`releases/{release}/data/chunk_{N}.parquet`, 10,000 entities each. Each chunk has one row
per entity: `id`, then `labels`, `descriptions`, `aliases`, `sitelinks`, `claims` and
`entity` as JSON strings. A line per chunk goes into `data/manifest.jsonl` (rows, bytes,
first and last id, and the field names seen), and `data/split.done` marks the end. A rerun
skips chunks already in the manifest. What the split changes in each entity is listed in
[Dump and routing](../reference/dump.md#split).

## 3. Route

`route-release` separates the scholarly works. Each chunk's entities whose "instance of"
(P31) is one of the scholarly classes go to the chunk of the same number in
`releases/{release}-scholar/data/`, and the rest stay in `releases/{release}/data/`. Each
set's chunks are then renumbered from 0 without gaps, and each set gets its own manifest,
`split.done` and `route.done`. For 20260928: 46,383,002 scholarly entities in 10,853
chunks, and 75,432,640 others in 12,182 chunks.

## 4. `just release`

The `release` recipe runs these in order:

1. `run-release {release} scholar`
2. `run-release {release} main`
3. `finalise-release {release} main`
4. `finalise-release {release} scholar`
5. `finalise-release {release} main`
6. `promote-release {release} {previous} main`
7. `promote-release {release} {previous} scholar`

Each recipe can also be run on its own, with `set` as `main` (the default) or `scholar`.

### run-release

For one set, each chunk is processed into the seven tables, partitioned by language, and
deleted. Partitioned chunks are uploaded in **groups** of contiguous chunks to a branch
`build-{release}` of each repo, which leaves `main` unchanged until promotion. Six chunks
are processed at once (`WIKIDATA_WORKERS`), and a group uploads in the background while
the next chunks run. See [Run loop and state](../reference/run.md) and
[Push](../reference/push.md).

### finalise-release

Once a set's chunks are all uploaded, each table is **compacted** (its many group files
rewritten into files of about 500 MB) and **sorted** by id, both on the build branch. The
claims go first. A release's `claims_labels` is built from the sorted claims and the labels
of **both** sets, because each set's statements refer to the other's entities. The first
`finalise-release main` therefore stops before `claims_labels`, since the scholarly
labels are not yet sorted. The scholarly finalise then builds its own, and the second
`finalise-release main` builds the main set's. Each set's last step renders and pushes its
dataset cards and writes `state/finalise.done`. See [Finalise](../reference/finalise.md).

### promote-release

For each repo of the set, `main`'s current files are tagged with `previous`, the build
branch's files replace them on `main`, `main` is tagged with the release, and the branch is
deleted. The scholarly repos are new, so their `main` has no previous release to tag.
Promotion refuses a set without `finalise.done`. See
[Hub branches and promotion](../reference/hub.md).

## Commands at a glance

| Recipe | Command run | Environment |
|---|---|---|
| `latest-dump` | `latest-dump` | |
| `download-dump r` | `download-dump` | `WIKIDATA_RELEASE=r` |
| `split-dump r` | `split-dump` | `WIKIDATA_RELEASE=r` |
| `route-release r` | `dump.run_route()` | `WIKIDATA_RELEASE=r` |
| `run-release r set` | `process-wikidata` | `WIKIDATA_RELEASE=r`, `WIKIDATA_SCHOLAR=1` for `scholar` |
| `finalise-release r set` | `finalise-wikidata` | as above |
| `promote-release r p set` | `promote-release` | as above, and `WIKIDATA_PREVIOUS_RELEASE=p` |
| `release r p` | the seven steps above | |

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `Justfile` | `b3f4fe372a609f225b19c87c219c80dbd40aa13e0247cf260a2d60ec06f8a7b8` |
    | `src/wikidata/dump.py` | `e80bbf872bee70d05bea89eecff0cd99c83bb6f9ce97da63a7cae9b0810d4467` |
    | `src/wikidata/main.py` | `578eeb24bd587fe45e004c8423ff02648c8235b58ddd35714351f9dd2e1ec03c` |
    | `src/wikidata/hub.py` | `7cbbddbea5f8ac38245f8e7c36b7bbfd942adeac7e9378de47858817b82111ff` |
