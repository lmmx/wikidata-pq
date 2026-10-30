lint: ty flake

run:
   process-wikidata

finalise:
   finalise-wikidata

download:
   download-wikidata

cards:
   render-cards

card-stats:
   card-stats

# The /loop coordinator's mode: .claude/loop.md is a copy of one of .claude/loops/
# (releasable: merges and releases; mergeable: merges only, releases left pending)
loop-releasable:
   cp .claude/loops/releasable.md .claude/loop.md && head -1 .claude/loop.md

loop-mergeable:
   cp .claude/loops/mergeable.md .claude/loop.md && head -1 .claude/loop.md

loop-mode:
   @head -1 .claude/loop.md 2>/dev/null || echo "no .claude/loop.md (a heartbeat restores mergeable)"

flake:
   flake8 src/wikidata --max-line-length=88 --extend-ignore=E203,E501,

ty *args:
   #!/usr/bin/env bash
   ty check {{args}} 2> >(grep -v "WARN ty is pre-release software" >&2)

t:
   just ty --output-format=concise

fmt:
   ruff format src/wikidata

# A release: an official Wikidata JSON dump by its date (e.g. 20260928), built into the
# datasets on a branch of each repo, then promoted to main (see src/wikidata/dump.py, hub.py)
latest-dump:
   latest-dump

download-dump release:
   WIKIDATA_RELEASE={{release}} download-dump

split-dump release:
   WIKIDATA_RELEASE={{release}} split-dump

# Both at once: the split follows the download as it is written (each resumable). A
# download that fails stops the split, which would otherwise wait for it
download-split release:
   #!/usr/bin/env bash
   set -euo pipefail
   export WIKIDATA_RELEASE={{release}}
   download-dump & download=$!
   split-dump & split=$!
   wait $download || { kill $split; wait $split || true; exit 1; }
   wait $split

# Number the parts of a split's chunks into each set's chunks (split-dump ends with it), or
# move an older split's scholarly works (src/wikidata/scholarly.py) into
# releases/{release}-scholar/data
route-release release:
   WIKIDATA_RELEASE={{release}} python -c 'from wikidata.dump import run_route; run_route()'

# A release's two sets: `main` (wikidata-{table}) and `scholar` (wikidata-scholar-{table});
# a recipe's `set` becomes WIKIDATA_SCHOLAR
# Process one set's chunks and upload them in groups (deleting each chunk once processed)
run-release release set="main":
   WIKIDATA_RELEASE={{release}} WIKIDATA_SCHOLAR={{ if set == "scholar" { "1" } else if set == "main" { "" } else { error("set is main or scholar") } }} process-wikidata

# Compact and sort one set's tables, build its claims_labels and push its cards
finalise-release release set="main":
   WIKIDATA_RELEASE={{release}} WIKIDATA_SCHOLAR={{ if set == "scholar" { "1" } else if set == "main" { "" } else { error("set is main or scholar") } }} finalise-wikidata

# Tag main's current files `previous` (20260507 the first time), then make the release main
promote-release release previous set="main":
   WIKIDATA_RELEASE={{release}} WIKIDATA_PREVIOUS_RELEASE={{previous}} WIKIDATA_SCHOLAR={{ if set == "scholar" { "1" } else if set == "main" { "" } else { error("set is main or scholar") } }} promote-release

# The download, the split and both sets' runs at once: each set's run follows the split
# (WIKIDATA_FOLLOW_SPLIT=1), processing and uploading its chunks as the split numbers them,
# at nice 10 so as not to slow the split, with WIKIDATA_OVERLAP_WORKERS chunk workers each
# (default 2). Each is resumable: run it again after an interruption. A split or download
# that fails stops the runs, which would otherwise wait for it. The scholarly run's pushes
# wait while the main run closes groups back to back, so the main set's uploads end first;
# once the main run ends, the main set's first finalise (`release`'s first
# finalise-release) runs, at nice 10 beside the scholarly run's last merges, its compute
# under that run's last uploads (WIKIDATA_FINALISE_EARLY=0: neither, both runs ending
# before any finalise). Then `release`, whose runs, and that finalise, find their work done.
# By default (WIKIDATA_FINALISE_CROSS_RUN=0 restores the serial runs), that finalise is the first of the
# release's three (finalise-release-runs: cross_run.py), each run started once the one
# before has written its last table, under its last commits; the scholarly set's once its
# run has ended with 0
overlap-release release:
   #!/usr/bin/env bash
   set -euo pipefail
   export WIKIDATA_RELEASE={{release}} WIKIDATA_FOLLOW_SPLIT=1
   export WIKIDATA_WORKERS=${WIKIDATA_OVERLAP_WORKERS:-2}
   just download-split {{release}} & split=$!
   WIKIDATA_SCHOLAR=1 nice -n 10 process-wikidata & scholar=$!
   WIKIDATA_SCHOLAR= nice -n 10 process-wikidata & main=$!
   wait $split || { kill $scholar $main; wait $scholar $main || true; exit 1; }
   code=0
   wait $main || code=1
   if [ $code = 0 ] && [ "${WIKIDATA_FINALISE_EARLY:-1}" != 0 ] && [ "${WIKIDATA_FINALISE_CROSS_RUN:-1}" != 0 ]; then
     gate=$(mktemp -d)
     WIKIDATA_SCHOLAR= nice -n 10 env -u WIKIDATA_FOLLOW_SPLIT -u WIKIDATA_WORKERS finalise-release-runs --after "$gate/scholar" & fin=$!
     s=0
     wait $scholar || s=1
     echo $s > "$gate/scholar.tmp" && mv "$gate/scholar.tmp" "$gate/scholar"
     [ $s = 0 ] || code=1
     wait $fin || code=1
     rm -rf "$gate"
     exit $code
   fi
   if [ $code = 0 ] && [ "${WIKIDATA_FINALISE_EARLY:-1}" != 0 ]; then
     WIKIDATA_SCHOLAR= nice -n 10 env -u WIKIDATA_FOLLOW_SPLIT -u WIKIDATA_WORKERS finalise-wikidata || code=1
   fi
   wait $scholar || code=1
   exit $code

# Everything after route-release, in order; resumable, so run it again after an interruption.
# Both sets are processed before either is finalised (no chunks left on disk during the
# sorts; after `overlap-release`, the main set's first finalise has run already, beside
# the scholarly run's last groups). claims_labels needs both sets' labels sorted, so the
# main set's first finalise stops before it and the third call finishes it. Promotion
# refuses an unfinalised set. By default (WIKIDATA_FINALISE_CROSS_RUN=0 restores the serial runs), the three
# finalise runs are finalise-release-runs' (cross_run.py): each started once the one before
# has written its last table, under its last commits
release release previous:
   #!/usr/bin/env bash
   set -euo pipefail
   just run-release {{release}} scholar
   just run-release {{release}} main
   if [ "${WIKIDATA_FINALISE_CROSS_RUN:-1}" != 0 ]; then
     WIKIDATA_RELEASE={{release}} WIKIDATA_SCHOLAR= finalise-release-runs
   else
     just finalise-release {{release}} main
     just finalise-release {{release}} scholar
     just finalise-release {{release}} main
   fi
   just promote-release {{release}} {{previous}} main
   just promote-release {{release}} {{previous}} scholar

# The compute benchmark (bench/full/README.md): a release's processing of a real dump on
# this machine, end to end, with every upload and download off the timed path (an
# on-disk stand-in Hub). `just bench-full fetch --release R` caches the dump (timed apart,
# its md5 and sha1 checked), then `just bench-full run --release R` runs it from disk
# (--source stream: fetched as it runs). Scaled: add --prefix-gb N or --windows K to both.
# Not the CI bench (bench/), which it leaves as it is
bench-full *args:
   uv run --frozen python bench/full/run.py {{args}}

# The native bz2 decoder's worker count through the real split (lmmx/mibz2#65;
# bench/bz2threads.py): the CI bench's split of its dump slice once for each count of decoder
# threads (WIKIDATA_BZ2_THREADS), interleaved and paired with the default (auto: every
# logical CPU), the parse workers and writer pools as they are, each split's parity checked.
# The bench data is checked out at ./bench-data first. Results in OUT/summary.md
bench-bz2-threads threads="auto,2,4,6,8,10" out="bench-bz2-threads" *args:
   #!/usr/bin/env bash
   set -euo pipefail
   data=$(sed -n 's/^  DATA_COMMIT: \([0-9a-f]*\).*/\1/p' .github/workflows/bench.yml)
   if [ ! -d bench-data ]; then
     git fetch --quiet origin bench-data
     git worktree add --detach bench-data "$data"
   fi
   (cd bench-data && sha256sum --quiet -c SHA256SUMS)
   uv run --frozen python bench/bz2threads.py --data bench-data --threads {{threads}} --out {{out}} {{args}}

# Then release-shaped, for the counts the sweep picks: bench-full's run (overlapped, from the
# cached dump: `just bench-full fetch --release R --windows 24 --window-mb 25` first) once a
# count a round, paired with auto, each run's published files' digest checked
bench-bz2-threads-full release threads="auto,4" scale="--windows 24 --window-mb 25" out="bench-bz2-threads-full" *args:
   uv run --frozen python bench/bz2threads.py --full "--release {{release}} {{scale}}" --threads {{threads}} --out {{out}} {{args}}

# The cap against 8 threads, where mibz2 plans 4 teams of 2 on 6 or more physical cores
# (docs/journal/2026-10-30-bz2-threads-teams.md, pre-registered): release-shaped, 8 rounds,
# paired with auto. About 30 minutes on the i9. Results in bench-bz2-teams/summary.md
bench-bz2-teams release="20261019":
   just bench-bz2-threads-full {{release}} auto,8 "--windows 24 --window-mb 25" bench-bz2-teams --rounds 8

# A release's claims schema grows as it runs (src/wikidata/release_schema.py): each field a
# chunk shows that it lacks is added, in auto mode (the default). With
# WIKIDATA_SCHEMA_EVOLUTION=strict, a run halts with the fields proposed: accept them as a
# version, then run again to resume
show-schema release:
   WIKIDATA_RELEASE={{release}} python -m wikidata.release_schema show

accept-schema release:
   WIKIDATA_RELEASE={{release}} python -m wikidata.release_schema accept

# After a release, the fields it added written into src/wikidata/schemas
# (claims.promoted.json, claims.avsc), for review as a diff: the next release starts from them
promote-schema release:
   WIKIDATA_RELEASE={{release}} python -m wikidata.release_schema promote
   git diff --stat -- src/wikidata/schemas

# Make a published SAE run (e.g. v1) the Space's default, and upload runs.json
default-run run:
   RUN={{run}} sae/default.sh

gdb-run:
   gdb -ex "set confirm off" -ex "run" -ex "bt full" -ex "quit" --args python misc/run_instrumented.py 2>&1 | tee misc/pipeline_gdb.log

# The docs site (mkdocs.yml), with the packages the site workflow installs (docs/docs-packages.txt)
mkdocs command="serve":
   uvx --from mkdocs --with-requirements docs/docs-packages.txt mkdocs {{command}}
