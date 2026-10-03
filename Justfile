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

# Move the scholarly works (src/wikidata/scholarly.py) into releases/{release}-scholar/data
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

# Everything after route-release, in order; resumable, so run it again after an interruption.
# Both sets are processed before either is finalised (no chunks left on disk during the
# sorts). claims_labels needs both sets' labels sorted, so the main set's first finalise
# stops before it and the third call finishes it. Promotion refuses an unfinalised set.
release release previous:
   just run-release {{release}} main
   just run-release {{release}} scholar
   just finalise-release {{release}} main
   just finalise-release {{release}} scholar
   just finalise-release {{release}} main
   just promote-release {{release}} {{previous}} main
   just promote-release {{release}} {{previous}} scholar

# Make a published SAE run (e.g. v1) the Space's default, and upload runs.json
default-run run:
   RUN={{run}} sae/default.sh

gdb-run:
   gdb -ex "set confirm off" -ex "run" -ex "bt full" -ex "quit" --args python misc/run_instrumented.py 2>&1 | tee misc/pipeline_gdb.log
