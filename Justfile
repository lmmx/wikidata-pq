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

run-release release:
   WIKIDATA_RELEASE={{release}} process-wikidata

finalise-release release:
   WIKIDATA_RELEASE={{release}} finalise-wikidata

# Tag main's current files `previous` (20260507 the first time), then make the release main
promote-release release previous:
   WIKIDATA_RELEASE={{release}} WIKIDATA_PREVIOUS_RELEASE={{previous}} promote-release

# Make a published SAE run (e.g. v1) the Space's default, and upload runs.json
default-run run:
   RUN={{run}} sae/default.sh

gdb-run:
   gdb -ex "set confirm off" -ex "run" -ex "bt full" -ex "quit" --args python misc/run_instrumented.py 2>&1 | tee misc/pipeline_gdb.log
