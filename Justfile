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

# Make a published SAE run (e.g. v1) the Space's default, and upload runs.json
default-run run:
   RUN={{run}} sae/default.sh

gdb-run:
   gdb -ex "set confirm off" -ex "run" -ex "bt full" -ex "quit" --args python misc/run_instrumented.py 2>&1 | tee misc/pipeline_gdb.log
