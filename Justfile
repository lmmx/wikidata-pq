lint: ty flake

run:
   process-wikidata

flake:
   flake8 src/wikidata --max-line-length=88 --extend-ignore=E203,E501,

ty *args:
   #!/usr/bin/env bash
   ty check {{args}} 2> >(grep -v "WARN ty is pre-release software" >&2)

t:
   just ty --output-format=concise

fmt:
   ruff format src/wikidata

gdb-run:
   gdb -ex "set confirm off" -ex "run" -ex "bt full" -ex "quit" --args python run_instrumented.py 2>&1 | tee pipeline_gdb.log
