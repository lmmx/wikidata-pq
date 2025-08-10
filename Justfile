lint: ty flake

run:
   process-wikidata

flake:
   flake8 src/wikidata --max-line-length=88 --extend-ignore=E203,E501,

ty:
   #!/usr/bin/env bash
   ty check 2> >(grep -v "WARN ty is pre-release software" >&2)

fmt:
   ruff format src/wikidata
