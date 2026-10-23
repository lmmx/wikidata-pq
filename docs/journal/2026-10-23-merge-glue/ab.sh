#!/usr/bin/env bash
# Full-pipeline A/B (pipeline_bench.py): each tree once per round, in an order shuffled
# afresh each round, primer.py before each run. Usage: ab.sh <tag> <rounds> <n> <group> <tree>...
S=${SCRATCH:?set SCRATCH to a directory holding envB/ (the Python env), cmp/chunks11.txt (chunk files) and cmp/primer.py}
tag=$1; rounds=$2; n=$3; group=$4; shift 4; trees=("$@")
cd $S/c5; mkdir -p $tag
files=$(awk '{print $2}' $S/cmp/chunks11.txt)
for round in $(seq 1 $rounds); do
  mapfile -t order < <(python3 -c "import random, sys; v = sys.argv[2:]; random.Random(sys.argv[1]).shuffle(v); print('\n'.join(v))" "$tag-$round" "${trees[@]}")
  for tree in "${order[@]}"; do
    python3 $S/cmp/primer.py 3
    printf "r%s %-10s " $round $tree
    $S/envB/bin/python $S/c5/pipeline_bench.py $S/$tree/src $tag/${tree}_r$round.json 8 6 $group $n $files 2>$tag/${tree}_r$round.err | tail -1
  done
done
echo done
