#!/usr/bin/env bash
# 66-chunk pool runs, variants tree:per, order shuffled afresh each round, primer before each.
# Usage: run.sh <tag> <rounds> <n> <tree:per>...
S=/tmp/scratch
tag=$1; rounds=$2; n=$3; shift 3; vs=("$@")
cd $S/cmp/pv; mkdir -p $tag
files=$(awk '{print $2}' $S/cmp/chunks11.txt)
for round in $(seq 1 $rounds); do
  mapfile -t order < <(python3 -c "import random, sys; v = sys.argv[2:]; random.Random(sys.argv[1]).shuffle(v); print('\n'.join(v))" "$tag-$round" "${vs[@]}")
  for v in "${order[@]}"; do
    tree=${v%%:*}; per=${v#*:}
    python3 $S/cmp/primer.py 3
    printf "r%s %-13s per=%-4s " $round $tree $per
    CLEAN=1 $S/envB/bin/python $S/cmp/pool_bench.py $S/$tree/src $tag/${tree}_${per}_r$round.json $per 6 $n $files 2>/dev/null | tail -1
  done
done
echo done
