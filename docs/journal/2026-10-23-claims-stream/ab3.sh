#!/usr/bin/env bash
# Three-arm pipeline A/B with disk accounting. Each round runs every arm once, in an order
# shuffled afresh, primer.py before each. Per run: pipeline_bench.py's line, then the peak
# rise in the disk's used space (sampled every 0.2 s), the bytes the disk device wrote
# (/proc/diskstats), and the temporary directories (/tmp/tmp*) made during the run that
# are left after it.
# SCRATCH holds the trees (<tree>/src), the envs, cmp/chunks11.txt, cmp/primer.py and
# c5/pipeline_bench.py (docs/journal/2026-10-23-merge-glue/pipeline_bench.py).
# Usage: ab3.sh <tag> <rounds> <arm=tree:env>...
S=${SCRATCH:?}; tag=$1; rounds=$2; shift 2; arms=("$@")
cd "$(dirname "$0")"; mkdir -p $tag
files=$(awk '{print $2}' $S/cmp/chunks11.txt)
dev=$(lsblk -no PKNAME "$(df --output=source / | tail -1)" 2>/dev/null); dev=${dev:-vda}
for round in $(seq 1 $rounds); do
  mapfile -t order < <(python3 -c "import random, sys; v = sys.argv[2:]; random.Random(sys.argv[1]).shuffle(v); print('\n'.join(v))" "$tag-$round" "${arms[@]}")
  for arm in "${order[@]}"; do
    name=${arm%%=*}; spec=${arm#*=}; tree=${spec%%:*}; env=${spec#*:}
    python3 $S/cmp/primer.py 3
    marker=$tag/${name}_r$round.start; touch $marker; done_f=$tag/${name}_r$round.done; rm -f $done_f
    ./disk_peak.sh $done_f > $tag/${name}_r$round.disk & sampler=$!
    w0=$(awk -v d=$dev '$3==d {print $10}' /proc/diskstats)
    printf "r%s %-7s " $round $name
    $S/$env/bin/python $S/c5/pipeline_bench.py $S/$tree/src $tag/${name}_r$round.json 8 6 22 66 $files 2>$tag/${name}_r$round.err | tail -1
    sync; w1=$(awk -v d=$dev '$3==d {print $10}' /proc/diskstats)
    touch $done_f; wait $sampler
    left=$(find /tmp -mindepth 1 -maxdepth 1 -newer $marker -name "tmp*" 2>/dev/null | wc -l)
    echo "    $(cat $tag/${name}_r$round.disk) | device wrote $(( (w1 - w0) * 512 / 1048576 )) MB | /tmp entries left: $left"
    rm -rf $tag/${name}_r$round.run
  done
done
echo done
