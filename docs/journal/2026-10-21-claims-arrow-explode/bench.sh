#!/usr/bin/env bash
# 3 rounds; in each, every chunk partitioned by base and new code (order alternating by round), each in a fresh process
P=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/env/bin/python
cd /tmp/claude-0/-home-user/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/cae; mkdir -p errs; : > bench.jsonl
for round in 1 2 3; do
  if [ $round = 2 ]; then order="new base"; else order="base new"; fi
  while read name f; do
    for v in $order; do
      out=$($P -I partition_one.py proc/$name/out runs/$name/$v/r$round src_$v/src 2> errs/${name}_${v}_r${round}.err) || out='{"error": true}'
      echo "{\"round\": $round, \"chunk\": \"$name\", \"variant\": \"$v\", \"result\": $out}" >> bench.jsonl
    done
  done < chunks.txt
done
