#!/usr/bin/env bash
# 3 rounds: the main set (chunk 0, r10-r90) and the scholarly set (s10-s90, and s90 again as chunk 91), 6 workers each
P=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/env/bin/python
cd /tmp/claude-0/-home-user/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/cae; : > pool_bench.jsonl; mkdir -p errs
MAIN="chunk_0=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/chunk/chunk_0.parquet $(for n in 10 30 50 70 90; do echo -n "chunk_$n=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/chunks/routed/chunk_$n.parquet "; done)"
SCHOLAR="$(for n in 10 30 50 70 90; do echo -n "chunk_$n=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/chunks/scholar/routed/chunk_$n.parquet "; done) chunk_91=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/chunks/scholar/routed/chunk_90.parquet"
for round in 1 2 3; do
  for set in main scholar; do
    if [ $set = main ]; then args=$MAIN; s=0; else args=$SCHOLAR; s=1; fi
    out=$(WIKIDATA_WORKERS=6 $P -I pool6.py pool/$set/r$round src_typed/src $s $args 2> errs/pool_${set}_r$round.err | tail -1)
    case "$out" in "{"*) ;; *) out='{"error": true}' ;; esac
    echo "{\"round\": $round, \"set\": \"$set\", \"result\": $out}" >> pool_bench.jsonl
  done
done
