#!/usr/bin/env bash
# A/B of process() on chunk 0: unchanged src against src with the chunk read limited to
# id, entity and sitelinks. 3 processes per side, alternating, 4 runs each (run 0 cold).
cd /tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof
for i in 1 2 3; do
  for side in $( [ $((i%2)) = 1 ] && echo "base proj" || echo "proj base"); do
    WPQ_SRC=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/src_$side KEEP_OUT=/tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/out_$side /tmp/claude-0/-home-user-polars-genson/a3c6f4e5-fac5-5911-974c-cd24e5aa55fb/scratchpad/reprof/env/bin/python -I profile_keep.py chunk/routed/chunk_0.parquet work_$side 4 ab_${side}_$i.json 2>&1 | grep -E '^run|Error|Traceback' | sed "s/^/$side $i /"
    rm -rf work_$side
  done
done
echo AB DONE
