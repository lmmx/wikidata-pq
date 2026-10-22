"""bench.jsonl as a table: per chunk and code, the median (range) of the claims table's
partition wall and CPU, of the whole partition (all tables), and the process peak RSS."""
import json, statistics as st
runs = [json.loads(l) for l in open("bench.jsonl")]
print("| chunk | code | claims wall (s) | claims CPU (s) | all tables wall (s) | peak RSS (MB) |\n|---|---|---|---|---|---|")
for chunk in dict.fromkeys(r["chunk"] for r in runs):
    for v in ("base", "new"):
        rs = [r["result"] for r in runs if r["chunk"] == chunk and r["variant"] == v]
        ok = [r for r in rs if "error" not in r]
        if not ok:
            print(f"| {chunk} | {v} | killed (out of memory), {len(rs)} of {len(rs)} runs | | | |"); continue
        f = lambda xs: f"{st.median(xs):.2f} ({min(xs):.2f}–{max(xs):.2f})"
        print(f"| {chunk} | {v} | {f([r['tables']['claims'][0] for r in ok])} | {f([r['tables']['claims'][1] for r in ok])} | "
              f"{f([sum(t[0] for t in r['tables'].values()) for r in ok])} | {min(r['maxrss_mb'] for r in ok):,}–{max(r['maxrss_mb'] for r in ok):,} |")
