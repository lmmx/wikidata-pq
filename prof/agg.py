"""Summarise a py-spy raw profile (--subprocesses) of bench/pipeline.py: finalise's
samples by worker role and by what they were doing."""
import collections
import sys

RATE = float(sys.argv[2])
ROLES = ["_sort_bucket", "sort_in_memory", "_bucket_source", "_pack_file", "_write_run", "process_chunk"]
PHASES = ["_check_whole", "_check_stable_sort", "_check_order", "_row_hashes", "_bucket_source", "_boundaries",
          "_sort_bucket", "_pack_file", "pack_key", "sort_in_memory", "_read_key", "_read_fragments", "_read_file",
          "_sorted", "_write_file_parallel", "check_compacted_copy", "write_metadata", "replace_local_copy",
          "_write_run", "_check_glued", "_sha256", "compact_table", "process", "partition"]
per = collections.defaultdict(collections.Counter)
seen_names = collections.defaultdict(set)
leaf = collections.Counter()
for line in open(sys.argv[1], errors="replace"):
    stack, _, n = line.rstrip().rpartition(" ")
    n = int(n)
    frames = stack.split(";")
    proc = frames[1][:24] if len(frames) > 1 and frames[1].startswith("process ") else "main"
    names = [f.split(" (")[0] for f in frames]
    seen_names[proc].update(names)
    if "_find_and_load" in names:
        cat = "IMPORT"
    elif names[-1] in ("take", "combine_chunks", "concat_tables", "sort_indices"):
        cat = "arrow." + names[-1]
    else:
        cat = next((p for p in reversed(names) if p in PHASES), "other:" + names[-1])
    per[proc][cat] += n
    leaf[" ".join(frames[-1].split()[:2])[:80]] += n
roles = collections.defaultdict(collections.Counter)
for proc, c in per.items():
    seen = [r for r in ROLES if r in seen_names[proc]] or (["main"] if proc == "main" else ["(other)"])
    role = "+".join(seen)
    roles[role].update(c)
    roles[role]["#procs"] += 1
s = lambda v: f"{v / RATE:8.1f}s"
grand = sum(v for c in roles.values() for k, v in c.items() if k != "#procs")
print(f"all samples {s(grand)}")
for role, c in sorted(roles.items(), key=lambda x: -sum(v for k, v in x[1].items() if k != "#procs")):
    tot = sum(v for k, v in c.items() if k != "#procs")
    print(f"\n== {role}: {c['#procs']} processes, {s(tot)} ({100 * tot / grand:.1f}%)")
    for k, v in c.most_common(14):
        if k != "#procs":
            print(f"  {s(v)} {100 * v / grand:5.1f}%  {k}")
print("\n== top leaves")
for k, v in leaf.most_common(30):
    print(f"  {s(v)} {100 * v / grand:5.1f}%  {k}")
