"""The published (sorted) files of every key of the four tables, from two groups' files
made both ways (row groups copied one by one, and merged), compared by sha256: each key
sorted in memory by sort_by_id.sort_in_memory from its two group files, as finalise does
(compaction links them; their rows checked as listed). Run with cwd = ROOT."""
import hashlib, json, os, sys
from pathlib import Path
from wikidata.partitioning import fused, rowgroups as r
from wikidata import sort_by_id as s

built = Path(sys.argv[1]).resolve()
groups = [g.split(",") for g in sys.argv[2:]]
root = Path.cwd()
summary = {}
for table in ["labels", "descriptions", "aliases", "links"]:
    files = {}
    for how in ("copy", "merge"):
        for gi, chunks in enumerate(groups):
            paths = [built / c / table / "chunk.parquet" for c in chunks]
            idx = [fused.built_index(p.parent, "chunk.parquet") for p in paths]
            F = r.Footer if how == "copy" else r.ChunkFooter
            footers = [F(p) if k else None for p, k in zip(paths, idx)]
            parts = {}
            for f, keys in zip(footers, idx):
                for key, _, _, _, _, first, n in keys:
                    parts.setdefault(key, []).append((f, first, n))
            name = f"chunks-{gi * 4:04d}-{gi * 4 + 3:04d}.parquet"
            for key, ps in parts.items():
                dst = root / how / "src" / table / key / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if how == "copy":
                    r.copy_row_groups(dst, ps)
                else:
                    r.merge_row_groups(dst, ps, 122_880)
                files.setdefault((how, key), []).append(dst)
    same = n = 0
    for (how, key), srcs in files.items():
        if how != "copy":
            continue
        out = {}
        for h in ("copy", "merge"):
            os.chdir(root / h)
            entries = s.sort_in_memory(table, key, files[(h, key)])
            out[h] = [(e["name"], e["rows"], e["sha256"]) for e in entries]
            os.chdir(root)
        n += 1
        same += out["copy"] == out["merge"]
    summary[table] = {"keys": n, "byte_identical": same}
    print(table, summary[table], flush=True)
print(json.dumps(summary))
