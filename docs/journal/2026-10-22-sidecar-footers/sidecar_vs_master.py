"""write_sidecar from a tree against master's, on a chunk's partition output: wall per table
(twice each, alternating) and whether the sidecar files are byte-identical.
Usage: sidecar_vs_master.py <src> <master src> <results dir> <stem>"""
import importlib.util, shutil, sys, time
from pathlib import Path
def load(src, name):
    sys.path.insert(0, src)
    for m in [m for m in sys.modules if m.startswith("wikidata")]: del sys.modules[m]
    from wikidata.partitioning import core
    sys.path.remove(src); return core
new = load(sys.argv[1], "new"); old = load(sys.argv[2], "old")
res, stem = Path(sys.argv[3]), sys.argv[4]
tot = {"old": 0.0, "new": 0.0}; same = True
for tbl in sorted(p.name for p in res.iterdir() if p.is_dir()):
    by = {"labels": "language", "descriptions": "language", "aliases": "language", "links": "site"}.get(tbl, "partition")
    out = {}
    for name, mod in (("old", old), ("new", new)) * 2:
        d = Path(f"/tmp/scratch/cmp/w/sc_{name}"); shutil.rmtree(d, ignore_errors=True)
        t = time.perf_counter(); mod.write_sidecar(by, dst_dir=res / tbl, source=Path(f"{stem}.parquet"), log_dir=d); tot[name] += time.perf_counter() - t
        out[name] = (d / f"{stem}.parquet").read_bytes()
    same &= out["old"] == out["new"]
    print(f"{tbl:14s} identical: {out['old'] == out['new']}")
print(f"all identical: {same}; total (2 runs each) master {tot['old']:.2f}s, this {tot['new']:.2f}s")
