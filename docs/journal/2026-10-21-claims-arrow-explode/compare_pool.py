"""Each six-worker run's partition outputs against the single-process new code's
(runs/<chunk>/new/r1): partition files byte for byte, and audit sidecars as frames with
their paths taken from the table directory on (the run dirs differ). s90's second copy,
chunk 91, is compared with s90."""
import json
from pathlib import Path
import polars as pl
SETS = {"main": ("20261012", {0: "c0", 10: "r10", 30: "r30", 50: "r50", 70: "r70", 90: "r90"}),
        "scholar": ("20261012-scholar", {10: "s10", 30: "s30", 50: "s50", 70: "s70", 90: "s90", 91: "s90"})}
def rel(p: str, root: str) -> str:
    return ("/" + p).split(f"/{root}/", 1)[1]
for line in open("pool_bench.jsonl"):
    r = json.loads(line)
    if "error" in r["result"]:
        print(f"{r['set']} r{r['round']}: run failed"); continue
    set_name, labels = SETS[r["set"]]
    base = Path(r["result"]["run"]) / "releases" / set_name
    bad, n = [], 0
    for idx, label in labels.items():
        single = Path("runs") / label / "new" / "r1"
        stem, sstem = f"chunk_{idx}", f"chunk_{90 if idx == 91 else idx}"
        mine = {str(p.relative_to(base / "results")).replace(stem + ".parquet", "X"): p for p in (base / "results").rglob(f"{stem}.parquet")}
        theirs = {str(p.relative_to(single / "out")).replace(sstem + ".parquet", "X"): p for p in (single / "out").rglob(f"{sstem}.parquet")}
        if mine.keys() != theirs.keys():
            bad.append(f"{label}: file sets differ ({len(mine)} vs {len(theirs)})"); continue
        bad += [f"{label}: {k}" for k in mine if mine[k].read_bytes() != theirs[k].read_bytes()]
        n += len(mine)
        for tdir in (base / "audit").iterdir():
            a, b = pl.read_parquet(tdir / f"{stem}.parquet"), pl.read_parquet(single / "audit" / tdir.name / f"{sstem}.parquet")
            norm = lambda df, root, s: df.with_columns(pl.col("path").map_elements(lambda p: rel(p, root).replace(s, "X"), return_dtype=pl.String))
            if not norm(a, "results", stem).equals(norm(b, "out", sstem)):
                bad.append(f"{label}: sidecar {tdir.name}")
            n += 1
    print(f"{r['set']} r{r['round']}: {n} files and sidecars, " + ("all identical" if not bad else f"DIFFER: {bad[:6]}"))
