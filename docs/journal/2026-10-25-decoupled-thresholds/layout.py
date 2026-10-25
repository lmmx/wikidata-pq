"""Each published key's files: an ordered fingerprint of its rows (as bench/parity.py),
its schema's hash, and each file's bytes, sha256 and rows a row group."""
import hashlib, json, sys
from pathlib import Path
import polars as pl, pyarrow.parquet as pq

def layout(root: Path) -> dict:
    keys = {}
    for d in sorted({p.parent for p in root.rglob("*.parquet")}):
        fs = sorted(d.glob("*.parquet"))
        lf = pl.scan_parquet(fs).with_row_index("_row")
        fp = lf.select(pl.len().alias("n"), pl.struct(pl.all()).hash(seed=0).sum().alias("h0"),
                       pl.struct(pl.all()).hash(seed=1).sum().alias("h1")).collect(engine="streaming").row(0)
        schema = hashlib.sha256(pq.read_schema(fs[0]).serialize().to_pybytes()).hexdigest()[:16]
        files = []
        for f in fs:
            m = pq.read_metadata(f)
            files.append({"name": f.name, "bytes": f.stat().st_size, "sha": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
                          "rgs": [m.row_group(g).num_rows for g in range(m.num_row_groups)]})
        keys[str(d.relative_to(root))] = {"fp": list(fp), "schema": schema, "files": files}
    return keys

if __name__ == "__main__":
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    hub = run / "releases/20261012/hub"
    Path(out).write_text(json.dumps({"first": layout(run / "keep/final"), "x3": layout(hub)}))
