"""Compare two pipeline_bench.py KEEP outputs, whose groups may close at different chunks:
for each table and partition key, the key's group files read in group order hold the same
rows in the same order (compact._fingerprint), with the same Arrow schema and metadata;
quarantine files byte-identical; audit sidecars equal but for the run directory in `path`.

Usage: keep_parity.py <src> <a.keep> <b.keep>"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import polars as pl  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from wikidata.compact import _fingerprint  # noqa: E402


def keys(root: Path) -> dict:
    """(table, key) -> its group files, in group order."""
    out = defaultdict(list)
    for p in sorted((root / "staging").rglob("*.parquet")):
        group, table, key = p.parts[-4], p.parts[-3], p.parts[-2]
        out[(table, key)].append((group, p))
    return {k: [p for _, p in sorted(v)] for k, v in out.items()}


a, b = Path(sys.argv[2]), Path(sys.argv[3])
ka, kb = keys(a), keys(b)
print(f"table/key pairs: {len(ka)} and {len(kb)}, same set: {set(ka) == set(kb)}")
groups = lambda r: sorted({p.name for p in (r / "staging").iterdir()})
print(f"groups: {groups(a)} vs {groups(b)}")
bad = 0
by_table = defaultdict(int)
for k in sorted(ka):
    fa, fb = ka[k], kb.get(k, [])
    same = (
        bool(fb)
        and _fingerprint(pl.scan_parquet(fa)) == _fingerprint(pl.scan_parquet(fb))
        and pq.read_schema(fa[0]).equals(pq.read_schema(fb[0]), check_metadata=True)
    )
    by_table[k[0]] += 1
    if not same:
        bad += 1
        print("DIFFERS", k)
print("keys compared per table:", dict(by_table))
for sub in ("quarantine",):
    qa = {p.relative_to(a): p.read_bytes() for p in (a / sub).rglob("*.parquet")} if (a / sub).exists() else {}
    qb = {p.relative_to(b): p.read_bytes() for p in (b / sub).rglob("*.parquet")} if (b / sub).exists() else {}
    print(f"{sub}: {len(qa)} files, byte-identical: {qa == qb}")
strip = lambda d: d.with_columns(pl.col("path").str.replace(r"^.*/results/", "results/"))
sa = sorted((a / "audit").rglob("*.parquet"))
same_audit = all(strip(pl.read_parquet(p)).equals(strip(pl.read_parquet(b / p.relative_to(a)))) for p in sa)
print(f"audit sidecars: {len(sa)}, equal but for run paths: {same_audit}")
print(f"keys differing: {bad}")
