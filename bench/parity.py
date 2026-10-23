"""Whether two kept runs of pipeline.py (KEEP=1) published the same data. Exits 1 if not.

- **Group files:** for each table and partition key, the key's files read in group order
  hold the same rows in the same order (`compact._fingerprint`: the row count and two
  sums of position-seeded row hashes), with the same Arrow schema and schema metadata.
- **Quarantine:** the same files, byte for byte.
- **Audit sidecars:** every column equal, the run directory stripped from `path`, but
  `file_size`: a partition file's bytes depend on how its writer laid out the same rows.

Prints a JSON report on the last line. Usage: parity.py <src> <a.keep> <b.keep>
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import polars as pl  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from wikidata.compact import _fingerprint  # noqa: E402

MAY_DIFFER = {"file_size"}


def group_files(root: Path) -> dict:
    """(table, key) -> its group files, in group order."""
    out = defaultdict(list)
    for p in sorted((root / "staging").rglob("*.parquet")):
        group, table, key = p.parts[-4], p.parts[-3], p.parts[-2]
        out[(table, key)].append((group, p))
    return {k: [p for _, p in sorted(v)] for k, v in out.items()}


def files(root: Path, sub: str) -> dict:
    return {p.relative_to(root): p.read_bytes() for p in (root / sub).rglob("*.parquet")}


def main() -> None:
    a, b = Path(sys.argv[2]), Path(sys.argv[3])
    ka, kb = group_files(a), group_files(b)
    differing = sorted(
        "/".join(k)
        for k in set(ka) | set(kb)
        if k not in ka
        or k not in kb
        or _fingerprint(pl.scan_parquet(ka[k])) != _fingerprint(pl.scan_parquet(kb[k]))
        or not pq.read_schema(ka[k][0]).equals(pq.read_schema(kb[k][0]), check_metadata=True)
    )
    strip = lambda d: d.with_columns(pl.col("path").str.replace(r"^.*/results/", "results/"))
    sa, sb = (sorted(p.relative_to(r / "audit") for p in (r / "audit").rglob("*.parquet")) for r in (a, b))
    sidecar_diff = Counter()
    for rel in sa if sa == sb else []:
        x, y = strip(pl.read_parquet(a / "audit" / rel)), strip(pl.read_parquet(b / "audit" / rel))
        if x.columns != y.columns or x.height != y.height:
            sidecar_diff[f"{rel.parts[0]}: shape"] += 1
            continue
        for c in x.columns:
            if not x[c].equals(y[c]):
                sidecar_diff[f"{rel.parts[0]}.{c}"] += 1
    report = {
        "keys": len(ka),
        "keys_per_table": dict(Counter(t for t, _ in ka)),
        "keys_differing": differing,
        "quarantine_files": len(files(a, "quarantine")),
        "quarantine_same": files(a, "quarantine") == files(b, "quarantine"),
        "sidecars": len(sa),
        "sidecars_same_set": sa == sb,
        "sidecar_columns_differing": dict(sidecar_diff),
    }
    report["ok"] = (
        not differing
        and report["quarantine_same"]
        and report["sidecars_same_set"]
        and all(k.split(".")[-1] in MAY_DIFFER for k in sidecar_diff)
    )
    print(json.dumps(report))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
