"""Whether two kept runs of pipeline.py (KEEP=1) published the same data, compared by
fingerprint, so a run can be checked against an earlier one's outputs without them.

A run's fingerprint:

- **Published files:** for each table and key, finalise's sorted part files read in
  order: the row count and two sums of position-seeded row hashes (as
  `compact._fingerprint`), so any changed, missing, extra or reordered row changes it, and
  the SHA-256 of the Arrow schema with its metadata. What a release publishes, exactly.
- **Group files:** for each table and partition key, the key's staged files: the row count
  and two sums of row hashes without positions (the same rows, in any order), and the
  schema's SHA-256. Their order is the intermediate layout, which the published files'
  order (each key stably sorted by id) checks wherever it shows: a change in the order
  of an id's rows changes the published files. Whether their order differs is reported.
- **Quarantine:** each file's SHA-256.
- **Audit sidecars:** each column's SHA-256 but those of the partition files' layout:
  `path`, `row_groups` and `file_size`, where the writer put each key's rows (a file per
  key, or a row group per key in one file a table) and how it encoded them. The other
  columns (each key, its rows and its ids' bounds) are compared, in order; the rows each
  sidecar row names are the group files', which are compared key by key, and a sidecar
  naming a key's rows wrongly fails the merge's check of each key's rows against it.

The row hashes are Polars', so fingerprints compare only when made with the same Polars
version, which each holds.

Usage: parity.py fingerprint <run.keep> <out.json>
       parity.py compare <a.json> <b.json>   (prints a JSON report; exits 1 if not the same)
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The partition files' layout (see the docstring)
LAYOUT = {"path", "row_groups", "file_size"}
sha = lambda b: hashlib.sha256(b).hexdigest()


def group_files(root: Path) -> dict:
    """table/key -> its group files, in group order."""
    out = defaultdict(list)
    for p in sorted((root / "staging").rglob("*.parquet")):
        group, table, key = p.parts[-4], p.parts[-3], p.parts[-2]
        out[f"{table}/{key}"].append((group, p))
    return {k: [p for _, p in sorted(v)] for k, v in out.items()}


def fingerprint(root: Path) -> dict:
    import polars as pl
    import pyarrow.parquet as pq

    def rows(files: list[Path], ordered: bool = True) -> list[int]:
        lf = pl.scan_parquet(files)
        if ordered:
            lf = lf.with_row_index("_row")
        row = pl.struct(pl.all())
        df = (
            lf.select(pl.len().alias("n"), row.hash(seed=0).sum().alias("h0"), row.hash(seed=1).sum().alias("h1"))
            .collect(engine="streaming")
        )
        return [int(df["n"][0]), int(df["h0"][0]), int(df["h1"][0])]

    schema = lambda f: sha(pq.read_schema(f).serialize().to_pybytes())
    final = defaultdict(list)
    for p in sorted((root / "final").rglob("*.parquet")):
        final[str(p.parent.relative_to(root / "final"))].append(p)

    sidecars = {}
    for p in sorted((root / "audit").rglob("*.parquet")):
        d = pl.read_parquet(p)
        d = d.drop(c for c in d.columns if c in LAYOUT)
        sidecars[str(p.relative_to(root / "audit"))] = {
            "height": d.height,
            "columns": {c: sha(json.dumps(d[c].to_list(), default=str).encode()) for c in d.columns},
        }
    return {
        "polars": pl.__version__,
        "keys": {k: rows(fs, ordered=False) + [schema(fs[0])] for k, fs in group_files(root).items()},
        "keys_ordered": {k: rows(fs) for k, fs in group_files(root).items()},
        "final": {k: rows(fs) + [schema(fs[0])] for k, fs in final.items()},
        "quarantine": {
            str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted((root / "quarantine").rglob("*.parquet"))
        },
        "sidecars": sidecars,
    }


def compare(a: dict, b: dict) -> dict:
    keys_differing = sorted(k for k in set(a["keys"]) | set(b["keys"]) if a["keys"].get(k) != b["keys"].get(k))
    final_differing = sorted(k for k in set(a["final"]) | set(b["final"]) if a["final"].get(k) != b["final"].get(k))
    order_differing = sorted(
        k for k in set(a["keys_ordered"]) & set(b["keys_ordered"])
        if a["keys_ordered"][k] != b["keys_ordered"][k] and k not in keys_differing
    )
    sidecar_diff = Counter()
    for rel in set(a["sidecars"]) & set(b["sidecars"]):
        x, y = a["sidecars"][rel], b["sidecars"][rel]
        table = rel.split("/")[0]
        if x["height"] != y["height"] or set(x["columns"]) != set(y["columns"]):
            sidecar_diff[f"{table}: shape"] += 1
            continue
        for c in x["columns"]:
            if x["columns"][c] != y["columns"][c]:
                sidecar_diff[f"{table}.{c}"] += 1
    report = {
        "keys": len(b["keys"]),
        "keys_per_table": dict(Counter(k.split("/")[0] for k in b["keys"])),
        "keys_differing": keys_differing,
        "published_keys": len(b["final"]),
        "published_differing": final_differing,
        "keys_reordered": order_differing,
        "quarantine_files": len(b["quarantine"]),
        "quarantine_same": a["quarantine"] == b["quarantine"],
        "sidecars": len(b["sidecars"]),
        "sidecars_same_set": set(a["sidecars"]) == set(b["sidecars"]),
        "sidecar_columns_differing": dict(sidecar_diff),
    }
    report["ok"] = (
        not keys_differing
        and not final_differing
        and len(b["final"]) > 0
        and report["quarantine_same"]
        and report["sidecars_same_set"]
        and not sidecar_diff
    )
    return report


def main() -> None:
    if sys.argv[1] == "fingerprint":
        Path(sys.argv[3]).write_text(json.dumps(fingerprint(Path(sys.argv[2]))))
    else:
        a, b = (json.loads(Path(p).read_text()) for p in sys.argv[2:4])
        report = compare(a, b)
        print(json.dumps(report))
        sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
