"""Time normalise_from_parquet on one chunk's claims with and without extract_invariants.

Usage: python bench_extract_invariants.py [chunk_prefix] [--keep DIR]   (default chunk_0-00004)
--keep DIR writes the outputs to DIR and leaves them there.
Needs a polars-genson with `extract_invariants` (not yet released) in the running Python.
Each run is a separate subprocess, so peak RSS is per run.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

DATA = Path(__file__).parent / "data/huggingface_hub/philippesaade/wikidata/data"
LABEL_FIELDS = {"labels", "property-labels", "unit-labels"}

WORKER = r"""
import json, resource, sys, time
from pathlib import Path
from polars_genson import normalise_from_parquet
src, mode, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
opts = dict(ndjson=True, map_threshold=0, unify_maps=True,
            force_field_types={"mainsnak": "record", "labels": "map"},
            force_scalar_promotion={"datavalue", "precision", "latitude", "longitude", "labels"},
            no_unify={"qualifiers"}, wrap_root="claims", max_builders=1000,
            output_column="claims", typed=True, keep_columns=["id"])
if mode == "extract":
    opts |= dict(extract_invariants={"labels": "id", "property-labels": "property", "unit-labels": "unit"},
                 lookup_output_path=out_dir / "lookup.parquet")
t = time.perf_counter()
normalise_from_parquet(src, "claims", out_dir / f"{mode}.parquet", **opts)
print(json.dumps({"mode": mode, "time_s": round(time.perf_counter() - t, 2),
                  "peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}))
"""


def diagnose(path: Path) -> None:
    """Describe a Parquet file Polars could not read: trailer, and footer via pyarrow."""
    with path.open("rb") as fh:
        fh.seek(-8, 2)
        trailer = fh.read(8)
    footer_len = int.from_bytes(trailer[:4], "little")
    print(f"    trailer magic {trailer[4:]!r}, footer {footer_len / 1e6:.2f} MB")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("    (pyarrow not installed, skipping footer inspection)")
        return
    try:
        md = pq.read_metadata(path)
    except Exception as e:  # noqa: BLE001 - report whatever pyarrow makes of it
        print(f"    pyarrow cannot read the footer either: {type(e).__name__}: {e}")
        return
    print(
        f"    pyarrow reads it: {md.num_rows} rows, {md.num_row_groups} row groups, "
        f"{md.num_columns} leaf columns, created_by={md.created_by!r}"
    )
    kv = md.metadata or {}
    print("    key/value metadata sizes:", {k.decode(): len(v) for k, v in kv.items()})
    stat_bytes, largest = 0, (0, "")
    for rg in range(md.num_row_groups):
        for c in range(md.num_columns):
            st = md.row_group(rg).column(c).statistics
            if st is None or not st.has_min_max:
                continue
            for v in (st.min, st.max):
                n = len(v) if isinstance(v, (bytes, str)) else 8
                stat_bytes += n
                if n > largest[0]:
                    largest = (n, md.row_group(rg).column(c).path_in_schema)
    print(
        f"    column statistics min/max total {stat_bytes / 1e6:.2f} MB, "
        f"largest single value {largest[0]} bytes in {largest[1]}"
    )


def strip(dt: pl.DataType) -> pl.DataType:
    """The dtype with the label-map fields removed, at any depth."""
    if isinstance(dt, pl.Struct):
        return pl.Struct(
            {f.name: strip(f.dtype) for f in dt.fields if f.name not in LABEL_FIELDS}
        )
    if isinstance(dt, pl.List):
        return pl.List(strip(dt.inner))
    return dt


args = sys.argv[1:]
keep = None
if "--keep" in args:
    i = args.index("--keep")
    keep = Path(args[i + 1])
    del args[i : i + 2]
prefix = args[0] if args else "chunk_0-00004"
(src,) = DATA.glob(f"{prefix}-*.parquet")
print(f"{src.name} ({src.stat().st_size / 1e9:.3f} GB)", flush=True)
with tempfile.TemporaryDirectory() as tmp:
    d = keep or Path(tmp)
    d.mkdir(parents=True, exist_ok=True)
    for mode in ("full", "extract"):
        r = subprocess.run(
            [sys.executable, "-c", WORKER, str(src), mode, str(d)],
            capture_output=True,
            text=True,
            check=False,
        )
        out = r.stdout.strip() or f"{mode}: FAILED rc={r.returncode} {r.stderr[-300:]}"
        print(out, flush=True)
    schemas = {}
    for name in ("full", "extract", "lookup"):
        path = d / f"{name}.parquet"
        if not path.exists():
            print(f"{name}.parquet: missing (its run failed)")
            continue
        size = path.stat().st_size / 1e6
        try:
            schemas[name] = pl.read_parquet_schema(path)
            status = "readable"
        except pl.exceptions.PolarsError as e:  # report every file, even if one fails
            status = f"UNREADABLE: {type(e).__name__}: {e}"
        print(f"{name}.parquet: {size:.1f} MB, {status}")
        if name not in schemas and path.exists():
            diagnose(path)
    if "full" in schemas and "extract" in schemas:
        match = strip(schemas["full"]["claims"]) == schemas["extract"]["claims"]
        print("dtype matches (full minus label fields):", match)
    if "lookup" in schemas:
        lookup = pl.read_parquet(d / "lookup.parquet")
        print("lookup rows:", json.dumps(dict(lookup.group_by("field").len().rows())))
    if keep:
        print(f"outputs kept in {d}")
