"""Column encodings on the bench sample's claims, written by the writers that write them.

- group: each claims group file the split wrote (split_sample.py's parts), rewritten as
  partitioning.core.write_table writes it (genson's writer, which is pq-glue's
  write_columns: zstd 1, adaptive dictionary, spec names, row groups of 122,880). The
  baseline must reproduce each file's size, which shows the emulation is faithful.
- published: each set's claims sorted by id (stable), written as
  compact._write_file_parallel writes them (pq-glue parallel, zstd 3, CDC, sorted by id,
  row groups of sort_by_id.row_group_rows).

For each variant it records the bytes (file, and each column of interest), the write's
CPU and wall, and pyarrow's and Polars' whole-read CPU (median of ROUNDS), and checks the
values read back equal the baseline's.

Usage: encmeasure.py WPQ_SRC PARTS_DIR OUT_DIR [ROUNDS]"""
import json, resource, sys, time
from pathlib import Path
from statistics import median

src, parts, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
sys.path.insert(0, str(src))
import types
stub = types.ModuleType("wikidata_reshape")
stub.__getattr__ = lambda name: None
sys.modules["wikidata_reshape"] = stub
import polars as pl
import pq_glue
import pyarrow as pa
import pyarrow.parquet as pq
from wikidata.sort_by_id import row_group_rows

D = "DELTA_BYTE_ARRAY"
REF = "references.list.element.hash"
QUAL = "qualifiers.list.element.value.list.element.hash"
SNAK = "references.list.element.snaks.list.element.value.list.element.hash"
VARIANTS = {
    "master": {},
    "statement_id": {"statement_id": D},
    "statement_id+ref_hash": {"statement_id": D, REF: D},
    "statement_id+qual_hash": {"statement_id": D, QUAL: D},
    "statement_id+snak_hashes": {"statement_id": D, SNAK: D, QUAL: D},
    "statement_id+id": {"statement_id": D, "id": D},
    "statement_id_dlba": {"statement_id": "DELTA_LENGTH_BYTE_ARRAY"},
}
COLUMNS = ["statement_id", "id", REF, QUAL, SNAK]


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def column_bytes(path) -> dict:
    md = pq.read_metadata(path)
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    out = {}
    for c in COLUMNS:
        if c in names:
            i = names.index(c)
            out[c] = sum(md.row_group(g).column(i).total_compressed_size for g in range(md.num_row_groups))
    return out


def run(kind, tables, write):
    """Each variant's figures for `tables`, as `write(table, dst, encoding)` writes them."""
    dst = out / f"{kind}.parquet"
    base = None
    for name, enc in VARIANTS.items():
        leaves = None
        wc, ww, rpa, rpl = [], [], [], []
        size, cols = 0, {}
        for r in range(rounds):
            size, cols = 0, {}
            for i, t in enumerate(tables):
                if leaves is None:
                    write(t, dst, {})
                    md = pq.read_metadata(dst)
                    leaves = {md.schema.column(j).path for j in range(md.num_columns)}
                e = {c: v for c, v in enc.items() if c in leaves}
                c0, w0 = cpu(), time.perf_counter()
                write(t, dst, e)
                wc.append(cpu() - c0); ww.append(time.perf_counter() - w0)
                size += dst.stat().st_size
                for c, n in column_bytes(dst).items():
                    cols[c] = cols.get(c, 0) + n
                c0 = cpu(); got = pq.read_table(dst); rpa.append(cpu() - c0)
                c0 = cpu(); pl.read_parquet(dst); rpl.append(cpu() - c0)
                if r == 0:
                    assert got.equals(t), (kind, name, i)
        n = len(tables)
        row = {"kind": kind, "variant": name, "mb": round(size / 1e6, 3),
               "cols_mb": {c: round(v / 1e6, 3) for c, v in cols.items()},
               # per write summed over the tables, median over the rounds
               "write_cpu": round(sum(sorted(wc[i::n])[len(wc[i::n]) // 2] for i in range(n)), 3),
               "write_wall": round(sum(sorted(ww[i::n])[len(ww[i::n]) // 2] for i in range(n)), 3),
               "read_cpu_pa": round(sum(sorted(rpa[i::n])[len(rpa[i::n]) // 2] for i in range(n)), 3),
               "read_cpu_pl": round(sum(sorted(rpl[i::n])[len(rpl[i::n]) // 2] for i in range(n)), 3)}
        if base is None:
            base = row
        row["vs_master"] = round(row["mb"] / base["mb"] - 1, 4)
        print(json.dumps(row), flush=True)
    dst.unlink(missing_ok=True)


out.mkdir(parents=True, exist_ok=True)
group_files = sorted(parts.rglob("claims/*.parquet"))
assert group_files, parts
tables = [pq.read_table(f) for f in group_files]
orig = sum(f.stat().st_size for f in group_files)


def group_write(t, dst, enc):
    pq_glue.write_parquet(t, dst, engine="parallel", row_group_size=122_880, compression="zstd",
                          compression_level=1, column_encoding=enc or None)


# The emulation reproduces the split's group files' sizes
again = 0
for f, t in zip(group_files, tables):
    group_write(t, out / "g.parquet", {})
    again += (out / "g.parquet").stat().st_size
print(json.dumps({"group_files": len(group_files), "orig_mb": orig / 1e6, "rewritten_mb": again / 1e6,
                  "rows": sum(t.num_rows for t in tables)}), flush=True)
run("group", tables, group_write)

# Published: each set's claims, sorted by id (stable)
sets = {}
for f, t in zip(group_files, tables):
    sets.setdefault("scholar" if "scholar" in str(f) else "main", []).append(t)
published = []
for name, ts in sorted(sets.items()):
    t = pa.concat_tables(ts, promote_options="permissive")
    idx = pl.from_arrow(t.select(["id"])).with_row_index("i").sort("id", maintain_order=True)["i"].to_arrow()
    published.append(t.take(idx).combine_chunks())
print(json.dumps({"published_rows": [t.num_rows for t in published]}), flush=True)


def published_write(t, dst, enc):
    rg = row_group_rows(t)
    pq_glue.write_parquet(t, dst, engine="parallel", row_group_size=rg, compression="zstd",
                          compression_level=3, content_defined_chunking=True,
                          sorting_columns=[(t.schema.get_field_index("id"), False, False)],
                          column_encoding=enc or None)


run("published", published, published_write)
