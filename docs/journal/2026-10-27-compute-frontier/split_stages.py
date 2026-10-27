"""The split's work, stage by stage, on one core's worth of scheduling: the slice decoded
once into memory, then each chunk's `dump.write_chunk` run one after another (not in the
split's thread pool), every function of interest wrapped to take its own process CPU
time (all threads: the reshape's rayon pool and genson's writer pool included) and wall
time, less the wrapped functions it calls. The outputs (each table's Parquet bytes, the
claims' Arrow bytes) are counted for the denominators.

Usage: split_stages.py SRC_DIR SLICE.json.bz2|LINES.jsonl OUT.json

SRC_DIR is the checkout's src/. Run in an empty working directory (it writes the parts
under ./releases). RAYON_NUM_THREADS / the reshape's own thread count are left as the
split runs them unless set in the environment."""

import collections
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

src, slice_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, src)

import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

from wikidata import claims_build, dump, entities_build, release_schema  # noqa: E402
from wikidata.partitioning import fused  # noqa: E402
import wikidata.claims_refs as claims_refs  # noqa: E402

own = collections.defaultdict(lambda: [0.0, 0.0, 0])  # name -> [cpu, wall, calls]
stack: list[list[float]] = []  # each open call's [child cpu, child wall]


def wrap(holder, name, label=None):
    f = getattr(holder, name)
    label = label or f"{getattr(holder, '__name__', holder)}.{name}"

    @functools.wraps(f)
    def g(*a, **kw):
        stack.append([0.0, 0.0])
        c0, w0 = time.process_time(), time.perf_counter()
        try:
            return f(*a, **kw)
        finally:
            c, w = time.process_time() - c0, time.perf_counter() - w0
            child = stack.pop()
            e = own[label]
            e[0] += c - child[0]
            e[1] += w - child[1]
            e[2] += 1
            if stack:
                stack[-1][0] += c
                stack[-1][1] += w

    setattr(holder, name, g)


# The reshape (Rust) and the arrays' assembly
wrap(claims_build, "entity_rows_built", "rust.entity_rows_built (scan, text columns, claims arrays)")
wrap(claims_build, "_arrays", "claims_build._arrays (nodes to pyarrow)")
wrap(claims_build, "reshape_built")
wrap(dump, "_reshape", "dump._reshape (own: frame, large strings)")
wrap(dump, "chunk_census")
# The claims' write
wrap(claims_build, "write_built", "claims_build.write_built (own: table, check_ids)")
wrap(claims_build, "canonical_coordinates")
wrap(claims_build, "statements", "partitioning.claims.statements (flatten to statements)")
wrap(release_schema, "tagged")
wrap(claims_build, "write_table", "genson write_table (claims Parquet encode, zstd)")
wrap(claims_build.pq_glue, "concat", "pq_glue.concat (row-group files glued)")
wrap(claims_refs, "write_refs")
# The terms and links (Rust build and Parquet), the entities, the part's rest
wrap(fused, "build_terms", "fused.build_terms (rust: terms/links Arrow + Parquet zstd)")
wrap(fused, "write_one_file", "fused.write_one_file (own: index, empty cells)")
wrap(entities_build, "write_entities", "entities_build.write_entities (polars)")
wrap(pl.DataFrame, "write_parquet", "polars write_parquet (the part's rest)")
wrap(dump, "route_parts", "dump.route_parts (own)")
wrap(dump, "write_chunk", "dump.write_chunk (own)")

# The decode, timed apart; the text held in memory (or, for a .jsonl of dump lines, the
# lines as they are: no decode)
t0, c0 = time.perf_counter(), time.process_time()
r0 = os.times()
if slice_path.suffix == ".jsonl":
    text = slice_path.read_bytes()
else:
    text = subprocess.run(["lbzip2", "-n1", "-dc", str(slice_path)], capture_output=True, check=True).stdout
r1 = os.times()
decode = {"wall": time.perf_counter() - t0, "cpu_child": r1.children_user + r1.children_system - r0.children_user - r0.children_system}
# Line splitting as _dump_lines does it (a readline loop, rstrip, startswith)
import io  # noqa: E402

c0, w0 = time.process_time(), time.perf_counter()
lines = []
for line in io.BufferedReader(io.BytesIO(text), buffer_size=16 << 20):
    line = line.rstrip(b",\r\n")
    if line.startswith(b"{"):
        lines.append(line)
split_lines = {"cpu": time.process_time() - c0, "wall": time.perf_counter() - w0}
json_bytes = len(text)
line_bytes = sum(map(len, lines))
del text

root = Path("releases")
sets = {"main": root / "20261012" / "data", "scholar": root / "20261012-scholar" / "data"}
for d in sets.values():
    (d / dump.ROUTED).mkdir(parents=True, exist_ok=True)
claims_arrow_bytes = 0
orig_reshape_built = claims_build.reshape_built


def counting(lines_):
    global claims_arrow_bytes
    r = orig_reshape_built(lines_)
    if r is not None:
        claims_arrow_bytes += sum(c.nbytes for c in r[1].claims.values())
    return r


claims_build.reshape_built = counting
c0, w0 = time.process_time(), time.perf_counter()
chunks = 0
for i in range(0, len(lines), dump.CHUNK_ENTITIES):
    dump.write_chunk(chunks, lines[i : i + dump.CHUNK_ENTITIES], *sets.values())
    chunks += 1
total = {"cpu": time.process_time() - c0, "wall": time.perf_counter() - w0}

out_bytes = collections.Counter()
for name, d in sets.items():
    for f in (d / dump.ROUTED).rglob("*.parquet"):
        rel = f.relative_to(d / dump.ROUTED)
        table = rel.parts[0] if len(rel.parts) > 1 else "part"
        out_bytes[table] += f.stat().st_size

result = {
    "bz2_bytes": slice_path.stat().st_size,
    "json_bytes": json_bytes,
    "line_bytes": line_bytes,
    "lines": len(lines),
    "chunks": chunks,
    "decode_lbzip2_n1": decode,
    "split_lines_python": split_lines,
    "write_chunks_total": total,
    "claims_arrow_bytes": claims_arrow_bytes,
    "parquet_bytes": dict(out_bytes),
    "own": {k: {"cpu": round(v[0], 3), "wall": round(v[1], 3), "calls": v[2]} for k, v in
            sorted(own.items(), key=lambda kv: -kv[1][0])},
}
out_path.write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
