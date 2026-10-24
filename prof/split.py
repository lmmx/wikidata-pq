"""The dump split's steps on a real slice of the dump (its first 48 MB of bz2, about a
chunk of entities), each timed in user CPU: decompression (bz2, and the gz dump's slice),
orjson, the reshape, the chunk's write at zstd 9, 3 and 1, and routing's read and test."""
import resource
import subprocess
import sys
import time
from pathlib import Path

import orjson
import polars as pl

from wikidata import dump
from wikidata.scholarly import is_scholarly

d = Path(sys.argv[1])


def user():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + c.ru_utime


for name, cmd in (("bz2", ["bzip2", "-dc", str(d / "head.json.bz2")]), ("gz", ["gzip", "-dc", str(d / "head.json.gz")])):
    c = user()
    out = subprocess.run(cmd, capture_output=True).stdout
    print(f"decompress {name}: user {user() - c:.2f}s for {len(out) / 1e6:.0f} MB out, "
          f"{len(out) / 1e6 / max(user() - c, 1e-9):.0f} MB/s", flush=True)
    if name == "bz2":
        raw = out
lines = [l.rstrip(b",") for l in raw.splitlines() if l not in (b"[", b"]") and l.strip()][:-1]
mb = sum(map(len, lines)) / 1e6
print(f"entities {len(lines)}, {mb:.0f} MB of JSON", flush=True)
c = user(); ents = [orjson.loads(l) for l in lines]; print(f"orjson.loads: {user() - c:.2f}s", flush=True)
m, f = [0], {"entity": set(), "sitelink": set()}
c = user(); rows = [dump.entity_row(e, m, f) for e in ents]; print(f"reshape + dumps: {user() - c:.2f}s", flush=True)
frame = pl.DataFrame(rows, schema={k: pl.String for k in dump.COLUMNS}, orient="row")
for lvl in (9, 3, 1):
    p = d / f"c{lvl}.parquet"
    c, w = user(), time.perf_counter()
    frame.write_parquet(p, compression="zstd", compression_level=lvl)
    print(f"write zstd {lvl}: user {user() - c:.2f}s wall {time.perf_counter() - w:.2f}s size {p.stat().st_size / 1e6:.1f} MB", flush=True)
c = user(); fr = pl.read_parquet(d / "c9.parquet"); print(f"route read: {user() - c:.2f}s", flush=True)
c = user(); fr.select(is_scholarly(pl.col("claims"))).to_series(); print(f"route is_scholarly: {user() - c:.2f}s", flush=True)
