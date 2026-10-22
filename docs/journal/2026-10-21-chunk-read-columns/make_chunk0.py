"""Chunk 0 of release 20261012, made as wikidata-pq's split and route make it: the dump's
first CHUNK_ENTITIES entity lines (the bz2 dump streamed, not downloaded whole),
reshaped by dump.write_chunk, then routed by dump.route_chunk (scholarly works apart).
Usage: python make_chunk0.py <out_dir>"""
import bz2, sys, time
from pathlib import Path
from wikidata import dump
RELEASE = "20261012"
out = Path(sys.argv[1]); (out / dump.ROUTED).mkdir(parents=True, exist_ok=True)
url = f"{dump.DUMPS_URL}/{RELEASE}/{dump.dump_name(RELEASE)}"
dec, lines, buf, read, t0 = bz2.BZ2Decompressor(), [], b"", 0, time.time()
with dump._open(url, timeout=120) as r:
    while len(lines) < dump.CHUNK_ENTITIES:
        block = r.read(1 << 20); read += len(block)
        buf += dec.decompress(block)
        *done, buf = buf.split(b"\n")
        lines += [l.rstrip(b",\r") for l in done if l.startswith(b"{")]
lines = lines[:dump.CHUNK_ENTITIES]
print(f"read {read/1e6:.1f} MB compressed, {sum(map(len, lines))/1e6:.1f} MB JSON, {len(lines)} entities in {time.time()-t0:.0f}s", flush=True)
entry = dump.write_chunk(0, lines, out)
print({k: v for k, v in entry.items() if k != "fields"})
print(dump.route_chunk(entry, out, out / "scholar"))
