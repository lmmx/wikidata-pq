"""The files' own pages recompressed at other zstd levels: each column chunk's pages read
from the file (their Thrift headers parsed), each page's body decompressed and compressed
again at each level, as the writer that wrote it would compress the same page bytes
(genson and pq-glue: arrow-rs, a page at a time; the fused writer: one zstd call a page).
Encoding, dictionaries, page boundaries (content-defined chunking's among them) are the
file's own, so only the compression changes. A data page v2's level bytes are not
compressed and are kept as they are.

Output, a JSON line per table and level: the files' bytes with every page at that level
(their headers' size change ignored: a few bytes a page), the compress CPU time of every
page at that level (one thread), and the decompress CPU time.

Usage: pages.py ROOT KIND(staging|final) LEVELS [ROUNDS]
"""

import json
import resource
import sys
from pathlib import Path
from statistics import median

import pyarrow as pa
import pyarrow.parquet as pq


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def varint(b: bytes, i: int) -> tuple[int, int]:
    n = s = 0
    while True:
        x = b[i]
        i += 1
        n |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return n, i


def zz(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def skip(b: bytes, i: int, t: int) -> int:
    """Skip a compact-protocol value of type t at i."""
    if t in (1, 2):  # bool true/false (in a struct field, no payload)
        return i
    if t == 3:
        return i + 1
    if t in (4, 5, 6):
        return varint(b, i)[1]
    if t == 7:
        return i + 8
    if t == 8:
        n, i = varint(b, i)
        return i + n
    if t in (9, 10):
        h = b[i]
        i += 1
        n, et = h >> 4, h & 0x0F
        if n == 15:
            n, i = varint(b, i)
        for _ in range(n):
            i = skip(b, i, et) if et not in (1, 2) else i + 1
        return i
    if t == 12:
        return struct(b, i, {})[1]
    raise ValueError(f"type {t}")


def struct(b: bytes, i: int, want: dict) -> tuple[dict, int]:
    """Parse a struct, returning the i32 fields in `want` (id: name) and nested structs
    named in `want` as (id: (name, subwant))."""
    out, last = {}, 0
    while True:
        h = b[i]
        i += 1
        if h == 0:
            return out, i
        d, t = h >> 4, h & 0x0F
        if d:
            fid = last + d
        else:
            v, i = varint(b, i)
            fid = zz(v)
        last = fid
        w = want.get(fid)
        if w is not None and t == 5:
            v, i = varint(b, i)
            out[w] = zz(v)
        elif isinstance(w, tuple) and t == 12:
            out[w[0]], i = struct(b, i, w[1])
        else:
            i = skip(b, i, t)


HEADER = {1: "type", 2: "uncompressed", 3: "compressed",
          8: ("v2", {5: "def_len", 6: "rep_len"})}


def pages(path: Path):
    """Every page's (body, levels prefix length) of the file."""
    data = path.read_bytes()
    m = pq.read_metadata(path)
    for g in range(m.num_row_groups):
        rg = m.row_group(g)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            start = cc.dictionary_page_offset if cc.has_dictionary_page and cc.dictionary_page_offset else cc.data_page_offset
            end = start + cc.total_compressed_size
            i = start
            while i < end:
                h, j = struct(data, i, HEADER)
                body = data[j:j + h["compressed"]]
                lv = 0
                if "v2" in h:
                    lv = h["v2"].get("def_len", 0) + h["v2"].get("rep_len", 0)
                yield body, lv, h["uncompressed"]
                i = j + h["compressed"]


def main() -> None:
    root, kind = Path(sys.argv[1]), sys.argv[2]
    levels = [int(x) for x in sys.argv[3].split(",")]
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    by: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*.parquet")):
        t = p.relative_to(root).parts[1 if kind == "staging" else 0]
        by.setdefault(t, []).append(p)
    dec = pa.Codec("zstd")
    for t, files in sorted(by.items()):
        raw = []  # (levels prefix, uncompressed body)
        size = sum(f.stat().st_size for f in files)
        comp = 0
        c0 = cpu()
        for f in files:
            for body, lv, unc in pages(f):
                comp += len(body)
                raw.append((body[:lv], dec.decompress(body[lv:], unc - lv, asbytes=True)))
        dcpu = cpu() - c0
        overhead = size - comp  # headers, footers
        for lv in levels:
            codec = pa.Codec("zstd", compression_level=lv)
            times, total = [], 0
            for _ in range(rounds):
                total = 0
                c0 = cpu()
                packed = []
                for pre, body in raw:
                    z = codec.compress(body, asbytes=True)
                    total += len(pre) + len(z)
                    packed.append(z)
                times.append(cpu() - c0)
            c0 = cpu()
            for (pre, body), z in zip(raw, packed):
                dec.decompress(z, len(body), asbytes=True)
            rd = cpu() - c0
            print(json.dumps({"kind": kind, "table": t, "files": len(files), "pages": len(raw),
                              "orig_mb": round(size / 1e6, 4), "level": lv,
                              "mb": round((overhead + total) / 1e6, 4),
                              "compress_cpu": round(median(times), 3), "decompress_cpu": round(rd, 3),
                              "orig_decompress_cpu": round(dcpu, 3)}), flush=True)


if __name__ == "__main__":
    main()
