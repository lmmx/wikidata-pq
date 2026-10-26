"""A Parquet file's pages, footer and page index, read from its bytes (stdlib only): a
minimal Thrift compact-protocol reader for the structs parquet.thrift defines
(FileMetaData, ColumnChunk, PageHeader, OffsetIndex), enough to find every page's byte
range and first row.

- `pages(data)`: per column chunk, its pages as (offset, length incl. header, first row,
  kind), from each chunk's OffsetIndex when written (the published files: the pipeline
  writes a page index) and its dictionary page from the page headers; else from walking
  the chunk's page headers.
- `footer_span(data)`: the footer's byte range (FileMetaData, its length, `PAR1`).
- `index_span(data)`: the page index's byte range (column and offset indexes, written
  between the last row group and the footer).
"""

import struct

STOP, TRUE, FALSE, I8, I16, I32, I64, DOUBLE, BINARY, LIST, SET, MAP, STRUCT = range(13)


class Reader:
    def __init__(self, buf: bytes, pos: int = 0):
        self.b, self.p = buf, pos

    def byte(self) -> int:
        v = self.b[self.p]
        self.p += 1
        return v

    def varint(self) -> int:
        shift = out = 0
        while True:
            x = self.byte()
            out |= (x & 0x7F) << shift
            if not x & 0x80:
                return out
            shift += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def value(self, t: int):
        if t == TRUE:
            return True
        if t == FALSE:
            return False
        if t == I8:
            return struct.unpack("b", bytes([self.byte()]))[0]
        if t in (I16, I32, I64):
            return self.zigzag()
        if t == DOUBLE:
            v = struct.unpack("<d", self.b[self.p:self.p + 8])[0]
            self.p += 8
            return v
        if t == BINARY:
            n = self.varint()
            v = self.b[self.p:self.p + n]
            self.p += n
            return v
        if t in (LIST, SET):
            h = self.byte()
            n, et = h >> 4, h & 0x0F
            if n == 15:
                n = self.varint()
            if et in (TRUE, FALSE):
                return [self.byte() == 1 for _ in range(n)]
            return [self.value(et) for _ in range(n)]
        if t == MAP:
            n = self.varint()
            if n == 0:
                return {}
            kv = self.byte()
            return {self.value(kv >> 4): self.value(kv & 0x0F) for _ in range(n)}
        if t == STRUCT:
            return self.struct()
        raise ValueError(f"thrift type {t}")

    def struct(self) -> dict:
        out, fid = {}, 0
        while True:
            h = self.byte()
            t = h & 0x0F
            if t == STOP:
                return out
            d = h >> 4
            fid = fid + d if d else self.zigzag()
            out[fid] = self.value(t)


def footer_span(data: bytes) -> tuple[int, int]:
    n = struct.unpack("<I", data[-8:-4])[0]
    return len(data) - 8 - n, len(data)


def metadata(data: bytes) -> dict:
    s, _ = footer_span(data)
    return Reader(data, s).struct()


def pages(data: bytes) -> list[dict]:
    """Every column chunk's pages: {"rg", "col", "pages": [(offset, length, first_row,
    kind)]}, kind "dict" or "data"."""
    md = metadata(data)
    out = []
    for rg_i, rg in enumerate(md.get(4, [])):
        for col_i, cc in enumerate(rg.get(1, [])):
            meta = cc.get(3, {})
            start = meta.get(11) or meta.get(9)  # dictionary_page_offset or data_page_offset
            end = start + meta[7]  # total_compressed_size
            ps = []
            if cc.get(4) is not None:  # the OffsetIndex: data pages' locations and first rows
                oi = Reader(data, cc[4]).struct()
                locs = [(loc[1], loc[2], loc[3]) for loc in oi.get(1, [])]
                if locs and locs[0][0] > start:
                    ps.append((start, locs[0][0] - start, 0, "dict"))
                ps += [(o, n, r, "data") for o, n, r in locs]
            else:
                p = start
                while p < end:
                    r = Reader(data, p)
                    h = r.struct()
                    n = (r.p - p) + h[3]
                    ps.append((p, n, None, "dict" if h[1] == 2 else "data"))
                    p += n
            out.append({"rg": rg_i, "col": col_i, "rows": rg.get(3), "pages": ps})
    return out


def index_span(data: bytes) -> tuple[int, int] | None:
    md = metadata(data)
    lo = hi = None
    for rg in md.get(4, []):
        for cc in rg.get(1, []):
            for off, ln in ((cc.get(4), cc.get(5)), (cc.get(6), cc.get(7))):
                if off is not None:
                    lo = off if lo is None else min(lo, off)
                    hi = off + ln if hi is None else max(hi, off + ln)
    return None if lo is None else (lo, hi)
