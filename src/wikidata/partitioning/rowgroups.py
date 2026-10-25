"""A parquet file made of other files' row groups, copied as bytes: no decoding, no
encoding. The group close (push.core._partition_table) makes a key's group file this way
from the row groups the split built for the key in each chunk's one file
(partitioning.fused, its index's spans), in place of a sink that decodes, partitions and
re-encodes them.

The footer is thrift's compact protocol, read here into a tree of fields, kept in order,
and written back with the copied row groups' offsets moved to where they now are. The
sources must share a schema and keep no data outside their row groups (no page indexes,
bloom filters or encryption), as the split's writer (reshape's partition::Writer) writes
them; anything else raises."""

import os
import struct
from pathlib import Path

MAGIC = b"PAR1"
STOP, TRUE, FALSE, BYTE, I16, I32, I64, DOUBLE, BINARY, LIST, SET, MAP, STRUCT = range(13)


def _varint(b: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        c = b[i]
        i += 1
        n |= (c & 0x7F) << shift
        if c < 0x80:
            return n, i
        shift += 7


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _value(b: bytes, i: int, t: int):
    if t in (TRUE, FALSE):  # (a list's bool: a byte each)
        return t == TRUE, i
    if t == BYTE:
        return struct.unpack_from("b", b, i)[0], i + 1
    if t in (I16, I32, I64):
        n, i = _varint(b, i)
        return _zigzag(n), i
    if t == DOUBLE:
        return struct.unpack_from("<d", b, i)[0], i + 8
    if t == BINARY:
        n, i = _varint(b, i)
        return b[i : i + n], i + n
    if t in (LIST, SET):
        h = b[i]
        i += 1
        n, et = h >> 4, h & 0x0F
        if n == 15:
            n, i = _varint(b, i)
        items = []
        for _ in range(n):
            if et in (TRUE, FALSE):
                items.append(b[i] == TRUE)
                i += 1
            else:
                v, i = _value(b, i, et)
                items.append(v)
        return (et, items), i
    if t == STRUCT:
        fields, fid = [], 0
        while True:
            h = b[i]
            i += 1
            if h == STOP:
                return fields, i
            ft, delta = h & 0x0F, h >> 4
            if delta:
                fid += delta
            else:
                n, i = _varint(b, i)
                fid = _zigzag(n)
            v, i = _value(b, i, ft)
            fields.append([fid, ft, v])
    raise ValueError(f"thrift type {t} not in a parquet footer")


def _put_varint(out: bytearray, n: int) -> None:
    while n >= 0x80:
        out.append(n & 0x7F | 0x80)
        n >>= 7
    out.append(n)


def _put(out: bytearray, t: int, v) -> None:
    if t == BYTE:
        out += struct.pack("b", v)
    elif t in (I16, I32, I64):
        _put_varint(out, (v << 1) ^ (v >> 63))
    elif t == DOUBLE:
        out += struct.pack("<d", v)
    elif t == BINARY:
        _put_varint(out, len(v))
        out += v
    elif t in (LIST, SET):
        et, items = v
        if len(items) < 15:
            out.append(len(items) << 4 | et)
        else:
            out.append(0xF0 | et)
            _put_varint(out, len(items))
        for x in items:
            if et in (TRUE, FALSE):
                out.append(TRUE if x else FALSE)
            else:
                _put(out, et, x)
    elif t == STRUCT:
        fid = 0
        for f, ft, x in v:
            wire = (TRUE if x else FALSE) if ft in (TRUE, FALSE) else ft
            if 0 < f - fid <= 15:
                out.append((f - fid) << 4 | wire)
            else:
                out.append(wire)
                _put_varint(out, (f << 1) ^ (f >> 15))
            fid = f
            if ft not in (TRUE, FALSE):
                _put(out, ft, x)
        out.append(STOP)
    else:
        raise ValueError(f"thrift type {t} not in a parquet footer")


def _get(fields: list, fid: int):
    return next((f for f in fields if f[0] == fid), None)


# Fields that would point outside the row groups: page indexes, bloom filters, encryption
_OUTSIDE = {"column chunk": (4, 6, 8, 9), "column meta": (14,), "file": (8, 9)}


class Footer:
    """A file's footer, read once: its fields, row groups and their byte spans."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            f.seek(-8, os.SEEK_END)
            n, magic = struct.unpack("<I4s", f.read(8))
            if magic != MAGIC:
                raise ValueError(f"{path}: not a parquet file")
            f.seek(-8 - n, os.SEEK_END)
            self.fields, _ = _value(f.read(n), 0, STRUCT)
        if any(_get(self.fields, k) for k in _OUTSIDE["file"]):
            raise ValueError(f"{path}: encrypted")
        self.groups = _get(self.fields, 4)[2][1] if _get(self.fields, 4) else []
        self.spans = []  # (start, end) of each row group's bytes
        for g in self.groups:
            start = end = None
            for cc in _get(g, 1)[2][1]:
                if any(_get(cc, k) for k in _OUTSIDE["column chunk"]):
                    raise ValueError(f"{path}: data outside its row groups")
                meta = _get(cc, 3)[2]
                if any(_get(meta, k) for k in _OUTSIDE["column meta"]):
                    raise ValueError(f"{path}: a bloom filter")
                first = min(_get(meta, k)[2] for k in (9, 11) if _get(meta, k))
                size = _get(meta, 7)[2]
                if end is not None and first != end:
                    raise ValueError(f"{path}: a row group's column chunks not contiguous")
                start = first if start is None else start
                end = first + size
            self.spans.append((start, end))

    def rows(self, first: int, n: int) -> int:
        return sum(_get(g, 3)[2] for g in self.groups[first : first + n])


def _moved(group: list, by: int, ordinal: int) -> list:
    """A row group's fields, its offsets moved `by` bytes and its ordinal set."""
    out = []
    for fid, ft, v in group:
        if fid == 1:  # columns
            ccs = []
            for cc in v[1]:
                cc = [list(f) for f in cc]
                for f in cc:
                    if f[0] == 2:  # file_offset
                        f[2] += by
                    elif f[0] == 3:  # meta_data
                        f[2] = [[k, t, x + by if k in (9, 10, 11) else x] for k, t, x in f[2]]
                ccs.append(cc)
            v = (v[0], ccs)
        elif fid == 5:  # file_offset
            v += by
        elif fid == 7:  # ordinal
            continue
        out.append([fid, ft, v])
    if _get(group, 7) is not None:
        out.append([7, I16, ordinal])
    return out


def copy_row_groups(dst: Path, parts: list[tuple[Footer, int, int]]) -> int:
    """Write `dst` from each part's row groups (its footer, first row group, row groups),
    in order, their bytes copied as they are; the first part's footer gives the rest of
    the metadata. Returns the rows written."""
    base = parts[0][0].fields
    schema = _get(base, 2)
    groups, rows = [], 0
    with open(dst, "wb", buffering=0) as out:
        _write(out, MAGIC)
        at = len(MAGIC)
        for footer, first, n in parts:
            if n == 0:
                continue
            if _get(footer.fields, 2) != schema:
                raise ValueError(f"{footer.path}: a schema not the first part's")
            start, end = footer.spans[first][0], footer.spans[first + n - 1][1]
            for g in footer.groups[first : first + n]:
                groups.append(_moved(g, at - start, len(groups)))
                rows += _get(g, 3)[2]
            with open(footer.path, "rb") as src:
                left, off = end - start, start
                while left:
                    copied = os.copy_file_range(src.fileno(), out.fileno(), left, off)
                    if not copied:
                        raise OSError(f"{footer.path}: ended before its row groups")
                    left -= copied
                    off += copied
            at += end - start
        fields = []
        for fid, ft, v in base:
            if fid == 3:
                v = rows
            elif fid == 4:
                v = (STRUCT, groups)
            fields.append([fid, ft, v])
        meta = bytearray()
        _put(meta, STRUCT, fields)
        _write(out, bytes(meta) + struct.pack("<I", len(meta)) + MAGIC)
    return rows


def _write(f, b: bytes) -> None:
    """All of `b` to an unbuffered file, at its position (where copy_file_range left it)."""
    view = memoryview(b)
    while view:
        view = view[os.write(f.fileno(), view) :]
