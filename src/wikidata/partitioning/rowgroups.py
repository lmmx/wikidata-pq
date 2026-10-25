"""A parquet file made of other files' row groups, copied as bytes: no decoding, no
encoding. The group close (push.core._partition_table) makes a key's group file this way
from the row groups the split built for the key in each chunk's one file
(partitioning.fused, its index's spans), in place of a sink that decodes, partitions and
re-encodes them.

Two ways, both of encoded bytes only:

- `merge_row_groups` (the close's): consecutive row groups, up to a number of rows, merged
  into one at the column-chunk level. Each column's pages are copied one source after
  another under the first source's dictionary page (the split's writer gives only the key
  column a dictionary, of its one value, so every source's is the same bytes: checked),
  and its metadata summed (values, sizes) or bounded (min, max). A key of a group of
  hundreds of chunks would otherwise have a row group a chunk, mostly of a few rows, each
  with its footer entry (~200-300 bytes) and its own reads.
- `copy_row_groups`: each row group copied as it is, with its own footer entry. For
  footers `merge_row_groups` does not read (`ChunkFooter` raises Unsupported).

The footer is thrift's compact protocol. `Footer` reads it into a tree of fields, kept in
order, written back with the copied row groups' offsets moved to where they now are.
`ChunkFooter` reads only the fields the split's writer (reshape's partition::Writer) writes,
in one pass, about ten times faster, and anything else raises Unsupported. The sources
must share a schema and keep no data outside their row groups (no page indexes, bloom
filters or encryption), as the split's writer writes them; anything else raises."""

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


# Merging row groups at the column-chunk level (merge_row_groups)


class Unsupported(ValueError):
    """A footer with a field ChunkFooter does not read: its row groups are copied as they
    are (copy_row_groups), not merged."""


def _uvarint(b: bytes, i: int) -> tuple[int, int]:
    c = b[i]
    if c < 0x80:
        return c, i + 1
    n, shift, i = c & 0x7F, 7, i + 1
    while True:
        c = b[i]
        i += 1
        n |= (c & 0x7F) << shift
        if c < 0x80:
            return n, i
        shift += 7


def _header(b: bytes, i: int, fid: int) -> tuple[int, int, int]:
    """A struct's next field header: (field id, type, position after it); type STOP at
    the struct's end."""
    h = b[i]
    i += 1
    if h == STOP:
        return fid, STOP, i
    if h >> 4:
        return fid + (h >> 4), h & 0x0F, i
    n, i = _uvarint(b, i)
    return _zigzag(n), h & 0x0F, i


def _list_header(b: bytes, i: int) -> tuple[int, int, int]:
    """A list's (size, element type, position after its header)."""
    h = b[i]
    i += 1
    n = h >> 4
    if n == 15:
        n, i = _uvarint(b, i)
    return n, h & 0x0F, i


# The integer fields read, by struct, and their types
_COLUMN_INTS = {1: I32, 4: I32, 5: I64, 6: I64, 7: I64, 9: I64, 11: I64}
_GROUP_INTS = {2: I64, 3: I64, 5: I64, 6: I64, 7: I16}


class Column:
    """A column chunk's metadata, as merge_row_groups needs it: its bytes `start` (its
    dictionary page's, if any, else its first data page's) to `end`."""

    __slots__ = ("type", "encodings", "path", "codec", "values", "unc", "comp", "data", "dictionary",
                 "nulls", "min", "max", "min_exact", "max_exact", "start", "end", "_end")

    def __init__(self, b: bytes, i: int) -> None:
        self.dictionary = self.nulls = self.min = self.max = None
        self.min_exact = self.max_exact = False
        ints, fid = {}, 0
        while True:
            h = b[i]  # (_header and _uvarint inlined: the close reads every chunk's footer)
            i += 1
            if h == STOP:
                break
            t = h & 0x0F
            if h >> 4:
                fid += h >> 4
            else:
                n, i = _uvarint(b, i)
                fid = _zigzag(n)
            if _COLUMN_INTS.get(fid) == t:
                n = b[i]
                if n < 0x80:
                    i += 1
                else:
                    n, i = _uvarint(b, i)
                ints[fid] = (n >> 1) ^ -(n & 1)
            elif fid == 2 and t == LIST:  # encodings
                n, et, i = _list_header(b, i)
                encodings = []
                for _ in range(n):
                    v, i = _uvarint(b, i)
                    encodings.append(_zigzag(v))
                self.encodings = tuple(encodings)
            elif fid == 3 and t == LIST:  # path in schema, kept as its encoded bytes
                start = i
                n, et, i = _list_header(b, i)
                for _ in range(n):
                    v, i = _uvarint(b, i)
                    i += v
                self.path = b[start:i]
            elif fid == 12 and t == STRUCT:
                i = self._statistics(b, i)
            else:
                raise Unsupported(f"column metadata field {fid}")
        try:
            self.type, self.codec, self.values = ints[1], ints[4], ints[5]
            self.unc, self.comp, self.data = ints[6], ints[7], ints[9]
            self.encodings, self.path  # (both read)
        except (KeyError, AttributeError):
            raise Unsupported("column metadata without a required field") from None
        self.dictionary = ints.get(11)
        if self.dictionary is not None and not self.dictionary < self.data:
            raise Unsupported("a dictionary page not before the data pages")
        self.start = self.data if self.dictionary is None else self.dictionary
        self.end = self.start + self.comp
        self._end = i  # (where the struct ended, for the caller)

    def _statistics(self, b: bytes, i: int) -> int:
        fid = 0
        while True:
            h = b[i]
            i += 1
            if h == STOP:
                return i
            t = h & 0x0F
            if h >> 4:
                fid += h >> 4
            else:
                n, i = _uvarint(b, i)
                fid = _zigzag(n)
            if fid == 3 and t == I64:
                n, i = _uvarint(b, i)
                self.nulls = _zigzag(n)
            elif fid in (5, 6) and t == BINARY:
                n, i = _uvarint(b, i)
                if fid == 5:
                    self.max = b[i : i + n]
                else:
                    self.min = b[i : i + n]
                i += n
            elif fid in (7, 8) and t in (TRUE, FALSE):
                if fid == 7:
                    self.max_exact = t == TRUE
                else:
                    self.min_exact = t == TRUE
            else:
                raise Unsupported(f"statistics field {fid}")



class RowGroup:
    """A row group's rows, columns and bytes `start` to `end`."""

    __slots__ = ("rows", "columns", "start", "end", "ordinal", "_end")

    def __init__(self, b: bytes, i: int) -> None:
        ints, fid, self.columns = {}, 0, []
        while True:
            fid, t, i = _header(b, i, fid)
            if t == STOP:
                break
            if fid in _GROUP_INTS and t == _GROUP_INTS[fid]:
                n, i = _uvarint(b, i)
                ints[fid] = _zigzag(n)
            elif fid == 1 and t == LIST:
                n, et, i = _list_header(b, i)
                for _ in range(n):  # ColumnChunk: its file_offset and meta_data only
                    cfid, offset, col = 0, None, None
                    while True:
                        cfid, ct, i = _header(b, i, cfid)
                        if ct == STOP:
                            break
                        if cfid == 2 and ct == I64:
                            v, i = _uvarint(b, i)
                            offset = _zigzag(v)
                        elif cfid == 3 and ct == STRUCT:
                            col = Column(b, i)
                            i = col._end
                        else:
                            raise Unsupported(f"column chunk field {cfid}")
                    if col is None or offset != col.start:
                        raise Unsupported("a column chunk's file_offset not its first page")
                    self.columns.append(col)
            else:
                raise Unsupported(f"row group field {fid}")
        self._end = i
        cols = self.columns
        if not cols or 3 not in ints:
            raise Unsupported("a row group without columns or rows")
        self.rows, self.ordinal = ints[3], ints.get(7)
        self.start, self.end = cols[0].start, cols[-1].end
        if any(a.end != c.start for a, c in zip(cols, cols[1:])):
            raise Unsupported("a row group's column chunks not contiguous")
        if (ints.get(5), ints.get(2), ints.get(6)) != (
            self.start, sum(c.unc for c in cols), sum(c.comp for c in cols)
        ):
            raise Unsupported("a row group's offset or sizes not its columns'")



class ChunkFooter:
    """A footer of the split's writer's fields, read in one pass: its row groups (RowGroup)
    and its other top-level fields kept as encoded bytes (`fields`: id, type, bytes, in
    order). Raises Unsupported for any other field. `fd`, if the caller opens the file,
    is read in place of opening it for each copy."""

    def __init__(self, path: Path):
        self.path, self.fd = path, None
        with open(path, "rb") as f:
            f.seek(-8, os.SEEK_END)
            n, magic = struct.unpack("<I4s", f.read(8))
            if magic != MAGIC:
                raise ValueError(f"{path}: not a parquet file")
            f.seek(-8 - n, os.SEEK_END)
            b = f.read(n)
        self.fields, self.groups, fid, i = [], [], 0, 0
        while True:
            fid, t, i = _header(b, i, fid)
            if t == STOP:
                break
            if fid == 4 and t == LIST:
                k, et, i = _list_header(b, i)
                for _ in range(k):
                    g = RowGroup(b, i)
                    i = g._end
                    self.groups.append(g)
            elif fid in (1, 2, 3, 5, 6, 7):  # version, schema, num_rows, metadata, created_by, column_orders
                _, end = _value(b, i, t)
                self.fields.append((fid, t, b[i:end]))
                i = end
            else:
                raise Unsupported(f"file metadata field {fid}")
        self.schema = next((v for f, _, v in self.fields if f == 2), None)


def _merged_column(sources: list[tuple[Column, memoryview]], at: int) -> tuple[list, list, int, int]:
    """One column's chunk of a merged row group from the sources' chunks of it (their
    metadata and bytes), to be written at `at`: its ColumnMetaData fields (its path in
    schema the first source's, written by _put_column_meta), its bytes (the first source's
    dictionary page, if any, then each source's data pages), and its compressed and
    uncompressed sizes."""
    first, first_bytes = sources[0]
    dict_len = first.data - first.start  # 0 without a dictionary
    pieces = [first_bytes[:dict_len]] if dict_len else []
    dict_unc = 0
    if dict_len:
        # the dictionary page's header and its uncompressed page size
        hdr, end = _value(bytes(first_bytes[:dict_len]), 0, STRUCT)
        page_unc = _get(hdr, 2)[2]
        dict_unc = end + page_unc
        dict_page = first_bytes[:dict_len]
    values = unc = comp = 0
    nulls = 0
    encodings: list[int] = []
    for col, data in sources:
        if (col.type, col.path, col.codec) != (first.type, first.path, first.codec):
            raise ValueError("a column's type, path or codec not the first source's")
        if col.data - col.start != dict_len or (dict_len and data[:dict_len] != dict_page):
            raise ValueError("a column's dictionary page not the first source's")
        pieces.append(data[dict_len:])
        values += col.values
        unc += col.unc - dict_unc
        comp += col.comp - dict_len
        nulls = None if nulls is None or col.nulls is None else nulls + col.nulls
        encodings += [e for e in col.encodings if e not in encodings]
    unc += dict_unc
    comp += dict_len
    meta = [[1, I32, first.type], [2, LIST, (I32, encodings)], [3, LIST, None], [4, I32, first.codec],
            [5, I64, values], [6, I64, unc], [7, I64, comp], [9, I64, at + dict_len]]
    if dict_len:
        meta.append([11, I64, at])
    stats = []
    if nulls is not None:
        stats.append([3, I64, nulls])
    if all(c.min is not None and c.max is not None for c, _ in sources):
        hi = max(c.max for c, _ in sources)
        lo = min(c.min for c, _ in sources)  # (bytes: unsigned, as a string column's order)
        stats += [[5, BINARY, hi], [6, BINARY, lo],
                  [7, TRUE, any(c.max_exact for c, _ in sources if c.max == hi)],
                  [8, TRUE, any(c.min_exact for c, _ in sources if c.min == lo)]]
    if stats:
        meta.append([12, STRUCT, stats])
    return meta, pieces, comp, unc


def _put_column_meta(out: bytearray, meta: list, path: bytes) -> None:
    """A ColumnMetaData struct, its path in schema copied as encoded."""
    fid = 0
    for f, ft, x in meta:
        wire = (TRUE if x else FALSE) if ft in (TRUE, FALSE) else ft
        out.append((f - fid) << 4 | wire)  # (ids ascending, at most 15 apart)
        fid = f
        if f == 3:
            out += path
        elif ft not in (TRUE, FALSE):
            _put(out, ft, x)
    out.append(STOP)


def merge_row_groups(dst: Path, parts: list[tuple[ChunkFooter, int, int]], max_rows: int) -> int:
    """Write `dst` from each part's row groups (its footer, first row group, row groups), in
    order: consecutive row groups merged into one while it has at most `max_rows` rows (a
    row group of more is alone), each column's pages copied as they are (see the module's
    docstring); the first part's footer gives the rest of the metadata. Returns the rows
    written. A column whose dictionary page is not the first source's raises ValueError."""
    base = parts[0][0]
    groups: list[bytes] = []  # each merged row group's RowGroup struct, encoded
    rows = 0
    run: list[tuple[RowGroup, memoryview]] = []
    run_rows = 0
    with open(dst, "wb", buffering=0) as out:
        _write(out, MAGIC)
        at = len(MAGIC)

        def flush() -> None:
            nonlocal at
            if not run:
                return
            n_cols = len(run[0][0].columns)
            encoded = bytearray()
            encoded.append(1 << 4 | LIST)  # field 1, columns
            encoded.append((n_cols << 4 | STRUCT) if n_cols < 15 else (0xF0 | STRUCT))
            if n_cols >= 15:
                _put_varint(encoded, n_cols)
            start, unc_total, comp_total, body = at, 0, 0, []
            for c in range(n_cols):
                sources = []
                for g, data in run:
                    col = g.columns[c]
                    sources.append((col, data[col.start - g.start : col.end - g.start]))
                meta, pieces, comp, unc = _merged_column(sources, at)
                # ColumnChunk: file_offset (2), meta_data (3)
                encoded.append(2 << 4 | I64)
                _put_varint(encoded, at << 1)
                encoded.append(1 << 4 | STRUCT)
                _put_column_meta(encoded, meta, sources[0][0].path)
                encoded.append(STOP)
                body += pieces
                at += comp
                unc_total += unc
                comp_total += comp
            # total_byte_size (2), num_rows (3), file_offset (5), total_compressed_size (6),
            # ordinal (7, if the sources have one), after the columns (1)
            fields = [(2, unc_total), (3, sum(g.rows for g, _ in run)), (5, start), (6, comp_total)]
            if run[0][0].ordinal is not None:
                fields.append((7, len(groups)))
            fid = 1
            for f, x in fields:
                encoded.append((f - fid) << 4 | (I16 if f == 7 else I64))
                _put_varint(encoded, x << 1)
                fid = f
            encoded.append(STOP)
            groups.append(bytes(encoded))
            _writev(out, body)
            run.clear()

        for footer, first, n in parts:
            if n == 0:
                continue
            if footer.schema != base.schema:
                raise ValueError(f"{footer.path}: a schema not the first part's")
            sel = footer.groups[first : first + n]
            start, end = sel[0].start, sel[-1].end
            fd = footer.fd if footer.fd is not None else os.open(footer.path, os.O_RDONLY)
            try:
                data = memoryview(os.pread(fd, end - start, start))
            finally:
                if footer.fd is None:
                    os.close(fd)
            if len(data) != end - start:
                raise OSError(f"{footer.path}: ended before its row groups")
            for g in sel:
                if run and run_rows + g.rows > max_rows:
                    flush()
                    run_rows = 0
                run.append((g, data[g.start - start : g.end - start]))
                run_rows += g.rows
                rows += g.rows
        flush()
        meta = bytearray()
        fid = 0
        for f, t, raw in [*[x for x in base.fields if x[0] < 3], (3, I64, None), (4, LIST, None),
                          *[x for x in base.fields if x[0] > 4]]:
            meta.append((f - fid) << 4 | t)
            fid = f
            if f == 3:
                _put_varint(meta, rows << 1)
            elif f == 4:
                if len(groups) < 15:
                    meta.append(len(groups) << 4 | STRUCT)
                else:
                    meta.append(0xF0 | STRUCT)
                    _put_varint(meta, len(groups))
                for g in groups:
                    meta += g
            else:
                meta += raw
        meta.append(STOP)
        _write(out, bytes(meta) + struct.pack("<I", len(meta)) + MAGIC)
    return rows


def _writev(f, pieces: list) -> None:
    """All of `pieces`, in order, to an unbuffered file at its position."""
    for i in range(0, len(pieces), 512):  # (under IOV_MAX)
        batch = [p for p in pieces[i : i + 512] if len(p)]
        want = sum(len(p) for p in batch)
        done = os.writev(f.fileno(), batch) if batch else 0
        if done != want:  # a short write: the rest one at a time
            _write(f, b"".join(bytes(p) for p in batch)[done:])
