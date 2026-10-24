"""Row fingerprints: one 64-bit value per row, from the row's whole content.

Each leaf value is hashed (by Polars) with its column path and its null-ness, and mixed,
column by column, with its position in every list it is in; a list's elements are
summed into their list, a struct's fields into the struct, a row's columns into the row.
So a changed, missing or extra value, a value moved to another row or column, and list
elements reordered each change the row's fingerprint. The same values give the same
fingerprints whichever reader decoded them (pyarrow, Polars or pq-glue) and however the
rows are chunked: a fingerprint depends on the row alone.

Computed a column at a time over Arrow buffers (numpy folding the leaves up through list
offsets), not row by row through the nested structs: on the claims, a third of the CPU
time of Polars' struct hash with one seed (docs/journal/2026-10-24-row-fingerprints.md).
"""
import numpy as np
import polars as pl
import pyarrow as pa

M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser, elementwise (wrapping uint64 arithmetic)."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * M1
        x = x ^ (x >> np.uint64(27))
        x = x * M2
        return x ^ (x >> np.uint64(31))


def _salt(path: str) -> np.uint64:
    h = 1469598103934665603
    for b in path.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return np.uint64(h)


def _validity(arr: pa.Array) -> np.ndarray:
    if arr.null_count == 0:
        return np.ones(len(arr), dtype=bool)
    return np.asarray(arr.is_valid())


def _fold(arr: pa.Array, path: str) -> np.ndarray:
    """One uint64 per element of `arr` (length len(arr)), covering its whole subtree."""
    ty = arr.type
    salt = _salt(path)
    valid = _validity(arr)
    if pa.types.is_struct(ty):
        acc = np.zeros(len(arr), dtype=np.uint64)
        children = arr.flatten()  # each child with the parent's offset (and nulls) applied
        for i in range(ty.num_fields):
            child = children[i]
            with np.errstate(over="ignore"):
                acc = acc + _fold(child, f"{path}.{ty.field(i).name}")
    elif pa.types.is_list(ty) or pa.types.is_large_list(ty) or pa.types.is_map(ty):
        offsets = np.asarray(arr.offsets)
        values = arr.values
        start, end = int(offsets[0]), int(offsets[-1])
        child = _fold(values.slice(start, end - start), path + "[]")
        # each element mixed with its position in its list, then summed per list
        lengths = np.diff(offsets)
        pos = np.arange(end - start, dtype=np.uint64) - np.repeat(
            (offsets[:-1] - start).astype(np.uint64), lengths
        )
        with np.errstate(over="ignore"):
            child = _mix(child ^ (pos * np.uint64(0x9E3779B97F4A7C15)))
        sums = np.zeros(len(arr), dtype=np.uint64)
        nonempty = lengths > 0
        if nonempty.any():
            red = np.add.reduceat(child, (offsets[:-1] - start)[nonempty])
            sums[nonempty] = red
        with np.errstate(over="ignore"):
            acc = sums + lengths.astype(np.uint64) * np.uint64(0x2545F4914F6CDD1D)
    elif pa.types.is_nested(ty):
        # a nested type not folded above would have its content skipped: refuse it
        raise TypeError(f"no row fingerprint for {path}'s type {ty}")
    else:
        acc = pl.from_arrow(arr).hash(seed=0).to_numpy().astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        out = _mix(acc ^ salt)
    return np.where(valid, out, salt ^ np.uint64(0xDEADBEEF))


def row_fingerprints(table: pa.Table) -> np.ndarray:
    """One uint64 per row of `table`, from every column's content."""
    out = np.zeros(table.num_rows, dtype=np.uint64)
    for name in table.column_names:
        col = table.column(name)
        parts = [_fold(c, name) for c in col.chunks] if col.num_chunks else [np.zeros(0, np.uint64)]
        with np.errstate(over="ignore"):
            out = out + np.concatenate(parts)
    return _mix(out)


def frame_fingerprints(lf: pl.LazyFrame, *keep: str) -> pl.DataFrame:
    """The `keep` columns and each row's fingerprint (`fp`, UInt64) of the rows `lf`
    reads, in order. Streaming: a batch at a time, so a file is never held decoded whole
    (each batch handed over as Arrow without a copy)."""
    keeps, fps = [], []
    for batch in lf.collect_batches():
        fps.append(row_fingerprints(batch.to_arrow(compat_level=pl.CompatLevel.newest())))
        if keep:
            keeps.append(batch.select(keep))
    fp = pl.Series("fp", np.concatenate(fps) if fps else [], dtype=pl.UInt64)
    if not keep:
        return fp.to_frame()
    out = pl.concat(keeps) if keeps else lf.select(keep).head(0).collect()
    return out.with_columns(fp)
