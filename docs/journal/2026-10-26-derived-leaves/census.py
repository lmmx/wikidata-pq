"""Census of entity-type and numeric-id against their sibling id, every datavalue site."""
import glob, sys, collections, re
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
import polars as pl

def sites(arr, typ, path):
    """Yield (path, logical datavalue struct array) for each struct named datavalue with id."""
    if pa.types.is_struct(typ):
        names = [typ.field(i).name for i in range(typ.num_fields)]
        if {"id", "entity-type", "numeric-id"} <= set(names):
            yield path, arr
            return
        for i, n in enumerate(names):
            # flatten masks by parent validity
            yield from sites(arr.field(n) if arr.null_count == 0 else pc.struct_field(arr, [i]), typ.field(i).type, f"{path}.{n}" if path else n)
    elif pa.types.is_large_list(typ) or pa.types.is_list(typ):
        yield from sites(arr.flatten(), typ.value_type, path + "[]")

counts = collections.Counter()
for f in sorted(glob.glob(sys.argv[1])):
    t = pq.read_table(f)
    sa = pa.StructArray.from_arrays([c.combine_chunks() for c in t.columns], names=t.column_names)
    for path, dv in sites(sa, sa.type, ""):
        # logical: struct_field masks by struct validity
        i = dv.type.get_field_index
        ids = pc.struct_field(dv, [i("id")]); et = pc.struct_field(dv, [i("entity-type")]); nid = pc.struct_field(dv, [i("numeric-id")])
        df = pl.DataFrame({"id": pl.from_arrow(ids), "et": pl.from_arrow(et), "nid": pl.from_arrow(nid)})
        df = df.with_columns(shape=pl.col("id").str.replace_all(r"\d+", "N"))
        g = df.group_by("shape", "et", (pl.col("nid").is_null()).alias("nid_null"),
                        (pl.col("nid") == pl.col("id").str.slice(1).cast(pl.Int64, strict=False)).alias("nid_eq")).len()
        for r in g.iter_rows():
            counts[(path,) + r[:-1]] += r[-1]
for k, v in sorted(counts.items(), key=lambda kv: (kv[0][0], str(kv[0][1:]))):
    print(v, k)
