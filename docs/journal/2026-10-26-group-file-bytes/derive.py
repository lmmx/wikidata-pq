"""Is each numeric-id and entity-type in the claims exactly a function of its id?"""
import os
import glob, polars as pl
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
files = sorted(glob.glob(f"{P}/parts/*/claims/chunk_*.parquet"))
df = pl.concat([pl.read_parquet(f) for f in files])
def check(dv: pl.DataFrame, where: str):
    dv = dv.filter(pl.col("id").is_not_null() | pl.col("numeric-id").is_not_null() | pl.col("entity-type").is_not_null())
    num = pl.when(pl.col("id").str.contains(r"^[QPLM]\d+$")).then(pl.col("id").str.slice(1).cast(pl.Int64))
    et = (pl.when(pl.col("id").str.contains(r"^Q\d+$")).then(pl.lit("item"))
          .when(pl.col("id").str.contains(r"^P\d+$")).then(pl.lit("property"))
          .when(pl.col("id").str.contains(r"^L\d+$")).then(pl.lit("lexeme"))
          .when(pl.col("id").str.contains(r"^L\d+-F\d+$")).then(pl.lit("form"))
          .when(pl.col("id").str.contains(r"^L\d+-S\d+$")).then(pl.lit("sense")).when(pl.col("id").str.contains(r"^E\d+$")).then(pl.lit("entity-schema")))
    bad_n = dv.filter(~pl.col("numeric-id").eq_missing(num))
    bad_e = dv.filter(~pl.col("entity-type").eq_missing(et))
    print(where, dv.height, "rows with an entity value; numeric-id not derivable:", bad_n.height, "entity-type not derivable:", bad_e.height)
    if bad_n.height: print(bad_n.head(5))
    if bad_e.height: print(bad_e.head(5))
check(df.select(pl.col("datavalue").struct.field("id", "numeric-id", "entity-type")), "mainsnak")
q = df.select(pl.col("qualifiers").explode().struct.field("value").explode().struct.field("datavalue")).unnest("datavalue")
check(q.select("id", "numeric-id", "entity-type"), "qualifiers")
# (the references' are checked by derive_refs.py: exploded here they took over 10 GB)
