"""Page size's effect on the claims' bytes, by pyarrow (pq-glue's writer has no page-size
option): the sample's claims files at zstd 1, statement_id DBA, 1 MiB (both writers'
default) against 4 and 16 MiB pages; relative sizes only."""
import glob, os
import pyarrow.parquet as pq
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
fs = sorted(glob.glob(f"{P}/parts/*/claims/chunk_*.parquet"))
meta = pq.read_metadata(fs[0]); paths = [meta.schema.column(i).path for i in range(meta.num_columns)]
for mib in (1, 4, 16):
    tot = 0
    for f in fs:
        t = pq.read_table(f)
        pq.write_table(t, f"{P}/ps.parquet", row_group_size=122_880, compression="zstd", compression_level=1,
                       data_page_size=mib << 20, dictionary_pagesize_limit=mib << 20,
                       use_dictionary=[p for p in paths if p != "statement_id"], column_encoding={"statement_id": "DELTA_BYTE_ARRAY"})
        tot += os.path.getsize(f"{P}/ps.parquet")
    print(mib, "MiB pages:", tot, flush=True)
os.unlink(f"{P}/ps.parquet")
