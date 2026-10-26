"""statement_id (36% of the published claims' bytes) under encodings pyarrow can write:
PLAIN (as today, its sampled values all distinct), DELTA_LENGTH_BYTE_ARRAY and
DELTA_BYTE_ARRAY, at zstd levels (3 and 9 by default); size, write and read CPU, and a
value check.

Usage: enc.py CLAIMS_FILE SCRATCH_FILE [LEVELS]"""
import resource, sys, time
import pyarrow as pa, pyarrow.parquet as pq
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
src, dst = sys.argv[1], sys.argv[2]
t = pq.read_table(src, columns=["id", "statement_id"])
col = t.select(["statement_id"])
print("rows", t.num_rows, "same entity prefix share",
      sum(1 for i, s in zip(t["id"].to_pylist()[:200000], t["statement_id"].to_pylist()[:200000]) if s.split("$")[0].upper() == i.upper()) / 200000)
for enc in ("PLAIN", "DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY"):
    for lv in [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "3,9").split(",")]:
        c0 = cpu()
        pq.write_table(col, dst, compression="zstd", compression_level=lv, use_dictionary=False,
                       column_encoding={"statement_id": enc}, row_group_size=120_000)
        w = cpu() - c0
        c0 = cpu(); back = pq.read_table(dst); r = cpu() - c0
        assert back["statement_id"].equals(col["statement_id"])
        print(f"{enc:24s} zstd {lv}: {pq.read_metadata(dst).row_group(0).column(0).total_compressed_size and sum(pq.read_metadata(dst).row_group(i).column(0).total_compressed_size for i in range(pq.read_metadata(dst).num_row_groups))/1e6:.2f} MB  write {w:.2f} CPU-s  read {r:.2f} CPU-s")
