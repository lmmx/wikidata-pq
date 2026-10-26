import sys, os
from collections import defaultdict
import pyarrow.parquet as pq
for f in sys.argv[1:]:
    m = pq.ParquetFile(f).metadata
    size = os.path.getsize(f)
    chunks = 0; unc = 0
    bycol = defaultdict(int); enc = {}
    stats_bytes = 0
    for g in range(m.num_row_groups):
        rg = m.row_group(g)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            chunks += cc.total_compressed_size; unc += cc.total_uncompressed_size
            bycol[cc.path_in_schema] += cc.total_compressed_size
            enc[cc.path_in_schema] = (cc.encodings, cc.compression)
            s = cc.statistics
            if s is not None and s.has_min_max:
                try: stats_bytes += len(str(s.min)) + len(str(s.max))
                except Exception: pass
    footer = m.serialized_size
    print(f"{f}: {size/1e6:.2f} MB rows={m.num_rows} rgs={m.num_row_groups} cols={m.num_columns} "
          f"chunks={chunks/1e6:.2f} MB (uncomp {unc/1e6:.2f}) footer={footer/1e3:.1f} KB "
          f"pageindex~={(size-chunks-footer-12)/1e3:.1f} KB statsminmax~={stats_bytes/1e3:.1f} KB")
    if os.environ.get("COLS"):
        for k, v in sorted(bycol.items(), key=lambda kv: -kv[1])[:25]:
            print(f"   {v/1e6:8.3f} MB {100*v/chunks:5.1f}% {k} {enc[k]}")
