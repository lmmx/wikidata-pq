"""A release-shaped claims group file: 40 copies of a release-sized chunk's claims file (one
row group, 45,546 rows) glued as merge_group glues them; its footer and page-index bytes."""
import os, json, glob, struct
import pyarrow.parquet as pq, pq_glue
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
src = f"{P}/parts/main/claims/chunk_1.parquet"
out = f"{P}/glued.parquet"
pq_glue.concat_many([([src] * 40, out)])
size = os.path.getsize(out)
m = pq.read_metadata(out)
chunks = sum(m.row_group(g).column(c).total_compressed_size for g in range(m.num_row_groups) for c in range(m.num_columns))
footer = m.serialized_size
print(f"glued: {size/1e6:.2f} MB, {m.num_row_groups} rgs, column chunks {chunks/1e6:.2f} MB, footer {footer/1e3:.1f} KB "
      f"({100*footer/size:.2f}%), other (page index) {(size-chunks-footer-12)/1e3:.1f} KB ({100*(size-chunks-footer-12)/size:.2f}%)")
kv = m.metadata or {}
print("key-value metadata bytes:", sum(len(k)+len(v) for k, v in kv.items()))
one = pq.read_metadata(src)
print(f"single: footer {one.serialized_size/1e3:.1f} KB; per extra row group in glued: {(footer-one.serialized_size)/39/1e3:.1f} KB of {os.path.getsize(src)/1e6:.2f} MB")
# statistics share: row group 0's column chunk stats
rg = m.row_group(0)
st = 0
for c in range(rg.num_columns):
    s = rg.column(c).statistics
    if s is not None and s.has_min_max:
        for v in (s.min, s.max):
            st += len(v.encode()) if isinstance(v, str) else 8
print("min/max bytes a row group (each written twice, legacy and new fields, at most):", 2 * st)
# redundant columns
b = json.load(open(f"{P}/chosen_l1.json"))["base"]["cols"]
tot = sum(sum(v.values()) for v in b.values())
for pat in ("numeric-id", "entity-type"):
    n = sum(v for s in b.values() for k, v in s.items() if k.endswith(pat))
    print(pat, f"{n/1e6:.2f} MB {100*n/tot:.2f}% of the claims' column bytes")
os.unlink(out)
