"""Finalise's per-file costs on a layout's group files (OUT/{layout}/{table}/{key}/g.parquet):
CPU of footer_size (plan_key), pq.read_metadata (compaction's link), sha256 (the link's
entry, check_compacted_copy) and _read_key (the sort's read), each over every key."""
import json, sys, time
from pathlib import Path
import pyarrow.parquet as pq
from wikidata import sort_by_id as s

root = Path(sys.argv[1])
for layout in sys.argv[2].split(","):
    for tdir in sorted((root / "cmp" / layout).iterdir()):
        files = sorted(tdir.glob("*.parquet"))
        out = {"layout": layout, "table": tdir.name, "files": len(files),
               "bytes": sum(f.stat().st_size for f in files),
               "rgs": sum(pq.read_metadata(f).num_row_groups for f in files)}
        for name, fn in [("footer_size", lambda f: s.footer_size([f])),
                         ("read_metadata", lambda f: pq.read_metadata(f).num_rows),
                         ("sha256", s._sha256),
                         ("read_key", lambda f: s._read_key([f], threads=False)), ("read_key_glue", lambda f: s._read_key([f], threads=True))]:
            best = None
            for _ in range(3):
                t0 = time.process_time()
                for f in files:
                    fn(f)
                c = time.process_time() - t0
                best = c if best is None else min(best, c)
            out[name] = round(best, 3)
        print(json.dumps(out), flush=True)
