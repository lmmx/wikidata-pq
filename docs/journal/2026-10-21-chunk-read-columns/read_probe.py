"""One read of the routed chunk, full or of the three columns process() uses; prints wall, user, sys and minor faults of the read."""
import json, resource, sys, time
import polars as pl
path, mode = sys.argv[1], sys.argv[2]
cols = None if mode == "full" else ["id", "entity", "sitelinks"]
r0, t = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
df = pl.read_parquet(path, columns=cols)
w, r1 = time.perf_counter() - t, resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({"mode": mode, "wall": round(w, 3), "user": round(r1.ru_utime - r0.ru_utime, 2), "sys": round(r1.ru_stime - r0.ru_stime, 2), "minflt": r1.ru_minflt - r0.ru_minflt, "cols": df.width}))
