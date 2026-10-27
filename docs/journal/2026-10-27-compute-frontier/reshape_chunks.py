import os, sys, time
sys.path.insert(0, sys.argv[1]); os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
from wikidata import claims_build
lines = [l for l in open(sys.argv[2], "rb").read().split(b"\n") if l]
n = int(sys.argv[3])
for i in range(0, len(lines), n):
    part = lines[i:i+n]
    c0, w0 = time.process_time(), time.perf_counter()
    r = claims_build.reshape_built(part)
    c, w = time.process_time() - c0, time.perf_counter() - w0
    b = sum(map(len, part))
    print(i, len(part), b, "built" if r is not None else "NONE", f"cpu {c:.3f} wall {w:.3f} cyB {c*2.1e9/b:.2f}")
