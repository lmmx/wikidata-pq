"""One call of the split's compiled reshape and claims build (claims_build.reshape_built)
on the lines of FILE (a dump line a line), for callgrind (--toggle-collect on the Rust
entry point) or a native timing of REPEAT calls.

Usage: reshape_once.py SRC_DIR FILE [REPEAT]"""
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
from wikidata import claims_build  # noqa: E402

lines = [l for l in open(sys.argv[2], "rb").read().split(b"\n") if l]
repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
claims_build.reshape_built(lines[:10])  # warm: the plan compiled, imports done
best = None
for _ in range(repeat):
    c0, w0 = time.process_time(), time.perf_counter()
    r = claims_build.reshape_built(lines)
    c, w = time.process_time() - c0, time.perf_counter() - w0
    best = c if best is None else min(best, c)
n = sum(map(len, lines))
print(f"{len(lines)} lines, {n} bytes: best cpu {best:.3f}s = {best * 2.1e9 / n:.2f} cycles/byte (2.1 GHz)", file=sys.stderr)
