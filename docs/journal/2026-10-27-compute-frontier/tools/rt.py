#!/usr/bin/env python3
"""rt.py CMD...: run CMD (stdout to /dev/null unless RT_KEEP), print wall, user, sys, max RSS of children."""
import os, resource, subprocess, sys, time
t = time.perf_counter()
out = None if os.environ.get("RT_KEEP") else subprocess.DEVNULL
p = subprocess.run(sys.argv[1:], stdout=out)
w = time.perf_counter() - t
r = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"{os.path.basename(sys.argv[1])}: wall {w:.3f}s user {r.ru_utime:.3f}s sys {r.ru_stime:.3f}s cpu {r.ru_utime+r.ru_stime:.3f}s maxrss {r.ru_maxrss/1024:.0f} MB minflt {r.ru_minflt} majflt {r.ru_majflt} nvcsw {r.ru_nvcsw} nivcsw {r.ru_nivcsw} rc {p.returncode}", file=sys.stderr)
