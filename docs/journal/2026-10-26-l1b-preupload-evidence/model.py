"""L1b of #167 (parts preuploaded as they are checked, published by the same commits),
re-modelled on master with L1a (#170) and L2 (#175) in, from L2's model
(`2026-10-26-finalise-before-uploads-end/model.py`), which keeps bench 17.2.0's.

Inputs: L2's (master's bench record, 26e9259, bench 17.3.0, quick tier, CI run
40265661727; `runs/quick/26e9259e13911801ec8984be45f5b81b281b74bb.json` on the
bench-results branch), each "after" arm's run, medians over them. Every figure printed is
modelled, none measured. Release seconds, bytes, bytes a second.

Kept from L2's model: the split and both sets' runs, the build branch's pushes (each set's
groups closed one at a time, the main set's turn held after the split: built), the main
set's first finalise started once its own run ends (built), the three finalise runs one
process after another, each table downloaded (unless kept) and computed, L1a's committer
(tables committed in order, the 64 GB bound), claims_labels' wait on the set's labels
and claims, the refs after the main set's claims.

Changed here, for L1b: a table's published bytes no longer go up in its commit. They go
up part by part, on one uploader (`preupload_lfs_files`, a call at a time), as each part
becomes eligible, in eligibility order across tables; the table's commit is then only
the commit request (taken as 0 s of link) once the table is `written` and each of its
parts is up. When a part becomes eligible (its checks passed):

- **the claims key** (the bucket path, one key a set): no part before the bucketing and
  the plan, the first `b` of the claims' compute; then its parts one after another,
  evenly, to the end of its compute (each part once `made` in `sort_buckets`: its
  `_check_sort`, `_check_written` or `_check_glued`, and its sha256 in `files.jsonl`).
  `strict`: every part only once the key's own checks pass (`_check_sorted`,
  `_check_whole`, its manifest entry), at the end of its compute, which is L1a for the
  claims;
- **every other table** (~600 keys sorted in memory, and claims_labels' languages): a
  key's parts once its manifest entry is written (`write_parts`' `_check_written`), the
  keys evenly over the table's compute.

`o`: the uploader's fixed cost a call (the /preupload request, the Xet token, the
session's shard upload and its finalise), as link time. A call takes every part of the
table eligible when it starts (`coalesce`; else one claims part, or one of the other
table's `calls` shares of its keys, a call). Unknown: the user's measurement
(`preupload_measure.py`) gives it.

Usage: python model.py runs/quick/26e9259e13911801ec8984be45f5b81b281b74bb.json
"""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

_l2 = Path(__file__).resolve().parents[1] / "2026-10-26-finalise-before-uploads-end" / "model.py"
_spec = importlib.util.spec_from_file_location("l2_model", _l2)
l2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l2)

from link import CLAIMS_LABELS_BUILD, DOWNLOAD_RATE  # noqa: E402  (bench/, on l2's path)

RATES = l2.RATES
PART_BYTES = 0.5e9  # a claims part file (COMPACT_FILE_BYTES; 86 parts of ~43 GB at a release)
BUILT = "gate, main first, held after the split"


def eligible(name: str, start: float, cpu: float, nbytes: float, *, b: float, strict: bool,
             calls: int) -> list[tuple[float, float]]:
    """(when eligible, bytes) of each upload call of a table computed from `start`."""
    if name == "claims":
        n = max(1, round(nbytes / PART_BYTES))
        if strict:
            return [(start + cpu, nbytes / n)] * n
        return [(start + cpu * (b + (1 - b) * (i + 1) / n), nbytes / n) for i in range(n)]
    return [(start + cpu * (i + 1) / calls, nbytes / calls) for i in range(calls)]


def one_l1b(k, tables, cl, group, pub, cpu, run, rate, kept, *, t0=0.0, link=None,
            b=0.10, strict=False, o=0.0, calls=20, coalesce=True):
    """One finalise run from t0 under L1b (L2's `one`, the commits' bytes moved to the
    uploader): its end."""
    pending = run["finalise_link_bytes"]["pending"]
    link = link or l2.Link()
    t = free = up = t0  # the writer's, the committer's and the uploader's clocks
    done, queued = {}, []
    for name in tables + (["claims_labels"] if cl else []):
        if name == "claims_labels":
            t = max(t, *(done.get(n, t0) for n in ("claims", "labels")))
            start, c = t, CLAIMS_LABELS_BUILD / 2
        else:
            while sum(m for c_, m in queued if c_ > t) > pending:
                t = min(c_ for c_, _ in queued if c_ > t)
            down = 0.0 if kept else group[k].get(name, 0) / DOWNLOAD_RATE
            start, c = t + down, cpu[k].get(name, 0)
        t = start + c
        nbytes = pub[k].get(name, 0)
        waiting = eligible(name, start, c, nbytes, b=b, strict=strict, calls=calls)
        while waiting:  # (a call takes every part eligible when it starts)
            s = max(up, waiting[0][0])
            batch = [p for w, p in waiting if w <= s] if coalesce else [waiting[0][1]]
            waiting = waiting[len(batch):]
            up = link.take(s, o + sum(batch) / rate)
        free = max(free, t, up)  # (the commit: in order, once written and every part up)
        done[name] = free
        if k == "main" and name == "claims" and not cl:
            free += run.get("refs_eta") or 0
        queued.append((free, nbytes))
    return max(t, free)


def release(run: dict, rate: float, kept: bool, l1b: dict | None) -> float:
    """The release's wall under L2 as built (L2's `release`, mode `BUILT`), finalise's
    runs as L1a (`l1b` None) or L1b."""
    if l1b is None:
        return l2.release(run, rate, kept, BUILT, True)["wall"]
    group, pub, cpu = l2.tables_of(run)
    ph = l2.ov._phase(run, True)
    tbl = {k: [t for t in l2.ORDER if t in group[k]] for k in group}
    link, end, _, _ = l2.build(ph, group, rate, merge=True, **l2.MODES[BUILT])
    start = max(ph.split_end, ph.compute_end, end["main"])
    r1 = one_l1b("main", tbl["main"], False, group, pub, cpu, run, rate, kept, t0=start, link=link, **l1b)
    r1 = max(r1, end["scholar"])
    r2 = one_l1b("scholar", tbl["scholar"], True, group, pub, cpu, run, rate, kept, t0=r1, **l1b)
    r3 = one_l1b("main", [], True, group, pub, cpu, run, rate, kept, t0=r2, **l1b)
    return r3


def main(path: str) -> None:
    runs = [r for rec in json.load(open(path)) for r in rec["runs"] if r["arm"] == "after"]

    def med(f) -> float:
        return statistics.median(f(r) for r in runs)

    head = "| | " + " | ".join(f"{r / 1e6:.0f} MB/s" for r in RATES) + " |"
    sep = "|---|" + "---|" * len(RATES)
    cases = [
        ("L1a + L2 (master)", None),
        ("**L1b: per part, b=0.10, o=0**", dict(b=0.10)),
        ("L1b: per part, b=0 (no bucketing gap: the bound)", dict(b=0.0)),
        ("L1b: per part, b=0.20", dict(b=0.20)),
        ("L1b: per part, b=0.30", dict(b=0.30)),
        ("L1b: per part, b=0.10, o=5 s a call", dict(b=0.10, o=5.0)),
        ("L1b: per part, b=0.10, o=30 s a call", dict(b=0.10, o=30.0)),
        ("L1b: per part, b=0.10, o=5 s, a part a call (no coalescing)", dict(b=0.10, o=5.0, coalesce=False)),
        ("L1b: strict (claims parts at the key's end)", dict(strict=True)),
    ]
    print(f"{len(runs)} runs of master (26e9259); medians.\n")
    for kept in (False, True):
        walls = {label: [med(lambda r: release(r, x, kept, kw)) for x in RATES] for label, kw in cases}
        print(f"Release walls, L3 {'on (group files kept)' if kept else 'off (the default)'}:\n")
        print(head + "\n" + sep)
        for label, w in walls.items():
            print(f"| {label} | " + " | ".join(l2.hms(x) for x in w) + " |")
        print(f"\nSavings against L1a + L2, L3 {'on' if kept else 'off'}:\n")
        print(head + "\n" + sep)
        ref = walls["L1a + L2 (master)"]
        for label, w in walls.items():
            if w is ref:
                continue
            print(f"| {label} | " + " | ".join(f"{(x - b_) / 60:+.0f} min" for b_, x in zip(ref, w)) + " |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
