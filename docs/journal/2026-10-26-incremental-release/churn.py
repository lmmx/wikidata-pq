"""Churn between consecutive weekly dumps, from aligned byte windows (stdlib only).

For each of N points, a window of the newest dump (B) is read in one shard at a fraction
of it, and decoded (bz2probe: each block rewrapped and decoded by libbz2, which checks its
CRC). Then, in each older dump (A1, A2, ...), the same shard's same id range is found by a
few small probes (the shards are in page-id order, monotone in id to within 2, see
2026-10-26-progressive-finalise.md) and read as one window. The entities of the shard
whose id falls in every window's range are compared by id:

- unchanged: the same line, byte for byte (after the separating comma);
- meta-only: a different line, but the same line once the page metadata (lastrevid,
  modified) are blanked: an edit and its revert, or a null edit;
- changed: the content differs; which top-level fields differ is recorded, and for the
  claims which properties;
- new: in B only (created, undeleted or un-redirected in between);
- gone: in A only (deleted or turned into a redirect: the dumps carry neither).

Nothing is written to disk but the per-entity table and the per-window log; every read
stays in memory.

    python3 -I churn.py --newest 20261019 --older 20261012 20261005 \\
        --boundaries ../2026-10-26-progressive-finalise/layout-20261019/boundaries.json \\
        --points 32 --bytes 1310720 --out raw
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bz2probe as bp  # noqa: E402

SCHOLARLY = {
    "Q13442814", "Q871232", "Q2782326", "Q815382", "Q187685", "Q1348305", "Q1907875", "Q18918145",
    "Q45182324", "Q23927052", "Q1402850", "Q15781350", "Q1266946", "Q7316896", "Q580922", "Q5246046",
    "Q30749496", "Q111475835", "Q56478376", "Q798134", "Q114613919", "Q92998777", "Q58901591",
    "Q10885494", "Q132115645", "Q59387148", "Q1385450", "Q130709863", "Q54670950", "Q51282918",
    "Q51282711", "Q15706459", "Q111475860", "Q51283092", "Q58900768", "Q58897583", "Q58898636",
    "Q591041", "Q193842", "Q58632367", "Q3099732", "Q637866", "Q21481766",
}
FIELDS = ("labels", "descriptions", "aliases", "sitelinks", "claims")
META = ("lastrevid", "modified")


def shard_of(eid: str, n: int = 8) -> int:
    return int(hashlib.sha1(eid.encode()).hexdigest()[:8], 16) % n


def size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": bp.UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def lines_at(url: str, offset: int, nbytes: int) -> list[bytes]:
    """The whole lines decoded from the first block at or after `offset` in `nbytes`."""
    buf = bp.fetch(url, offset, nbytes)
    blocks = bp.bit_positions(buf, bp.BLOCK, limit=2)
    if not blocks:
        return []
    text = bp.decode_blocks(buf, blocks[0], want=1 << 62)
    return [x.rstrip(b",") for x in text.split(b"\n")[1:-1] if x.startswith(b"{")]


def head_id(line: bytes) -> tuple[str, int] | None:
    m = bp.ID.match(line)
    if not m:
        return None
    return m.group(2).decode() + m.group(3).decode(), int(m.group(3))


def shard_q(lines: list[bytes], s: int) -> list[tuple[int, bytes]]:
    out = []
    for line in lines:
        h = head_id(line)
        if h and h[0][0] == "Q" and shard_of(h[0]) == s:
            out.append((h[1], line))
    return out


def locate(url: str, s: int, target: int, guess: int, density: float, total: int, log: list) -> tuple[int, int]:
    """An offset in `url` from which shard `s`'s ids start at or a little below `target`,
    found by small probes; returns (offset, first id there)."""
    p = max(0, min(guess, total - (1 << 20)))
    for _ in range(10):
        q = shard_q(lines_at(url, p, 192 << 10), s)
        log.append(p)
        if not q:
            # outside the shard (or no whole line): ids of another shard tell which way
            other = [head_id(x) for x in lines_at(url, p, 192 << 10)]
            sh = [shard_of(h[0]) for h in other if h]
            if sh and sum(1 for x in sh if x < s) > len(sh) / 2:
                p += 4 << 20
            else:
                p -= 4 << 20
            continue
        m = min(n for n, _ in q[:20])
        if m <= target and target - m < (320 << 10) / density:  # within ~320 KB of bz2 below
            return p, m
        p = max(0, p + int((target - m) * density) - (48 << 10))
    raise SystemExit(f"could not locate Q{target} in shard {s} of {url}")


def blank_meta(e: dict) -> dict:
    return {k: v for k, v in e.items() if k not in META}


def p31(e: dict) -> list[str]:
    c = e.get("claims") or {}
    out = []
    for st in c.get("P31", []) if isinstance(c, dict) else []:
        v = st.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(v, dict) and "id" in v:
            out.append(v["id"])
    return out


def claims_diff(a: dict, b: dict) -> tuple[list[str], int, int]:
    """Properties whose statement lists differ, and the statements added and removed (by
    statement id)."""
    ca, cb = a.get("claims") or {}, b.get("claims") or {}
    props = sorted(p for p in set(ca) | set(cb) if ca.get(p) != cb.get(p))
    ida = {st.get("id") for v in ca.values() for st in v}
    idb = {st.get("id") for v in cb.values() for st in v}
    return props, len(idb - ida), len(ida - idb)


def term_diff(a: dict, b: dict, f: str) -> list[str]:
    ta, tb = a.get(f) or {}, b.get(f) or {}
    return sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))


def compare(pair: str, point: int, s: int, a: dict[int, bytes], b: dict[int, bytes], lo: int, hi: int, w) -> dict:
    tally: dict[str, int] = {}
    for n in sorted(set(a) | set(b)):
        if not lo <= n <= hi:
            continue
        la, lb = a.get(n), b.get(n)
        row = {"pair": pair, "point": point, "shard": s, "num": n, "line_a": len(la or b""), "line_b": len(lb or b"")}
        if la is None or lb is None:
            e = json.loads(lb or la)
            row.update(status="new" if la is None else "gone", scholarly=int(bool(set(p31(e)) & SCHOLARLY)),
                       lastrevid_b=e.get("lastrevid") if lb else "", modified_b=e.get("modified") if lb else "")
        elif la == lb:
            row.update(status="unchanged")
        else:
            ea, eb = json.loads(la), json.loads(lb)
            row.update(scholarly=int(bool(set(p31(eb)) & SCHOLARLY)), lastrevid_a=ea.get("lastrevid"),
                       lastrevid_b=eb.get("lastrevid"), modified_b=eb.get("modified"),
                       same_rev=int(ea.get("lastrevid") == eb.get("lastrevid")))
            if blank_meta(ea) == blank_meta(eb):
                row.update(status="meta_only")
            else:
                row.update(status="changed")
                for f in FIELDS[:4]:
                    row[f"d_{f}"] = ";".join(term_diff(ea, eb, f))
                props, add, rem = claims_diff(ea, eb)
                row.update(d_claims=";".join(props), st_added=add, st_removed=rem,
                           d_other=";".join(sorted(k for k in set(ea) | set(eb)
                                                   if k not in FIELDS + META and ea.get(k) != eb.get(k))))
                row["p31"] = ";".join(p31(eb)[:3])
        tally[row["status"]] = tally.get(row["status"], 0) + 1
        w.writerow(row)
    return tally


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--newest", required=True)
    ap.add_argument("--older", nargs="+", required=True)
    ap.add_argument("--boundaries", required=True)
    ap.add_argument("--points", type=int, default=32)
    ap.add_argument("--bytes", type=int, default=1280 << 10)
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ub = bp.URL.format(r=a.newest)
    bnd = json.load(open(a.boundaries))
    tb = bnd["bytes"]
    edges = [0] + [x["before"] for x in bnd["boundaries"]] + [tb]
    olds = {r: (bp.URL.format(r=r), size(bp.URL.format(r=r))) for r in a.older}
    cols = ["pair", "point", "shard", "num", "status", "scholarly", "line_a", "line_b", "lastrevid_a", "lastrevid_b",
            "same_rev", "modified_b", "d_labels", "d_descriptions", "d_aliases", "d_sitelinks", "d_claims",
            "st_added", "st_removed", "d_other", "p31"]
    f = gzip.open(f"{a.out}/entities.csv.gz", "wt", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    logf = open(f"{a.out}/windows.jsonl", "w")
    for j in range(a.points):
        if a.only and j not in a.only:
            continue
        s = j % 8
        frac = (j + 0.5) / a.points
        ob = edges[s] + int((edges[s + 1] - edges[s]) * frac)
        r0 = bp.READS[0]
        qb = shard_q(lines_at(ub, ob, a.bytes), s)
        nums_b = [n for n, _ in qb]
        lo_b, hi_b = min(nums_b[:20]), max(nums_b[-20:])
        density = a.bytes / max(1, hi_b - lo_b)  # bz2 bytes a unit of id number, here
        b = dict(qb)
        rec = {"point": j, "shard": s, "frac": frac, "offset_b": ob, "lo_b": lo_b, "hi_b": hi_b, "n_b": len(qb),
               "bytes_b": bp.READS[0] - r0, "older": {}}
        wins = {}
        for r, (ua, ta) in olds.items():
            r1 = bp.READS[0]
            probes: list[int] = []
            pa, ma = locate(ua, s, lo_b, int(ob * ta / tb), density, ta, probes)
            need = int((hi_b - ma) * density * 1.04) + (96 << 10)
            qa = shard_q(lines_at(ua, pa, need), s)
            wins[r] = dict(qa)
            nums_a = [n for n, _ in qa]
            rec["older"][r] = {"offset": pa, "probes": len(probes), "lo": min(nums_a[:20]), "hi": max(nums_a[-20:]),
                               "n": len(qa), "bytes": bp.READS[0] - r1}
        # the common range, 5 ids in from each end (the order is monotone to within 2)
        lo = max([lo_b] + [v["lo"] for v in rec["older"].values()]) + 5
        hi = min([hi_b] + [v["hi"] for v in rec["older"].values()]) - 5
        rec.update(lo=lo, hi=hi, tally={})
        chain = [a.newest] + a.older  # newest first
        for x, y in zip(chain[1:], chain[:-1]):  # (older, newer) consecutive pairs
            rec["tally"][f"{x}-{y}"] = compare(f"{x}-{y}", j, s, wins[x], b if y == a.newest else wins[y], lo, hi, w)
        if len(a.older) > 1:
            x = a.older[-1]
            rec["tally"][f"{x}-{a.newest}"] = compare(f"{x}-{a.newest}", j, s, wins[x], b, lo, hi, w)
        rec["bytes_total"] = bp.READS[0] - r0
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        f.flush()
        print(j, s, f"Q{lo}-Q{hi}", rec["tally"], rec["bytes_total"] >> 10, "KiB", flush=True)
    f.close()
    print("read", bp.READS[0])


if __name__ == "__main__":
    main()
