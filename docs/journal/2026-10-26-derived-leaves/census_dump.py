"""The derived leaves' rule (src/wikidata/claims_derived.py) on a real dump, by byte-range
reads of dumps.wikimedia.org (stdlib only; nothing kept on disk): `--windows` windows
spread evenly through each dump file given, each `--mib` MiB read, its whole bz2 blocks
decoded (#180's bz2probe), and every entity line parsed. Every snak's datavalue (a main
snak's, each qualifier's, each reference snak's; and a lexeme's forms' and senses' own
statements) is counted by its site, its `id`'s shape (digits as N), its entity-type, and
whether its numeric-id is the rule's; a snak with no datavalue (somevalue, novalue) by
its snaktype. Every value the rule does not give is listed (`exceptions`).

The rule, as claims_derived.derive gives it, restated here apart from it: entity-type
item/property/lexeme/entity-schema for Q/P/L/E and 1-18 ASCII digits, form/sense for
L…-F…/L…-S…, else null; numeric-id the digits of a Q/P/L id of that shape, else null.

    python3 -I census_dump.py --dump all:20261019 --dump lexemes:20261019 --windows 40 --mib 6 \\
        --out results/census_dump.json
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-26-progressive-finalise"))
import bz2probe as bp  # noqa: E402

URLS = {"all": "https://dumps.wikimedia.org/wikidatawiki/entities/{r}/wikidata-{r}-all.json.bz2",
        "lexemes": "https://dumps.wikimedia.org/wikidatawiki/entities/{r}/wikidata-{r}-lexemes.json.bz2"}
TYPES = {"Q": "item", "P": "property", "L": "lexeme", "E": "entity-schema"}
PLAIN = re.compile(r"^([QPLE])([0-9]{1,18})$")
PART = re.compile(r"^L[0-9]{1,18}-([FS])[0-9]+$")


def rule(i):
    if not isinstance(i, str):
        return None, None
    m = PLAIN.match(i)
    if m:
        return TYPES[m.group(1)], int(m.group(2)) if m.group(1) in "QPL" else None
    m = PART.match(i)
    if m:
        return {"F": "form", "S": "sense"}[m.group(1)], None
    return None, None


def shape(i):
    return None if i is None else re.sub(r"[0-9]+", "N", str(i))


class Census:
    def __init__(self):
        self.values = Counter()  # (site, shape, entity-type, numeric-id as the rule) -> n
        self.none = Counter()  # (site, snaktype, datavalue type) -> n, no entity id
        self.exceptions = []
        self.entities = Counter()

    def snak(self, site, s, where):
        dv = s.get("datavalue")
        if dv is None:
            self.none[(site, s.get("snaktype"), None)] += 1
            return
        v = dv.get("value")
        if not isinstance(v, dict) or not ({"id", "entity-type", "numeric-id"} & set(v)):
            self.none[(site, s.get("snaktype"), dv.get("type"))] += 1
            return
        i, et, nid = v.get("id"), v.get("entity-type"), v.get("numeric-id")
        want_et, want_nid = rule(i)
        ok_et, ok_nid = et == want_et, nid == want_nid
        self.values[(site, shape(i), et, "rule" if ok_nid else f"other:{nid is None}")] += 1
        if not (ok_et and ok_nid) and len(self.exceptions) < 200:
            self.exceptions.append({"where": where, "site": site, "value": v})

    def statements(self, claims, where, prefix=""):
        for prop, ss in (claims or {}).items():
            for st in ss:
                self.snak(prefix + "mainsnak", st["mainsnak"], where)
                for q in (st.get("qualifiers") or {}).values():
                    for s in q:
                        self.snak(prefix + "qualifier", s, where)
                for r in st.get("references") or []:
                    for x in r.get("snaks", {}).values():
                        for s in x:
                            self.snak(prefix + "reference", s, where)

    def entity(self, e):
        self.entities[e.get("type")] += 1
        self.statements(e.get("claims"), e.get("id"))
        for kind in ("forms", "senses"):
            for sub in e.get(kind) or []:
                self.statements(sub.get("claims"), sub.get("id"), f"{kind[:-1]} ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, help="kind:release, e.g. all:20261019")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--mib", type=float, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    report = {"dumps": {}, "reads": []}
    c = Census()
    for d in a.dump:
        kind, r = d.split(":")
        url = URLS[kind].format(r=r)
        n = bp.size(url) if hasattr(bp, "size") else None
        if n is None:
            import urllib.request
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": bp.UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                n = int(resp.headers["Content-Length"])
        windows = a.windows if kind == "all" else max(2, a.windows // 10)
        length = int(a.mib * 2**20)
        report["dumps"][d] = {"url": url, "bytes": n, "windows": windows}
        for w in range(windows):
            off = int((n - length) * (w + 0.5) / windows)
            t = time.time()
            buf = bp.fetch(url, off, length)
            blocks = bp.bit_positions(buf, bp.BLOCK, limit=2)
            text = bp.decode_blocks(buf, blocks[0], want=1 << 62) if blocks else b""
            del buf
            lines = [x.rstrip(b",") for x in text.split(b"\n")[1:-1] if x.startswith(b"{")]
            del text
            got = 0
            for line in lines:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                c.entity(e)
                got += 1
            report["reads"].append({"dump": d, "offset": off, "bytes": length, "entities": got,
                                    "first": lines[0][:40].decode(errors="replace") if lines else None,
                                    "s": round(time.time() - t, 1)})
            print(json.dumps(report["reads"][-1]), flush=True)
    report["bytes_read"] = bp.READS[0]
    report["entities"] = dict(c.entities)
    report["values"] = [[*k, v] for k, v in sorted(c.values.items(), key=lambda kv: (kv[0][0], -kv[1]))]
    report["no_entity_id"] = [[*k, v] for k, v in sorted(c.none.items(), key=lambda kv: (kv[0][0], -kv[1]))]
    report["exceptions"] = c.exceptions
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({"bytes_read": report["bytes_read"], "entities": report["entities"],
                      "exceptions": len(c.exceptions)}))


if __name__ == "__main__":
    main()
