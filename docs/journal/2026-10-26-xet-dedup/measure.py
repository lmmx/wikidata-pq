"""How much of release B's upload Xet deduplicates against release A's (releases.py's
builds), file by file, by Xet's own chunker (xet.py).

For each kind of file (`groups`: the build branch's group files, pushed while the runs
run; `published`: the sorted files finalise commits), by set and table:

- **bytes**: B's bytes.
- **identical files**: bytes of B's files byte-identical to A's file at the same path.
- **pages**: bytes in Parquet pages byte-identical to a page of A's same table (pqpages),
  what #188's model takes to dedup: "an untouched page re-encodes to the same bytes".
- **predicted**: bytes in pages none of whose rows is of an entity changed, new or gone
  between A and B (from the page index's first rows and the `id` column), #188's
  premise applied to the real pages. Footers and page indexes are never predicted.
- **chunks**: bytes of B in Xet chunks whose hash is among A's chunks (any of A's
  files): the most any client could dedup.
- **xet**: bytes Xet's client would not upload, by its own dedup (xet.Uploader: ranges
  of consecutive chunks of one xorb, the fragmentation guard), with every one of A's
  uploads known to it: A uploaded from the same machine, its shards in the local shard
  cache (3 weeks; 16 GB). A's uploads are themselves simulated first, upload session by
  upload session, to form its xorbs.
- **xet_global_xorb**, **xet_global_session**: the same with no local cache (another
  machine), A's chunks found only by global dedup queries (xet.GlobalIndex), the Hub's
  answer to a hit being only the xorb holding the chunk, or the whole upload session's
  shard (the Hub decides; not in the client's source).

B's uploads run in the order a release makes them: the group files (each group's push
its own session), then the published files (each table's commits a session), each
session's shard known to the sessions after it (it goes to the cache once uploaded).

    python measure.py --a A --b B --changed B.json --out results/dedup-1x.json
"""

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pqpages  # noqa: E402
import xet  # noqa: E402

KINDS = ("groups", "published")


def table_of(path: str) -> tuple[str, str]:
    """(set, table) of a stand-in Hub path `<owner>__wikidata-[scholar-]<table>/<rev>/...`."""
    repo = path.split("/")[0].split("__", 1)[1].removeprefix("wikidata-")
    return ("scholar", repo.removeprefix("scholar-")) if repo.startswith("scholar-") else ("main", repo)


def norm(path: str) -> str:
    """The path without its branch (build-<release>), to pair A's files with B's."""
    p = path.split("/")
    return "/".join([p[0], *p[2:]])


def files(rel: Path, kind: str) -> dict[str, Path]:
    root = rel / kind
    return {norm(str(p.relative_to(root))): p for p in sorted(root.rglob("*.parquet"))}


def chunked(rel: Path) -> dict[str, dict[str, list]]:
    """Each file's Xet chunks (hash, size), cached beside the release."""
    cache = rel / "chunks.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    out = {k: {n: xet.chunks_of(p.read_bytes()) for n, p in files(rel, k).items()} for k in KINDS}
    cache.write_bytes(pickle.dumps(out))
    return out


def page_hashes(data: bytes) -> list[tuple[bytes, int, int, int | None, int]]:
    """Each page's (sha256, offset, length, first row, rows in its row group)."""
    out = []
    for c in pqpages.pages(data):
        for off, n, first, _ in c["pages"]:
            out.append((hashlib.sha256(data[off:off + n]).digest()[:16], off, n, first, c["rows"]))
    return out


def page_rows(data: bytes, path: Path) -> list[tuple[int, int, int, int]]:
    """Each data page's (offset, length, first row, end row) in file rows, from the page
    index, for the predicted figure."""
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).metadata
    starts, s = [], 0
    for i in range(md.num_row_groups):
        starts.append(s)
        s += md.row_group(i).num_rows
    out = []
    for c in pqpages.pages(data):
        base, rows = starts[c["rg"]], c["rows"]
        ps = [p for p in c["pages"] if p[3] == "data" and p[2] is not None]
        for j, (off, n, first, _) in enumerate(ps):
            end = ps[j + 1][2] if j + 1 < len(ps) else rows
            out.append((off, n, base + first, base + end))
        for off, n, _, kind in c["pages"]:
            if kind == "dict":
                out.append((off, n, base, base + rows))  # (a dictionary covers its row group)
    return out


def simulate(order: list[tuple[str, str, list]], known: xet.Index, global_index: xet.GlobalIndex | None,
             register: bool = True) -> dict[tuple[str, str], xet.Metrics]:
    """Upload sessions in order: (session, file, chunks). Each session's xorbs join
    `known` once it ends (its shard in the cache) and, given one, the global index."""
    out: dict[tuple[str, str], xet.Metrics] = {}
    sessions: dict[str, list] = defaultdict(list)
    for s, f, ch in order:
        sessions[s].append((f, ch))
    for s, fs in sessions.items():
        up = xet.Uploader(known=known, global_index=global_index)
        for f, ch in fs:
            out[(s, f)] = up.upload_chunks(ch)
        if register:
            for x in up.xorbs:
                known.add_xorb(x)
            if global_index is not None:
                global_index.add_session(up.xorbs, up.firsts)
    return out


def session_of(kind: str, path: str) -> str:
    """A release's upload sessions: each group's push of a table (its files' group in the
    name, `chunks-<first>-<last>`), each table's commits."""
    repo = path.split("/")[0]
    if kind == "groups":
        return f"groups/{repo}/{Path(path).stem}"
    return f"published/{repo}"


def order_of(ch: dict[str, dict[str, list]]) -> list[tuple[str, str, list]]:
    out = []
    for kind in KINDS:
        for n in sorted(ch[kind], key=lambda n: (session_of(kind, n), n)):
            out.append((session_of(kind, n), f"{kind}:{n}", ch[kind][n]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--changed", type=Path, help="slice-b's record: the changed, new and gone ids")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cha, chb = chunked(a.a), chunked(a.b)
    rec = json.loads(a.changed.read_text()) if a.changed else {}
    touched = set(rec.get("changed_ids", [])) | set(rec.get("new_ids", [])) | set(rec.get("gone_ids", []))

    # A's uploads from scratch, to form its xorbs; then B's, three ways
    a_order = order_of(cha)
    gi_x, gi_s = xet.GlobalIndex("xorb"), xet.GlobalIndex("session")
    known_a = xet.Index()
    simulate(a_order, known_a, None)
    for gi in (gi_x, gi_s):  # (A's sessions again, for the global index's shards)
        simulate(a_order, xet.Index(), gi)
    b_order = order_of(chb)
    runs = {"xet": simulate(b_order, known_a, None)}
    for name, gi in (("xet_global_xorb", gi_x), ("xet_global_session", gi_s)):
        runs[name] = simulate(b_order, xet.Index(), gi)

    a_chunks = {h for kind in KINDS for ch in cha[kind].values() for h, _ in ch}
    rows = defaultdict(lambda: defaultdict(float))
    import pyarrow.parquet as pq

    for kind in KINDS:
        fa, fb = files(a.a, kind), files(a.b, kind)
        a_pages: dict[tuple[str, str], set] = defaultdict(set)
        for n, p in fa.items():
            a_pages[table_of(n)] |= {h for h, *_ in page_hashes(p.read_bytes())}
        for n, p in fb.items():
            st = table_of(n)
            key = (kind, *st)
            data = p.read_bytes()
            r = rows[key]
            r["files"] += 1
            r["bytes"] += len(data)
            if n in fa and fa[n].read_bytes() == data:
                r["identical_files"] += len(data)
            r["pages"] += sum(ln for h, _, ln, _, _ in page_hashes(data) if h in a_pages[st])
            r["chunks"] += sum(s for h, s in chb[kind][n] if h in a_chunks)
            for name, res in runs.items():
                m = res[(session_of(kind, n), f"{kind}:{n}")]
                r[name] += m.total - m.new  # (total counts a refused range twice: the bytes not sent)
                r[name + "_refused"] += m.refused
            if touched and "id" not in pq.ParquetFile(p).schema_arrow.names:
                r["predicted_na"] += len(data)  # (claims_labels: keyed by `ref`, not by entity)
            elif touched:
                ids = pq.read_table(p, columns=["id"]).column("id").to_pylist()
                gone = sorted(rec.get("gone_ids", [])) if kind == "published" else []
                for off, ln, r0, r1 in page_rows(data, p):
                    if r1 <= r0 or touched.intersection(ids[r0:r1]):
                        continue
                    if any(ids[r0] <= g <= ids[r1 - 1] for g in gone):
                        continue
                    r["predicted"] += ln
    out = []
    for (kind, set_, table), r in sorted(rows.items()):
        out.append({"kind": kind, "set": set_, "table": table, **{k: round(v) for k, v in r.items()}})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    show(out)


def show(out: list[dict]) -> None:
    cols = ["identical_files", "pages", "predicted", "chunks", "xet", "xet_global_xorb", "xet_global_session"]
    print(f"{'kind':10} {'set':8} {'table':14} {'MB':>7} " + " ".join(f"{c[:12]:>12}" for c in cols))
    tot = defaultdict(lambda: defaultdict(float))
    for r in out:
        for g in (r["kind"], f"{r['kind']}:{r['set']}"):
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    tot[g][k] += v
        print(f"{r['kind']:10} {r['set']:8} {r['table']:14} {r['bytes'] / 1e6:7.2f} "
              + " ".join(f"{100 * r.get(c, 0) / max(1, r['bytes']):11.1f}%" for c in cols))
    for g, r in tot.items():
        print(f"{g:34} {r['bytes'] / 1e6:7.2f} " + " ".join(f"{100 * r.get(c, 0) / max(1, r['bytes']):11.1f}%" for c in cols))


if __name__ == "__main__":
    main()
