"""Where B's published bytes dedup and where they don't, by row group: the mechanism behind
measure.py's figures, at the published files' own geometry (their row groups are a
release's: ~128 MiB of decoded rows each, COMPACT_ROW_GROUP_BYTES; their pages the
writer's CDC pages).

For each published file of B with a file at the same path in A (same set, table, key):

- **Which entities' rows changed in this table** (`touched`): each entity's rows hashed
  (polars hash_rows, both releases in one process), compared by id (`ref` for
  claims_labels); an id in one release only is touched.
- **Each of B's row groups is:**
  - **aligned** if A has a row group starting at the same row index with the same row
    (by its hash): no row was added or removed before it in the file (row groups are cut
    every `row_group_rows` rows from the file's start, so one row more anywhere before
    shifts it);
  - else **shifted**;
  - and **touched** if it holds a touched entity's row.
- **Its bytes that dedup:** page level (a page byte-identical to one of A's same table),
  and Xet chunk level (a chunk, by its start, whose hash is among A's chunks); the
  footer and page index apart.
- **By column encoding:** dictionary-encoded column chunks (a dictionary page) apart from
  the rest, in shifted untouched row groups: a dictionary is built in the order values
  first appear, so a shift reorders it and every index changes.

    python rowgroups.py --a A --b B --out results/rowgroups-1x.json
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
from measure import files, table_of  # noqa: E402


def touched_ids(fa: Path, fb: Path, key: str) -> tuple[set, dict, list, list]:
    """The touched ids, counts, and each release's row hashes (for row groups' first rows)."""
    import polars as pl

    hashes = []

    def digest(p):
        df = pl.read_parquet(p)
        h = pl.Series("h", df.hash_rows(seed=7))
        hashes.append(h.to_list())
        return df.select(pl.col(key), h).group_by(key).agg(pl.col("h").sort().alias("hs"), pl.len().alias("n"))

    a, b = digest(fa), digest(fb)
    j = a.join(b, on=key, how="full", coalesce=True, suffix="_b")
    t = j.filter(pl.col("n").is_null() | pl.col("n_b").is_null() | (pl.col("hs") != pl.col("hs_b")))
    count_changed = t.filter(pl.col("n").fill_null(0) != pl.col("n_b").fill_null(0)).height
    return set(t[key].to_list()), {"touched": t.height, "count_changed": count_changed, "entities_b": b.height,
                                   "rows_b": int(b["n"].sum())}, hashes[0], hashes[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    cha = pickle.loads((a.a / "chunks.pkl").read_bytes())
    chb = pickle.loads((a.b / "chunks.pkl").read_bytes())
    a_chunks = {h for kind in cha for ch in cha[kind].values() for h, _ in ch}
    fa, fb = files(a.a, "published"), files(a.b, "published")
    a_pages: dict[tuple, set] = defaultdict(set)
    for n, p in fa.items():
        d = p.read_bytes()
        for c in pqpages.pages(d):
            for o, ln, _, _ in c["pages"]:
                a_pages[table_of(n)].add(hashlib.sha256(d[o:o + ln]).digest())
    agg: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    for n, pb in sorted(fb.items()):
        if n not in fa:
            continue
        st = table_of(n)
        r = agg[st]
        key = "id" if "id" in pq.ParquetFile(pb).schema_arrow.names else "ref"
        db = pb.read_bytes()
        tid, info, ha, hb = touched_ids(fa[n], pb, key)
        for k, v in info.items():
            r[k] += v
        idb = pq.read_table(pb, columns=[key]).column(key).to_pylist()
        mda, mdb = pq.ParquetFile(fa[n]).metadata, pq.ParquetFile(pb).metadata
        starts_a, s = {}, 0
        for i in range(mda.num_row_groups):
            starts_a[s] = (ha[s], mda.row_group(i).num_rows)
            s += mda.row_group(i).num_rows
        cats, s = [], 0
        for i in range(mdb.num_row_groups):
            m = mdb.row_group(i).num_rows
            # aligned: A has a row group starting at the same row with the same row, and
            # (the last) as many rows
            aligned = s in starts_a and starts_a[s][0] == hb[s] and (starts_a[s][1] == m or i + 1 < mdb.num_row_groups)
            touched = bool(tid.intersection(idb[s:s + m]))
            cats.append(("aligned" if aligned else "shifted") + ("_touched" if touched else "_untouched"))
            r[f"rgs_{cats[-1]}"] += 1
            s += m
        spans = []
        for c in pqpages.pages(db):
            cat = cats[c["rg"]]
            dict_chunk = any(k == "dict" for _, _, _, k in c["pages"])
            for o, ln, _, _ in c["pages"]:
                same = hashlib.sha256(db[o:o + ln]).digest() in a_pages[st]
                r[f"{cat}_bytes"] += ln
                r[f"{cat}_pages_same"] += ln if same else 0
                if cat == "shifted_untouched":
                    e = "dict" if dict_chunk else "plain"
                    r[f"shifted_untouched_{e}_bytes"] += ln
                    r[f"shifted_untouched_{e}_same"] += ln if same else 0
                spans.append((o, o + ln, cat))
        spans.sort()
        # Xet chunks, each by the category of the page its start falls in (else footer)
        import bisect

        starts = [x[0] for x in spans]
        off = 0
        for h, size in chb["published"][n]:
            j = bisect.bisect_right(starts, off) - 1
            cat = spans[j][2] if j >= 0 and off < spans[j][1] else "footer"
            r[f"{cat}_chunk_bytes"] += size
            r[f"{cat}_chunks_same"] += size if h in a_chunks else 0
            off += size
        r["bytes"] += len(db)
        r["files"] += 1
    out = [{"set": s, "table": t, **{k: round(v) for k, v in r.items()}} for (s, t), r in sorted(agg.items())]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    cats = ("aligned_untouched", "aligned_touched", "shifted_untouched", "shifted_touched")
    print(f"{'set':8} {'table':14} {'MB':>6} {'touched':>8} {'cnt chg':>7} " +
          " ".join(f"{c:>26}" for c in cats) + f" {'shifted dict/plain same':>24}")
    for r in out:
        cells = []
        for c in cats:
            b = r.get(f"{c}_bytes", 0)
            cells.append(f"{r.get(f'rgs_{c}', 0):3d} rg {b / 1e6:6.2f}MB {100 * r.get(f'{c}_pages_same', 0) / b if b else 0:5.1f}%"
                         f"/{100 * r.get(f'{c}_chunks_same', 0) / max(1, r.get(f'{c}_chunk_bytes', 0)):5.1f}%")
        sd, sp = r.get("shifted_untouched_dict_bytes", 0), r.get("shifted_untouched_plain_bytes", 0)
        print(f"{r['set']:8} {r['table']:14} {r['bytes'] / 1e6:6.2f} {r.get('touched', 0):8d} {r.get('count_changed', 0):7d} "
              + " ".join(f"{x:>26}" for x in cells)
              + f" {100 * r.get('shifted_untouched_dict_same', 0) / sd if sd else 0:9.1f}% of {sd / 1e6:.2f}"
              + f" / {100 * r.get('shifted_untouched_plain_same', 0) / sp if sp else 0:5.1f}% of {sp / 1e6:.2f}")


if __name__ == "__main__":
    main()
