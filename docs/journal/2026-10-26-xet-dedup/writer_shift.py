"""The published writer's layout and Xet's dedup: what moves between releases, and a
layout that wouldn't. Everything written by the pipeline's own writer
(compact._write_file_parallel: pq-glue's arrow-rs engine, zstd 3, CDC pages, a page
index, the sort column declared), its row groups sized by sort_by_id.row_group_rows
(~128 MiB of decoded rows) as the sort sizes them.

**Real claims** (`claims`): A's and B's published claims tables (each set's, all rows),
rewritten as:
- **master**: as the sort writes them, each release's own `row_group_rows` (from its
  own rows' decoded bytes), row groups cut every that many rows from the file's start.
  This must give back the published files byte for byte (checked);
- **master, A's row_group_rows**: B with A's rows a row group, to part the shift of rows
  from the change of `row_group_rows` itself;
- **content-defined row groups**: a row group cut at the first entity boundary, at least
  half the target rows in, whose id's hash is 0 mod M (M fixed for the table: A's
  entities a row group over 2), or at twice the target; so a boundary moves only where
  the rows around it change.

**Big language tables** (`terms`): the slice's language tables are a page or two a
file, where a release's large languages are many row groups (millions of rows). For each
of labels, descriptions, aliases and links, a table of 4M rows (id, language, value): ids
`Q1`..., sorted as strings, the values the slice's own (that table's, every language),
drawn at random (seeded), one language. B is A with the table's measured weekly churn
(TERMS_CHURN, 1x, 2x, 5x) applied at random rows (`--placement uniform`), or 60% of it in
sweeps of 5-60 consecutive rows within 4 of 32 equal windows and the rest at random
(`clustered`: #188's measured concentration). Written by master's layout, with
content-defined row groups, and with smaller CDC pages.

For each: B's bytes in Xet chunks already among A's (xet.chunks_of), and in pages
byte-identical to A's, with the footer and page index apart.

    python writer_shift.py --a A --b B --scratch SCRATCH --out results/writer_shift.json
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pqpages  # noqa: E402
import xet  # noqa: E402


def presence(fa: list[Path], fb: list[Path]) -> dict:
    """B's files' bytes in chunks among A's, in pages among A's, and in footers."""
    ach = {h for p in fa for h, _ in xet.chunks_of(p.read_bytes())}
    apg = set()
    for p in fa:
        d = p.read_bytes()
        for c in pqpages.pages(d):
            for o, n, _, _ in c["pages"]:
                apg.add(hashlib.sha256(d[o:o + n]).digest())
    out = {"bytes": 0, "chunks_same": 0, "pages_same": 0, "footer_index": 0, "rgs": 0}
    for p in fb:
        d = p.read_bytes()
        out["bytes"] += len(d)
        out["chunks_same"] += sum(s for h, s in xet.chunks_of(d) if h in ach)
        pg = pqpages.pages(d)
        out["rgs"] += len({c["rg"] for c in pg})
        out["pages_same"] += sum(n for c in pg for o, n, _, _ in c["pages"]
                                 if hashlib.sha256(d[o:o + n]).digest() in apg)
        f0, _ = pqpages.footer_span(d)
        ix = pqpages.index_span(d)
        out["footer_index"] += len(d) - f0 + (ix[1] - ix[0] if ix else 0)
    out["chunk_dedup"] = round(out["chunks_same"] / out["bytes"], 4)
    out["page_dedup"] = round(out["pages_same"] / out["bytes"], 4)
    out["footer_share"] = round(out["footer_index"] / out["bytes"], 4)
    return out


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def cd_cuts(ids: list[str], target: int, m: int) -> list[int]:
    """Content-defined row group ends: at an entity's first row, at least target/2 rows
    in, whose id hash % m == 0; or the first entity boundary at 2 * target rows."""
    cuts, start = [], 0
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            continue
        n = i - start
        if (n >= target // 2 and h64(ids[i]) % m == 0) or n >= 2 * target:
            cuts.append(i)
            start = i
    cuts.append(len(ids))
    return cuts


def write_fixed(t, dst: Path, rg_rows: int) -> None:
    from wikidata import compact, sort_by_id

    compact._write_file_parallel(dst, t.schema, t.to_batches(max_chunksize=rg_rows), rg_rows,
                                 sort_by_id._sorting(t.schema, sort_by_id.sort_column("claims")))


def write_cuts(t, dst: Path, cuts: list[int], scratch: Path) -> None:
    """Row groups ending at `cuts`, each written by the same engine and settings as
    _write_file_parallel (pq-glue's, which with CDC encodes whole row groups apart and
    glues them), then glued in order (pq_glue.concat)."""
    import pq_glue

    parts, s = [], 0
    for i, e in enumerate(cuts):
        p = scratch / f"{dst.stem}.rg{i:05d}.parquet"
        pq_glue.write_parquet(t.slice(s, e - s), str(p), engine="native", row_group_size=e - s, compression="zstd",
                              compression_level=3, content_defined_chunking=True, sorting_columns=[(0, False, False)])
        parts.append(str(p))
        s = e
    pq_glue.concat(parts, str(dst))
    for p in parts:
        Path(p).unlink()


def claims(a: argparse.Namespace, out: dict) -> None:
    import pyarrow.parquet as pq
    from wikidata import sort_by_id

    for set_, repo in (("main", "wikidata-claims"), ("scholar", "wikidata-scholar-claims")):
        fa = next((a.a / "published").glob(f"*__{repo}/*/all/part-0-of-1.parquet"))
        fb = next((a.b / "published").glob(f"*__{repo}/*/all/part-0-of-1.parquet"))
        ta, tb = pq.read_table(fa), pq.read_table(fb)
        ra, rb = sort_by_id.row_group_rows(ta), sort_by_id.row_group_rows(tb)
        d = a.scratch / f"claims-{set_}"
        d.mkdir(parents=True, exist_ok=True)
        res = {"rows_a": ta.num_rows, "rows_b": tb.num_rows, "row_group_rows_a": ra, "row_group_rows_b": rb}
        write_fixed(ta, d / "A.parquet", ra)
        write_fixed(tb, d / "B.parquet", rb)
        res["rewrite_identical"] = (d / "A.parquet").read_bytes() == fa.read_bytes() and \
            (d / "B.parquet").read_bytes() == fb.read_bytes()
        res["master"] = presence([d / "A.parquet"], [d / "B.parquet"])
        write_fixed(tb, d / "B_ra.parquet", ra)
        res["master_a_rows"] = presence([d / "A.parquet"], [d / "B_ra.parquet"])
        ida = ta.column("id").to_pylist()
        idb = tb.column("id").to_pylist()
        ents = len(set(ida))
        m = max(1, round(ents * ra / ta.num_rows / 2))
        res["cd_modulus"] = m
        write_cuts(ta, d / "A_cd.parquet", cd_cuts(ida, ra, m), d)
        write_cuts(tb, d / "B_cd.parquet", cd_cuts(idb, ra, m), d)
        res["content_defined"] = presence([d / "A_cd.parquet"], [d / "B_cd.parquet"])
        out[f"claims_{set_}"] = res
        print(set_, json.dumps(res), flush=True)
        shutil.rmtree(d)


# Each language table's weekly churn on the slice at 1x (rowgroups.py on B1: the share
# of the table's rows of an entity whose rows in that language changed, and the share of
# those whose row count changed), main set
TERMS_CHURN = {"labels": ("wikidata-labels", 0.000097, 0.84),
               "descriptions": ("wikidata-descriptions", 0.000207, 0.88),
               "aliases": ("wikidata-aliases", 0.000291, 0.96),
               "links": ("wikidata-links", 0.000071, 0.83),
               # (the scholarly set's: B2 against A over 2, its 1x too few to count)
               "scholar descriptions": ("wikidata-scholar-descriptions", 0.000018, 0.11)}


def write_pq(t, dst: Path, rg_rows: int, cdc=True) -> None:
    """_write_file_parallel's call (pq-glue, engine "parallel", zstd 3, the sort column
    declared), with the CDC options given: True for the writer's defaults (256 KiB-1 MiB
    pages), or a dict."""
    import pq_glue

    pq_glue.write_parquet(t, str(dst), engine="parallel", row_group_size=rg_rows, compression="zstd",
                          compression_level=3, content_defined_chunking=cdc, sorting_columns=[(0, False, False)])


def terms(a: argparse.Namespace, out: dict) -> None:
    """A big language table of each kind (its values the slice's own, shuffled) at each
    multiple of its measured churn: B is A with `share` of its rows touched at random
    (uniformly: no sweep), `count` of those a row added (a new term for an entity; half
    of them) or removed, the rest a value edited. Written by master's layout, with
    content-defined row groups, and with master's row groups and smaller CDC pages
    (32-128 KiB)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from wikidata import sort_by_id

    n = a.terms_rows
    res_all = out.setdefault("terms" if a.placement == "uniform" else "terms_clustered", {})
    for table, (repo, share, count) in TERMS_CHURN.items():
        if table in res_all:
            continue
        vals = []
        for p in sorted((a.a / "published").glob(f"*__{repo}/*/*/part-*.parquet")):
            col = "value" if "value" in pq.ParquetFile(p).schema_arrow.names else "title"  # (links: its title)
            vals += pq.read_table(p, columns=[col]).column(col).to_pylist()
        vals = [v if isinstance(v, str) else json.dumps(v) for v in vals]
        rng = random.Random(7)
        ida = sorted(f"Q{i}" for i in range(1, n + 1))
        va = [vals[rng.randrange(len(vals))] for _ in range(n)]
        sch = pa.schema([("id", pa.large_string()), ("language", pa.large_string()), ("value", pa.large_string())])
        mk = lambda i, v: pa.table({"id": i, "language": ["xx"] * len(i), "value": v}, schema=sch)  # noqa: E731
        ta = mk(ida, va)
        ra = sort_by_id.row_group_rows(ta)
        d = a.scratch / f"terms-{table.replace(' ', '-')}"
        d.mkdir(parents=True, exist_ok=True)
        variants = {"master": lambda t, p, r: write_pq(t, p, r),
                    "content_defined": None,
                    "small_pages": lambda t, p, r: write_pq(t, p, r, {"min_chunk_size": 32 << 10,
                                                                       "max_chunk_size": 128 << 10, "norm_level": 0})}
        m = max(1, ra // 2)
        files_a = {}
        for v, w in variants.items():
            files_a[v] = d / f"A_{v}.parquet"
            if w is None:
                write_cuts(ta, files_a[v], cd_cuts(ida, ra, m), d)
            else:
                w(ta, files_a[v], ra)
        res = {"rows": n, "row_group_rows": ra, "share": share, "count_share": count}
        for mult in (1, 2, 5):
            rng = random.Random(100 + mult)
            k = max(1, round(share * mult * n))
            if a.placement == "uniform":
                pos = sorted(rng.sample(range(n), k))
            else:  # (#188's concentration: the 4 busiest of 32 windows hold ~60% of the
                # changes, as sweeps over consecutive entities; the rest at random)
                busy = rng.sample(range(32), 4)
                sel: set[int] = set(rng.sample(range(n), k - round(0.6 * k)))
                while len(sel) < k:
                    w = busy[rng.randrange(4)]
                    start = w * (n // 32) + rng.randrange(n // 32)
                    sel.update(range(start, min(n, start + rng.randint(5, 60))))
                pos = sorted(sel)[:k] if len(sel) > k else sorted(sel)
            idb, vb = [], []
            j = 0
            kinds = [("add" if rng.random() < 0.5 else "remove") if rng.random() < count else "edit" for _ in pos]
            todo = dict(zip(pos, kinds))
            for i in range(n):
                kind = todo.get(i)
                if kind == "remove":
                    continue
                idb.append(ida[i])
                vb.append(va[i] + " (edited)" if kind == "edit" else va[i])
                if kind == "add":  # (a new term: an id that sorts right after this one)
                    idb.append(ida[i] + "0")
                    vb.append(vals[rng.randrange(len(vals))])
                    j += 1
            order = sorted(range(len(idb)), key=idb.__getitem__)
            idb, vb = [idb[i] for i in order], [vb[i] for i in order]
            tb = mk(idb, vb)
            rb = sort_by_id.row_group_rows(tb)
            r = {"touched": k, "row_group_rows_b": rb}
            for v, w in variants.items():
                fb = d / f"B_{v}.parquet"
                if w is None:
                    write_cuts(tb, fb, cd_cuts(idb, ra, m), d)
                else:
                    w(tb, fb, rb)
                r[v] = presence([files_a[v]], [fb])
                fb.unlink()
            res[f"{mult}x"] = r
            print(table, mult, json.dumps(r), flush=True)
        res_all[table] = res
        shutil.rmtree(d)
        a.out.write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--terms-rows", type=int, default=4_000_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=["claims", "terms"])
    ap.add_argument("--placement", choices=["uniform", "clustered"], default="uniform")
    a = ap.parse_args()
    sys.path.insert(0, str(a.src.resolve() / "src"))
    a.scratch.mkdir(parents=True, exist_ok=True)
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    if a.only in (None, "claims"):
        claims(a, out)
    if a.only in (None, "terms"):
        terms(a, out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
