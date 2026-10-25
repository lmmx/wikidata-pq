#!/usr/bin/env python
"""claims_labels' refs from the sidecars (claims_refs.py) against the refs read again from
the claims, on the benchmark's chunk files (run with WIKIDATA_RELEASE set, from the repo):

    WIKIDATA_RELEASE=20261012 python scripts/test_claims_refs.py <chunk files...>

In a scratch directory (the release's work directories in it), each chunk's claims
partition file is written as the split writes it (claims_build.write_claims, a batch of
statements at a time) and, for another copy, as the pipeline's JSON path does
(process.normalise_claims_direct, all at once). Checks:

- per chunk: the sidecar both write is the same file, and holds (its ids put back) the refs
  `file_refs` before sidecars (copied below as `file_refs_before`) takes from the claims
  file, as does `file_refs` now;
- the chunks' claims sorted by id into two files (the local copy of the sorted claims):
  `sidecar_refs` gives the frame `collect_refs` reads from them, and refs.parquet written
  from each is the same bytes;
- claims_labels' language files (`write_groups`, with the chunks' labels partitioned by
  language as the pipeline partitions them) from each refs.parquet: the same bytes;
- a sidecar missing, or a stray one, and `sidecar_refs` gives None (collect_refs is used).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def file_refs_before(path: Path) -> pl.DataFrame:
    """file_refs as it was before the sidecars (master at fa959e4), verbatim (_snak_refs
    and _refs_leaves are unchanged)."""
    from wikidata.claims_labels import _refs_leaves, _snak_refs

    def snak_table(snaks):
        return pa.table({"property": pc.struct_field(snaks, "property"),
                         "datavalue": pc.struct_field(snaks, "datavalue")})
    f = pq.ParquetFile(path)
    t = f.read(columns=_refs_leaves(f))
    qualifiers = pc.list_flatten(pc.struct_field(pc.list_flatten(t["qualifiers"]), "value"))
    groups = pc.list_flatten(pc.struct_field(pc.list_flatten(t["references"]), "snaks"))
    references = pc.list_flatten(pc.struct_field(groups, "value"))
    parts = [t.select(["property", "datavalue"]), snak_table(qualifiers), snak_table(references)]
    return pl.concat([_snak_refs(pl.from_arrow(p).lazy()) for p in parts]).unique().collect()


def canon(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("field", "ref", "id")


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def main() -> None:
    files = [Path(f).resolve() for f in sys.argv[1:]]
    scratch = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    os.chdir(scratch)  # the release's work directories are relative to it

    from wikidata import claims_build, claims_labels as cl, claims_refs, process
    from wikidata.config import HUB_COPY_DIR, ROOT_DATA_DIR, OUTPUT_DIR, WORK_DIR, Table
    from wikidata.partitioning import partition_parquet, prepare_for_partition

    sidecars = WORK_DIR / claims_refs.SIDECAR_DIR
    split_claims = ROOT_DATA_DIR / "routed" / claims_build.CLAIMS_DIR
    labels_dir = HUB_COPY_DIR / Table.LABEL
    claim_files = []
    for i, path in enumerate(files):
        name = f"chunk_{i}.parquet"
        part = pl.read_parquet(path, columns=["id", "claims"])
        built = split_claims / name
        assert claims_build.write_claims(part["id"], claims_build.text_of(part), None, built) is None, path.name
        side = sorted(sidecars.glob("*.parquet"))
        by_split = {p.name: p.read_bytes() for p in side}
        json_claims = OUTPUT_DIR / Table.CLAIMS / name
        tmp = process.normalise_claims_direct(path, json_claims, scratch / "lookup.parquet",
                                             scratch / "q.parquet", process.n_ids(part))
        tmp.replace(json_claims)
        # (the same content: the split writes its file a row group at a time, each encoded on
        # its own, so not the same bytes)
        check(f"{path.name}: the split's and the pipeline's claims files hold the same statements",
              pq.read_table(built).equals(pq.read_table(json_claims)))
        after = {p.name: p.read_bytes() for p in sorted(sidecars.glob("*.parquet"))}
        check(f"{path.name}: the pipeline's sidecar is the split's (same name, same bytes)", after == by_split)
        ids = pq.read_table(built, columns=["id"]).column("id")
        rows = len(ids)
        mine = [sidecars / f"{rows}-{ids[0]}-{ids[-1]}.parquet"]
        sidecar = pl.read_parquet(mine[0])
        want = canon(file_refs_before(built))
        check(f"{path.name}: sidecar = file_refs before ({want.height:,} refs, {rows:,} statements, "
              f"{mine[0].stat().st_size / 1024:.0f} KiB beside {built.stat().st_size / 1024**2:.1f} MiB)",
              canon(claims_refs.with_id(sidecar)).equals(want) and sidecar.columns == ["field", "ref"] and claims_refs._rows(mine[0]) == rows)
        check(f"{path.name}: file_refs now = file_refs before", canon(cl.file_refs(built)).equals(want))
        claim_files.append(built)
        # The chunk's labels, partitioned by language as the pipeline partitions them
        labels = scratch / "labels" / name
        process.normalise_map_direct(path, labels, key="labels")
        partition_parquet("language", prepare_for_partition(labels, Table.LABEL), name, labels_dir, scratch / "audit")

    # The sorted claims: every chunk's statements stably sorted by id, in two files
    claims_dir = HUB_COPY_DIR / Table.CLAIMS / "all"
    claims_dir.mkdir(parents=True)
    table = pa.concat_tables([pq.read_table(f) for f in claim_files])
    table = table.take(pc.sort_indices(table, [("id", "ascending")]))
    half = table.num_rows // 2
    pq.write_table(table.slice(0, half), claims_dir / "part-0.parquet")
    pq.write_table(table.slice(half), claims_dir / "part-1.parquet")
    del table

    from_sidecars = claims_refs.sidecar_refs(sidecars, claims_dir)
    from_claims = cl.collect_refs(claims_dir)
    check(f"union of sidecars = collect_refs from the sorted claims ({from_claims.height:,} refs)",
          from_sidecars is not None and from_sidecars.equals(from_claims))
    check("union of sidecars, deduplicated as read, = union all at once",
          claims_refs.union(sorted(sidecars.glob("*.parquet")), every=1).equals(from_sidecars))
    a, b = scratch / "a" / "refs.parquet", scratch / "b" / "refs.parquet"
    for refs, p in ((from_sidecars, a), (from_claims, b)):
        p.parent.mkdir()
        refs.write_parquet(p)
    check("refs.parquet: the same bytes", a.read_bytes() == b.read_bytes())

    outs = []
    for p in (a, b):
        out = p.parent / "groups"
        n = cl.write_groups(p, [labels_dir], out, "chunks-0000-0010")
        outs.append({f.relative_to(out): f.read_bytes() for f in sorted(out.rglob("*.parquet"))})
        print(f"     {n:,} claims_labels rows in {len(outs[-1])} language files", flush=True)
    check("claims_labels' language files: the same names and bytes", outs[0] == outs[1] and len(outs[0]) > 0)

    # Restart safety: a missing sidecar, or a stray one, and the sorted claims are read
    first = sorted(sidecars.glob("*.parquet"))[0]
    held = first.read_bytes()
    first.unlink()
    check("a sidecar missing: None", claims_refs.sidecar_refs(sidecars, claims_dir) is None)
    first.write_bytes(held)
    shutil.copy(first, sidecars / "1-Q0-Q0.parquet")
    check("a stray sidecar: None", claims_refs.sidecar_refs(sidecars, claims_dir) is None)
    (sidecars / "1-Q0-Q0.parquet").unlink()
    check("restored: the union again", claims_refs.sidecar_refs(sidecars, claims_dir).equals(from_claims))
    check("no sidecar written outside a release's work directories",
          claims_refs.sidecar_dir(scratch.parent / "elsewhere.parquet") is None)
    shutil.rmtree(scratch)
    print("all checks passed")


if __name__ == "__main__":
    main()
