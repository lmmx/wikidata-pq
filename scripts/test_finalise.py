#!/usr/bin/env python3
"""Test finalise's worker-pool steps and its single upload, on small local data (no Hub):

- compaction: group files of a few languages rewritten one output file per job, resumed
  from `files.jsonl`, and handed to the sort's local copy (`keep_as_hub_copy`)
- the sort's input check against compaction's manifest, and a changed file caught
- keys sorted in memory as jobs (the larger ones apart), rows kept in stable id order
- the sort's commit operations: a key's group files replaced, an unknown file refused
- claims_labels: refs of a sample of claims taken per file (resumed from the files
  written) and each language's rows written as a job, both equal to a one-process run
- one refs job on a full claims file: its peak memory, times 2 workers, under 80 GiB,
  and its time under 2 min (about 50 s one at a time before 2026-10-05)

Synthetic labels, and claims sampled from the local copy of the sorted claims. Runs in a
temporary directory; reads the claims files only.

Usage: python scripts/test_finalise.py [claims key dir] [rows per file]
(default releases/20260928/hub/claims/all, 20,000)
"""

import json
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pyarrow.parquet as pq

CLAIMS = Path(sys.argv[1] if len(sys.argv) > 1 else "releases/20260928/hub/claims/all").resolve()
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
LANGS = {"en": 40_000, "de": 15_000, "fr": 9_000, "xx": 300}  # rows per language


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def raises(fn) -> bool:
    try:
        fn()
    except RuntimeError:
        return True
    return False


def labels(lang: str, n: int, seed: int) -> pl.DataFrame:
    """n label rows with ids out of order and some ids repeated."""
    ids = pl.Series([f"Q{(i * 7919 + seed) % (n // 2 + 1)}" for i in range(n)])
    return pl.DataFrame({"id": ids, "language": lang, "value": [f"{lang} {i}" for i in range(n)]})


def make_groups(src: Path) -> dict[str, pl.DataFrame]:
    """Each language's rows split into 5 group files, as the grouped upload leaves them."""
    rows = {}
    for k, (lang, n) in enumerate(LANGS.items()):
        df = labels(lang, n, k)
        rows[lang] = df
        (src / lang).mkdir(parents=True)
        for g, part in enumerate(df.iter_slices(-(-n // 5))):
            part.write_parquet(src / lang / f"chunks-{g * 10:02d}-{g * 10 + 9:02d}.parquet")
    return rows


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        os.environ["WIKIDATA_RELEASE"] = "finalisetest"
        os.environ["WIKIDATA_FINALISE_WORKERS"] = "3"
        from wikidata import claims_labels as cl
        from wikidata import compact
        from wikidata import sort_by_id as sbi
        from wikidata.config import Table

        # Compaction
        rows = make_groups(compact._src_dir(Table.LABEL))
        en_src = list((compact._src_dir(Table.LABEL) / "en").glob("*.parquet"))
        compact.COMPACT_FILE_BYTES = sum(p.stat().st_size for p in en_src) // 3
        compact.rewrite_table(Table.LABEL, 49)
        manifest = compact.read_manifest(Table.LABEL)
        check("every key compacted", set(manifest) == set(LANGS))
        check("en compacted into several files", len(manifest["en"]["files"]) > 1)
        same = all(
            pl.concat([pl.read_parquet(compact._out_dir(Table.LABEL) / k / f["name"]) for f in e["files"]]).equals(rows[k])
            for k, e in manifest.items()
        )
        check("each key's compacted files hold its rows in order", same)
        names = {f["name"] for e in manifest.values() for f in e["files"]}
        check("output names padded to one width", len({len(n) for n in names}) == 1)

        mpath = compact._manifest_path(Table.LABEL)
        lines = mpath.read_text().splitlines()
        mpath.write_text("\n".join(line for line in lines if json.loads(line)["key"] != "en") + "\n")
        last = compact._out_dir(Table.LABEL) / "en" / manifest["en"]["files"][-1]["name"]
        last.unlink()
        before = len(compact._files_path(Table.LABEL).read_text().splitlines())
        compact.rewrite_table(Table.LABEL, 49)
        after = len(compact._files_path(Table.LABEL).read_text().splitlines())
        check("a restart rewrites only the missing file", after == before + 1)
        check("the resumed key is in the manifest again", compact.read_manifest(Table.LABEL)["en"] == manifest["en"])

        compact.keep_as_hub_copy(Table.LABEL)
        sbi.check_compacted_copy(Table.LABEL)
        check("the sort's input matches compaction's manifest", True)
        victim = sbi._src_dir(Table.LABEL) / "de" / manifest["de"]["files"][0]["name"]
        good = victim.read_bytes()
        victim.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
        check("a changed compacted file is caught", raises(lambda: sbi.check_compacted_copy(Table.LABEL)))
        victim.write_bytes(good)

        # Sort, keys in memory as jobs
        sbi.SORT_IN_MEMORY_BYTES = 8 * max(sum(p.stat().st_size for p in s) for s in sbi._local_keys(Table.LABEL).values()) - 1
        sbi.write_table(Table.LABEL)
        sorted_manifest = sbi.read_manifest(Table.LABEL)
        check("every key sorted", set(sorted_manifest) == set(LANGS))
        same = all(
            pl.concat([pl.read_parquet(sbi._out_dir(Table.LABEL) / k / f["name"]) for f in e["files"]]).equals(
                rows[k].sort("id", maintain_order=True)
            )
            for k, e in sorted_manifest.items()
        )
        check("each key sorted by id, stably", same)

        # Commit operations: the Hub has the group files
        entry = sorted_manifest["de"]
        groups = set(manifest["de"]["sources"])
        remote = {n: SimpleNamespace() for n in groups}
        ops = sbi._key_operations(Table.LABEL, entry, remote, groups)
        adds = {o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationAdd"}
        deletes = {o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationDelete"}
        check("commit adds the sorted files", adds == {f"de/{f['name']}" for f in entry["files"]})
        check("commit deletes the group files", deletes == {f"de/{n}" for n in groups})
        remote["chunks-99-99.parquet"] = SimpleNamespace()
        check("commit refuses an unknown file", raises(lambda: sbi._key_operations(Table.LABEL, entry, remote, groups)))

        # claims_labels: refs per claims file, then languages
        claims_dir = Path("claims")
        claims_dir.mkdir()
        files = sorted(CLAIMS.glob("*.parquet"))
        for k, f in enumerate(files[:: max(1, len(files) // 3)][:3]):
            rows_k = pq.ParquetFile(f).read_row_group(0).slice(0, ROWS)
            pq.write_table(rows_k, claims_dir / f"part-{k}.parquet", row_group_size=ROWS // 4)
        def whole(f: Path) -> pl.DataFrame:  # one file read whole, as before row groups
            return cl._refs(pl.scan_parquet(f)).collect()

        want = pl.concat([whole(f) for f in sorted(claims_dir.glob("*.parquet"))]).unique().sort("field", "ref")
        got = cl.collect_refs(claims_dir)
        check(f"refs per file job equal whole files read in one process ({got.height:,} refs)", got.equals(want))
        first = sorted(cl.REFS_DIR.glob("*.parquet"))[0]
        first.unlink()
        check("refs resumed from the files written", cl.collect_refs(claims_dir).equals(want))

        cl.BUILD_DIR.mkdir(parents=True, exist_ok=True)
        want.write_parquet(cl.REFS_PATH)
        ids = want["id"].unique().to_list()
        label_dirs = [Path("labels_a"), Path("labels_b")]
        for j, d in enumerate(label_dirs):
            for lang in ("en", "de", "zz"):
                if lang == "zz" and j == 0:
                    continue
                (d / lang).mkdir(parents=True)
                n = len(ids) // (2 + j)
                pl.DataFrame({"id": ids[j * n : (j + 1) * n], "language": lang, "value": [f"{lang}{j} {i}" for i in range(n)]}).write_parquet(d / lang / "part-0.parquet")
        out = Path("groups")
        n = cl.write_groups(cl.REFS_PATH, label_dirs, out, "chunks-00-49")
        langs = {p.parent.name for p in out.glob("*/*.parquet")}
        check("one file per language with labels", langs == {"en", "de", "zz"})
        expect = {}
        for lang in langs:
            lab = pl.concat([pl.read_parquet(d / lang / "part-0.parquet") for d in label_dirs if (d / lang).is_dir()])
            expect[lang] = (
                want.join(lab.rename({"value": "label"}), on="id").select("field", "ref", "language", "label").sort("ref", "field")
            )
        same = all(
            pl.read_parquet(out / lang / "chunks-00-49.parquet").sort("ref", "field", "label").equals(expect[lang].sort("ref", "field", "label"))
            for lang in langs
        )
        check(f"each language's rows equal a join of refs and labels ({n:,} rows)", same)
        (out / "de" / "chunks-00-49.parquet").unlink()
        check("languages resumed: only the missing one written", cl.write_groups(cl.REFS_PATH, label_dirs, out, "chunks-00-49") == expect["de"].height)

        # Memory of one refs job on a full claims file, as a worker runs it
        ctx = multiprocessing.get_context("spawn")
        t0 = time.time()
        with ProcessPoolExecutor(1, mp_context=ctx) as pool:
            pool.submit(cl.file_refs, files[0]).result()
        secs = time.time() - t0
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2  # KiB to GiB
        workers = 2  # FINALISE_LARGE_WORKERS: refs jobs at once
        check(
            f"refs of a full claims file ({files[0].name}) peaked at {peak:.1f} GiB;"
            f" {workers} at once: {workers * peak:.0f} GiB",
            workers * peak < 80,
        )
        # One at a time before 2026-10-05: about 50 s a file
        check(f"refs of a full claims file took {secs:.0f} s (about 50 s before)", secs < 120)
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
