#!/usr/bin/env python3
"""Test finalise's worker-pool steps and its single upload, on small local data (no Hub):

- compaction: group files of a few languages rewritten one output file per job, resumed
  from `files.jsonl`, and handed to the sort's local copy (`keep_as_hub_copy`)
- a key sorted in memory not glued: its group files linked one by one, resumed mid-key,
  and sorted to the same part files, byte for byte, as from glued files; each file's
  rows read checked against compaction's
- the sort's input check against compaction's manifest, and a changed file caught
- keys sorted in memory as jobs (the larger ones apart), rows kept in stable id order
- the sort's commit operations: a key's group files replaced, an unknown file refused
- claims_labels written as its published files (build_language): the same part files,
  by name, order and sha256, as its group files compacted and sorted, at once and with
  every label three times over (deduplicated as compaction would); a key the sort would
  take through buckets written as its group file, with the same files in the end; a
  restart after a crash mid-part, after the parts but before their manifest entry, and
  with a listed part changed, rebuilding only that language, to the same bytes; rows not
  as the sort would leave them (`ref` out of order, a null `ref`) refused for the direct
  path
- claims_labels' languages joined with the refs once a batch of languages
  (build_batch): the same files however they are batched, and as a join a language; a
  batch killed after two languages rerun to the same files; an id with two labels in a
  language (rows of the same (`ref`, `field`) that differ) taken from the language's own
  join, the same rows as compacted and sorted
- claims_labels: refs of a sample of claims taken per file (resumed from the files
  written) and each language's rows written as a job, both equal to a one-process run
- one refs job on a full claims file: its peak memory, times 2 workers, under 80 GiB,
  and its time under 15 s (2 s in scripts/refs_bench.py)

- the worker pool with 60 jobs on 3 workers, workers kept and replaced after each job,
  and heavy jobs started in order while their estimated memory fits a budget

Synthetic labels, and claims sampled from the local copy of the sorted claims. Runs in a
temporary directory; reads the claims files only.

Usage: python scripts/test_finalise.py [claims key dir] [rows per file]
(default releases/20260928/hub/claims/all, 20,000)
"""

import hashlib
import json
import multiprocessing
import operator
import os
import resource
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

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
        from wikidata import parallel
        from wikidata.parallel import in_parallel, memory_budget

        # The pool, with far more jobs than workers, in both modes: replacing workers after
        # 8 jobs each hung after 6 x 8 (2026-10-05), so a hang fails here, by the alarm
        signal.signal(signal.SIGALRM, lambda *_: sys.exit("FAIL the pool hung"))
        for per_child in (None, 1):
            signal.alarm(120)
            jobs = {i: (i, 1) for i in range(60)}
            got = dict(in_parallel(operator.add, jobs, f"pool, tasks_per_child={per_child}", "job", tasks_per_child=per_child))
            signal.alarm(0)
            check(f"60 jobs on 3 workers, tasks_per_child={per_child}", got == {i: i + 1 for i in range(60)})
        signal.alarm(120)
        jobs = {i: (i, 1) for i in range(12)}
        got = dict(in_parallel(operator.add, jobs, "pool, a memory budget", "job", tasks_per_child=1,
                               memory={i: 1 for i in jobs}))
        signal.alarm(0)
        check("12 jobs on 3 workers with a memory budget", got == {i: i + 1 for i in range(12)})

        # The memory budget: jobs started in order while they fit, at least one at a time
        for budget, sizes, most in ((None, [1] * 9, 3), (10, [1] * 9, 3), (2, [1] * 9, 2),
                                    (3, [2, 2, 1, 1, 5, 1], 2), (1, [5, 5, 5], 1)):
            lock, now, peak, order = threading.Lock(), [0], [0], []

            def job(i):
                with lock:
                    now[0] += 1
                    peak[0] = max(peak[0], now[0])
                time.sleep(0.05)
                with lock:
                    now[0] -= 1
                return i

            parallel.memory_budget = lambda b=budget: b
            sizes_ = dict(enumerate(sizes))
            with ThreadPoolExecutor(3) as tp, tqdm(total=len(sizes), disable=True) as bar:
                submit = tp.submit
                tp.submit = lambda fn, i: (order.append(i), submit(fn, i))[1]
                got = dict(parallel._budgeted(tp, job, {i: (i,) for i in sizes_}, sizes_, 3, "budget", bar))
            check(f"memory budget {budget}, jobs of {sizes}: at most {most} at once, in order",
                  (got, peak[0], order) == ({i: i for i in sizes_}, most, sorted(sizes_)))
        parallel.memory_budget = memory_budget

        # Compaction
        rows = make_groups(compact._src_dir(Table.LABEL))
        en_src = list((compact._src_dir(Table.LABEL) / "en").glob("*.parquet"))
        compact.COMPACT_FILE_BYTES = sum(p.stat().st_size for p in en_src) // 3
        # Glued, as for a key the sort takes through buckets (the threshold below every key)
        sbi.SORT_IN_MEMORY_BYTES = 0
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
        # (every key in memory, the largest a larger key: over 3/16 of the threshold)
        sbi.SORT_IN_MEMORY_BYTES = 5 * max(sbi.plan_key(Table.LABEL, s).decoded for s in sbi._local_keys(Table.LABEL).values())
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

        # A key sorted in memory, not glued: the same group files as another table's
        make_groups(compact._src_dir(Table.DESC))
        compact.rewrite_table(Table.DESC, 49)
        linked = compact.read_manifest(Table.DESC)
        check("each group file of a key sorted in memory linked as a file of its own",
              all(len(e["files"]) == len(e["sources"]) == 5 for e in linked.values()))
        check("the links are the group files",
              all(os.path.samefile(compact._src_dir(Table.DESC) / k / s, compact._out_dir(Table.DESC) / k / f["name"])
                  for k, e in linked.items() for s, f in zip(e["sources"], e["files"], strict=True)))
        mpath = compact._manifest_path(Table.DESC)
        lines = mpath.read_text().splitlines()
        mpath.write_text("\n".join(line for line in lines if json.loads(line)["key"] != "en") + "\n")
        (compact._out_dir(Table.DESC) / "en" / linked["en"]["files"][2]["name"]).unlink()
        before = len(compact._files_path(Table.DESC).read_text().splitlines())
        compact.rewrite_table(Table.DESC, 49)
        after = len(compact._files_path(Table.DESC).read_text().splitlines())
        check("a restart mid-key links only the missing file", after == before + 1)
        check("the resumed key's entry as before", compact.read_manifest(Table.DESC)["en"] == linked["en"])
        compact.keep_as_hub_copy(Table.DESC)
        sbi.check_compacted_copy(Table.DESC)
        de = sbi._local_keys(Table.DESC)["de"]
        listed = sbi._listed_rows(linked["de"])
        counts = [pq.read_metadata(f).num_rows for f in de]
        check("each file's rows read as compaction listed them pass", not raises(lambda: sbi._check_read(de, counts, listed)))
        check("a file read with a row fewer is caught", raises(lambda: sbi._check_read(de, [counts[0] - 1, *counts[1:]], listed)))
        check("a file read that compaction did not list is caught",
              raises(lambda: sbi._check_read(de, counts, {**listed, de[0].name: None})))
        sbi.write_table(Table.DESC)
        desc = sbi.read_manifest(Table.DESC)
        check("each key's part files from its group files the same, by name and byte, as from its glued files",
              set(desc) == set(sorted_manifest)
              and all([(f["name"], f["sha256"]) for f in desc[k]["files"]] == [(f["name"], f["sha256"]) for f in e["files"]]
                      for k, e in sorted_manifest.items()))

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

        # claims_labels written as its published files, against its group files compacted
        # and sorted (the path before 2026-10-25, cl.DIRECT = False)
        T = Table.CLAIMS_LABELS
        url = cl.ENTITY_URL
        refs = pl.concat([
            pl.DataFrame({"field": "labels", "ref": [f"Q{i}" for i in range(4000)]}),
            pl.DataFrame({"field": "labels", "ref": [f"P{i}" for i in range(30)]}),
            pl.DataFrame({"field": "property-labels", "ref": [f"P{i}" for i in range(60)]}),
            pl.DataFrame({"field": "unit-labels", "ref": [f"{url}Q{i}" for i in range(0, 600, 7)]}),
        ]).with_columns(id=pl.col("ref").str.strip_prefix(url)).sort("field", "ref")
        cl_labels = [Path("cl_labels_a"), Path("cl_labels_b")]  # (the two sets')
        ids = [f"Q{i}" for i in range(4000)] + [f"P{i}" for i in range(60)]
        for j, d in enumerate(cl_labels):
            for lang, every in (("en", 1), ("de", 2), ("fr", 3), ("xx", 97)):
                (d / lang).mkdir(parents=True)
                mine = [x for k, x in enumerate(ids) if k % 2 == j and k % every == 0][::-1]  # (out of order)
                pl.DataFrame({"id": mine, "language": lang, "value": [f"{lang} {x}" for x in mine]}).write_parquet(
                    d / lang / "part-0.parquet")
        cl.labels_dirs = lambda: cl_labels
        cl.ensure_build_branch = lambda *a, **k: None
        api = SimpleNamespace(create_repo=lambda *a, **k: None)
        sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()

        def published(tag: str, direct: bool) -> dict[str, list[tuple[str, str]]]:
            """claims_labels built, compacted and sorted (the local stages of each), and its
            published files by key, in order: (name, sha256)."""
            cl.DIRECT = direct
            for d in (compact._src_dir(T), compact._out_dir(T), sbi._src_dir(T), sbi._out_dir(T), cl.BUILD_DIR):
                shutil.rmtree(d, ignore_errors=True)
            state = Path(f"state-claims-labels-{tag}")
            state.mkdir()
            cl.BUILD_DIR.mkdir(parents=True)
            refs.write_parquet(cl.REFS_PATH)
            cl.record_stage(state, "refs")
            cl.build_claims_labels("unused", state, 49, api)
            compact.rewrite_table(T, 49)  # (compact_table's stages but its metadata's)
            compact.keep_as_hub_copy(T)
            sbi.check_compacted_copy(T)
            sbi.write_table(T)
            sbi.replace_local_copy(T)
            return {k.name: [(f.name, sha(f)) for f in sorted(k.glob("*.parquet"), key=sbi._name_order)]
                    for k in sorted(sbi._src_dir(T).iterdir()) if k.is_dir()}

        # (en in several part files)
        file_bytes = sbi.COMPACT_FILE_BYTES
        en_decoded = 4 * sbi.SORT_STRING_BYTES * refs.height
        sbi.COMPACT_FILE_BYTES = int(en_decoded * sbi.SORT_HUB_BYTES_PER_DECODED[str(T)] / 3)
        before = published("before", False)
        direct = published("direct", True)
        check(f"claims_labels as its published files: {len(direct)} keys, {len(direct['en'])} part files of en,"
              " the same by name, order and sha256 as compacted and sorted", direct == before and len(direct["en"]) > 1)
        built = sbi.read_manifest(T)
        check("each language's part files listed in the sort's manifest, from no source",
              set(built) == set(direct) and all(e["sources"] == [] for e in built.values()))

        # One join a batch of languages (build_batch, cl.BATCHED: the default, so `direct`'s,
        # where with 3 workers each language is a batch of its own, of at least 12), against
        # one join a language: the same files however the languages are batched (with a
        # batch a worker and 1 worker, all four in one; en alone, as a language over the
        # cap, de, and fr with xx; each alone)
        check("batches: the most rows first, into the batch with the fewest that it fits",
              cl._batches({"en": 9, "de": 5, "fr": 3, "xx": 1, "yy": 1}, 6, 2) == [["en"], ["de"], ["fr", "xx", "yy"]])
        cl.BATCHED = False
        unbatched = published("unbatched", True)
        cl.BATCHED = True
        workers, batch_cap, a_worker = cl.FINALISE_WORKERS, cl._batch_cap, cl._BATCHES_A_WORKER
        cl._BATCHES_A_WORKER = 1
        for w, cap in ((1, None), (1, 3000), (3, 1)):
            cl.FINALISE_WORKERS = w
            if cap:
                cl._batch_cap = lambda _, cap=cap: cap
            check(f"batched with {w} worker(s), at most {cl._batch_cap(sbi.SORT_IN_MEMORY_BYTES):,} labels a batch:"
                  " the same files as a join a language", published(f"batched-{w}-{cap}", True) == unbatched == direct)
            cl._batch_cap = batch_cap
        cl.FINALISE_WORKERS = workers

        # Restarts: a crash mid-part (a part cut short, a temporary file left, no entry), a
        # crash after the parts and before their entry, a listed part changed since
        lines = [json.loads(line) for line in sbi._manifest_path(T).read_text().splitlines()]
        shutil.rmtree(sbi._out_dir(T), ignore_errors=True)
        for e in lines:  # (the files moved into the local copy, back where the build left them)
            (sbi._out_dir(T) / e["key"]).mkdir(parents=True)
            for f in e["files"]:
                shutil.copy(sbi._src_dir(T) / e["key"] / f["name"], sbi._out_dir(T) / e["key"] / f["name"])
        keep = [e for e in lines if e["key"] not in ("en", "de")]
        sbi._manifest_path(T).write_text("".join(json.dumps(e) + "\n" for e in keep))
        cl.BUILD_DIR.mkdir(parents=True, exist_ok=True)
        refs.write_parquet(cl.REFS_PATH)
        en_dir = sbi._out_dir(T) / "en"
        first = en_dir / direct["en"][0][0]
        first.write_bytes(first.read_bytes()[: first.stat().st_size // 2])
        first.with_suffix(".tmp").write_bytes(b"partial")
        fr = sbi._out_dir(T) / "fr" / direct["fr"][0][0]
        fr.write_bytes(fr.read_bytes()[:-1] + bytes([fr.read_bytes()[-1] ^ 1]))
        out = Path("cl-groups")
        rows_en_de_fr = sum(pq.read_metadata(sbi._src_dir(T) / k / n).num_rows for k in ("en", "de", "fr") for n, _ in direct[k])
        n = cl.write_groups(cl.REFS_PATH, cl_labels, out, "chunks-0000-0049", direct=True)
        again = sbi.read_manifest(T)
        check(f"a restart rebuilds en (cut mid-part), de (no entry) and fr (a part changed) alone ({n:,} rows)",
              n == rows_en_de_fr and not list(out.glob("*/*.parquet")))
        check("to the same files, by name and sha256",
              {k: [(f["name"], f["sha256"]) for f in e["files"]] for k, e in again.items()} == direct
              and all(sha(sbi._out_dir(T) / k / name) == h for k, fs in direct.items() for name, h in fs)
              and not list(en_dir.glob("*.tmp")))

        # A crash mid-batch: the batch's first two languages written, its job dead before its
        # parent listed them; the rerun rebuilds them to the same files
        sbi._manifest_path(T).write_text("")
        shutil.rmtree(sbi._out_dir(T), ignore_errors=True)
        limits = (sbi.SORT_IN_MEMORY_BYTES, sbi.COMPACT_FILE_BYTES)
        batch = [(lang, [d / lang for d in cl_labels], out / lang / "chunks-0000-0049.parquet") for lang in sorted(direct)]
        build_rows, done = cl._build_rows, []

        def dying(rows, lang, *a):
            if len(done) == 2:
                raise RuntimeError("killed mid-batch")
            done.append(lang)
            return build_rows(rows, lang, *a)

        cl._build_rows = dying
        check("a batch killed after two languages", raises(lambda: cl.build_batch(cl.REFS_PATH, batch, *limits)))
        cl._build_rows = build_rows
        left = sorted(d.name for d in sbi._out_dir(T).iterdir())
        cl.FINALISE_WORKERS = 1  # (one batch of all four)
        n = cl.write_groups(cl.REFS_PATH, cl_labels, out, "chunks-0000-0049", direct=True)
        cl.FINALISE_WORKERS = workers
        again = sbi.read_manifest(T)
        check(f"rerun after it ({left} written, unlisted): every language rebuilt ({n:,} rows), to the same files",
              left == done and {k: [(f["name"], f["sha256"]) for f in e["files"]] for k, e in again.items()} == direct
              and all(sha(sbi._out_dir(T) / k / name) == h for k, fs in direct.items() for name, h in fs)
              and not list(out.glob("*/*.parquet")))

        # Every label three times over: deduplicated as compaction would
        for d in cl_labels:
            for f in d.glob("*/part-0.parquet"):
                pl.concat([pl.read_parquet(f)] * 3).write_parquet(f)
        before3 = published("before-x3", False)
        direct3 = published("direct-x3", True)
        check("with every label three times over, the same files, and those of the labels once",
              direct3 == before3 == direct)

        # A key the sort would take through buckets: written as its group file, the files
        # the same in the end
        sbi.SORT_IN_MEMORY_BYTES, threshold = en_decoded * 3 // 4, sbi.SORT_IN_MEMORY_BYTES
        before_b = published("before-buckets", False)
        direct_b = published("direct-buckets", True)
        built = sbi.read_manifest(T)
        check("a key over the in-memory threshold written as its group file, the files the same in the end",
              direct_b == before_b and built["en"]["sources"] != [] and built["de"]["sources"] == [])
        sbi.SORT_IN_MEMORY_BYTES = threshold

        # A language with an id of two labels: rows of the same (ref, field) that differ,
        # refused by a batch's join (of all four) and taken from the language's own,
        # as before. (Not at a release: an id has a label a language.) Their order among
        # themselves is the join's and Polars' sort's, which is not the same from run to run
        # (as before: the same rows, by `ref`), so de is compared by its rows, every other
        # language by its files
        de_a = cl_labels[0] / "de" / "part-0.parquet"
        de_a_rows = pl.read_parquet(de_a)
        pl.concat([de_a_rows, pl.DataFrame({"id": ["Q0", "P0"], "language": "de", "value": ["de other Q0", "de other P0"]})]
                  ).write_parquet(de_a)
        cl.BUILD_DIR.mkdir(parents=True, exist_ok=True)
        refs.write_parquet(cl.REFS_PATH)
        de_tied = cl._language_rows(cl.REFS_PATH, [d / "de" for d in cl_labels])
        check("rows of the same (ref, field) that differ: not alike", not cl._ties_alike(de_tied))

        def de_rows(files: dict) -> pl.DataFrame:  # (the published files' rows, in order)
            return pl.concat([pl.read_parquet(sbi._src_dir(T) / "de" / n) for n, _ in files["de"]])

        before_t = published("before-ties", False)
        before_de = de_rows(before_t)
        cl.FINALISE_WORKERS = 1
        direct_t = published("direct-ties", True)
        cl.FINALISE_WORKERS = workers
        got = de_rows(direct_t)
        cols = ["ref", "field", "language", "label"]
        check(f"an id of two labels in de ({got.height:,} rows): batched, the same rows as compacted and sorted, by"
              " ref, and every other language's files the same",
              [n for n, _ in direct_t["de"]] == [n for n, _ in before_t["de"]] and got["ref"].is_sorted()
              and got.sort(cols).equals(before_de.sort(cols)) and got.height == before_de.height
              and {k: v for k, v in direct_t.items() if k != "de"} == {k: v for k, v in before_t.items() if k != "de"})
        de_a_rows.write_parquet(de_a)
        cl._BATCHES_A_WORKER = a_worker
        tie = pl.DataFrame({"field": ["labels"] * 3, "ref": ["Q1", "Q1", "Q2"], "language": "en", "label": ["a", "b", "c"]})
        check("ties of the same row alike, of rows that differ not",
              cl._ties_alike(pl.concat([tie.head(1), tie]).sort("ref", "field").filter(pl.col("label") != "b"))
              and not cl._ties_alike(tie))

        # Rows not as the sort would leave them are refused for the direct path
        ok = pl.DataFrame({"field": ["labels"] * 3, "ref": ["Q1", "Q2", "Q3"], "language": "en", "label": ["a", "b", "c"]})
        check("rows in order taken", cl._published_rows(ok, "test")[0] is not None)
        check("a ref out of order refused", cl._published_rows(ok.reverse(), "test")[0] is None)
        null = ok.with_columns(pl.when(pl.col("ref") == "Q2").then(None).otherwise(pl.col("ref")).alias("ref"))
        check("a null ref refused", cl._published_rows(null, "test")[0] is None)
        twice = pl.concat([ok, ok]).sort("ref", "field")
        t2, _, _ = cl._published_rows(twice, "test")
        check("rows twice over deduplicated", t2 is not None and t2.num_rows == 3)
        shutil.rmtree(cl.BUILD_DIR, ignore_errors=True)
        sbi.COMPACT_FILE_BYTES = file_bytes

        # claims_labels: refs per claims file, then languages
        claims_dir = Path("claims")
        claims_dir.mkdir()
        files = sorted(CLAIMS.glob("*.parquet"))
        for k, f in enumerate(files[:: max(1, len(files) // 3)][:3]):
            rows_k = pq.ParquetFile(f).read_row_group(0).slice(0, ROWS)
            pq.write_table(rows_k, claims_dir / f"part-{k}.parquet", row_group_size=ROWS // 4)
        def whole(f: Path) -> pl.DataFrame:  # every column read by Polars, as before 2026-10-05
            return cl._refs(pl.scan_parquet(f)).collect()

        want = pl.concat([whole(f) for f in sorted(claims_dir.glob("*.parquet"))]).unique().sort("field", "ref")
        got = cl.collect_refs(claims_dir)
        check(f"refs from the needed leaves, per file job, equal every column read in one process ({got.height:,} refs)", got.equals(want))
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
        # scripts/refs_bench.py: 2 s for a 529 MB file (snaks exploded by Polars: 29 s)
        check(f"refs of a full claims file took {secs:.0f} s (2 s in the benchmark)", secs < 15)
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
