"""claims_labels' largest languages' jobs at a fraction of a release (bench 17.11.0): run
apart from the timed run, after it (ab.run_once), so none of its figures take this in.

A release's claims_labels build joins the set's refs with each language's labels of both
sets. A language with more labels than a batch holds (claims_labels._batch_cap: 4.2M at a
release's settings, 40 languages) is a job of its own, FINALISE_LARGE_WORKERS (2) at once,
and those jobs set the build's peak (docs/journal/2026-10-25-claims-labels-one-join.md).
The sample has no such language: its claims_labels stage (pipeline.claims_labels_stage)
batches every language, so it measures the batches and never these jobs. Here the
largest two are built at a fraction of their release's size, each by the commit's own
`claims_labels.build_language` in a process of its own (its peak RSS, VmHWM, its CPU time
and wall time its own), from inputs made at a release's proportions:

- **en:** the main set's labels, 48,749,798 (docs/dataset_cards_metadata.json, the
  2026-05-07 build's main set) times 1.2 (a release's main set: #164); the scholarly
  set's, one a scholarly entity (RELEASE_SCHOLAR_ENTITIES: not on record, assumed); the
  main set's 20,697,204 refs (#164); and 17,342,525 of them with an en label (the cards'
  en rows of claims_labels).
- **nl:** the main set's labels, 22,423,008 times 1.2; none of the scholarly set's
  (assumed); the same refs, 9,819,685 of them with an nl label.

The labels' values are the sample's own en labels, the main set's from its main-set
chunk files and the scholarly set's from its scholarly ones (`s*`), drawn at random; the
ids distinct random Q-ids, each set's labels sorted by id and written as the sort writes
a key's files (zstd, row groups of about 128 MiB of decoded rows); the refs with a label
drawn at random from both sets' labels (so they name scholarly works in proportion to
their labels: for the rows' labels, the longer of the likely shares), the rest ids without
one, a unit ref one in 5,000 and a property-labels ref a property (12,000 at a release,
each with a label). en is built at 1/8 and 1/4 of these, nl at 1/4. The line through
two fractions is a model: a job's peak is not quite linear in its inputs (measured at
1/16, 1/8 and 1/4, the line through the smaller two overstated the join of the whole
labels at 1/4 by 16% and understated the join a row group at a time by 14%), so the
larger fractions are taken, and the whole's figures are a comparison of two commits
more than a forecast.

Figures (`labels_job_{lang}_{n}_{rss_mb,cpu,wall}`, 1/n of a release, and its labels,
refs and rows), and modelled (CHANGELOG 17.11.0):

- **en's job at a release** (`release_labels_job_en_rss_mb`, `_cpu`): on the line through
  its two fractions, at the whole: a fixed part (the process, a row group's reading) and
  a part in proportion to the inputs.
- **nl's** (`release_labels_job_nl_rss_mb`): en's fixed part, and the rest of its 1/4's
  four times.
- **The two at once** (`release_labels_jobs_rss_mb`): the largest two jobs side by side,
  as FINALISE_LARGE_WORKERS runs them at most.

Parity (`labels_job_parity_ok`): each job's published files hold the rows a plain join of
the inputs gives (an eager Polars inner join on `id`, sorted by (`ref`, `field`)), in
order, and (ab.py) every arm's files have the same SHA-256s as the first's.

Usage: pipeline.py --labels-job <src> <out.json> <chunk file>...
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from release import RELEASE_SCHOLAR_ENTITIES

ENTITY_URL = "http://www.wikidata.org/entity/"
MAIN_GROWTH = 1.2  # (a release's main set against the 2026-05-07 build's, #164)
REFS = 20_697_204
PROPERTIES = 12_000
UNIT_EVERY = 5_000
ROW_GROUP_BYTES = 128 * 1024**2  # (config.COMPACT_ROW_GROUP_BYTES)
# A language's labels in each set and its refs with a label, at a release
LANGS = {
    "en": (round(48_749_798 * MAIN_GROWTH), RELEASE_SCHOLAR_ENTITIES, 17_342_525),
    "nl": (round(22_423_008 * MAIN_GROWTH), 0, 9_819_685),
}
JOBS = [("en", 8), ("en", 4), ("nl", 4)]

# The job, in a process of its own: build_language as a worker runs it, with a release's
# plan settings (config's), and its peak RSS (VmHWM), CPU time and wall time
JOB = """
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from wikidata import claims_labels as cl, config, sort_by_id as sbi
from wikidata.config import Table
refs, dirs = Path(sys.argv[2]), [Path(d) for d in sys.argv[3:]]
t = time.perf_counter()
built = cl.build_language(refs, dirs, "en", Path("group/en.parquet"), config.SORT_IN_MEMORY_BYTES, config.COMPACT_FILE_BYTES)
wall = time.perf_counter() - t
r = resource.getrusage(resource.RUSAGE_SELF)
hwm = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM"))
files = [str(sbi._out_dir(Table.CLAIMS_LABELS) / "en" / f["name"]) for f in built["files"] or []]
print(json.dumps({"rows": built["rows"], "files": files, "rss_mb": round(hwm / 1024), "cpu": round(r.ru_utime + r.ru_stime, 2),
                  "wall": round(wall, 2)}))
"""


def pools(files: list[Path]) -> tuple[list[str], list[str]]:
    """The sample's en labels: its main-set chunk files' and its scholarly ones' (`s*`)."""
    import orjson
    import pyarrow.parquet as pq

    main, scholar = [], []
    for f in files:
        to = scholar if f.name.startswith("s") else main
        for labels in pq.read_table(f, columns=["labels"]).column("labels").to_pylist():
            if (v := orjson.loads(labels).get("en")) is not None:
                to.append(v)
    return main, scholar


def inputs(d: Path, lang: str, n: int, main_pool: list[str], scholar_pool: list[str]) -> dict:
    """1/n of the language's job at a release (LANGS) in `d`: its labels of each set
    (`main/en`, `scholar/en`) and its refs (`refs.parquet`, as collect_refs_stage writes
    them). Deterministic (seeded by the language and the fraction)."""
    import numpy as np
    import polars as pl
    import pyarrow.parquet as pq

    rng = np.random.default_rng([ord(c) for c in lang] + [n])
    n_main, n_scholar, n_hits = (round(x / n) for x in LANGS[lang])
    n_refs, n_props = round(REFS / n), round(PROPERTIES / n)
    n_misses = n_refs - n_hits
    q = rng.choice(150_000_000, n_main + n_scholar + n_misses, replace=False) + 1
    qid = lambda a: pl.select(pl.concat_str(pl.lit("Q"), pl.Series(a).cast(pl.String))).to_series()  # noqa: E731
    main_ids = pl.concat([qid(q[:n_main]), pl.Series([f"P{i}" for i in range(1, n_props + 1)])])
    scholar_ids = qid(q[n_main:n_main + n_scholar])
    miss_ids = qid(q[n_main + n_scholar:])
    out = {"dirs": []}
    for name, ids, pool in (("main", main_ids, main_pool), ("scholar", scholar_ids, scholar_pool)):
        out[f"labels_{name}"] = len(ids)
        if not len(ids):  # (a set with none of the language's labels has no directory of them)
            continue
        values = pl.Series(pool).gather(rng.integers(0, len(pool), len(ids)))
        labels = pl.DataFrame({"id": ids, "language": "en", "value": values}).sort("id")
        decoded = labels.select(pl.all().str.len_bytes().sum() + 4 * pl.len()).sum_horizontal().item()
        (d / name / "en").mkdir(parents=True)
        pq.write_table(labels.to_arrow(), d / name / "en" / "part-0-of-1.parquet", compression="zstd",
                       row_group_size=max(1, int(ROW_GROUP_BYTES * len(ids) / decoded)))
        out["dirs"].append(d / name / "en")
    # Refs: those with a label drawn from both sets' labels (the properties among them),
    # the rest ids without one; a unit ref one in UNIT_EVERY
    labelled = pl.concat([main_ids.slice(0, n_main), scholar_ids])
    hits = labelled.gather(rng.choice(len(labelled), n_hits - n_props, replace=False))
    ids = pl.concat([hits, miss_ids])
    unit = rng.integers(0, UNIT_EVERY, len(ids)) == 0
    refs = pl.concat([
        pl.DataFrame({"field": np.where(unit, "unit-labels", "labels"), "id": ids})
        .with_columns(ref=pl.when(pl.col("field") == "unit-labels").then(ENTITY_URL + pl.col("id")).otherwise(pl.col("id"))),
        pl.DataFrame({"field": "property-labels", "id": main_ids.slice(n_main)}).with_columns(ref=pl.col("id")),
    ]).select("field", "ref", "id").unique().sort("field", "ref")
    refs.write_parquet(d / "refs.parquet")
    out["refs"] = refs.height
    return out


def reference(d: Path, dirs: list[Path]):
    """The rows a plain join of the inputs gives: claims_labels' rows of the language."""
    import polars as pl

    labels = pl.concat([pl.scan_parquet(x / "part-0-of-1.parquet") for x in dirs])
    return (pl.scan_parquet(d / "refs.parquet").join(labels.rename({"value": "label"}), on="id", how="inner")
            .select("field", "ref", "language", "label").sort("ref", "field", maintain_order=True)
            .collect(engine="streaming"))


def labels_job_stage(src: Path, out: Path, files: list[Path]) -> None:
    import polars as pl

    main_pool, scholar_pool = pools(files)
    res, ok, shas = {}, True, {}
    env = {k: v for k, v in os.environ.items() if k not in ("WIKIDATA_PHASES",)}
    for lang, n in JOBS:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            made = inputs(d, lang, n, main_pool, scholar_pool)
            job = subprocess.run([sys.executable, "-c", JOB, str(src), str(d / "refs.parquet"), *map(str, made["dirs"])],
                                 cwd=d, env=env, capture_output=True, text=True)
            if job.returncode:
                raise SystemExit(f"labels job {lang} 1/{n}:\n{job.stdout}\n{job.stderr}")
            got = json.loads(job.stdout.strip().splitlines()[-1])
            parts = [d / f for f in got["files"]]
            rows = pl.concat([pl.read_parquet(f) for f in parts]) if parts else None
            want = reference(d, made["dirs"])
            same = rows is not None and rows.equals(want)
            ok &= same
            shas[f"{lang}_{n}"] = [hashlib.sha256(f.read_bytes()).hexdigest() for f in parts]
            key = f"labels_job_{lang}_{n}"
            res |= {f"{key}_rss_mb": got["rss_mb"], f"{key}_cpu": got["cpu"], f"{key}_wall": got["wall"],
                    f"{key}_rows": got["rows"], f"{key}_labels": made["labels_main"] + made["labels_scholar"],
                    f"{key}_refs": made["refs"]}
            print(f"labels job {lang} at 1/{n} of a release: {res[f'{key}_labels']:,} labels, {made['refs']:,} refs, "
                  f"{got['rows']:,} rows: {got['rss_mb']:,} MB, {got['cpu']} CPU-s, {got['wall']} s, "
                  f"parity {'passed' if same else 'FAILED'}", flush=True)
    res |= release_model(res)
    res["labels_job_parity_ok"] = ok
    res["labels_job_sha256"] = shas
    out.write_text(json.dumps(res))


def release_model(res: dict) -> dict:
    """The jobs at a release (the module's docstring): en's on the line through its two
    fractions; nl's with en's fixed part; the two at once."""
    (small, large), (nl_n,) = [n for lang, n in JOBS if lang == "en"], [n for lang, n in JOBS if lang == "nl"]
    (a, x), (b, y) = [(1 / n, res[f"labels_job_en_{n}_rss_mb"]) for n in (small, large)]
    slope = (y - x) / (b - a)
    fixed = x - slope * a
    en = fixed + slope
    nl = fixed + (res[f"labels_job_nl_{nl_n}_rss_mb"] - fixed) * nl_n
    (_, cx), (_, cy) = [(1 / n, res[f"labels_job_en_{n}_cpu"]) for n in (small, large)]
    cpu_slope = (cy - cx) / (b - a)
    return {"release_labels_job_en_rss_mb": round(en), "release_labels_job_nl_rss_mb": round(nl),
            "release_labels_jobs_rss_mb": round(en + nl), "labels_job_fixed_rss_mb": round(fixed),
            "release_labels_job_en_cpu": round(cx - cpu_slope * a + cpu_slope, 1)}
