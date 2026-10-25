"""Local finalise on a kept bench run's group files, as the bench runs it (its first run and
three times the rows, bench/pipeline.finalise_at_scale), then each published key's layout
and ordered fingerprint (layout.py). Usage: fin_driver.py <src> <bench dir> <staging> <run dir> <out.json> <data files>"""
import json, os, shutil, sys, time
from pathlib import Path

def main():
    src, bench, staging, run, out = sys.argv[1:6]
    out = str(Path(out).resolve())
    files = [str(Path(f).resolve()) for f in sys.argv[6:]]
    src, staging, run = Path(src).resolve(), Path(staging).resolve(), Path(run).resolve()
    os.environ.update(WIKIDATA_RELEASE="20261012", HF_HUB_OFFLINE="1", WIKIDATA_FINALISE_WORKERS="4",
                      PYTHONPATH=str(src))
    sys.path.insert(0, str(src)); sys.path.insert(0, str(Path(bench).resolve()))
    shutil.rmtree(run, ignore_errors=True)
    work = run / "releases" / "20261012"
    (work / "state").mkdir(parents=True)
    for i in range(4):
        (work / "state" / f"chunk_{i}.jsonl").write_text("")
    os.chdir(run)
    import pipeline as bp  # noqa
    from wikidata import compact, sort_by_id
    from wikidata.config import COMPACT_DIR, HUB_COPY_DIR, Table
    assert Path(sort_by_id.__file__).is_relative_to(src), sort_by_id.__file__
    for f in staging.rglob("*.parquet"):
        group, table, key, name = f.relative_to(staging).parts
        to = COMPACT_DIR / "src" / table / key / name
        to.parent.mkdir(parents=True, exist_ok=True)
        os.link(f, to)
    keep = run / "keep"
    t = time.perf_counter()
    fin, eta = bp.finalise_at_scale(compact, sort_by_id, Table, COMPACT_DIR, HUB_COPY_DIR, (work / "state").resolve(),
                                    4, files, 10.0, 4, keep)
    wall = time.perf_counter() - t

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from layout import layout
    res = {"fin": fin, "shape": {k: v for k, v in eta.items() if k.startswith(("sort_", "x3_"))},
           "wall": round(wall, 1), "first": layout(keep / "final"), "x3": layout(HUB_COPY_DIR)}
    Path(out).write_text(json.dumps(res))
    shutil.rmtree(run)
    print("done", fin, res["shape"])

if __name__ == "__main__":
    main()
