"""The bench's pipeline and finalise locally, without the split: each chunk processed and
partitioned (main.process_and_partition), one group closed (merge_group), its group files
moved where finalise's compaction takes them, finalise as the bench runs it
(bench/pipeline.finalise_at_scale: thresholds scaled, then three times the rows), then each
published key's layout (layout.py) and each group file's sha256.
Usage: pipe_driver.py <src> <bench dir> <run dir> <out.json> <chunk files...>
(wikidata_reshape stubbed on PYTHONPATH: the chunks' claims decoded from their JSON)"""
import hashlib, json, os, shutil, sys
from pathlib import Path


def main():
    src, bench, run, out = (Path(p).resolve() for p in sys.argv[1:5])
    files = [str(Path(f).resolve()) for f in sys.argv[5:]]
    os.environ.update(WIKIDATA_RELEASE="20261012", HF_HUB_OFFLINE="1", WIKIDATA_FINALISE_WORKERS="4")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])
    sys.path.insert(0, str(src)); sys.path.insert(0, str(bench)); sys.path.insert(0, str(Path(__file__).resolve().parent))
    shutil.rmtree(run, ignore_errors=True)
    work = run / "releases" / "20261012"
    data, state, results = work / "data", work / "state", work / "results"
    for d in (data, state, results):
        d.mkdir(parents=True)
    for i, f in enumerate(files):
        (data / f"chunk_{i}.parquet").symlink_to(f)
    sizes = [os.path.getsize(f) for f in files]
    os.chdir(run)
    import pipeline as bp  # noqa
    from wikidata import compact, sort_by_id
    from wikidata.config import AUDIT_DIR, COMPACT_DIR, HUB_COPY_DIR, STAGING_DIR, Table
    from wikidata.main import process_and_partition
    from wikidata.push.core import merge_group
    from wikidata.push.groups import Group, record_partitioned, record_stage
    from layout import layout
    assert Path(sort_by_id.__file__).is_relative_to(src)
    n = len(files)
    for i in range(n):
        process_and_partition(i, data_dir=data.resolve(), output_dir=results.resolve(), repo_id="unused",
                              state_dir=state.resolve())
        record_partitioned(state.resolve(), i, sizes[i])
    group = Group(0, n - 1, n - 1)
    record_stage(state.resolve(), group, "closed")
    merge_group(group, AUDIT_DIR, STAGING_DIR)
    record_stage(state.resolve(), group, "merged")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    groups = {}
    for table in Table:
        staged = STAGING_DIR / table
        if not staged.exists():
            continue
        for f in sorted(staged.rglob("*.parquet")):
            groups[str(f.relative_to(STAGING_DIR))] = sha(f)
            to = COMPACT_DIR / "src" / table / f.relative_to(staged)
            to.parent.mkdir(parents=True, exist_ok=True)
            f.rename(to)
        shutil.rmtree(staged, ignore_errors=True)
    keep = run / "keep"
    fin, eta = bp.finalise_at_scale(compact, sort_by_id, Table, COMPACT_DIR, HUB_COPY_DIR, state.resolve(),
                                    n, files, 10.0, 4, keep)
    res = {"fin": fin, "groups": groups, "first": layout(keep / "final"), "x3": layout(HUB_COPY_DIR),
           "shape": {k: v for k, v in eta.items() if k.startswith(("sort_", "x3_"))}}
    out.write_text(json.dumps(res))
    shutil.rmtree(run)
    print("done", fin, len(groups), "group files")


if __name__ == "__main__":
    main()
