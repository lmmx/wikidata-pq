"""Release A then release B uploaded by hf_xet itself (1.6.0, the lock's), to a local
content-addressed store on disk (`local://`: xet-core's LocalClient, its own CAS, shards
and global dedup table in a directory): the real client's chunking, dedup, fragmentation
guard, shard cache and global dedup queries, with no network and no Hub. Its per-file
DeduplicationMetrics and its own log lines are recorded.

Each upload session (an XetUploadCommit) is one of a release's (measure.session_of):
each group's push of a table, each table's commits. In the order a release makes them.

- `--fresh-cache`: before B, the local shard cache (for a local store, under its
  directory: `<cas>/xet/shard-cache`) is emptied, as on another machine: B then finds A's
  chunks only by global dedup queries, which LocalClient answers with the whole shard
  holding the chunk (its xorbs: `local_client.rs:1480-1506`).
- Otherwise B runs with A's shards in the cache, as on the machine that uploaded A.

    python xetlocal.py --a A --b B --work SCRATCH --out results/xetlocal-1x.json [--fresh-cache]

hf_xet writes its log to `<work>/logs` (HF_XET_LOG_DEST); xet_logs.py reads it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

UPLOAD = r"""
import json, os, sys
import hf_xet
sessions = json.load(open(sys.argv[1]))
endpoint = "local://" + sys.argv[2]
out = []
session = hf_xet.XetSession()
for s in sessions:
    commit = session.new_upload_commit(endpoint=endpoint)
    handles = [(f, commit.start_upload_file(p)) for f, p in s["files"]]
    report = commit.wait_to_finish()
    for f, h in handles:
        m = h.result().dedup_metrics
        out.append({"session": s["session"], "file": f, **{k: getattr(m, k) for k in (
            "total_bytes", "deduped_bytes", "new_bytes", "defrag_prevented_dedup_bytes",
            "deduped_bytes_by_global_dedup", "total_chunks", "new_chunks")}})
json.dump(out, open(sys.argv[3], "w"))
"""


def sessions_of(rel: Path) -> list[dict]:
    sys.path.insert(0, str(HERE))
    from measure import KINDS, files, session_of

    out: dict[str, list] = {}
    for kind in KINDS:
        for n, p in sorted(files(rel, kind).items(), key=lambda x: (session_of(kind, x[0]), x[0])):
            out.setdefault(session_of(kind, n), []).append([f"{kind}:{n}", str(p.resolve())])
    return [{"session": s, "files": fs} for s, fs in out.items()]


def upload(rel: Path, cas: Path, work: Path, label: str, py: str) -> list[dict]:
    (work / f"{label}.sessions.json").write_text(json.dumps(sessions_of(rel)))
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(work / "hf-home"),
           "HF_XET_CACHE": str(work / "xet-cache"), "HF_XET_LOG_DEST": str(work / "logs") + os.sep}
    env.pop("HF_TOKEN", None)
    res = work / f"{label}.result.json"
    subprocess.run([py, "-I", "-c", UPLOAD, str(work / f"{label}.sessions.json"), str(cas), str(res)],
                   env=env, check=True)
    return json.loads(res.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh-cache", action="store_true")
    a = ap.parse_args()
    work = a.work.resolve()
    shutil.rmtree(work, ignore_errors=True)
    (work / "logs").mkdir(parents=True)
    cas = work / "cas"
    py = sys.executable
    ra = upload(a.a, cas, work, "A", py)
    if a.fresh_cache:
        shutil.rmtree(cas / "xet" / "shard-cache", ignore_errors=True)
        shutil.rmtree(work / "xet-cache", ignore_errors=True)
    rb = upload(a.b, cas, work, "B", py)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"fresh_cache": a.fresh_cache, "A": ra, "B": rb}))
    tot = lambda r, k: sum(x[k] for x in r)  # noqa: E731
    for name, r in (("A", ra), ("B", rb)):
        print(f"{name}: {tot(r, 'total_bytes') / 1e6:.1f} MB, new {tot(r, 'new_bytes') / 1e6:.1f} MB, "
              f"deduped {tot(r, 'deduped_bytes') / 1e6:.1f} MB (defrag refused {tot(r, 'defrag_prevented_dedup_bytes') / 1e6:.1f}, "
              f"by global {tot(r, 'deduped_bytes_by_global_dedup') / 1e6:.1f})")


if __name__ == "__main__":
    main()
