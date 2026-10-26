"""Finalise on the bench's quick-tier sample, serial against overlapped, with an emulated
link: main.finalise run for real by scripts/test_finalise_overlap.py's child (compaction,
the sort, commit_table and verify_table against its stand-in Hub, whose commits sleep for
their bytes at a stated rate), from the group files the bench's pipeline kept
(`KEEP=1 python bench/pipeline.py ...`, its `.keep/staging`).

The rate is scaled: each case is named by the release's upload rate whose upload time,
against finalise's compute, the sample's emulated link reproduces. At a release, the main
set's run computes ~4,100 s (model.py: its claims, its other tables, their downloads) and
uploads ~59 GB, so its upload is that many times its compute: 59,000 / rate / 4,100. The
sample's link is set to take the same multiple of the sample's own serial compute.

Usage: python measure.py KEEP_DIR WORK_DIR [rounds]
"""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / "scripts" / "test_finalise_overlap.py"
RELEASE_COMPUTE = 4_100  # s, the main set's finalise run (model.py)
RELEASE_UPLOAD_MB = 59_000


def run(work: Path, keep: Path, env: dict) -> dict:
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {**os.environ, "WIKIDATA_RELEASE": "finaliseoverlap", "HF_HUB_OFFLINE": "1",
           "FINALISE_SAMPLE": str(keep), "WIKIDATA_FINALISE_WORKERS": "4", **env}
    with open(work.parent / f"{work.name}-child.log", "w") as log:  # (its own session, killed with its workers)
        p = subprocess.Popen([sys.executable, TEST, "--child", work, ""], env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        code = p.wait()
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if code:
        sys.exit((work.parent / f"{work.name}-child.log").read_text()[-3000:])
    ev = [json.loads(line) for line in (work / "fakehub" / "events.jsonl").read_text().splitlines()]
    t0 = next(x["t"] for x in ev if x["event"] == "start")
    end = next(x for x in ev if x["event"] == "end")
    commit = sum(b["t"] - a["t"] for a, b in zip([x for x in ev if x["event"] == "commit_start"],
                                                 [x for x in ev if x["event"] == "commit_end"]))
    out = {"wall": end["t"] - t0, "commit": commit, "rss_mb": end["rss_mb"], "mb": end["uploaded_mb"]}
    shutil.rmtree(work, ignore_errors=True)
    return out


def main() -> None:
    keep, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    # The sample's compute alone: serial, a link with no time
    base = [run(work, keep, {"WIKIDATA_FINALISE_PENDING_UPLOAD_GB": "0", "FAKE_RATE": "1e15"}) for _ in range(rounds)]
    compute, mb = median(b["wall"] for b in base), base[0]["mb"]
    print(f"sample: {mb:.1f} MB committed, finalise's compute {compute:.2f} s (serial, no link time), "
          f"peak RSS of finalise's process {max(b['rss_mb'] for b in base)} MB")
    print("| release rate emulated | sample's link | upload alone | serial (=0) | overlapped (64 GB) | saving | peak RSS serial / overlapped |")
    print("|---|---|---|---|---|---|---|")
    for rate in (7, 30, 60, 120):
        upload = compute * RELEASE_UPLOAD_MB / rate / RELEASE_COMPUTE
        sample_rate = mb * 1e6 / upload
        walls = {}
        for mode, gb in (("serial", "0"), ("overlap", "64")):
            rs = [run(work, keep, {"WIKIDATA_FINALISE_PENDING_UPLOAD_GB": gb, "FAKE_RATE": str(sample_rate)})
                  for _ in range(rounds)]
            walls[mode] = (median(r["wall"] for r in rs), max(r["rss_mb"] for r in rs))
        s, o = walls["serial"][0], walls["overlap"][0]
        print(f"| {rate} MB/s | {sample_rate / 1e6:.2f} MB/s | {upload:.2f} s | {s:.2f} s | {o:.2f} s | "
              f"−{s - o:.2f} s (−{(s - o) / s:.0%}) | {walls['serial'][1]} / {walls['overlap'][1]} MB |", flush=True)


if __name__ == "__main__":
    main()
