"""`finalise-release-runs [--after FILE]` (wikidata.cross_run.run) as the compute benchmark
runs it: the same orchestrator (cross_run.run_release) and the same runs
(`finalise-wikidata` with WIKIDATA_FINALISE_CROSS_RUN=1, the set's WIKIDATA_SCHOLAR), each
run under wrap.py so its own wall, CPU and I/O are recorded, and each set's card figures
in its own directory (WIKIDATA_CARD_FIGURES_DIR, from BENCH_FULL_CARD_FIGURES).

With WIKIDATA_FINALISE_CROSS_RUN=0 (as `just release` then runs them): the three runs one
after another instead (`just finalise-release` main, scholar, main).

Usage: finalise_runs.py OUT_DIR PREFIX [--after FILE]
"""

import os
import subprocess
import sys
from pathlib import Path

from wikidata import cross_run

HERE = Path(__file__).resolve().parent
BIN = Path(sys.executable).parent


def main() -> int:
    out, prefix, *args = sys.argv[1:]
    after = Path(args[args.index("--after") + 1]) if "--after" in args else None
    figures = os.environ.get("BENCH_FULL_CARD_FIGURES")

    def env_of(set_: str, scholar: bool, cross: bool) -> dict:
        env = {**os.environ, "WIKIDATA_SCHOLAR": "1" if scholar else ""}
        if cross:
            env["WIKIDATA_FINALISE_CROSS_RUN"] = "1"
        if figures:
            env["WIKIDATA_CARD_FIGURES_DIR"] = str(Path(figures) / set_)
        return env

    def cmd(i: int, set_: str) -> list[str]:
        return [sys.executable, "-I", str(HERE / "wrap.py"), str(Path(out) / f"{prefix}_run{i + 1}_{set_}.json"),
                "--", str(BIN / "finalise-wikidata")]

    if (os.environ.get("WIKIDATA_FINALISE_CROSS_RUN") or "1") != "0":
        def start(i: int, set_: str, scholar: bool) -> subprocess.Popen:
            return subprocess.Popen(cmd(i, set_), env=env_of(set_, scholar, True))

        return cross_run.run_release(start, after=after)
    for i, (set_, scholar) in enumerate(cross_run.RUNS):
        if subprocess.run(cmd(i, set_), env=env_of(set_, scholar, False)).returncode:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
