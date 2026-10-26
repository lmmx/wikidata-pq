"""Two releases of the bench slice through the real pipeline, for Xet's dedup between them.

- `slice-b`: release B's dump slice. Release A's is the bench's own (bench/dumpslice.py of
  the quick tier's chunk files: 20261012's entities). B has the same entities in the same
  order (`slice_b` says which change), each one's line:
  - **changed**: its real other version where it has one: 20261019's line (windows.py),
    else 20261005's (the week before's edit, undone), else a synthetic sweep edit;
  - **gone** (not in 20261019's stretch of its shard): left out, always;
  - **unchanged**: A's line;
  - **new**: the week's share of entities created after 20261012 (20261019's shard 7
    tail, windows.py), after the last line, where the shards' new entities are.
- `build`: a release of a slice as `just release` runs one, through the bench's own
  harness: the split (split-dump, routing), the scholarly set's run and the main set's
  (main.run, `process-wikidata`), each set's groups pushed to the stand-in Hub's build
  branch by the real close_group (bench/overlap.py's FakeHub, the overlap stage's
  sequential mode), then finalise's three runs (main, scholar, main) as bench/link.py's
  finalise_stage runs them without a link: compaction from the build branch, the sort,
  each table's commit to the build branch, claims_labels. Kept: the build branch's files
  after the runs (the group files) and after finalise (the published files), with the
  run's log; everything else is deleted.

Everything runs offline: HF_HUB_OFFLINE=1, the Hub a directory (FakeHub, LinkHub).

    python releases.py slice-b --a A.json.bz2 --lines LINES --churn 1 --files c0.parquet ... --out B1.json.bz2
    python releases.py build --src WORKTREE --slice A.json.bz2 --release 20261012 --out OUT
"""

import argparse
import bz2
import gzip
import json
import os
import re
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

REV = re.compile(rb'"lastrevid":(\d+)')
ID = re.compile(rb'^\{"type":"\w+",(?:"datatype":"[^"]*",)?"id":"([A-Z]\d+)"')


def read_slice(path: Path) -> list[bytes]:
    """A dump slice's entity lines, in order (without the separating commas)."""
    text = bz2.decompress(path.read_bytes())
    return [x.rstrip(b",") for x in text.split(b"\n") if x.startswith(b"{")]


def write_slice(path: Path, lines: list[bytes]) -> None:
    """The dump's framing, as bench/dumpslice.py writes it: `[`, a line an entity,
    comma-separated, `]`, bzip2 at level 9 (lbzip2 when installed, as dumpslice.py)."""
    import subprocess

    body = b"[\n" + b",\n".join(lines) + b"\n]\n"
    if shutil.which("lbzip2"):
        with open(path, "wb") as f:
            subprocess.run(["lbzip2", "-9", "-n", "2"], input=body, stdout=f, check=True)
    else:
        path.write_bytes(bz2.compress(body, 9))


def read_lines(path: Path) -> dict[str, bytes]:
    out = {}
    with gzip.open(path, "rt") as f:
        for row in f:
            e, line = row.rstrip("\n").split("\t", 1)
            out[e] = line.encode()
    return out


def rev(line: bytes) -> int | None:
    m = REV.search(line)
    return int(m.group(1)) if m else None


def sweep(line: bytes, k: int, rng: random.Random) -> bytes:
    """A synthetic edit, as the measured bot sweeps make them (#188: descriptions in a
    language over a class's items, census claims over villages): a new description, or a
    statement's value changed, and a new `lastrevid`."""
    e = json.loads(line)
    e["lastrevid"] = int(e.get("lastrevid") or 0) + 1_000_000 + k
    claims = e.get("claims") or {}
    if claims and rng.random() < 0.4:
        p = rng.choice(sorted(claims))
        st = claims[p][0]
        st["id"] = st["id"][:-4] + f"{k % 10000:04d}"
    else:
        lang = rng.choice(["ha", "as", "ig", "yo", "sw"])
        e.setdefault("descriptions", {})
        if isinstance(e["descriptions"], list):
            e["descriptions"] = {}
        e["descriptions"][lang] = {"language": lang, "value": f"synthetic sweep {k}"}
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
    from dumpslice import dump_escaped

    return dump_escaped(json.dumps(e, ensure_ascii=False, separators=(",", ":")).encode())


# The weekly rates of #188 (2026-10-26-incremental-release.md, "By set and table", the
# entities table: every edit), 20261012 -> 20261019: the share of each set's entities
# changed in a week. The bench's c0 and r50 stand for the main set, s30 and s90 for the
# scholarly set (bench/README.md).
WEEKLY = {"main": 0.0110, "scholar": 0.0002}
SETS = {"c0": "main", "r50": "main", "s30": "scholar", "s90": "scholar"}
# New entities a week: 71,896 of 121.8M (#188's tails.json), 0.059%
NEW_WEEKLY = 71_896 / 121_815_642


def slice_b(a: argparse.Namespace) -> None:
    """B's lines (see the module docstring). Which entities change: at `--churn real`,
    every one 20261019 changed; else, in each chunk file, `churn` times its set's weekly
    rate (WEEKLY) of its entities, drawn at random from those 20261019 really changed,
    then (if short) from those 20261005 changed (their edit undone), then synthetic
    sweeps over runs of consecutive entities. A changed entity's line is its real other
    version, where it has one. Gone entities (20261019 has no line) are always gone."""
    import pyarrow.parquet as pq

    lines_a = read_slice(a.a)
    by_id = {ID.match(x).group(1).decode(): x for x in lines_a}
    dump = lambda d: {k: v for f in sorted(a.lines.glob(f"{d}.*.lines.gz")) for k, v in read_lines(f).items()}  # noqa: E731
    new19, old05 = dump("20261019"), dump("20261005")
    rng = random.Random(a.seed)
    real = a.churn == "real"
    k = 0.0 if real else float(a.churn)
    stats = {"churn": a.churn, "entities_a": len(lines_a), "files": {}, "changed_ids": [], "gone_ids": [],
             "new_ids": []}
    edit: dict[str, bytes] = {}
    for f in a.files:
        ids = pq.read_table(f, columns=["id"]).column("id").to_pylist()
        name = Path(f).stem
        gone = [e for e in ids if e not in new19]
        r19 = [e for e in ids if e in new19 and rev(new19[e]) != rev(by_id[e])]
        r05 = [e for e in ids if e in new19 and e not in r19 and e in old05 and rev(old05[e]) != rev(by_id[e])]
        need = len(r19) if real else round(k * WEEKLY[SETS[name]] * len(ids))
        take19 = r19 if real else rng.sample(r19, min(need, len(r19)))
        take05 = [] if real else rng.sample(r05, min(need - len(take19), len(r05)))
        for e in take19:
            edit[e] = new19[e]
        for e in take05:
            edit[e] = old05[e]
        synth = []
        short = need - len(take19) - len(take05)
        free = [i for i, e in enumerate(ids) if e not in edit and e in new19]
        while len(synth) < short and free:
            j = rng.randrange(len(free))
            run = free[j:j + rng.randint(5, 60)][:short - len(synth)]
            for i in run:
                edit[ids[i]] = sweep(by_id[ids[i]], i, rng)
                synth.append(ids[i])
            free = [i for i in free if ids[i] not in edit]
        stats["files"][name] = {"set": SETS[name], "entities": len(ids), "real_19": len(r19), "real_05": len(r05),
                                "target": need, "from_19": len(take19), "from_05": len(take05),
                                "synthetic": len(synth), "gone": len(gone),
                                "changed_share": round((len(take19) + len(take05) + len(synth)) / len(ids), 5)}
        stats["gone_ids"] += gone
    out = []
    for x in lines_a:
        e = ID.match(x).group(1).decode()
        if e not in new19:
            continue
        if e in edit:
            out.append(edit[e])
            stats["changed_ids"].append(e)
        else:
            out.append(x)
    n_new = round(max(k, 1.0) * NEW_WEEKLY * len(lines_a))
    new = list(read_lines(a.lines / "new.lines.gz").items())[:n_new]
    out += [x for _, x in new]
    stats["new_ids"] = [e for e, _ in new]
    write_slice(a.out, out)
    stats["entities_b"] = len(out)
    a.out.with_suffix(".json").write_text(json.dumps(stats))
    print({k: v for k, v in stats.items() if not k.endswith("_ids")})


FINALISE_PATCH = """
import os
from wikidata import config
if os.environ.get("XETDEDUP_ROW_GROUP_BYTES"):
    config.COMPACT_ROW_GROUP_BYTES = int(os.environ["XETDEDUP_ROW_GROUP_BYTES"])
"""


def snapshot(hub: Path, dst: Path) -> list[dict]:
    """The stand-in Hub's files (Parquet and JSON), copied to `dst`, with their sizes."""
    out = []
    for p in sorted(hub.rglob("*")):
        if p.is_file() and p.suffix in (".parquet", ".json") and not p.name.endswith(".part"):
            rel = p.relative_to(hub)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst / rel)
            out.append({"path": str(rel), "bytes": p.stat().st_size})
    return out


def build(a: argparse.Namespace) -> None:
    src = a.src.resolve()
    sys.path.insert(0, str(src / "bench"))
    os.environ["WIKIDATA_RELEASE"] = a.release
    os.environ["HF_HUB_OFFLINE"] = "1"
    for k in ("WIKIDATA_PHASES", "KEEP"):
        os.environ.pop(k, None)
    if a.row_group_bytes:
        os.environ["XETDEDUP_ROW_GROUP_BYTES"] = str(a.row_group_bytes)
    import link
    import overlap

    link.FINALISE = link.FINALISE.replace("from wikidata import main", FINALISE_PATCH + "from wikidata import main", 1)
    out = a.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    stage = Path(tempfile.mkdtemp(dir=a.scratch))
    t = time.time()
    try:
        with open(out / "run.log", "w") as log:
            root = stage / "run"
            dump_dir = root / "releases" / a.release / "dump"
            dump_dir.mkdir(parents=True)
            shutil.copyfile(a.slice, dump_dir / f"wikidata-{a.release}-all.json.bz2")
            steps = [[("split", overlap._env("split", a.workers, False), 0)],
                     [("scholar", overlap._env("scholar", a.workers, False), 0)],
                     [("main", overlap._env("main", a.workers, False), 0)]]
            wall_runs, _ = overlap._run(src, root, steps, log)
            groups = snapshot(root / "fakehub", out / "groups")
            work = stage / "finalise"
            work.mkdir()
            f = link.finalise_stage(src, root, work, None, {"main": 1.0, "scholar": 1.0}, 1, a.workers, log)
            published = snapshot(work / "root" / "fakehub", out / "published")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    rec = {"release": a.release, "slice": str(a.slice), "row_group_bytes": a.row_group_bytes,
           "runs_wall": wall_runs, "finalise_wall": f["wall"], "wall": round(time.time() - t, 1),
           "groups": groups, "published": published, "digest": f["digest"]}
    (out / "build.json").write_text(json.dumps(rec, indent=1))
    print(f"{a.release}: runs {wall_runs}s, finalise {f['wall']}s; {len(groups)} files after the runs, "
          f"{len(published)} after finalise")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("slice-b")
    b.add_argument("--a", type=Path, required=True)
    b.add_argument("--lines", type=Path, required=True)
    b.add_argument("--churn", default="1", help="a multiple of #188's weekly rates, or `real`")
    b.add_argument("--files", nargs="+", required=True, help="the bench's chunk files, for their ids")
    b.add_argument("--seed", type=int, default=20261019)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("build")
    r.add_argument("--src", type=Path, required=True)
    r.add_argument("--slice", type=Path, required=True)
    r.add_argument("--release", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--row-group-bytes", type=int, default=0)
    r.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    a = ap.parse_args()
    {"slice-b": slice_b, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
