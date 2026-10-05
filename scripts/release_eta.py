"""Where `just release` has got to: each step, in the order the recipe runs them, marked
done, in progress ("you are here", with its progress and ETA) or to do.

Steps are read from the ledgers each stage writes in its set's state dir (compact.jsonl,
sort.jsonl, claims_labels_build.jsonl, groups.jsonl, finalise.done) and progress from the
files the step in progress writes, so it needs no access to the running process. Rates
are bytes per hour over the files finished in the last WINDOW minutes (from their mtimes),
so a restart does not skew them.

- Processing: chunk sizes from the set's manifest (data/manifest.jsonl), a chunk done
  once its claims audit file is written (audit/claims/chunk_{N}.parquet)
- Compaction: download into compact/src/{table} (its total from the Hub's file list, if
  huggingface_hub is importable), then writing compact/out/{table}, in source bytes
  rewritten (files.jsonl and manifest.jsonl)
- Sort: download into hub/{table}, against the compacted bytes; then keys sorted
  (manifest.jsonl), and a bucketed key (claims) by buckets sorted and files packed
- Uploads to the Hub (commit stages), the claims refs, claims_labels' build and promotion
  leave no local progress record: they are shown as in progress or to do, without an ETA

Usage: python scripts/release_eta.py 20260928 [WINDOW minutes, default 30]
"""

import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

RELEASES = Path(__file__).parent.parent / "releases"
HF_USER = "permutans"
CHUNK_RE = re.compile(r"chunk_(\d+)\.parquet$")
# Finalise order: claims first, claims_labels once both sets' labels are sorted
TABLES = ["claims", "labels", "descriptions", "aliases", "links", "entities"]
COMPACT_FILE_BYTES = 500 * 1024**2  # config.COMPACT_FILE_BYTES
GB = 1e9


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def last_stage(ledger: Path, table: str | None = None) -> str | None:
    rows = [r for r in read_jsonl(ledger) if table is None or r.get("table") == table]
    return rows[-1]["stage"] if rows else None


def parquet_bytes(d: Path, pattern: str = "*/*.parquet") -> dict[Path, int]:
    return {p: p.stat().st_size for p in d.glob(pattern)} if d.is_dir() else {}


def rate(events: list[tuple[float, float]], window: float) -> float | None:
    """Bytes per hour between the first and last of the events (mtime, bytes) in the
    window: the events after the first took that long."""
    recent = sorted(e for e in events if e[0] > time.time() - window)
    if len(recent) < 2 or recent[-1][0] <= recent[0][0]:
        return None
    return sum(b for _, b in recent[1:]) / (recent[-1][0] - recent[0][0]) * 3600


def progress(what: str, done: float, total: float | None, r: float | None, window: float) -> str:
    """`what: done of total (pct), rate, ETA` for a step measured in bytes."""
    if total is None:
        line = f"{what}: {done / GB:.1f} GB so far"
    else:
        line = f"{what}: {done / GB:.1f} of {total / GB:.1f} GB ({100 * done / max(total, 1):.0f}%)"
    if r is None:
        return f"{line}; nothing finished in the last {window / 60:.0f} min to measure a rate"
    line += f", {r / GB:.2f} GB/h"
    if total is not None:
        left = max(total - done, 0) / r
        line += f", ETA {left:.1f} h ({datetime.fromtimestamp(time.time() + left * 3600):%a %H:%M})"
    return line


def repo(set_name: str, table: str) -> str:
    return f"{HF_USER}/wikidata-{'scholar-' if set_name == 'scholar' else ''}{table}"


def hub_bytes(repo_id: str, revision: str, pattern: str) -> int | None:
    """Total size of the repo's files matching `pattern` on `revision`, or None without
    huggingface_hub or the Hub."""
    try:
        from huggingface_hub import HfApi

        tree = HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True, revision=revision)
        return sum(f.size for f in tree if re.fullmatch(pattern, f.path))
    except Exception:
        return None


class Set:
    def __init__(self, release: str, name: str, window: float):
        self.release, self.name, self.window = release, name, window
        self.dir = RELEASES / (f"{release}-scholar" if name == "scholar" else release)
        self.state = self.dir / "state"

    # Processing

    def processing(self) -> tuple[bool, str]:
        sizes = {e["chunk"]: e["bytes"] for e in read_jsonl(self.dir / "data" / "manifest.jsonl")}
        done = {
            int(CHUNK_RE.search(f.name).group(1)): f.stat().st_mtime
            for f in (self.dir / "audit" / "claims").glob("chunk_*.parquet")
        }
        latest = {(g["first"], g["last"]): g["stage"] for g in read_jsonl(self.state / "groups.jsonl")}
        uploaded = all(s == "done" for s in latest.values())
        total, done_bytes = sum(sizes.values()), sum(sizes.get(c, 0) for c in done)
        # Done once the group with the last chunk is uploaded (the audit files may be gone)
        if sizes and uploaded and max((g[1] for g in latest), default=-1) == max(sizes):
            return True, f"{len(sizes):,} chunks, {total / GB:.1f} GB"
        r = rate([(t, sizes.get(c, 0)) for c, t in done.items()], self.window)
        line = progress(f"{len(done):,} of {len(sizes):,} chunks", done_bytes, total, r, self.window)
        if len(done) == len(sizes) and not uploaded:
            line = f"all {len(sizes):,} chunks processed; uploading the last group"
        return False, line

    # Compaction

    def compact(self, table: str) -> tuple[bool, str]:
        stage = last_stage(self.state / "compact.jsonl", table)
        src, out = self.dir / "compact" / "src" / table, self.dir / "compact" / "out" / table
        if stage == "done":
            return True, ""
        if stage is None:
            files = parquet_bytes(src, "*/chunks-*.parquet")
            partial = sum(p.stat().st_size for p in src.glob(".cache/huggingface/download/**/*.incomplete"))
            total = hub_bytes(repo(self.name, table), f"build-{self.release}", r"[^/]+/chunks-\d+-\d+\.parquet")
            r = rate([(p.stat().st_mtime, b) for p, b in files.items()], self.window)
            return False, progress("downloading", sum(files.values()) + partial, total, r, self.window)
        if stage == "downloaded":
            sizes = {(p.parent.name, p.name): b for p, b in parquet_bytes(src, "*/chunks-*.parquet").items()}
            done: set[tuple[str, str]] = set()
            events = []
            for e in read_jsonl(out / "files.jsonl"):  # per file, tables not deduplicated
                dst = out / e["key"] / e["name"]
                new = {(e["key"], s) for s in e["sources"]} - done
                done |= new
                if dst.exists():
                    events.append((dst.stat().st_mtime, sum(sizes.get(k, 0) for k in new)))
            for e in read_jsonl(out / "manifest.jsonl"):  # per key, all tables
                new = {(e["key"], s) for s in e["sources"]} - done
                done |= new
                key_files = list((out / e["key"]).glob("*.parquet"))
                if key_files and new:
                    t = max(p.stat().st_mtime for p in key_files)
                    events.append((t, sum(sizes.get(k, 0) for k in new)))
            done_bytes = sum(sizes.get(k, 0) for k in done)
            r = rate(events, self.window)
            return False, progress("writing", done_bytes, sum(sizes.values()), r, self.window)
        nxt = {"written": "uploading to the Hub", "committed": "verifying on the Hub",
               "verified": "writing metadata"}[stage]
        return False, f"{nxt} (no local progress record)"

    # Sort

    def sort(self, table: str) -> tuple[bool, str]:
        stage = last_stage(self.state / "sort.jsonl", table)
        copy = self.dir / "hub" / table
        out = self.dir / "compact" / "sort" / "out" / table
        if stage == "done":
            return True, ""
        if stage is None:
            compacted = read_jsonl(self.dir / "compact" / "out" / table / "manifest.jsonl")
            latest = {e["key"]: e for e in compacted}
            total = sum(f["bytes"] for e in latest.values() for f in e["files"]) or None
            files = parquet_bytes(copy)
            r = rate([(p.stat().st_mtime, b) for p, b in files.items()], self.window)
            return False, progress("downloading the local copy", sum(files.values()), total, r, self.window)
        if stage == "sourced":
            return False, self._sort_writing(table, copy, out)
        nxt = {"written": "uploading to the Hub", "committed": "verifying on the Hub",
               "verified": "writing metadata, replacing the local copy"}[stage]
        return False, f"{nxt} (no local progress record)"

    def _sort_writing(self, table: str, copy: Path, out: Path) -> str:
        key_bytes: dict[str, int] = {}
        for p, b in parquet_bytes(copy).items():
            key_bytes[p.parent.name] = key_bytes.get(p.parent.name, 0) + b
        manifest = {e["key"]: e for e in read_jsonl(out / "manifest.jsonl")}
        events = []
        for key in manifest:
            parts = list((out / key).glob("*.parquet"))
            if parts:
                events.append((max(p.stat().st_mtime for p in parts), key_bytes.get(key, 0)))
        buckets_root = self.dir / "compact" / "sort" / "buckets" / table
        for bdir in sorted(buckets_root.glob("*/")) if buckets_root.is_dir() else []:
            if bdir.name in manifest:
                continue
            return f"{len(manifest):,} of {len(key_bytes):,} keys sorted; key {bdir.name}: " + self._bucketed(table, bdir, out / bdir.name)
        done = sum(key_bytes.get(k, 0) for k in manifest)
        r = rate(events, self.window)
        return progress(f"{len(manifest):,} of {len(key_bytes):,} keys sorted", done, sum(key_bytes.values()), r, self.window)

    def _bucketed(self, table: str, bdir: Path, out_key: Path) -> str:
        record = bdir / "buckets.json"
        if not record.exists():
            return "bucketing (see the run's own progress bar)"
        n = len(json.loads(record.read_text())["rows"])
        sorted_ = read_jsonl(bdir / "sorted.jsonl")
        if len(sorted_) < n:
            events = [((bdir / e["name"]).stat().st_mtime, 1) for e in sorted_ if (bdir / e["name"]).exists()]
            return self._count_eta(f"sorting buckets, {len(sorted_):,} of {n:,}", len(sorted_), n, events)
        files = math.ceil(sum(e["bytes"] for e in sorted_) / COMPACT_FILE_BYTES)
        packed = list(out_key.glob("part-*.parquet"))
        events = [(p.stat().st_mtime, 1) for p in packed]
        return self._count_eta(f"packing files, {len(packed):,} of about {files:,}", len(packed), files, events)

    def _count_eta(self, what: str, done: int, total: int, events: list[tuple[float, float]]) -> str:
        r = rate(events, self.window)
        if r is None:
            return f"{what}; nothing finished in the last {self.window / 60:.0f} min to measure a rate"
        left = max(total - done, 0) / r
        return f"{what}, {r:.0f}/h, ETA {left:.1f} h ({datetime.fromtimestamp(time.time() + left * 3600):%a %H:%M})"

    # Ledger-only steps

    def refs(self) -> tuple[bool, str]:
        stage = last_stage(self.state / "claims_labels_build.jsonl")
        return stage is not None, "collecting refs from the sorted claims (no local progress record)"

    def claims_labels_build(self) -> tuple[bool, str]:
        stage = last_stage(self.state / "claims_labels_build.jsonl")
        nxt = {None: "collecting refs", "refs": "writing", "written": "uploading to the Hub"}.get(stage, "")
        return stage == "uploaded", f"claims_labels: {nxt} (no local progress record)"

    def cards(self) -> tuple[bool, str]:
        return (self.state / "finalise.done").exists(), "dataset cards: computing figures, pushing cards"


def steps(release: str, window: float) -> list[tuple[str, callable]]:
    """Each step of `just release`, in order, as (label, check)."""
    main, scholar = Set(release, "main", window), Set(release, "scholar", window)
    out = [("process scholar", scholar.processing), ("process main", main.processing)]

    def tables(s: Set, label: str) -> list:
        rows = []
        for t in TABLES:
            rows += [(f"{label}: compact {t}", lambda s=s, t=t: s.compact(t)),
                     (f"{label}: sort {t}", lambda s=s, t=t: s.sort(t))]
            if t == "claims":
                rows.append((f"{label}: claims refs", s.refs))
        return rows

    def claims_labels(s: Set, label: str) -> list:
        return [
            (f"{label}: build claims_labels", s.claims_labels_build),
            (f"{label}: compact claims_labels", lambda: s.compact("claims_labels")),
            (f"{label}: sort claims_labels", lambda: s.sort("claims_labels")),
            (f"{label}: dataset cards", s.cards),
        ]

    out += tables(main, "finalise main")
    out += tables(scholar, "finalise scholar") + claims_labels(scholar, "finalise scholar")
    out += claims_labels(main, "finalise main")
    return out


def main() -> None:
    release = sys.argv[1]
    window = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 30 * 60
    rows = steps(release, window)
    width = max(len(label) for label, _ in rows)
    print(f"release {release}, {datetime.now():%a %H:%M}\n")
    here = False
    for label, check in rows:
        if here:
            print(f"  ·  {label}")
            continue
        done, detail = check()
        if done:
            print(f"  ✓  {label:{width}}  {detail}".rstrip())
        else:
            here = True
            print(f"  ▶  {label:{width}}  {detail}   <- you are here")
    for s in ("main", "scholar"):
        print(f"  {'·' if here else '?'}  promote {s} (on the Hub: tags, not recorded locally)")


if __name__ == "__main__":
    main()
