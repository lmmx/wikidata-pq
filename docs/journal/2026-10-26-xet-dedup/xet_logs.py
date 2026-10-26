"""Offline: how much each upload deduplicated, from hf_xet's own log files (stdlib only;
no Hub call, nothing sent). For the user to run on the release machine.

hf_xet (1.6.0) logs to `~/.cache/huggingface/xet/logs` by default (HF_XET_LOG_DEST;
`$HF_XET_CACHE/logs` or `$HF_HOME/xet/logs` when those are set), one file a process, JSON
lines, at INFO, kept up to 2 weeks and 250 MB (xet_runtime/src/config/groups/log.rs;
logging/init.rs; logging/constants.rs). After each file is chunked and deduplicated,
the client logs one line (xet_data/src/processing/file_cleaner.rs:274-282):

    {"timestamp":...,"level":"INFO","fields":{"action":"clean","file_name":"<path>",
     "file_size_count":<total_bytes>,"new_bytes_count":<new_bytes>,"start_ts":...,
     "end_processing_ts":...},...}

- `new_bytes_count` is the bytes that went into new xorbs: what the upload sent (before
  xorb compression, ~nothing on zstd Parquet). The rest was deduplicated.
- `file_size_count` is DeduplicationMetrics.total_bytes. It over-counts a file whose
  dedup the fragmentation guard refused for a range (that range is counted once as
  matched and again as new: file_deduplication.rs:200-203, 236-237), so the file's own
  size is taken when the file is still on disk.
- Each line's file is classified by its path as the pipeline's uploads are laid out:
  `releases/<release>[-scholar]/staging/<table>/...` a group file (close_group's push to
  the build branch), anything else under `releases/` a published file (finalise's
  commits), with its table.

    python xet_logs.py                     # the default log directory
    python xet_logs.py --log-dir DIR --out xet_dedup.json

The same lines are read by `read_dedup`, for #187's throughput_measure.py `logs` mode
(throughput_measure.logs.patch).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

TABLES = ("claims_labels", "claims", "labels", "descriptions", "aliases", "links", "entities")
# (the pretty, non-JSON form: HF_XET_LOG_FORMAT other than json)
TEXT = {k: re.compile(rf'{k}[=:]\s*"?([^",\s}}]+)') for k in ("file_name", "file_size_count", "new_bytes_count")}


def default_dir() -> Path:
    if os.environ.get("HF_XET_LOG_DEST"):
        return Path(os.environ["HF_XET_LOG_DEST"])
    if os.environ.get("HF_XET_CACHE"):
        return Path(os.environ["HF_XET_CACHE"]) / "logs"
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "xet" / "logs"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface" / "xet" / "logs"


def classify(path: str) -> tuple[str, str, str, str]:
    """(release, kind, set, table) of an uploaded file's local path."""
    m = re.search(r"releases/(\d{8})(-scholar)?/(.*)", path)
    if not m:
        return "", "other", "", ""
    release, scholar, rest = m.group(1), bool(m.group(2)), m.group(3)
    kind = "groups" if rest.startswith("staging/") else "published"
    parts = rest.split("/")
    table = next((p for p in parts[:-1] if p in TABLES), "")
    return release, kind, "scholar" if scholar else "main", table


def read_dedup(paths: list[Path]) -> list[dict]:
    """Every `clean` line: its log file, time, file, sizes, and (release, kind, set,
    table)."""
    out = []
    for p in sorted(paths):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if '"clean"' not in line and "clean" not in line:
                continue
            f = None
            if line.startswith("{"):
                try:
                    f = json.loads(line).get("fields", {})
                except ValueError:
                    f = None
                if not f or f.get("action") != "clean":
                    continue
                name, total, new, t = f.get("file_name", ""), f.get("file_size_count"), f.get("new_bytes_count"), \
                    f.get("end_processing_ts")
            else:
                if "action" not in line or not all(r.search(line) for r in TEXT.values()):
                    continue
                name, total, new = (TEXT[k].search(line).group(1) for k in TEXT)
                t = None
            try:
                total, new = int(total), int(new)
            except (TypeError, ValueError):
                continue
            size = os.path.getsize(name) if name and os.path.exists(name) else total
            out.append({"log": p.name, "t": t, "file": name, "size": size, "logged_total": total, "new": new,
                        **dict(zip(("release", "kind", "set", "table"), classify(name)))})
    return out


def by_log(recs: list[dict]) -> list[dict]:
    """By log file (a process: a run's pushes, a finalise run's commits): its first and
    last line's time, files, bytes, bytes sent, dedup."""
    g: dict[str, dict] = {}
    for r in recs:
        x = g.setdefault(r["log"], {"log": r["log"], "first": r["t"], "last": r["t"], "files": 0, "bytes": 0, "new": 0})
        x["last"] = r["t"] or x["last"]
        x["files"] += 1
        x["bytes"] += r["size"]
        x["new"] += min(r["new"], r["size"])
    print("| log file (process) | first, last | files | GB | GB sent | dedup |")
    print("|---|---|---|---|---|---|")
    for x in g.values():
        x["dedup"] = round(1 - x["new"] / x["bytes"], 4) if x["bytes"] else None
        print(f"| {x['log']} | {x['first']}, {x['last']} | {x['files']} | {x['bytes'] / 1e9:.2f} | {x['new'] / 1e9:.2f} | "
              f"{100 * (x['dedup'] or 0):.1f}% |")
    return list(g.values())


def summarise_dedup(recs: list[dict]) -> list[dict]:
    """By release, kind, set and table: files, bytes, bytes sent, dedup."""
    g: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in recs:
        for key in ((r["release"], r["kind"], r["set"], r["table"]), (r["release"], r["kind"], "all", ""),
                    (r["release"], "all", "", "")):
            x = g[key]
            x[0] += 1
            x[1] += r["size"]
            x[2] += min(r["new"], r["size"])
    rows = []
    print("| release | kind | set | table | files | GB | GB sent | dedup |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(g):
        n, size, new = g[key]
        rows.append({"release": key[0], "kind": key[1], "set": key[2], "table": key[3], "files": n,
                     "bytes": size, "new": new, "dedup": round(1 - new / size, 4) if size else None})
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {n} | {size / 1e9:.2f} | {new / 1e9:.2f} | "
              f"{100 * (1 - new / size) if size else 0:.1f}% |")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("xet_dedup.json"))
    a = ap.parse_args()
    root = a.log_dir or default_dir()
    paths = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    print(f"{len(paths)} log files under {root}")
    recs = read_dedup(paths)
    print(f"{len(recs)} files' clean lines")
    logs = by_log(recs)
    print()
    rows = summarise_dedup(recs)
    a.out.write_text(json.dumps({"files": recs, "by_log": logs, "summary": rows}, indent=1) + "\n")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
