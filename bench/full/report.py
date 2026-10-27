"""A compute-benchmark run's records (run.py's OUT_DIR) made into result.json and
summary.md, and two runs compared (a local and a streaming run, say).

Measured figures and modelled ones are kept apart: every modelled figure is under a key
or a heading that says so, with what it assumes.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

RELEASE_DUMP_BYTES = 103_488_697_311  # (bench/release.py)
# The CI bench's release model, for comparison: bench 17.12.0's at master 8f7720b, from 6
# runs on AMD EPYC 7763 runners with 4 cores (docs/journal/2026-10-27-compute-frontier.md,
# "Current State"): each stage's modelled wall at a release, its CPU time on the bench's
# 76 MB slice scaled to the release by bytes (the split by the dump's bz2 bytes) and
# spread over 4 cores. Modelled, not measured.
CI_MODEL = {
    "what": "bench 17.12.0's release model at master 8f7720b (6 runs, AMD EPYC 7763, 4 cores): "
            "a stage's sample CPU time scaled to the release and spread over 4 cores",
    "dump_bytes": RELEASE_DUMP_BYTES,
    "split_s": 3 * 3600 + 20 * 60,
    "pipeline_s": 1 * 3600 + 35 * 60,
    "finalise_s": 2 * 3600 + 13 * 60,
    # (the slice's JSON a bz2 byte, and the real dump's, 12 windows: #194's journal)
    "slice_json_per_bz2": 13.69,
    "real_json_per_bz2_194": 17.46,
    "real_line_cost_vs_slice_194": 1.27,
}
CORES_MODEL = 4


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


# -- after a run, before its work directory goes (run.py)
def post_checks(work: Path, hub: Path, release: str, digest: bool = True) -> dict:
    """The stand-in's linked files unchanged since placed (the pipeline wrote into none),
    and the published files on the build branches: bytes, files and digest."""
    placed = _jsonl(hub / ".bench" / "placed.jsonl")
    changed = checked = 0
    last = {}
    for r in placed:
        last[r["path"]] = r
    for path, r in last.items():
        try:
            st = Path(path).stat()
        except OSError:
            continue  # (deleted since: a commit's deletion, or the local copy's clean-up)
        checked += 1
        if st.st_ino != r["ino"] or st.st_size != r["size"] or st.st_mtime_ns != r["mtime_ns"]:
            changed += 1
    out = {"placed_files": len(placed), "placed_copies": sum(r["how"] == "copy" for r in placed),
           "placed_checked": checked, "placed_changed": changed}
    pub: dict = {"main": {}, "scholar": {}}
    h = hashlib.sha256()
    files = sorted(p for p in hub.glob(f"*/build-{release}/**/*.parquet")) if hub.is_dir() else []
    for p in files:
        repo = p.relative_to(hub).parts[0].split("__", 1)[1]
        set_ = "scholar" if repo.startswith("wikidata-scholar-") else "main"
        table = repo.removeprefix("wikidata-").removeprefix("scholar-")
        t = pub[set_].setdefault(table, {"files": 0, "bytes": 0})
        t["files"] += 1
        t["bytes"] += p.stat().st_size
        if digest:
            h.update(f"{p.relative_to(hub)}\0".encode())
            h.update(bytes.fromhex(_sha256(p)))
    out["published"] = pub
    if digest:
        out["published_digest"] = h.hexdigest()
    return out


def outputs(work: Path, release: str) -> dict:
    """What the split made: its chunks and entities, and each set's parts."""
    out: dict = {}
    data = work / "releases" / release / "data"
    if (done := data / "split.done").exists():
        words = done.read_text().split()
        out["split_chunks"], out["entities"] = int(words[0]), int(words[2])
    for set_, d in (("main", release), ("scholar", f"{release}-scholar")):
        rows = _jsonl(work / "releases" / d / "data" / "manifest.jsonl")
        out[set_] = {"parts": len(rows), "rows": sum(r.get("rows", 0) for r in rows)}
    return out


# -- the figures
def _window(samples: list[dict], t0: float, t1: float) -> list[dict]:
    inside = [s for s in samples if t0 <= s["t"] <= t1]
    before = [s for s in samples if s["t"] < t0][-1:]
    after = [s for s in samples if s["t"] > t1][:1]
    return before + inside + after


def _cpu_split(win: list[dict]) -> dict | None:
    if len(win) < 2:
        return None
    d = [b - a for a, b in zip(win[0]["cpu"], win[-1]["cpu"])]
    total = sum(d) or 1
    user, nice, system, idle, iowait, irq, softirq, steal = d
    return {"user": round((user + nice) / total, 3), "system": round((system + irq + softirq) / total, 3),
            "iowait": round(iowait / total, 3), "idle": round(idle / total, 3), "steal": round(steal / total, 3)}


USER_HZ = 100  # (/proc/stat's unit: USER_HZ, 100 on Linux)


def _busy_s(win: list[dict]) -> float | None:
    """The machine's busy CPU seconds over a window (every process's, ours and others')."""
    if len(win) < 2:
        return None
    d = [b - a for a, b in zip(win[0]["cpu"], win[-1]["cpu"])]
    user, nice, system, idle, iowait, irq, softirq, steal = d
    return (user + nice + system + irq + softirq) / USER_HZ


def _bound(cores: float, ncpu: int, split: dict | None, load: float | None) -> str:
    """A heuristic, from the stage's cores and the machine's counters over its window."""
    if cores >= 0.85 * ncpu:
        return "CPU, every core"
    if split and split["iowait"] >= 0.15:
        return f"disk (iowait {split['iowait']:.0%} of the machine)"
    if split and split["idle"] + split["iowait"] < 0.1:
        return f"CPU on {cores:.1f} of {ncpu} cores, the machine full (other stages, or other work)"
    return f"CPU on {cores:.1f} of {ncpu} cores (serial parts, or waiting on another stage)"


def build(out: Path) -> dict:
    meta = json.loads((out / "meta.json").read_text())
    samples = _jsonl(out / "timeline.jsonl")
    ncpu = meta["machine"]["cpu"].get("affinity") or meta["machine"]["cpu"]["logical_cpus"]
    t_run = meta.get("t0") or (samples[0]["t"] if samples else 0)
    stages = []
    for name in meta.get("stages_order", []):
        p = out / f"stage_{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        win = _window(samples, r["t0"], r["t1"])
        split = _cpu_split(win)
        cores = r["cpu"] / r["wall"] if r["wall"] else 0.0
        peak = max((s["rss"].get(name, 0) for s in win), default=0)
        loads = [s["load"] for s in win if "load" in s]
        load = statistics.fmean(loads) if loads else None
        busy = _busy_s(win)
        stages.append({"stage": name, "rc": r["rc"],
                       # (the machine's CPU besides this stage's over its window: other work,
                       # or, overlapped, the other stages)
                       "other_cpu_s": None if busy is None else round(busy - r["cpu"], 1), "start_s": round(r["t0"] - t_run, 1),
                       "end_s": round(r["t1"] - t_run, 1), "wall_s": r["wall"], "cpu_s": r["cpu"],
                       "user_s": r["user"], "sys_s": r["sys"], "cores": round(cores, 2),
                       "peak_rss_tree_mb": round(peak / 1e6), "maxrss_one_process_mb": r["maxrss_mb"],
                       "read_bytes": r["read_bytes"], "write_bytes": r["write_bytes"],
                       "majflt": r["majflt"], "minflt": r["minflt"],
                       "machine_cpu": split, "load": load and round(load, 2),
                       "bound": _bound(cores, ncpu, split, load)})
    by = {s["stage"]: s for s in stages}
    runs = []
    for p in sorted(out.glob("*finalise_run*_*.json")):
        r = json.loads(p.read_text())
        runs.append({"run": p.stem, "wall_s": r["wall"], "cpu_s": r["cpu"], "start_s": round(r["t0"] - t_run, 1),
                     "end_s": round(r["t1"] - t_run, 1), "maxrss_one_process_mb": r["maxrss_mb"],
                     "read_bytes": r["read_bytes"], "write_bytes": r["write_bytes"]})
    # finalise's steps and the group closes (spans.jsonl): wall and CPU (the process's own
    # and its reaped workers'), by set, step and table
    spans: dict = {}
    for s in _jsonl(out / "spans.jsonl"):
        key = f"{s['set']}/{s['fn']}" + (f"/{s['of']}" if s["fn"] in ("compact_table", "sort_table") else "")
        agg = spans.setdefault(key, {"calls": 0, "wall_s": 0.0, "cpu_s": 0.0})
        agg["calls"] += 1
        agg["wall_s"] = round(agg["wall_s"] + s["t1"] - s["t0"], 2)
        agg["cpu_s"] = round(agg["cpu_s"] + s["cpu_self"] + s["cpu_children"], 2)
    claims_labels = {}
    for set_ in ("main", "scholar"):
        parts = [spans.get(f"{set_}/build_claims_labels"), spans.get(f"{set_}/compact_table/claims_labels"),
                 spans.get(f"{set_}/sort_table/claims_labels")]
        if any(parts):
            claims_labels[set_] = {"wall_s": round(sum(p["wall_s"] for p in parts if p), 2),
                                   "cpu_s": round(sum(p["cpu_s"] for p in parts if p), 2),
                                   "build_wall_s": (parts[0] or {}).get("wall_s"),
                                   "build_cpu_s": (parts[0] or {}).get("cpu_s")}
    # the transfers a release would make, logged by the stand-in, never timed
    transfers: dict = {"upload": {}, "download": {}}
    for way, f in (("upload", "hub_uploads.jsonl"), ("download", "hub_downloads.jsonl")):
        for r in _jsonl(out / f):
            k = f"{r['set']}/{r['kind']}"
            t = transfers[way].setdefault(k, {"calls": 0, "files": 0, "bytes": 0})
            t["calls"] += 1
            t["files"] += r["files"]
            t["bytes"] += r["bytes"]
    hashes = _jsonl(out / "hub_hashes.jsonl")
    standin = {"hashed_bytes": sum(h["bytes"] for h in hashes), "hash_wall_s": round(sum(h["s"] for h in hashes), 2),
               "hash_cpu_s": round(sum(h["cpu"] for h in hashes), 2)}
    # the whole run
    top = [s for s in stages]
    total = {"wall_s": meta.get("wall_s"), "cpu_s": round(sum(s["cpu_s"] for s in top), 1),
             "read_bytes": sum(s["read_bytes"] for s in top), "write_bytes": sum(s["write_bytes"] for s in top)}
    if samples:
        free0 = samples[0]["free"]
        total |= {"peak_rss_tree_mb": round(max(s["rss_total"] for s in samples) / 1e6),
                  "peak_rss_one_process_mb": round(max(s.get("rss_max_proc", 0) for s in samples) / 1e6),
                  "memory_used_peak_mb": round((meta["machine"]["memory"].get("MemTotal", 0)
                                                - min(s["mem_avail"] for s in samples if s["mem_avail"])) / 1e6),
                  "min_mem_available_mb": round(min(s["mem_avail"] for s in samples if s["mem_avail"]) / 1e6),
                  "peak_disk_fall_bytes": max(free0 - s["free"] for s in samples),
                  "peak_work_tree_bytes": max((s["du"] for s in samples if "du" in s), default=None),
                  "machine_cpu": _cpu_split(samples),
                  # (the machine's busy CPU over the run besides the run's own: other work on the
                  # machine, and this harness's monitor; near 0 on a machine to itself)
                  "other_cpu_s": round(_busy_s(samples) - total["cpu_s"], 1),
                  "load_mean": round(statistics.fmean(s["load"] for s in samples), 2) if "load" in samples[0] else None}
        d = [s["disk"] for s in samples if s.get("disk")]
        if d:
            total |= {"device_read_bytes": d[-1][0] - d[0][0], "device_written_bytes": d[-1][1] - d[0][1],
                      "device_busy_s": round((d[-1][2] - d[0][2]) / 1000, 1)}
        n = [s["net"] for s in samples]
        total |= {"net_received_bytes": n[-1][0] - n[0][0], "net_sent_bytes": n[-1][1] - n[0][1]}
    # the time to the first processed chunk (the split's first manifest line), and each
    # set's first part numbered for its run, from the run's start
    first = {}
    for k in ("split_chunks", "main_parts", "scholar_parts"):
        hit = next((s["t"] for s in samples if (s.get("lines") or {}).get(k)), None)
        first[k] = None if hit is None else round(hit - t_run, 1)
    total["first_chunk_s"] = first["split_chunks"]
    total["first_part_s"] = {"main": first["main_parts"], "scholar": first["scholar_parts"]}
    # streaming: the bytes received over time, and the download against the split
    stream = None
    if meta["source"] == "stream":
        sizes = [(s["t"], (s["watch"].get("dump") or s["watch"].get("part") or 0)) for s in samples]
        rates, series = [], []
        for i in range(0, len(sizes) - 10, 10):
            (ta, a), (tb, b) = sizes[i], sizes[i + 10]
            if b > a or (series and b == a and a < sizes[-1][1]):
                rates.append((b - a) / (tb - ta))
            series.append([round(ta - t_run, 1), round((b - a) / (tb - ta) / 1e6, 2)])
        dl, sp = by.get("download"), by.get("split")
        got = sizes[-1][1] if sizes else 0
        stream = {"bytes": got, "download_wall_s": dl and dl["wall_s"],
                  "mean_mb_s": dl and round(got / dl["wall_s"] / 1e6, 2),
                  "rate_mb_s_10s": {"p10": round(_q(rates, 0.1) / 1e6, 2), "p50": round(_q(rates, 0.5) / 1e6, 2),
                                    "p90": round(_q(rates, 0.9) / 1e6, 2)} if rates else None,
                  "series_mb_s_10s": series,
                  "download_end_s": dl and dl["end_s"], "split_end_s": sp and sp["end_s"],
                  "split_after_download_s": dl and sp and round(sp["end_s"] - dl["end_s"], 1),
                  "first_chunk_s": total["first_chunk_s"],
                  "first_chunk_before_download_end": bool(dl and total["first_chunk_s"] is not None
                                                          and total["first_chunk_s"] < dl["end_s"]),
                  "at_download_end": _at(samples, dl and dl["end_s"] + t_run),
                  "net_received_bytes": total.get("net_received_bytes")}
        if (src := out / "stream_source.json").exists():
            stream["source"] = {k: v for k, v in json.loads(src.read_text()).items() if k != "ranges"}
    # per GB of the dump's bz2, and the CI bench's model beside it
    bz2 = (meta["dump"].get("bytes") or (stream or {}).get("bytes") or 0)
    js = meta["dump"].get("json_bytes")
    per_gb = None
    if bz2:
        g = bz2 / 1e9
        groups = {"split": ["split"], "pipeline": ["run_scholar", "run_main"],
                  "finalise": ["finalise", "finalise_early", "tail_finalise"]}
        per_gb = {"bz2_gb": round(g, 4), "json_per_bz2": js and round(js / bz2, 2)}
        for k, names in groups.items():
            xs = [by[n] for n in names if n in by]
            if not xs:
                continue
            cpu = sum(x["cpu_s"] for x in xs)
            row = {"cpu_s_per_gb": round(cpu / g, 1),
                   "cpu_over_4_cores_s_per_gb": round(cpu / CORES_MODEL / g, 1),
                   "model_s_per_gb": round(CI_MODEL[f"{k}_s"] / (RELEASE_DUMP_BYTES / 1e9), 1)}
            row["measured_over_model"] = round(row["cpu_over_4_cores_s_per_gb"] / row["model_s_per_gb"], 2)
            if js:  # (and per GB of JSON, against the model's on the CI bench's slice)
                row["cpu_s_per_json_gb"] = round(cpu / (js / 1e9), 2)
                row["model_cpu_s_per_json_gb"] = round(CI_MODEL[f"{k}_s"] * CORES_MODEL / (
                    RELEASE_DUMP_BYTES / 1e9 * CI_MODEL["slice_json_per_bz2"]), 2)
                row["measured_over_model_per_json_byte"] = round(
                    row["cpu_s_per_json_gb"] / row["model_cpu_s_per_json_gb"], 2)
            if meta["orchestration"] == "serial" or k == "split":
                row["wall_s_per_gb"] = round(sum(x["wall_s"] for x in xs) / g, 1)
            per_gb[k] = row
        per_gb["total"] = {"wall_s_per_gb": round((meta.get("wall_s") or 0) / g, 1),
                           "cpu_s_per_gb": round(total["cpu_s"] / g, 1)}
        if meta["label"] != "full":  # (modelled: the scaled run's rates, linear in bz2 bytes)
            per_gb["modelled_release_linear"] = {
                "what": "this scaled run's figures times the release's bz2 bytes over its own: linear in bytes, "
                        "which a sample of the dump's mix only approximates; modelled, not measured",
                "wall_s": round((meta.get("wall_s") or 0) * RELEASE_DUMP_BYTES / bz2),
                "cpu_s": round(total["cpu_s"] * RELEASE_DUMP_BYTES / bz2),
                "peak_disk_fall_gb": samples and round(total["peak_disk_fall_bytes"] * RELEASE_DUMP_BYTES / bz2 / 1e9),
                "disk_factor_measured": samples and round(total["peak_disk_fall_bytes"] / bz2, 2)}
    res = {"name": meta["name"], "ok": meta.get("ok"), "label": meta["label"], "release": meta["release"],
           "source": meta["source"], "orchestration": meta["orchestration"], "cards": meta.get("cards"),
           "dump": {k: v for k, v in meta["dump"].items() if k not in ("ranges",)},
           "machine": meta["machine"], "code": meta["code"], "env": meta["env"], "config": meta["config"],
           "preflight": meta["preflight"], "finalised": meta.get("finalised"), "outputs": meta.get("outputs"),
           "post": meta.get("post"), "total": total, "stages": stages, "finalise_runs": runs,
           "claims_labels": claims_labels, "steps": spans, "transfers_not_timed": transfers,
           "standin_overhead": standin, "stream": stream, "per_gb": per_gb, "ci_model": CI_MODEL}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "summary.md").write_text(summary(res))
    return res


def _at(samples: list[dict], t: float | None) -> dict | None:
    """What was done by time `t`: the split's chunks, each set's parts, the CPU so far."""
    if t is None or not samples:
        return None
    s = [x for x in samples if x["t"] <= t][-1:] or samples[:1]
    return {"split_chunks": (s[0].get("lines") or {}).get("split_chunks"),
            "main_parts": (s[0].get("lines") or {}).get("main_parts"),
            "scholar_parts": (s[0].get("lines") or {}).get("scholar_parts")}


def _q(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _gb(b) -> str:
    return "–" if b is None else f"{b / 1e9:.2f}"


def summary(r: dict) -> str:
    m, t = r["machine"], r["total"]
    d = r["dump"]
    lines = [f"# Compute benchmark: {r['name']}", "",
             f"- **Result:** {'ok' if r['ok'] else 'FAILED'}; both sets finalised: {r['finalised']}",
             f"- **Dump:** release {r['release']}, {r['label']}: {_gb(d.get('bytes'))} GB of bz2"
             + (f", {_gb(d.get('json_bytes'))} GB of JSON" if d.get("json_bytes") else "")
             + (f"; the whole dump's md5 {d['source'].get('md5')}, sha1 {d['source'].get('sha1')}" if d.get("source") else ""),
             f"- **Source:** {r['source']} ({'no network on the timed path' if r['source'] == 'local' else 'fetched as the run goes, the split following it'}); "
             f"orchestration: {r['orchestration']}; cards: {r['cards']}",
             f"- **Machine:** {m['cpu'].get('model')}, {m['cpu'].get('affinity')} CPUs, "
             f"{_gb(m['memory'].get('MemTotal'))} GB RAM, {m['disk'].get('fs')} on {m['disk'].get('device')} "
             f"({m['disk'].get('model') or 'model unknown'}), {m['os']}, kernel {m['kernel']}",
             f"- **Code:** {r['code']['describe']} ({r['code']['commit']}), "
             + ", ".join(f"{k} {v}" for k, v in r["code"]["packages"].items() if v),
             f"- **Decoder:** {(r['config'].get('main') or {}).get('decoder')}; WIKIDATA_BZ2_DECODER="
             f"{(r['config'].get('main') or {}).get('bz2_decoder_env')}, WIKIDATA_BZ2_THREADS="
             f"{(r['config'].get('main') or {}).get('bz2_threads_env')}, WIKIDATA_SPLIT_WRITER_THREADS="
             f"{(r['config'].get('main') or {}).get('SPLIT_WRITER_THREADS')}",
             "", "## Measured", "",
             f"- **Wall, end to end:** {t['wall_s']:.0f} s; **CPU:** {t['cpu_s']:.0f} s "
             f"({t['cpu_s'] / max(t['wall_s'] or 1, 1):.2f} cores on average)",
             f"- **Time to the first processed chunk** (the split's first chunk written): {t.get('first_chunk_s')} s; "
             f"each set's first part ready for its run: {t.get('first_part_s')} s (±1 s, the monitor's interval)",
             f"- **Peak RSS, every process at once:** {t.get('peak_rss_tree_mb')} MB (the largest process "
             f"{t.get('peak_rss_one_process_mb')} MB); the machine's memory in use at its peak "
             f"{t.get('memory_used_peak_mb')} MB (least available {t.get('min_mem_available_mb')} MB, other work included)",
             f"- **Disk:** peak {_gb(t.get('peak_disk_fall_bytes'))} GB taken (free space's fall), work tree peak "
             f"{_gb(t.get('peak_work_tree_bytes'))} GB; the device read {_gb(t.get('device_read_bytes'))} GB "
             f"and wrote {_gb(t.get('device_written_bytes'))} GB; the stages' own I/O read "
             f"{_gb(t.get('read_bytes'))} GB and wrote {_gb(t.get('write_bytes'))} GB",
             f"- **Network:** received {_gb(t.get('net_received_bytes'))} GB, sent {_gb(t.get('net_sent_bytes'))} GB "
             "(the machine's, every process)",
             f"- **Other work on the machine:** {t.get('other_cpu_s')} CPU s besides the run's own "
             f"({(t.get('other_cpu_s') or 0) / max(t['wall_s'] or 1, 1):.2f} cores on average; mean load "
             f"{t.get('load_mean')}). Near 0, the run had the machine to itself", "",
             "| stage | start s | wall s | CPU s | cores | other CPU s | peak RSS MB (tree) | read GB | written GB | machine: user / sys / iowait / idle | bound |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|"]
    for s in r["stages"]:
        c = s["machine_cpu"] or {}
        lines.append(f"| {s['stage']} | {s['start_s']:.0f} | {s['wall_s']:.1f} | {s['cpu_s']:.1f} | {s['cores']:.2f} | "
                     f"{s['other_cpu_s']} | "
                     f"{s['peak_rss_tree_mb']} | {_gb(s['read_bytes'])} | {_gb(s['write_bytes'])} | "
                     f"{c.get('user', 0):.0%} / {c.get('system', 0):.0%} / {c.get('iowait', 0):.0%} / {c.get('idle', 0):.0%} | {s['bound']} |")
    if r["orchestration"] == "overlap":
        lines += ["", "(Overlapped stages share the machine: their machine-wide split is the window's, not theirs alone.)"]
    if r["finalise_runs"]:
        lines += ["", "| finalise run | start s | wall s | CPU s | largest process MB |", "|---|--:|--:|--:|--:|"]
        lines += [f"| {x['run']} | {x['start_s']:.0f} | {x['wall_s']:.1f} | {x['cpu_s']:.1f} | {x['maxrss_one_process_mb']:.0f} |"
                  for x in r["finalise_runs"]]
    if r["claims_labels"]:
        lines += ["", "**claims_labels** (its build, compaction and sort, within finalise's runs): "
                  + "; ".join(f"{k} {v['wall_s']:.1f} s wall, {v['cpu_s']:.1f} s CPU" for k, v in r["claims_labels"].items())]
    if r["steps"]:
        lines += ["", "<details><summary>finalise's steps and the group closes (wall, CPU incl. workers)</summary>", "",
                  "| step | calls | wall s | CPU s |", "|---|--:|--:|--:|"]
        lines += [f"| {k} | {v['calls']} | {v['wall_s']:.1f} | {v['cpu_s']:.1f} |" for k, v in sorted(r["steps"].items())]
        lines += ["", "</details>"]
    tr = r["transfers_not_timed"]
    lines += ["", "## Transfers a release would make (logged by the stand-in Hub, not timed)", "",
              "| way | set/kind | calls | files | GB |", "|---|---|--:|--:|--:|"]
    for way in ("upload", "download"):
        for k, v in sorted(tr[way].items()):
            lines.append(f"| {way} | {k} | {v['calls']} | {v['files']} | {_gb(v['bytes'])} |")
    up = sum(v["bytes"] for v in tr["upload"].values())
    lines += ["", f"Uploads in all: {_gb(up)} GB (at 7 MB/s, the one rate on record: {up / 7e6 / 3600:.1f} h; at 60 MB/s: "
              f"{up / 60e6 / 3600:.1f} h; modelled, not measured).",
              f"The stand-in's own cost on the timed path (the Hub-side sha256 it computes): "
              f"{r['standin_overhead']['hash_cpu_s']:.1f} s CPU over {_gb(r['standin_overhead']['hashed_bytes'])} GB."]
    if (p := r.get("post")):
        lines += [f"Linked files changed after placing: {p.get('placed_changed')} of {p.get('placed_checked')} checked "
                  f"({p.get('placed_copies')} copied, not linked). Published digest: {p.get('published_digest')}"]
    if (s := r.get("stream")):
        lines += ["", "## Streaming", "",
                  f"- received {_gb(s['bytes'])} GB in {s['download_wall_s']} s: {s['mean_mb_s']} MB/s on average; "
                  f"10-s rates p10/p50/p90 {s['rate_mb_s_10s']}",
                  f"- the download ended at {s['download_end_s']} s, the split at {s['split_end_s']} s "
                  f"({s['split_after_download_s']} s after it)",
                  f"- the first chunk was processed at {s['first_chunk_s']} s, "
                  f"{'before' if s['first_chunk_before_download_end'] else 'NOT before'} the download ended; by its end "
                  f"the split had written {(s['at_download_end'] or {}).get('split_chunks')} chunks and numbered "
                  f"{(s['at_download_end'] or {}).get('main_parts')} main and "
                  f"{(s['at_download_end'] or {}).get('scholar_parts')} scholarly parts for the runs"]
    if (g := r.get("per_gb")):
        lines += ["", "## Per GB of bz2, against the CI bench's model", "",
                  f"{g['bz2_gb']} GB of bz2; JSON {g.get('json_per_bz2')}x its bz2 bytes (the CI bench's slice "
                  f"{CI_MODEL['slice_json_per_bz2']}x, #194's 12 windows of the real dump {CI_MODEL['real_json_per_bz2_194']}x).",
                  "", "| stage | CPU s / GB | CPU / 4 cores, s / GB (measured) | model s / GB (CI bench) | measured / model | wall s / GB | measured / model, a JSON byte |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for k in ("split", "pipeline", "finalise"):
            if k in g:
                x = g[k]
                lines.append(f"| {k} | {x['cpu_s_per_gb']} | {x['cpu_over_4_cores_s_per_gb']} | {x['model_s_per_gb']} | "
                             f"{x['measured_over_model']} | {x.get('wall_s_per_gb', '–')} | "
                             f"{x.get('measured_over_model_per_json_byte', '–')} |")
        lines += ["", f"The model: {CI_MODEL['what']}. Its figures are modelled; this run's are measured. A JSON "
                  f"byte's: the model's CPU over its slice's JSON ({CI_MODEL['slice_json_per_bz2']}x its bz2 bytes)."]
        if (mr := g.get("modelled_release_linear")):
            lines += ["", f"**Modelled** (this run's rates, linear in bz2 bytes, to the release's 103.5 GB): wall "
                      f"{mr['wall_s'] / 3600:.1f} h, CPU {mr['cpu_s'] / 3600:.1f} h, peak disk ~{mr['peak_disk_fall_gb']} GB "
                      f"({mr['disk_factor_measured']}x the bz2). {mr['what']}."]
    return "\n".join(lines) + "\n"


def compare(a: Path, b: Path) -> str:
    """Two runs side by side; with a local and a streaming run, how much of the download
    the processing hid."""
    ra, rb = (json.loads((p / "result.json").read_text()) for p in (a, b))
    local, stream = (ra, rb) if ra["source"] == "local" else (rb, ra)
    rows = ["| | " + " | ".join(f"{r['label']}, {r['source']}, {r['orchestration']}" for r in (ra, rb)) + " |",
            "|---|--:|--:|"]
    for k, f in (("wall s", lambda r: r["total"]["wall_s"]), ("CPU s", lambda r: r["total"]["cpu_s"]),
                 ("first chunk s", lambda r: r["total"].get("first_chunk_s")),
                 ("other work, CPU s", lambda r: r["total"].get("other_cpu_s")),
                 ("download wall s", lambda r: (r.get("stream") or {}).get("download_wall_s")),
                 ("split wall s", lambda r: next((s["wall_s"] for s in r["stages"] if s["stage"] == "split"), None)),
                 ("peak RSS MB", lambda r: r["total"].get("peak_rss_tree_mb")),
                 ("peak disk GB", lambda r: round(r["total"].get("peak_disk_fall_bytes", 0) / 1e9, 2))):
        rows.append(f"| {k} | {f(ra)} | {f(rb)} |")
    for k in ("split", "pipeline", "finalise"):  # (CPU a GB of bz2 and of JSON: the inputs' cost, the machine's the same)
        ga, rb_ = (ra.get("per_gb") or {}).get(k), (rb.get("per_gb") or {}).get(k)
        if ga and rb_:
            rows.append(f"| {k} CPU s / GB bz2 | {ga['cpu_s_per_gb']} | {rb_['cpu_s_per_gb']} |")
            if ga.get("cpu_s_per_json_gb") and rb_.get("cpu_s_per_json_gb"):
                rows.append(f"| {k} CPU s / GB JSON | {ga['cpu_s_per_json_gb']} | {rb_['cpu_s_per_json_gb']} |")
    rows.append(f"| JSON / bz2 | {(ra.get('per_gb') or {}).get('json_per_bz2')} | {(rb.get('per_gb') or {}).get('json_per_bz2')} |")
    out = "\n".join(rows)
    if local["source"] == "local" and stream["source"] == "stream" and stream.get("stream"):
        dl = stream["stream"]["download_wall_s"] or 0
        extra = stream["total"]["wall_s"] - local["total"]["wall_s"]
        out += (f"\n\nStreaming took {extra:+.1f} s against local; its download took {dl:.1f} s, so "
                f"{(1 - extra / dl) if dl else 0:.0%} of the download was hidden behind the processing "
                f"(1 - extra wall / download wall).")
    return out


if __name__ == "__main__":
    import sys

    print(compare(Path(sys.argv[1]), Path(sys.argv[2])) if len(sys.argv) > 2 else
          summary(build(Path(sys.argv[1]))))
