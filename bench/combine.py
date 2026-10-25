"""Rounds of the benchmark (ab.py's results.json files, one or more rounds each) as one
result: per round, the commit before's and after's figures on that round's runner, and
from them, over the rounds, each figure's mean ± standard deviation, and the change from
before to after, which is paired within each round, so runners of different speeds can
share the work.

Usage: combine.py <total rounds> <summary.md> <results.json>...   (BENCH_LOST: lost.py's file,
the jobs lost and why)
"""

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

from release import (RELEASE, RELEASE_CHUNKS, RELEASE_CLAIMS_BYTES, RELEASE_CLAIMS_ROWS, RELEASE_CLAIMS_WORKERS, RELEASE_DUMP_BYTES,
                     RELEASE_DUMP_GZ_BYTES, RELEASE_ENTITIES, RELEASE_FINALISE_WORKERS, RELEASE_GROUPS,
                     RELEASE_MAIN_TABLE_ROWS, RELEASE_MAIN_TABLES_GROWTH, RELEASE_SOURCE_BYTES)
from version import VERSION

METRICS = [  # label, results key, kind, decimals measured to
    ("measured, end to end: CPU (split + pipeline + finalise)", "measured_cpu", "s", 1),
    ("measured, end to end: wall (split + pipeline + finalise)", "measured_wall", "s", 2),
    ("ETA, a release", "eta", "s", 0), ("ETA: pipeline", "pipeline_eta", "s", 0),
    ("ETA: finalise", "finalise_eta", "s", 0), ("ETA: split", "split_eta", "s", 0),
    # (claims_labels' refs stage, apart from the figures above: bench 14.1.0)
    ("measured, end to end, with the claims refs: CPU", "measured_cpu_refs", "s", 1),
    ("measured, end to end, with the claims refs: wall", "measured_wall_refs", "s", 2),
    ("ETA, a release, with the claims refs", "eta_refs", "s", 0),
    ("ETA: claims refs, modelled", "refs_eta", "s", 0),
    ("claims refs", "refs_wall", "s", 2), ("CPU, claims refs", "refs_cpu", "s", 1),
    ("claims refs, 3× the rows", "refs_x3_wall", "s", 2), ("CPU, claims refs, 3× the rows", "refs_x3_cpu", "s", 1),
    ("peak RSS, claims refs: all processes", "refs_peak_rss_mb", "MB", 0),
    ("peak RSS, claims refs: one worker", "refs_worker_peak_mb", "MB", 0),
    ("peak RSS, claims refs 3×: all processes", "refs_peak_rss_mb_x3", "MB", 0),
    ("peak RSS, claims refs 3×: one worker", "refs_worker_peak_mb_x3", "MB", 0),
    ("wall, with finalise", "total_wall", "s", 2), ("wall", "wall", "s", 2),
    ("finalise: compact", "compact_wall", "s", 2), ("finalise: sort", "sort_wall", "s", 2),
    ("finalise, 3× the rows", "finalise_x3_wall", "s", 2),
    ("split: split", "split_wall", "s", 2), ("split: route", "route_wall", "s", 2), ("CPU, split", "split_cpu_total", "s", 1),
    ("peak RSS, split", "split_peak_rss_mb", "MB", 0),
    ("ETA: split from gz", "gz_split_eta", "s", 0), ("CPU, split from gz", "gz_split_cpu_total", "s", 1),
    ("throughput (entities/s)", "throughput", "rate", 0),
    ("CPU, all processes", "cpu_total", "s", 1),
    ("CPU, process", "process_cpu", "s", 1), ("CPU, partition", "partition_cpu", "s", 1),
    ("CPU, group merges", "close_cpu", "s", 1), ("CPU, finalise", "finalise_cpu", "s", 1), ("peak RSS, all processes", "peak_sum_rss_mb", "MB", 0),
    ("peak RSS, one worker", "worker_peak_mb", "MB", 0),
    ("peak RSS, finalise: all processes", "finalise_peak_rss_mb", "MB", 0),
    ("peak RSS, finalise: one worker", "finalise_worker_peak_mb", "MB", 0),
    ("peak RSS, finalise 3×: all processes", "finalise_peak_rss_mb_x3", "MB", 0),
    ("peak RSS, finalise 3×: one worker", "finalise_worker_peak_mb_x3", "MB", 0),
    ("peak RSS, finalise at a release, modelled: a bucket sort", "release_bucket_worker_rss_mb", "MB", 0),
    ("peak RSS, finalise at a release, modelled: all processes", "release_finalise_rss_mb", "MB", 0),
    ("CPU, a claims worker outside its jobs", "claims_worker_cpu", "s", 2),
    ("CPU, the pipeline's claims stages", "pipeline_claims_cpu", "s", 1),
    ("CPU, a pipeline worker outside its chunks", "pipeline_worker_cpu", "s", 2),
    ("disk used, peak rise", "disk_peak_mb", "MB", 0),
    ("device bytes written", "device_written_mb", "MB", 0),
]
# Figures kept with the metrics but not shown in the table: the fit's prediction of the
# finalise run on three times the rows (see pipeline.finalise_at_scale), the split
# from gz's against the split from bz2's (render_gz), and the bucket path as each
# finalise run took it, measured (pipeline.sort_shape)
SHAPE = ["sort_buckets", "sort_bucket_rows", "sort_spilled_share", "sort_flush_rows"]
# (and from bench 8.0.0, the claims key's compacted source files and part files, and the
# compaction target of the run on three times the rows)
FILES = ["sort_sources", "sort_parts"]
# (and from bench 9.0.0, the claims key's workers and finalise at a release's size:
# pipeline.release_model)
RELEASE_MODEL = ["finalise_eta_8", "finalise_cpu_per_mb_8", "claims_workers", "claims_workers_x3",
                 "claims_worker_startup_cpu", "release_claims_workers", "release_claims_workers_eta",
                 "bucket_worker_rss_mb", "bucket_worker_rss_mb_x3", "release_bucket_rows", "sort_bucket_rows_max",
                 "sort_bucket_rows_max_x3"]
# (and from bench 10.0.0, the claims' work by set, measured, and the claims parts scaled
# to a release by the claims' own rows and bytes: pipeline.claims_model)
CLAIMS_MODEL = [f"pipeline_claims_{k}_{s}" for s in ("main", "scholar") for k in ("cpu", "rows", "bytes", "source")] + [
    "finalise_claims_share", "release_claims_rows_scholar", "release_claims_mb", "release_pipeline_cpu",
    "claims_rows_scale_main", "claims_rows_scale_scholar", "claims_bytes_scale", "pipeline_eta_9", "finalise_eta_9",
    "eta_9"]
# (and from bench 11.0.0, the pipeline's workers, measured, their starts at a release,
# modelled (pipeline.starts_model), and the split on bench 10.0.0's slice)
STARTS_MODEL = ["scale", "pipeline_workers", "pipeline_worker_startup_cpu", "pipeline_workers_counted",
                "release_chunks_per_worker", "release_pipeline_workers", "release_pipeline_workers_cpu",
                "pipeline_workers_cpu_scaled_10", "release_pipeline_cpu_10", "pipeline_eta_10", "split_eta_10", "eta_10",
                "orjson_split_cpu_total", "orjson_split_wall_total", "orjson_split_source_mb", "finalise_wall"]
# (and from bench 15.1.0, the other tables' group files, measured, and finalise's
# per-file stages scaled by a release's files: pipeline.files_model)
FILES_MODEL = ["finalise_files", "finalise_keys", "finalise_file_stages_cpu", "finalise_file_hashed_cpu",
               "finalise_file_ms", "release_files", "release_files_eta", "finalise_eta_15", "eta_15"]
# (and from bench 15.2.0, the other tables' per-byte stages scaled by their own rows at a
# release: pipeline.tables_model; and the pipeline's claims build, measured)
TABLES_MODEL = [k for t in RELEASE_MAIN_TABLE_ROWS
                for k in (f"finalise_{t}_bytes_cpu", f"release_{t}_rows", f"{t}_rows_scale")] + [
    "release_tables_eta", "pipeline_eta_15_1", "finalise_eta_15_1", "eta_15_1", "pipeline_claims_build_cpu"]
FIGURES = ["finalise_x3_wall_fit", *FILES_MODEL, *TABLES_MODEL, "refs_cpu_per_mb", "refs_fixed", *RELEASE_MODEL, *CLAIMS_MODEL, *STARTS_MODEL, *FILES, *(f"{k}_x3" for k in FILES), "x3_compact_file_mb",
           "x3_other_key_max_mb", "split_source_mb", "split_wall_total", "gz_split_source_mb", "gz_split_wall_total",
           "gz_split_peak_rss_mb", "bz2_decode_wall", "bz2_decode_cpu", "gz_decode_wall", "gz_decode_cpu", *SHAPE, *(f"{k}_x3" for k in SHAPE)]
# Sizes' units, smallest first: their size in MB, and the decimals they add to keep the
# measurement's precision (1234 MB is 1.234 GB)
SIZES = [("MB", 1, 0), ("GB", 1000, 3), ("TB", 10**6, 6)]


def duration(seconds: float, decimals: int, sign: bool = False) -> str:
    """1h23m45s, 10m23s, 11.42s: seconds keep their decimals only under a minute."""
    lead = ("+" if seconds > 0 else "-" if seconds < 0 else "") if sign else "-" if seconds < 0 else ""
    v = abs(seconds)
    if round(v) < 60:
        return f"{lead}{v:.{decimals}f}s"
    h, rest = divmod(round(v), 3600)
    m, s = divmod(rest, 60)
    return f"{lead}{h}h{m:02d}m{s:02d}s" if h else f"{lead}{m}m{s:02d}s"


def formatter(kind: str, before: float, decimals: int):
    """The row's format, set by its value before: a duration, or the largest size unit
    in which the value before is at least 1. Returns (unit label, format)."""
    if kind == "s":
        return "", lambda v, sign=False: duration(v, decimals, sign)
    if kind == "rate":
        return "", lambda v, sign=False: f"{v:{'+' if sign else ''},.{decimals}f}"
    name, size, extra = ([u for u in SIZES if abs(before) >= u[1]] or SIZES[:1])[-1]
    return f" ({name})", lambda v, sign=False: f"{v / size:{'+' if sign else ''},.{decimals + extra}f}"


def relative(ratio: float) -> str:
    """A change as a percentage, or as a multiple once it more than doubles (2.3×) or
    falls below half (÷2.5)."""
    return f"{ratio:.1f}×" if ratio > 2 else f"÷{1 / ratio:.1f}" if 0 < ratio < 0.5 else f"{100 * (ratio - 1):+.1f}%"


# Student's t at 97.5% for n - 1 degrees of freedom, by n (from 2; the last for n >= 10)
T95 = [None, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26]


def noise(diffs: list[float]) -> bool:
    """Whether paired changes are inside the 95% confidence interval of their mean: noise."""
    n = len(diffs)
    return n > 1 and abs(statistics.fmean(diffs)) <= T95[min(n, len(T95)) - 1] * statistics.stdev(diffs) / n**0.5


def interval(values: list[float]) -> float:
    """The half-width of the 95% confidence interval of the values' mean (0 for one value)."""
    n = len(values)
    return T95[min(n, len(T95)) - 1] * statistics.stdev(values) / n**0.5 if n > 1 else 0.0


def ratios_of(rounds: list[dict], k: str) -> list[float]:
    """Each round's paired ratio of figure `k`, after over before: each half's on its own
    runner, averaged over the round's halves, so a fast and a slow runner count alike (a
    record without halves: its figures' ratio)."""
    before, after = rounds[0]["names"]
    out = []
    for r in rounds:
        v = r.get("ratios", {}).get(k)
        if v is None and r["figures"][before].get(k) and r["figures"][after].get(k) is not None:
            v = r["figures"][after][k] / r["figures"][before][k]
        out.append(v)
    return out


def percent(ratios: list[float]) -> str:
    """The mean paired change as a percentage and its 95% interval: "+1.2% ±2.0%"."""
    return f"{100 * (statistics.fmean(ratios) - 1):+.1f}% ±{100 * interval(ratios):.1f}%"


def spread(values: list[float], f) -> str:
    """" ± <standard deviation>", in the values' format; nothing for one value."""
    return f" ± {f(statistics.stdev(values))}" if len(values) > 1 else ""


def shape(rounds: list[dict], before: str, after: str) -> str:
    """The bucket path as finalise's two runs took it (pipeline.sort_shape), measured, for
    each commit that records it: buckets, rows a bucket, the share of rows spilled and
    rows a spill flush, at the sample's rows and at three times them (a run with nothing
    spilled has no flushes: its rows a flush is not shown)."""
    flush = lambda v: f"{v:,.0f}" if v else "no"  # (nothing spilled, no flushes)

    def one(a: str) -> str | None:
        got = [[r["figures"][a].get(f"{k}{x}") for r in rounds] for x in ("", "_x3") for k in SHAPE]
        if any(None in v for i, v in enumerate(got) if i % 4 != 3):
            return None
        got = [[x or 0 for x in v] for v in got]
        b, rb, sp, rf, b3, rb3, sp3, rf3 = (statistics.median(v) for v in got)
        files = [[r["figures"][a].get(f"{k}{x}") for r in rounds] for x in ("", "_x3") for k in FILES]
        told = ""
        if not any(None in v for v in files):
            s1, p1, s3, p3 = (statistics.median(v) for v in files)
            told = f"{s1:.0f} and {s3:.0f} source files, {p1:.0f} and {p3:.0f} part files, "
        return (f"{a}: {told}{b:.0f} and {b3:.0f} buckets, {rb:,.0f} and {rb3:,.0f} rows a bucket, "
                f"{sp:.0%} and {sp3:.0%} of rows spilled, {flush(rf)} and {flush(rf3)} rows a spill flush")
    told = [t for a in (before, after) if (t := one(a))]
    return ("The bucket path, measured (the sample's rows, then three times them): " + "; ".join(told)
            + ". ") if told else ""


def release_notes(rounds: list[dict], before: str, after: str) -> str:
    """The ETA's worker starts and finalise's peak RSS at a release (pipeline.release_model),
    for the commit after (the commit before's where it alone has them): measured, and
    modelled from it."""
    arm = next((a for a in (after, before) if all(r["figures"][a].get("claims_worker_cpu") is not None
                                                  for r in rounds)), None)
    got = lambda k: [r["figures"][a].get(k) for r in rounds for a in [arm or after]]
    mean = lambda k: statistics.fmean(v) if (v := got(k)) and None not in v else None
    out = ""
    if arm:
        eta8 = lambda a: (duration(statistics.fmean(v), 0) if None not in (v := [r["figures"][a].get("finalise_eta_8")
                                                                              for r in rounds]) else "–")
        out += (f"The claims key's workers (a worker a job: compaction, bucketing, bucket sorts, packing), measured "
                f"({arm}): {mean('claims_workers'):.0f} in the first run and {mean('claims_workers_x3'):.0f} on three "
                f"times the rows, each {mean('claims_worker_cpu'):.2f} CPU-s outside its jobs "
                f"({mean('claims_worker_startup_cpu'):.2f} of them to its first job: the interpreter and the imports). "
                "Every worker's CPU time outside its jobs is taken out of the part per byte, which carried it scaled "
                f"to the release's bytes, and the release's claims workers added back: "
                f"{mean('release_claims_workers'):,.0f} (the main set's {RELEASE_CLAIMS_WORKERS['main']} measured, the "
                f"scholarly set's {RELEASE_CLAIMS_WORKERS['scholar']} modelled, and its pools' workers started with no "
                f"job), modelled: {duration(mean('release_claims_workers_eta'), 0)} of the finalise ETA. With every "
                f"start in the part per byte (bench 8.0.0's model), finalise would be {eta8(before)} → {eta8(after)}. ")
    if (rss := mean("release_bucket_worker_rss_mb")) is not None:
        gb = lambda v: f"{v / 1000:.2f} GB"
        out += (f"Finalise's peak RSS at a release, modelled: a bucket-sort worker at {mean('release_bucket_rows'):,.0f} "
                f"rows (the release's rows a bucket), from the largest bucket sorts measured, "
                f"{gb(mean('bucket_worker_rss_mb'))} at {mean('sort_bucket_rows_max'):,.0f} rows and "
                f"{gb(mean('bucket_worker_rss_mb_x3'))} at {mean('sort_bucket_rows_max_x3'):,.0f} (each worker's own "
                f"peak at its exit), on the line through them (or, for runs under 20% apart in rows, in proportion "
                f"to the larger's rows): {gb(rss)}")
        if (tree := mean("release_finalise_rss_mb")) is not None:
            out += (f"; {RELEASE_FINALISE_WORKERS} of them (FINALISE_WORKERS' default) at their peak at once, with the "
                    f"pipeline's process: {gb(tree)}, summed over the processes")
        out += ". "
    return out


def claims_notes(rounds: list[dict], before: str, after: str) -> str:
    """The claims parts of the ETA (pipeline.claims_model): measured, modelled from it, and
    bench 9.0.0's ETAs, every part scaled by entities, on the same runs."""
    arm = next((a for a in (after, before) if all(r["figures"][a].get("pipeline_eta_9") is not None
                                                  for r in rounds)), None)
    if not arm:
        return ""
    mean = lambda k, a=arm: statistics.fmean(v) if None not in (v := [r["figures"][a].get(k) for r in rounds]) else None
    nine = lambda k: " → ".join(duration(v, 0) if (v := mean(k, a)) is not None else "–" for a in (before, after))
    share = mean("pipeline_claims_cpu") / mean("cpu_total")
    placed = (b := mean("pipeline_claims_build_cpu")) is not None and round(b, 1) == 0  # (bench 15.2.0: none built)
    entities = (f"entities ({RELEASE_ENTITIES / e:,.0f}×)" if (e := rounds[0]["workload"].get("entities"))
                else "entities")
    rows = (f"the main set's rows {mean('claims_rows_scale_main'):,.0f}× ({RELEASE_CLAIMS_ROWS / 1e6:,.0f} million, "
            f"modelled), the scholarly set's {mean('claims_rows_scale_scholar'):,.0f}× "
            f"({mean('release_claims_rows_scholar') / 1e6:,.0f} million, modelled from its chunk files' "
            f"{RELEASE_SOURCE_BYTES['scholar'] / 1e9:.1f} GB, measured)")
    key = (f"the claims key's bytes {mean('claims_bytes_scale'):,.0f}× ({mean('release_claims_mb') / 1000:.1f} GB: the "
           f"main set's {RELEASE_CLAIMS_BYTES / 1e9:.1f} GB measured, the scholarly set's modelled)")
    if placed:  # the split builds the claims: in its CPU time, scaled with it by the dump's bytes
        head = (f"The claims, measured ({arm}): built by the split, so in its CPU time and scaled with it by the dump's "
                f"bytes; the pipeline only places each chunk's claims file ({mean('pipeline_claims_cpu'):.2f}s, scaled "
                f"by entities with the rest of the pipeline); and {mean('finalise_claims_share'):.0%} of finalise's "
                f"part per byte, scaled to the release by their own bytes, not by {entities}: {key}. (Their rows at a "
                f"release: {rows}.)")
    else:
        head = (f"The claims, measured ({arm}): {share:.0%} of the pipeline's CPU time "
                f"({mean('pipeline_claims_cpu'):.1f}s: the main set's chunks {mean('pipeline_claims_cpu_main'):.1f}s on "
                f"{mean('pipeline_claims_rows_main'):,.0f} rows, the scholarly set's {mean('pipeline_claims_cpu_scholar'):.1f}s "
                f"on {mean('pipeline_claims_rows_scholar'):,.0f}) and {mean('finalise_claims_share'):.0%} of finalise's part "
                f"per byte. They are scaled to the release by their own rows (the pipeline) and bytes (finalise), not by "
                f"{entities}: {rows}, and {key}.")
    return (head + " Every other part but the pipeline's workers' starts and finalise's other tables (below) is "
            "scaled by entities, as before. With every part scaled by entities (bench 9.0.0's model), the ETA "
            f"would be {nine('eta_9')}: the pipeline {nine('pipeline_eta_9')}, finalise {nine('finalise_eta_9')}. ")


def files_notes(rounds: list[dict], before: str, after: str) -> str:
    """finalise's per-file stages in the ETA (pipeline.files_model): measured, modelled from
    it, and bench 15.0.0's ETAs, those stages scaled by entities, on the same runs."""
    arm = next((a for a in (after, before) if all(r["figures"][a].get("finalise_eta_15") is not None
                                                  for r in rounds)), None)
    if not arm:
        return ""
    mean = lambda k, a=arm: statistics.fmean(v) if None not in (v := [r["figures"][a].get(k) for r in rounds]) else None
    old = lambda k: " → ".join(duration(v, 0) if (v := mean(k, a)) is not None else "–" for a in (before, after))
    return (f"The other tables' per-file stages (compaction's plan and links, the local copy's check, the sort's "
            f"plans), measured ({arm}): {mean('finalise_files'):,.0f} group files of {mean('finalise_keys'):,.0f} keys in "
            f"the first run, three times them in the second, which took {mean('finalise_file_stages_cpu'):.2f} CPU-s "
            f"more in those stages; of it {mean('finalise_file_hashed_cpu'):.2f} is their bytes' sha256 (at the claims' "
            f"rate), kept in the part per byte, and the rest {mean('finalise_file_ms'):.2f} ms a file. They are scaled "
            f"by a release's files, not by entities: {mean('release_files'):,.0f}, modelled (every key in each of "
            f"{' + '.join(str(g) for g in RELEASE_GROUPS.values())} groups, both sets'), in finalise's process a file "
            f"at a time: {duration(mean('release_files_eta'), 0)} of the finalise ETA. With them scaled by entities "
            f"(bench 15.0.0's model), the ETA would be {old('eta_15')}: finalise {old('finalise_eta_15')}. ")


def tables_notes(rounds: list[dict], before: str, after: str) -> str:
    """finalise's other tables in the ETA (pipeline.tables_model): their per-byte stages'
    CPU time, measured, each table's rows at a release, modelled, and bench 15.1.0's ETAs,
    those stages scaled by entities, on the same runs."""
    arm = next((a for a in (after, before) if all(r["figures"][a].get("finalise_eta_15_1") is not None
                                                  for r in rounds)), None)
    if not arm:
        return ""
    mean = lambda k, a=arm: statistics.fmean(v) if None not in (v := [r["figures"][a].get(k) for r in rounds]) else None
    old = lambda k: " → ".join(duration(v, 0) if (v := mean(k, a)) is not None else "–" for a in (before, after))
    each = "; ".join(f"{t} {mean(f'finalise_{t}_bytes_cpu'):.2f} CPU-s, {mean(f'release_{t}_rows') / 1e6:,.0f} million "
                     f"rows at a release ({mean(f'{t}_rows_scale'):,.0f}×)" for t in RELEASE_MAIN_TABLE_ROWS)
    return (f"The other tables' per-byte stages (the decode, the sort and its check, the encode and the written "
            f"file's check), measured ({arm}), on three times the rows less the first, and scaled by each table's own "
            f"rows at a release, not by entities: {each}. A release's rows are modelled: the main set's the 20260507 "
            f"build's ({sum(RELEASE_MAIN_TABLE_ROWS.values()) / RELEASE_MAIN_TABLES_GROWTH / 1e6:,.0f} million, "
            f"measured) grown by {RELEASE_MAIN_TABLES_GROWTH - 1:.0%}, the scholarly set's at the sample's rows a "
            f"scholarly entity. They come to {duration(mean('release_tables_eta'), 0, sign=True)} of the finalise ETA. "
            f"With them scaled by entities (bench 15.1.0's model), the ETA would be {old('eta_15_1')}: the pipeline "
            f"{old('pipeline_eta_15_1')} (the claims files' placing scaled by the claims' rows), finalise "
            f"{old('finalise_eta_15_1')}. ")


def change_of(rounds: list[dict], k: str) -> str:
    """A figure's paired change, or that it is within noise."""
    ratios = [v for v in ratios_of(rounds, k) if v is not None]
    if len(ratios) < len(rounds) or not ratios:
        return "no change to give"
    return "no change beyond noise" if noise([v - 1 for v in ratios]) else percent(ratios)


def measured_line(rounds: list[dict]) -> list[str]:
    """The sample end to end, measured, nothing modelled: the split's, the pipeline's and
    finalise's CPU time and wall time summed, before and after, with their paired change.
    Work moved from one stage to another shows here as a move, whatever each stage's
    scale to a release."""
    before, after = rounds[0]["names"]
    if not all(r["figures"][a].get("measured_cpu") for r in rounds for a in (before, after)):
        return []
    m = lambda a, k: statistics.fmean(r["figures"][a][k] for r in rounds)
    pair = lambda k, d: f"{duration(m(before, k), d)} → {duration(m(after, k), d)}"
    return ["", f"**Measured, the sample end to end** (the split, the pipeline and finalise's first run, summed; "
            f"nothing modelled): CPU {pair('measured_cpu', 1)} ({change_of(rounds, 'measured_cpu')}), wall "
            f"{pair('measured_wall', 2)} ({change_of(rounds, 'measured_wall')}). CPU by stage: the split "
            f"{pair('split_cpu_total', 1)}, the pipeline {pair('cpu_total', 1)}, finalise {pair('finalise_cpu', 1)}."]


def stages_table(rounds: list[dict]) -> list[str]:
    """The release ETA by stage, each stage's measured CPU time on the sample beside it,
    and how the stage is scaled to a release (modelled): what an ETA second costs in CPU
    seconds on the sample, part by part, so work moved between stages (or between a
    stage's parts) shows as a move: its CPU time leaves one row and enters another at
    that row's rate."""
    before, after = rounds[0]["names"]
    need = ("split_eta", "split_cpu_total", "release_pipeline_cpu", "scale", "claims_rows_scale_main",
            "claims_bytes_scale", "release_pipeline_workers", "pipeline_worker_cpu")
    if not all(r["figures"][a].get(k) for r in rounds for a in (before, after) for k in need) or not all(
            r["figures"][a].get("pipeline_claims_cpu") is not None for r in rounds for a in (before, after)):
        return []
    m = lambda k, a=after: statistics.fmean(r["figures"][a][k] for r in rounds)
    pair = lambda k, d=1: f"{duration(m(k, before), d)} → {duration(m(k), d)}"
    cores = rounds[0]["workload"].get("workers") or rounds[0]["machine"]["cores"]
    per_cpu = m("wall") / m("cpu_total")  # (the pipeline ETA is its wall scaled by its CPU time at a release)
    # The claims' build in the pipeline (a part the split left as JSON), scaled by their rows;
    # the claims' placements (a part the split built: bench 15.2.0) are in the rest
    built = lambda a: statistics.fmean(r["figures"][a].get("pipeline_claims_build_cpu", r["figures"][a]["pipeline_claims_cpu"])
                                       for r in rounds)
    rest = lambda a: m("cpu_total", a) - built(a) - m("pipeline_workers", a) * m("pipeline_worker_cpu", a)
    claims_row = round(built(before), 1) > 0 or round(built(after), 1) > 0
    tables = all(r["figures"][after].get("release_tables_eta") is not None for r in rounds)  # (bench 15.2.0)
    x = lambda v: f"{v:,.0f}" if v >= 10 else f"{v:,.1f}"
    by_rows = ", ".join(f"{t} {x(m(f'{t}_rows_scale'))}×" for t in RELEASE_MAIN_TABLE_ROWS) if tables else ""
    gb = RELEASE_DUMP_BYTES / 1e9
    files = all(r["figures"][after].get("release_files") for r in rounds)  # (bench 15.1.0)
    return [
        "", "**The ETA by stage**, modelled from each stage's measured figures (rates: the ETA's seconds a CPU-second "
        f"of the sample's, {after}'s; a CPU-second moved from one part to another changes the ETA by the difference "
        "of their rates):", "",
        "| stage | CPU on the sample, measured | scaled to a release by | rate | ETA |", "|---|---|---|---|---|",
        f"| split | {pair('split_cpu_total')} | its CPU time a byte of the slice's bz2 "
        f"({m('split_source_mb'):.1f} MB), × the dump's {gb:.1f} GB, over {cores} cores | "
        f"{x(m('split_eta') / m('split_cpu_total'))} | {pair('split_eta', 0)} |",
        *([f"| pipeline: the claims' build | {duration(built(before), 1)} → {duration(built(after), 1)} | the "
           f"claims' rows in each set (main {m('claims_rows_scale_main'):,.0f}×, scholarly "
           f"{m('claims_rows_scale_scholar'):,.0f}×), × wall / CPU | {x(m('claims_rows_scale_main') * per_cpu)}, "
           f"{x(m('claims_rows_scale_scholar') * per_cpu)} | |"] if claims_row else []),
        f"| pipeline: its workers' starts | {m('pipeline_workers', before):.0f} → {m('pipeline_workers'):.0f} workers "
        f"at {m('pipeline_worker_cpu', before):.2f} → {m('pipeline_worker_cpu'):.2f} CPU-s | a release's starts, "
        f"counted ({m('release_pipeline_workers'):,.0f}: its {sum(RELEASE_CHUNKS.values()):,} chunks, "
        f"{m('release_chunks_per_worker'):.0f} a worker, {cores} at once), × wall / CPU | "
        f"{x(m('release_pipeline_workers') / m('pipeline_workers') * per_cpu)} | |",
        f"| pipeline: the rest{'' if claims_row else ', the claims files placed among it (the split builds them)'} | "
        f"{duration(rest(before), 1)} → {duration(rest(after), 1)} | entities "
        f"({m('scale'):,.0f}×), × wall / CPU | {x(m('scale') * per_cpu)} | |",
        f"| pipeline | {pair('cpu_total')} | (its CPU time at a release {duration(m('release_pipeline_cpu', before), 0)} → "
        f"{duration(m('release_pipeline_cpu'), 0)}, × its wall / its CPU, {per_cpu:.2f}) | "
        f"{x(m('pipeline_eta') / m('cpu_total'))} | {pair('pipeline_eta', 0)} |",
        f"| finalise | {pair('finalise_cpu')} | its time per key as measured; its CPU time per source byte (3× the "
        f"rows less the first run) × the release's size, the claims' bytes ({x(m('claims_bytes_scale'))}×), "
        + (f"the other tables' rows ({by_rows}) " if tables else "")
        + f"and the rest by entities ({m('scale'):,.0f}×), and the release's claims workers' starts, over {cores} cores"
        + (f"; the other tables' per-file stages × the release's files ({x(m('release_files') / m('finalise_files'))}× "
           f"the first run's), on one core" if files else "")
        + f" | {x(m('claims_bytes_scale') / cores)}, "
        + (", ".join(x(m(f"{t}_rows_scale") / cores) for t in RELEASE_MAIN_TABLE_ROWS) + ", " if tables else "")
        + f"{x(m('scale') / cores)}"
        + (f", {x(m('release_files') / m('finalise_files'))}" if files else "")
        + f" a CPU-s per byte{' or file' if files else ''} | {pair('finalise_eta', 0)} |",
        f"| all | {pair('measured_cpu')} | | | {pair('eta', 0)} |",
    ]


def starts_notes(rounds: list[dict], before: str, after: str) -> str:
    """The pipeline's worker starts in the ETA (pipeline.starts_model), and bench 10.0.0's
    ETA on the same runs (its split on its own slice, in orjson's form)."""
    arm = next((a for a in (after, before) if all(r["figures"][a].get("release_pipeline_workers") is not None
                                                  for r in rounds)), None)
    if not arm:
        return ""
    mean = lambda k, a=arm: statistics.fmean(v) if None not in (v := [r["figures"][a].get(k) for r in rounds]) else None
    ten = lambda k: " → ".join(duration(v, 0) if (v := mean(k, a)) is not None else "–" for a in (before, after))
    counted = mean("pipeline_workers_counted")
    return (f"The pipeline's workers, measured ({arm}): {mean('pipeline_workers'):.0f} (pool.process_chunks starts a "
            f"worker for every {rounds[0]['workload'].get('chunks_per_worker', 8)} chunks; counted, "
            f"{counted:.0f}), each {mean('pipeline_worker_cpu'):.2f} CPU-s outside its chunks "
            f"({mean('pipeline_worker_startup_cpu'):.2f} of them to its first: the interpreter and the imports). "
            "They are taken out of the part scaled by entities, which stood for "
            f"{mean('pipeline_workers') * mean('scale'):,.0f} starts ({duration(mean('pipeline_workers_cpu_scaled_10'), 0)} "
            f"of CPU), and a release's added back: {mean('release_pipeline_workers'):,.0f} "
            f"({RELEASE_CHUNKS['main']:,} and {RELEASE_CHUNKS['scholar']:,} chunks, measured, "
            f"{mean('release_chunks_per_worker'):.0f} a worker: CHUNKS_PER_WORKER), modelled: "
            f"{duration(mean('release_pipeline_workers_cpu'), 0)} of CPU. With bench 10.0.0's model (every start "
            f"scaled by entities, and the split on its slice in orjson's escaping), the ETA would be {ten('eta_10')}: the "
            f"split {ten('split_eta_10')}, the pipeline {ten('pipeline_eta_10')}. ")


def rounds_of(res: dict) -> list[dict]:
    """ab.py's results as one record per round: each commit's figures on its runner."""
    before, after = list(res["arms"])
    out = []
    for rnd in sorted({r["round"] for r in res["runs"] if r["arm"] == after}):
        figures, gz_split = {}, {}
        for arm in (before, after):
            # the commit before, when cached, has its runs from the store, whatever their round
            runs = [r for r in res["runs"] if r["arm"] == arm and (r["round"] == rnd or (arm == before and res.get("before_cached")))]
            figures[arm] = {k: statistics.median(v) if (v := [r[k] for r in runs if r.get(k) is not None]) else None
                            for k in [m[1] for m in METRICS] + FIGURES}
            if timed := [r["phases"] for r in runs if r.get("phases")]:  # where the time goes (phases.py)
                figures[arm]["phases"] = mean_phases(timed)
            if lines := [r["timeline"] for r in runs if r.get("timeline")]:  # where the wall goes (timeline.py)
                figures[arm]["timeline"] = mean_timeline(lines)
            if res["workload"].get("entities"):  # the sample's throughput, derived from its wall time
                figures[arm]["throughput"] = res["workload"]["entities"] / figures[arm]["wall"]
            # The split from gz (pipeline.split_gz): its own parity, and its decoder
            if gz := [r for r in runs if "gz_split_parity_ok" in r]:
                gz_split[arm] = {"parity_ok": all(r["gz_split_parity_ok"] for r in gz), "decoder": gz[0]["gz_decoder"],
                                 "bz2_parity_ok": all(r.get("split_parity_ok", True) for r in gz)}
        out.append({
            "cpu": res["machine"]["cpu"], "machine": res["machine"], "before_cached": res.get("before_cached", False),
            "parity_ok": res["parity_ok"], "parity": res["parity"][after],
            "arms": {a: {"sha": res["arms"][a]["sha"], "packages": res["arms"][a]["packages"]} for a in (before, after)},
            "names": [before, after], "figures": figures, "gz_split": gz_split,
            "workload": res["workload"], "data": res["data_sha256sums"], "bench": res.get("bench", "1.0.0"),
            "round": res.get("round"), "order": res.get("order"), "cpus": [res["machine"]["cpu"]],
        })
    return out


def mean_phases(runs: list[dict]) -> dict:
    """Runs' phases (span -> function -> own CPU seconds) averaged, a function a run lacks
    counted as 0."""
    spans = {s for r in runs for s in r}
    return {s: {f: statistics.fmean(r.get(s, {}).get(f, 0.0) for r in runs)
                for f in {f for r in runs for f in r.get(s, {})}} for s in spans}


# A stage's figures that add up over its stages, and those that are a mean over its wall
TIMELINE_SUMS = ["wall", "cpu", "idle_core_s"]
TIMELINE_MEANS = ["cores", "tree_cores", "iowait_cores", "disk_busy", "start", "end", "jobs", "workers",
                  "peak_running", "mean_running", "longest", "alone_at_end"]


def mean_timeline(runs: list[dict]) -> dict:
    """Runs' timelines (timeline.summarise: span -> its figures and its stages) averaged,
    stage by stage by name (a name twice in a run, summed), each span and each stage with
    the figures of TIMELINE_SUMS and TIMELINE_MEANS that its runs have."""
    out = {}
    for span in {s for r in runs for s in r}:
        got = [r[span] for r in runs if span in r]
        merged = []
        for t in got:
            by = {}
            for st in t["stages"]:
                if st["stage"] in by:
                    for k in TIMELINE_SUMS:
                        by[st["stage"]][k] += st.get(k, 0.0)
                else:
                    by[st["stage"]] = dict(st)
            merged.append(by)
        mean = lambda ds, k: (statistics.fmean(v) if (v := [d[k] for d in ds if d.get(k) is not None]) else None)
        names = sorted({n for m in merged for n in m}, key=lambda n: mean([m[n] for m in merged if n in m], "start"))
        out[span] = {**{k: mean(got, k) for k in TIMELINE_SUMS + TIMELINE_MEANS if any(k in t for t in got)},
                     "stages": [{"stage": n, **{k: mean([m[n] for m in merged if n in m], k)
                                                for k in TIMELINE_SUMS + TIMELINE_MEANS}} for n in names]}
    return out


def paired(halves: list[dict]) -> list[dict]:
    """A round's forward and reverse halves (ab.py --order) as one round: each figure the
    mean of the two, so a change is the mean of the two halves' changes, and what running
    first or second does cancels. A round with one half in is left out until the other
    is; a record from before the pairs (no order) is a round of its own."""
    out, halves_of = [], {}
    for h in halves:
        if h.get("order") is None:
            out.append(h)
        else:
            halves_of.setdefault(h["round"], {})[h["order"]] = h
    for rnd in sorted(halves_of):
        pair = halves_of[rnd]
        if len(pair) < 2:
            continue
        f, r = pair["forward"], pair["reverse"]
        avg = lambda x, y: None if x is None or y is None else (x + y) / 2
        before, after = f["names"]
        ratio = lambda h, k: (h["figures"][after][k] / h["figures"][before][k]
                              if h["figures"][before][k] and h["figures"][after][k] is not None else None)
        out.append({**f, "order": "pair", "cpus": f["cpus"] + r["cpus"], "halves": [f, r],
                    "ratios": {k: avg(ratio(f, k), ratio(r, k)) for k in f["figures"][after]
                               if k not in ("phases", "timeline")},
                    "parity_ok": f["parity_ok"] and r["parity_ok"],
                    "parity": f["parity"] if f["parity_ok"] or not r["parity_ok"] else r["parity"],
                    "before_cached": f["before_cached"] and r["before_cached"],
                    "gz_split": {a: {**g, **{k: g[k] and r.get("gz_split", {}).get(a, g)[k]
                                             for k in ("parity_ok", "bz2_parity_ok")}}
                                 for a, g in f.get("gz_split", {}).items()},
                    "figures": {a: {k: (mean_phases([f["figures"][a][k], r["figures"][a][k]]) if k == "phases"
                                        else mean_timeline([f["figures"][a][k], r["figures"][a][k]]) if k == "timeline"
                                        else avg(f["figures"][a][k], r["figures"][a][k]))
                                    for k in f["figures"][a] if k not in ("phases", "timeline") or k in r["figures"][a]}
                                for a in f["figures"]}})
    return out


# Where the time goes: each span's stages, a stage shown if it is this share of its span's
# instrumented CPU time in either commit (the rest summed as "other")
PHASE_SPANS = [
    ("the split and routing of the dump slice", lambda p: p.get("split")),
    ("the pipeline, every chunk", lambda p: p.get("pipeline")),
    ("finalise, per byte (the run on three times the rows less the first; what the ETA scales)",
     lambda p: None if "finalise" not in p or "finalise_x3" not in p else
     {f: p["finalise_x3"].get(f, 0.0) - p["finalise"].get(f, 0.0)
      for f in set(p["finalise"]) | set(p["finalise_x3"])}),
]
PHASE_SHARE = 0.01


def render_phases(rounds: list[dict]) -> list[str]:
    """The table of where the CPU time goes, by stage, before and after: each stage's own
    CPU time (phases.py) and share, and its paired change where it is beyond noise."""
    before, after = rounds[0]["names"]
    if not all(r["figures"][a].get("phases") for r in rounds for a in (before, after)):
        return []
    out = ["", "**Where the CPU goes**, by stage: each stage's own CPU time in its process (its worker's "
           "native threads included, the stages it calls apart), the mean over the rounds, and its share of "
           "the span's.", "",
           f"| stage | {before} | {after} | change |", "|---|---|---|---|"]
    for title, span in PHASE_SPANS:
        per = [(span(r["figures"][before]["phases"]), span(r["figures"][after]["phases"])) for r in rounds]
        if any(b is None or a is None for b, a in per):
            continue
        stages = {f for b, a in per for f in (*b, *a)}
        mb = {f: statistics.fmean(b.get(f, 0.0) for b, _ in per) for f in stages}
        ma = {f: statistics.fmean(a.get(f, 0.0) for _, a in per) for f in stages}
        tb, ta = sum(mb.values()) or 1.0, sum(ma.values()) or 1.0
        shown = sorted((f for f in stages if max(mb[f] / tb, ma[f] / ta) >= PHASE_SHARE), key=lambda f: -ma[f])
        out.append(f"| *{title}*: {tb:.1f}s → {ta:.1f}s | | | |")
        rest_b, rest_a = tb - sum(mb[f] for f in shown), ta - sum(ma[f] for f in shown)
        for f, vb, va in [*((f, mb[f], ma[f]) for f in shown), ("other", rest_b, rest_a)]:
            if f == "other" and max(vb / tb, va / ta) < 0.001:
                continue
            ratios = [a.get(f, 0.0) / b[f] for b, a in per if f != "other" and b.get(f, 0.0) > 0]
            change = ""
            if len(ratios) == len(per) and not noise([v - 1 for v in ratios]):
                change = f"**{percent(ratios)}**"
            elif vb > 0 and va == 0:
                change = "**gone**"
            elif vb == 0 and va > 0:
                change = "**new**"
            out.append(f"| `{f}` | {vb:.1f}s ({100 * vb / tb:.0f}%) | {va:.1f}s ({100 * va / ta:.0f}%) | {change} |")
    return out


# A stage is shown if it is this share of its span's wall time in either commit
TIMELINE_SHARE = 0.02


def render_timeline(rounds: list[dict]) -> list[str]:
    """Where finalise's wall time goes, beside its CPU time (timeline.py), before and
    after: on three times the rows, and per byte (that run less the first, what the ETA
    scales), each stage's wall time, the cores busy over it, and the core-seconds left
    idle (the cores × its wall less the machine's CPU time over it); then, in a fold, each
    stage's jobs on three times the rows."""
    before, after = rounds[0]["names"]
    if not all(r["figures"][a].get("timeline", {}).get(s) for r in rounds for a in (before, after)
               for s in ("finalise", "finalise_x3")):
        return []
    cores = rounds[0]["machine"]["cores"]
    mean = lambda vs: statistics.fmean(vs) if vs else 0.0

    def stages(a: str, span: str) -> dict[str, dict]:
        return {st["stage"]: st for st in mean_timeline([r["figures"][a]["timeline"] for r in rounds])[span]["stages"]}

    def total(a: str, span: str, k: str) -> float:
        return mean([r["figures"][a]["timeline"][span][k] for r in rounds])

    def per_byte(a: str) -> dict[str, dict]:
        one, three = stages(a, "finalise"), stages(a, "finalise_x3")
        return {n: {k: three.get(n, {}).get(k, 0.0) - one.get(n, {}).get(k, 0.0) for k in TIMELINE_SUMS}
                for n in {*one, *three}}

    cell = lambda st: (f"{duration(st['wall'], 2)}, {st['cpu'] / st['wall']:.1f} busy, {st['idle_core_s']:.0f} idle"
                       if st and st["wall"] > 0 else "–")
    delta = lambda st: (f"{duration(st['wall'], 2, sign=True)}, CPU {st['cpu']:+.0f}s, idle {st['idle_core_s']:+.0f}s"
                        if st else "–")
    out = ["", f"**Where finalise's wall time goes**, beside its CPU (timeline.py): each stage's wall time, the cores "
           f"busy over it (the machine's CPU time over its wall, of {cores}) and the core-seconds left idle "
           f"({cores} × its wall less that CPU time), the mean over the rounds.", "",
           f"| stage | {before} | {after} |", "|---|---|---|"]
    for title, span in (("finalise, 3× the rows", "finalise_x3"),):
        tb, ta = total(before, span, "wall"), total(after, span, "wall")
        out.append(f"| *{title}* | {cell({k: total(before, span, k) for k in TIMELINE_SUMS})} | "
                   f"{cell({k: total(after, span, k) for k in TIMELINE_SUMS})} |")
        sb, sa = stages(before, span), stages(after, span)
        for n in sorted({*sb, *sa}, key=lambda n: (sa.get(n) or sb[n])["start"]):
            if max(sb.get(n, {}).get("wall", 0) / tb, sa.get(n, {}).get("wall", 0) / ta) >= TIMELINE_SHARE:
                out.append(f"| `{n}` | {cell(sb.get(n))} | {cell(sa.get(n))} |")
    pb, pa = per_byte(before), per_byte(after)
    tot = lambda a: {k: total(a, "finalise_x3", k) - total(a, "finalise", k) for k in TIMELINE_SUMS}
    out.append(f"| *per byte: 3× the rows less the first run* | {delta(tot(before))} | {delta(tot(after))} |")
    big = lambda d, n: abs(d.get(n, {}).get("wall", 0.0)) >= 0.25 or abs(d.get(n, {}).get("idle_core_s", 0.0)) >= 1
    for n in sorted((n for n in {*pb, *pa} if big(pb, n) or big(pa, n)), key=lambda n: -pa.get(n, pb.get(n))["wall"]):
        out.append(f"| `{n}` | {delta(pb.get(n))} | {delta(pa.get(n))} |")
    out += ["", f"<details><summary>Each stage's jobs, on three times the rows ({after})</summary>", "",
            "| stage | start → end | jobs (workers) | running at once: mean, peak | longest | last alone | "
            "cores waiting on I/O | disk busy |", "|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=1, unit="": "–" if v is None else f"{v:.{d}f}{unit}"
    for n, st in sorted(stages(after, "finalise_x3").items(), key=lambda kv: kv[1]["start"]):
        if st["wall"] < 0.1:
            continue
        jobs = f"{st['jobs']:.0f} ({st['workers']:.0f})" if st.get("jobs") else "–"
        out.append(f"| `{n}` | {st['start']:.1f}s → {st['end']:.1f}s | {jobs} | {f(st.get('mean_running'))}, "
                   f"{f(st.get('peak_running'), 0)} | {f(st.get('longest'), 1, 's')} | {f(st.get('alone_at_end'), 1, 's')} | "
                   f"{f(st.get('iowait_cores'), 2)} | {f(None if st.get('disk_busy') is None else 100 * st['disk_busy'], 0, '%')} |")
    return out + ["", "</details>"]


def render_gz(rounds: list[dict]) -> list[str]:
    """The split from the gz dump (pipeline.split_gz) against the split from the bz2, for
    each commit whose split reads gz: at the release's size, on the slice, each decoder
    alone, and the gz split's own parity (its chunks the sample's, its manifests the bz2
    split's)."""
    arms = [a for a in rounds[0]["names"] if all(a in r.get("gz_split", {}) for r in rounds)]
    if not arms:
        return []
    mean = lambda a, k: statistics.fmean(r["figures"][a][k] for r in rounds)
    t = lambda v: duration(v, 0)
    gb = lambda n: f"{n / 1e9:.1f} GB"
    decoder = rounds[0]["gz_split"][arms[-1]]["decoder"]
    out = ["", f"**The split from the gz dump** (`WIKIDATA_DUMP_FORMAT=gz`, decoded by {decoder}) against the split "
           "from the bz2, run after the timed run and outside its figures. The ETA is the split's CPU time per byte "
           f"of the slice's dump, scaled to the newest dump on record ({gb(RELEASE_DUMP_BYTES)} bz2, "
           f"{gb(RELEASE_DUMP_GZ_BYTES)} gz) over the cores; decoding alone is the decoder by itself on the slice, "
           "its wall time scaled the same way (lbzip2 decodes on every core, gz decoders on one), a floor under the "
           "split's wall time that the ETA does not take in.", "",
           "| | " + " | ".join(f"{a}, bz2 | {a}, gz" for a in arms) + " |", "|---" * (1 + 2 * len(arms)) + "|"]
    rows = [
        ("ETA: split", lambda a: (t(mean(a, "split_eta")), t(mean(a, "gz_split_eta")))),
        ("decoding alone, wall", lambda a: (t(mean(a, "bz2_decode_wall") / mean(a, "split_source_mb") / 1e6 * RELEASE_DUMP_BYTES),
                                           t(mean(a, "gz_decode_wall") / mean(a, "gz_split_source_mb") / 1e6 * RELEASE_DUMP_GZ_BYTES))),
        ("the slice (MB)", lambda a: (f"{mean(a, 'split_source_mb'):.1f}", f"{mean(a, 'gz_split_source_mb'):.1f}")),
        ("CPU, split, the slice", lambda a: (f"{mean(a, 'split_cpu_total'):.1f}s", f"{mean(a, 'gz_split_cpu_total'):.1f}s")),
        ("CPU, decoding alone, the slice", lambda a: (f"{mean(a, 'bz2_decode_cpu'):.1f}s", f"{mean(a, 'gz_decode_cpu'):.1f}s")),
        ("wall, split, the slice", lambda a: (f"{mean(a, 'split_wall_total'):.2f}s", f"{mean(a, 'gz_split_wall_total'):.2f}s")),
        ("peak RSS, split (MB)", lambda a: (f"{mean(a, 'split_peak_rss_mb'):.0f}", f"{mean(a, 'gz_split_peak_rss_mb'):.0f}")),
        ("parity, split", lambda a: tuple("passed" if all(r["gz_split"][a][k] for r in rounds) else "**FAILED**"
                                          for k in ("bz2_parity_ok", "parity_ok"))),
    ]
    for label, cells in rows:
        out.append(f"| {label} | " + " | ".join(c for a in arms for c in cells(a)) + " |")
    return out + ["", "The split's parity: its chunks hold exactly the sample's rows; from gz, its manifests are also "
                  "those of the split from bz2."]


def lost_jobs(path: str | None = None) -> list[dict]:
    """The round jobs lost (lost.py's file, BENCH_LOST), none if it is not there."""
    path = path or os.environ.get("BENCH_LOST")
    try:
        return json.loads(Path(path).read_text()) if path else []
    except (OSError, ValueError):
        return []


def render_lost(lost: list[dict]) -> list[str]:
    """Each lost job: its round and order, the run it was in, its last stage and memory
    (ab.Heartbeat), and why it ended, as far as its log tells (lost.py)."""
    if not lost:
        return []
    out = ["", f"**Lost:** {len(lost)} job" + ("s" if len(lost) != 1 else "") + ", their rounds left out "
           "(a round needs both its jobs):", ""]
    for e in lost:
        where = f" in `{e['run']}`'s run" if e.get("run") else ""
        last = (f", last seen at {e['stage']} ({duration(e['after_s'], 0)} in) with "
                f"{e['memory_available_gb']:.1f} GB of memory available at its lowest"
                if e.get("stage") is not None else "")
        link = f" ([job]({e['job']}))" if e.get("job") else ""
        out.append(f"- round {e['round']}, {e['order']}{where}: {e.get('reason', 'lost')}{last}{link}")
    return out


def render(rounds: list[dict], total: int, lost: list[dict] | None = None) -> str:
    n = len(rounds)
    before, after = rounds[0]["names"]
    sha = lambda a: (rounds[0]["arms"][a]["sha"] or "?")[:9]
    fig = lambda a, k: [r["figures"][a][k] for r in rounds]
    mean = statistics.fmean
    lost = lost or []
    pending = (f" ({len(lost)} job{'s' if len(lost) != 1 else ''} lost)" if lost
               else f" (pending 🔁 {n}/{total})" if n < total else "")
    pm = lambda v, f: f(mean(v)) + spread(v, f)

    # The score: the whole run, finalise included, when every round has it
    whole = "total_wall" if all(r["figures"][a].get("total_wall") for r in rounds for a in (before, after)) else "wall"
    speeds = [1 / v for v in ratios_of(rounds, whole)]
    rounds_note = f"{n} round" + ("s" if n != 1 else "") + (f" of {total}" if n < total else "")
    w = rounds[0]["workload"]
    tier = f"{w.get('tier', 'quick')} tier, " if w.get("tier") else ""
    size = f" ({w['entities']:,} entities, {w['source_mb']:.0f} MB of source)" if w.get("entities") else ""
    out = [f"## Benchmark: wikidata-pq pipeline, {tier}{w['chunks']} chunks{size}, {rounds_note}{pending}", ""]
    if lost:
        out += render_lost(lost)[1:] + [""]
    out.append(f"**{after} ran at {pm(speeds, lambda v: f'{v:.2f}×')} the speed of {before}**"
               + f", {relative(1 / mean(speeds))} wall time")
    out += measured_line(rounds)
    if all(r["figures"][a].get("eta") for r in rounds for a in (before, after)):
        etas = ratios_of(rounds, "eta")
        change = "no change beyond noise" if noise([v - 1 for v in etas]) else percent(etas)
        part = lambda a, k: duration(mean(fig(a, k)), 0)
        halves = [h for r in rounds for h in r.get("halves", [r])]
        models = sorted({h["cpu"] for h in halves})
        by_cpu = lambda c, a, k: duration(mean([h["figures"][a][k] for h in halves if h["cpu"] == c]), 0)
        fits = [abs(f / m - 1) for a in (before, after) for f, m in zip(fig(a, "finalise_x3_wall_fit"), fig(a, "finalise_x3_wall"))]
        has_split = all(r["figures"][a].get("split_eta") for r in rounds for a in (before, after))
        split_part = (f"the split {part(before, 'split_eta')} → {part(after, 'split_eta')}, " if has_split else "")
        out += ["", f"**Full Wikidata ETA:** {part(before, 'eta')} → {part(after, 'eta')} ({change}): {split_part}the pipeline "
                f"{part(before, 'pipeline_eta')} → {part(after, 'pipeline_eta')}, finalise "
                f"{part(before, 'finalise_eta')} → {part(after, 'finalise_eta')}", *stages_table(rounds), "",
                "<details><summary>The ETA by runner CPU, and how it is made</summary>", "",
                f"{RELEASE}'s {RELEASE_ENTITIES:,} entities, on {rounds[0]['machine']['cores']} cores. The pipeline's "
                "wall time scaled by its CPU time at the release's size over the sample's: its claims stages' by the "
                "claims' rows in each set (where the pipeline builds them), its workers' starts by a release's count, the rest by entities; finalise as it runs at the release's size (a key over "
                f"{RELEASE}'s bucket threshold sorted through buckets), measured on the sample and on three times its "
                "rows: its time per key as measured, its CPU time per source byte scaled to the release's bytes over "
                f"the cores (the claims key's by its bytes, the other tables' by their rows, the rest by entities), the "
                f"other tables' per-file stages by a release's files. The line through the two runs is within {100 * max(fits):.0f}% of the second's wall time. "
                + ("The split (split-dump, then route-release) run on a slice of a dump made of the sample's own "
                   "entities: its CPU time per byte of the slice's bz2, scaled to the release's dump over the cores. "
                   "It leaves out downloading the dump, and every upload. " if has_split else
                   "It leaves out downloading, splitting and routing the dump, and every upload. ")
                + shape(rounds, before, after)
                + release_notes(rounds, before, after)
                + claims_notes(rounds, before, after)
                + files_notes(rounds, before, after)
                + tables_notes(rounds, before, after)
                + starts_notes(rounds, before, after)
                + "The runners differ in "
                "speed by up to about 1.7×; the change is paired, each commit against the other on one runner.", "",
                "| runner CPU | runs | ETA, before | ETA, after | pipeline, before → after | finalise, before → after |",
                "|---|---|---|---|---|---|"]
        for c in models:
            k = sum(h["cpu"] == c for h in halves)
            out.append(f"| {c} | {k} | {by_cpu(c, before, 'eta')} | {by_cpu(c, after, 'eta')} | "
                       f"{by_cpu(c, before, 'pipeline_eta')} → {by_cpu(c, after, 'pipeline_eta')} | "
                       f"{by_cpu(c, before, 'finalise_eta')} → {by_cpu(c, after, 'finalise_eta')} |")
        out += ["", "</details>"]
    elif w.get("entities"):
        # Before bench 4.0.0: the pipeline alone, scaled by entities
        scale = lambda a: mean(fig(a, "wall")) / w["entities"] * RELEASE_ENTITIES
        walls = [a - b for a, b in zip(fig(after, "wall"), fig(before, "wall"))]
        change = "≈, within noise" if noise(walls) else duration(scale(after) - scale(before), 0, sign=True)
        out += ["", f"**Full Wikidata ETA (pipeline only):** {duration(scale(before), 0)} → "
                f"{duration(scale(after), 0)} ({change})."]
    out += render_gz(rounds)
    parity_ok = all(r["parity_ok"] for r in rounds)
    cached = sum(r["before_cached"] for r in rounds)
    out += ["", f"**Parity:** {'passed' if parity_ok else 'FAILED'}", "",
            f"**{before}:** " + ("measured in this run" if not cached else
                                 f"in {cached} of {n} rounds an earlier measurement of the same pipeline on the "
                                 "same CPU model, from the results store; otherwise measured in the run"), ""]
    rows, quiet = [], 0  # (quiet: the figures whose change is noise)
    for label, k, kind, decimals in METRICS:
        vb, va = fig(before, k), fig(after, k)
        if None in vb or None in va:  # a figure these results do not have
            continue
        unit, f = formatter(kind, mean(vb), decimals)
        cb, ca = pm(vb, f), pm(va, f)
        diffs = [a - b for a, b in zip(va, vb)]
        ratios = [v for v in ratios_of(rounds, k) if v is not None]
        if ca == cb or all(d == 0 for d in diffs):  # the same as shown: no change to give
            change = ""
        elif not ratios:
            change = f"{f(mean(diffs), sign=True)}"
        elif noise([v - 1 for v in ratios]):
            quiet += 1  # its 95% interval holds no change: noise, the cell left blank
            change = ""
        else:
            r = mean(ratios)
            change = f"**{f(mean(diffs), sign=True)} ({percent(ratios) if '%' in relative(r) else relative(r)})**"
        rows.append(f"| {label}{unit} | {cb} | {'=' if ca == cb else ca} | {change} |")
    out += [f"| | {before} (`{sha(before)}`) | {after} (`{sha(after)}`) | change |", "|---|---|---|---|", *rows]
    out += render_phases(rounds)
    out += render_timeline(rounds)
    note = ("Each figure is the mean over the rounds ± their standard deviation, which takes in the runners' "
            "differences in speed. A round is a pair of runners, one running the commit before first and one the "
            "commit after. Each change is paired: each runner's after over its before, averaged over the round's two "
            "runners, then over the rounds, given with its 95% confidence interval (±). A change is given only "
            "when its interval holds no zero; otherwise the rounds cannot tell it from noise and the cell is blank.")
    cpus = Counter(c for r in rounds for c in r.get("cpus", [r["cpu"]]))
    m = rounds[0]["machine"]
    pk = lambda a: ", ".join(f"{k} {v}" for k, v in rounds[0]["arms"][a]["packages"].items() if v)
    pa, pb = rounds[0]["arms"][after]["packages"], rounds[0]["arms"][before]["packages"]
    out += [
        "", note if n > 1 else "One round.", "",
        "**Runners:** " + ", ".join(f"{c}" + (f" ×{k}" if k > 1 else "") for c, k in cpus.items())
        + f"; {m['cores']} cores, {m['memory_gb']} GB, Python {m['python']}  ",
        f"**Packages, {before}:** {pk(before)}  ",
        f"**Packages, {after}:** " + ("=" if pa == pb else ", ".join(f"{k} {v}" for k, v in pa.items() if v != pb.get(k))) + "  ",
        f"**Data:** the 11 chunk files, SHA256SUMS hashing to `{rounds[0]['data'][:16]}`  ",
        f"**Harness:** bench {rounds[0].get('bench', VERSION)}", "",
    ]
    shown = next((r for r in rounds if not r["parity_ok"]), rounds[0])
    out.append(f"<details><summary>Parity, {after} against {before}</summary>\n\n```json\n"
               f"{json.dumps(shown['parity'], indent=1)}\n```\n</details>")
    return "\n".join(out) + "\n"


def stored_rounds(path: Path) -> list[dict]:
    """A store's runs/<commit>.json (one results.json, or a list of them) as rounds."""
    data = json.loads(path.read_text())
    return paired([r for res in (data if isinstance(data, list) else [data]) for r in rounds_of(res)])


def render_pending(stored: list[dict], total: int, after_sha: str) -> str:
    """The table before any round is in: the commit before's figures from its own run in
    the results store (it was that run's commit after), the commit after pending."""
    n = len(stored)
    name = stored[0]["names"][1]
    fig = lambda k: [r["figures"][name][k] for r in stored]
    sha = (stored[0]["arms"][name]["sha"] or "?")[:9]
    out = [f"## Benchmark: wikidata-pq pipeline, {stored[0]['workload']['chunks']} chunks, 0 rounds of {total} "
           f"(pending 🔁 0/{total})", "",
           "**after:** pending ⏳", "",
           f"**before:** its figures from its own run in the results store ({n} round{'s' if n != 1 else ''}, "
           + ", ".join(sorted({r['cpu'] for r in stored})) + "), until this run's rounds measure it again", "",
           f"| | before (`{sha}`) | after (`{after_sha[:9]}`) | change |", "|---|---|---|---|"]
    for label, k, kind, decimals in METRICS:
        v = fig(k)
        if None in v:
            continue
        unit, f = formatter(kind, statistics.fmean(v), decimals)
        out.append(f"| {label}{unit} | {f(statistics.fmean(v))}{spread(v, f)} | pending ⏳ | pending |")
    return "\n".join(out) + "\n"


def main() -> None:
    total, dest = int(sys.argv[1]), Path(sys.argv[2])
    rounds = paired([r for p in sys.argv[3:] for r in rounds_of(json.loads(Path(p).read_text()))])
    lost = lost_jobs()
    dest.write_text(render(rounds, total, lost) if rounds else
                    "\n".join(["No round has both its halves.", *render_lost(lost)]) + "\n")


if __name__ == "__main__":
    main()
