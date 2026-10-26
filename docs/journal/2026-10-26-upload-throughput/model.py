"""Upload throughput: what rate our uploads reach for a given link, and the release wall
at those rates (docs/journal/2026-10-26-upload-throughput.md). Nothing here is measured,
and nothing calls the Hub.

Two parts:

1. **The release wall at a rate** (`wall`): bench 17.9.0's model (bench/overlap.py's
   `release_wall`, its link section), with the build branch's pushes and finalise's
   commits each at a rate of their own. With both at one rate it gives the bench's
   `release_wall_link_{r}_modelled` to the second (checked at start-up, on every run).
2. **Our uploads' effective rate** (`push_rate`, `finalise_rate`), from the code's
   concurrency (push/core.py, sort_by_id.commit_table, huggingface_hub 1.32.0, hf_xet
   1.6.0) and a link:
   - `U`, the uplink: what every stream together can send;
   - `r`, one stream's rate (one HTTP/1.1 connection: hf_xet is `http1_only`);
   - `c0`, a commit's fixed cost on the link (the /preupload request, the Xet token, the
     shard after the last xorb, the commit request and the Hub's commit);
   - `q`, the round trip of a file's first global-dedup query (one a small file), at
     most F at once (F, the files Xet ingests at once: 8, or 100 in high-performance mode);
   - `api`, a plain API round trip (create_repo, list_repo_refs, file_exists,
     get_paths_info, list_repo_tree pages).
   A commit of B bytes in f files takes c0 + max(B / min(U, N r), ceil(f / F) q), where N,
   its streams, is the fewer of its xorbs (ceil(B / 64 MiB): small files share xorbs) and
   the controller's permits. The controller ramps to its maximum if a 64 MiB xorb at the
   next concurrency is predicted under its 60 s target (with one stream's rate r, from
   its start s: (64 MiB / r)(s + 1)/s < 60 s), else stays at s; with N r over U, the link
   is full either way.

The inventory (`inventory`): a release's bytes by set and table are the bench record's
(its sample's group and published files, scaled to the release's source bytes, as
`finalise_link` scales them); the files by table are the main set's keys
(docs/dataset_cards_metadata.json: labels 621, descriptions 594, aliases 587, links 955,
claims_labels 618; claims and entities one key), taken for both sets, an upper bound
(about 166k group files at a release, against #167's ~140-160k). A set's group has a file
of every key; finalise's claims and entities keys a part per 500 MiB.

How each path runs today (the journal has the citations):
- **a group's push** (push_group): its tables one after another, an `upload_folder` each
  (huggingface_hub's pipelined upload: commits of 250 files, a batch's commit overlapping
  the next batch's upload), after create_repo, list_repo_refs and file_exists; then
  verify_group's get_paths_info, 100 paths a call. One group at a time a set;
- **finalise** (commit_table, on the committer thread): a create_commit per batch of up
  to 50 additions (COMPACT_COMMIT_MAX_ADDS), one after another; then verify_table's
  listing. One table at a time a run.

Usage: python model.py RECORD.json, RECORD being master's bench record (b110f78, bench
17.9.0: `runs/quick/b110f78160a96ddaa771ae7a0cedaf36c07c5968.json` on the bench-results
branch). Every figure printed is modelled.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

BENCH = Path(__file__).resolve().parents[3] / "bench"
sys.path.insert(0, str(BENCH))
import overlap as ov  # noqa: E402
from release import RELEASE_GROUPS, RELEASE_SOURCE_BYTES  # noqa: E402

H = 3600
MB = 1e6
XORB = 64 * 1024**2  # a xorb's most bytes (xet_core_structures xorb_object/constants.rs)
TARGET_RTT = 60.0  # the controller's target for a reference xorb (client.rs ac_target_rtt)
PART = 500 * 1024**2  # finalise's part files (config.COMPACT_FILE_BYTES)
PIPELINE_BATCH = 250  # upload_folder's first commit size (_upload_pipeline.py COMMIT_SIZE_SCALE[6])
VERIFY_PATHS = 100  # verify_group's get_paths_info batch
KEYS = {"labels": 621, "descriptions": 594, "aliases": 587, "links": 955, "claims_labels": 618,
        "claims": 1, "entities": 1}


@dataclass(frozen=True)
class Link:
    U: float  # the uplink, bytes a second
    r: float  # one stream's rate, bytes a second
    c0: float = 5.0  # a commit's fixed cost, s
    q: float = 0.2  # a small file's dedup-query round trip, s
    api: float = 0.3  # an API round trip, s


@dataclass(frozen=True)
class Settings:
    name: str
    F: int = 8  # files ingested at once (HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION)
    s: int = 2  # the controller's start (HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY)
    m: int = 64  # its maximum (HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY)
    adds: int = 50  # finalise's additions a commit (COMPACT_COMMIT_MAX_ADDS)
    tables_at_once: bool = False  # a group's tables pushed at once (a thread each)
    fin_k: int = 1  # finalise's commits in flight (tables of different repos)
    turn: bool = True  # the turn lock: after the split, one set pushes at a time


DEFAULT = Settings("today (defaults)")
HP = Settings("HF_XET_HIGH_PERFORMANCE=1", F=100, s=16, m=124)


def streams(link: Link, st: Settings) -> int:
    """The controller's settled permits while one stream's rate is below the link's."""
    return st.m if (XORB / link.r) * (st.s + 1) / st.s < TARGET_RTT else st.s


def transfer(link: Link, st: Settings, nbytes: float, files: int) -> float:
    """A commit's bytes on the link, or its files' dedup queries, whichever is longer."""
    n = min(streams(link, st), max(1, math.ceil(nbytes / XORB)))
    return max(nbytes / min(link.U, n * link.r), math.ceil(files / st.F) * link.q)


def inventory(run: dict) -> tuple[dict, dict]:
    """A release's bytes by set and table: the build branch's and the published, as
    finalise_link scales them."""
    lb = run["finalise_link_bytes"]
    scale = {k: RELEASE_SOURCE_BYTES[k] / n for k, n in lb["sample"].items() if n}
    group = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["group"].items()}
    pub = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["published"].items()}
    return group, pub


def push_time(group: dict, link: Link, st: Settings) -> tuple[float, float]:
    """The build branch's pushes: (bytes, link seconds), every group of both sets."""
    nbytes = secs = 0.0
    for k, tables in group.items():
        n = RELEASE_GROUPS[k]
        times, gbytes, gfiles = [], 0.0, 0
        for t, b in tables.items():
            b /= n
            f = KEYS[t]
            batches = math.ceil(f / PIPELINE_BATCH)
            setup = 3 * link.api  # create_repo, list_repo_refs, file_exists
            # (batches pipelined: their transfers in turn, the last batch's commit after)
            times.append(setup + batches * transfer(link, st, b / batches, math.ceil(f / batches)) + link.c0)
            gbytes, gfiles = gbytes + b, gfiles + f
        verify = math.ceil(gfiles / VERIFY_PATHS) * link.api
        if st.tables_at_once:  # (one process: its F ingestion slots shared by the tables)
            g = max(gbytes / link.U, max(times), math.ceil(gfiles / st.F) * link.q + link.c0) + verify
        else:
            g = sum(times) + verify
        nbytes, secs = nbytes + n * gbytes, secs + n * g
    return nbytes, secs


def finalise_time(pub: dict, link: Link, st: Settings) -> tuple[float, float]:
    """finalise's commits and verifies: (bytes, link seconds), both sets."""
    nbytes = secs = 0.0
    for k, tables in pub.items():
        for t, b in tables.items():
            adds = math.ceil(b / PART) if t in ("claims", "entities") else KEYS[t]
            commits = math.ceil(adds / st.adds)
            listing = 2 * math.ceil(adds / 1000) * link.api  # commit_table's and verify_table's
            t_ = commits * (link.c0 + transfer(link, st, b / commits, math.ceil(adds / commits))) + listing
            nbytes, secs = nbytes + b, secs + t_
    if st.fin_k > 1:
        secs = max(nbytes / link.U, secs / st.fin_k)
    return nbytes, secs


def push_rate(run, link, st):
    """The build branch's effective rate at a release. Both sets' runs (two processes)
    push at once while both have groups waiting, as during the split: each one's fixed
    costs filled by the other's bytes, b / max(b / U, s / 2). After the split, under the
    turn lock (push/turn.py), one set at a time: b / s. The share pushed after the split
    is the bench's timeline's (`_uploads_end` at that rate), iterated to a fixed point."""
    b, s = push_time(inventory(run)[0], link, st)
    one, two = b / s, b / max(b / link.U, s / 2)
    if not st.turn:
        return two
    ph, rate = ov._phase(run, True), two
    seconds = run["upload_eta_modelled"] * 7e6  # (link seconds at 1 B/s)
    for _ in range(20):
        end = ov._uploads_end(ph, seconds / rate)
        after = min(1.0, max(0.0, (end - ph.split_end) / (seconds / rate)))
        rate = 1 / ((1 - after) / two + after / one)
    return rate


def finalise_rate(run, link, st):
    b, s = finalise_time(inventory(run)[1], link, st)
    return b / s


def record_rate(link: Link) -> float:
    """What 2026-10-05's observation would read on this link: the main set's claims (86
    parts of 500 MiB) committed by commit_table at the defaults, 50 parts and then 36."""
    secs = sum(link.c0 + transfer(link, DEFAULT, n * PART, n) for n in (50, 36))
    return 86 * PART / secs


def wall(run: dict, push: float, fin: float, turn: bool = True) -> float:
    """bench 17.9.0's release wall (overlap.release_wall's link section), the build
    branch's pushes at `push` and finalise's transfers at `fin`, bytes a second. Without
    the turn (`turn` False) the main set's pushes no longer end first: its finalise starts
    once both sets' pushes end (L2's early start lost, its commits then alone on the link)."""
    ph = ov._phase(run, True)
    seconds = run["upload_eta_modelled"] * 7e6 / push  # (its bytes at the bench's 7 MB/s)
    uploaded = ov._uploads_end(ph, seconds)
    if turn and (run.get("finalise_link_bytes") or {}).get("early"):
        start = max(ph.split_end, ph.compute_end, ov._main_uploads_end(ph, seconds))
        f = ov.finalise_link(run, fin, busy=max(0.0, uploaded - start))
    else:
        start = max(ph.split_end, ph.compute_end, uploaded)
        f = ov.finalise_link(run, fin)
    return start + f["wall"]


def hm(s: float) -> str:
    s = round(s)
    return f"{s // H}h{s % H // 60:02d}m"


def med(runs, fn) -> float:
    return statistics.median(fn(r) for r in runs)


def main(path: str) -> None:
    record = json.loads(Path(path).read_text())
    runs = [r for e in record for r in e["runs"] if r.get("arm") == "after"]
    for r in runs:  # (the bench's own walls, to the second)
        for rate in ov.RELEASE_LINK_RATES:
            want = r[f"release_wall_link_{rate / 1e6:.0f}_modelled"]
            assert round(wall(r, rate, rate)) == want, (rate, want)
    print(f"{len(runs)} after-arm runs; the bench's walls reproduced at {[int(x / 1e6) for x in ov.RELEASE_LINK_RATES]} MB/s\n")

    group, pub = inventory(runs[0])
    gfiles = sum(RELEASE_GROUPS[k] * sum(KEYS[t] for t in v) for k, v in group.items())
    pcommits = sum(RELEASE_GROUPS[k] * sum(math.ceil(KEYS[t] / PIPELINE_BATCH) for t in v) for k, v in group.items())
    fcommits = sum(math.ceil((math.ceil(b / PART) if t in ("claims", "entities") else KEYS[t]) / 50)
                   for v in pub.values() for t, b in v.items())
    print("## A release's uploads (modelled inventory)\n")
    print(f"- build branch: {sum(map(sum, (v.values() for v in group.values()))) / 1e9:.1f} GB, "
          f"{sum(RELEASE_GROUPS.values())} groups, ~{gfiles / 1e3:.0f}k files, ~{pcommits} commits")
    print(f"- finalise: {sum(map(sum, (v.values() for v in pub.values()))) / 1e9:.1f} GB, ~{fcommits} commits (50 additions each)\n")

    print("## The wall if one stream is 7 MB/s and N streams add up to the uplink U (bench model, no fixed cost)\n")
    Us = (30e6, 60e6, 120e6)
    print("| N streams | " + " | ".join(f"U = {u / 1e6:.0f} MB/s" for u in Us) + " |")
    print("|---|" + "---|" * len(Us))
    for n in (1, 2, 4, 8, 16, 32):
        cells = []
        for u in Us:
            rate = min(u, n * 7e6)
            cells.append(f"{rate / 1e6:.0f} MB/s: {hm(med(runs, lambda r: wall(r, rate, rate)))}")
        print(f"| {n} | " + " | ".join(cells) + " |")

    print("\n## Our paths' effective rates and the wall, by link and settings\n")
    links = {
        "uplink 7 MB/s, one stream fills it": Link(U=7e6, r=7e6),
        "uplink 7 MB/s, a stream 2 MB/s": Link(U=7e6, r=2e6),
        "a stream 7 MB/s, uplink 30": Link(U=30e6, r=7e6),
        "a stream 7 MB/s, uplink 60": Link(U=60e6, r=7e6),
        "a stream 7 MB/s, uplink 120": Link(U=120e6, r=7e6),
        "a stream 1.5 MB/s, uplink 60": Link(U=60e6, r=1.5e6),
    }
    rec_push = replace(HP, name="", tables_at_once=True)
    rec_fin = replace(DEFAULT, name="", adds=250, fin_k=2)
    settings = [(st.name, st, st) for st in (
        DEFAULT,
        HP,
        replace(DEFAULT, name="no turn lock (both sets at once after the split; L2's early finalise lost)", turn=False),
        replace(DEFAULT, name="a group's tables at once", tables_at_once=True),
        replace(DEFAULT, name="finalise: 250 additions a commit", adds=250),
        replace(DEFAULT, name="finalise: 2 commits in flight", fin_k=2),
    )] + [
        ("HP for the pipeline's runs only", HP, DEFAULT),
        ("recommended: HP for the runs and their tables at once; finalise 250 additions, 2 in flight",
         rec_push, rec_fin),
    ]
    print("| link | 2026-10-05 would read | settings | push MB/s | finalise MB/s | wall | bench at U |")
    print("|---|---|---|---|---|---|---|")
    for lname, link in links.items():
        bench = hm(med(runs, lambda r: wall(r, link.U, link.U)))
        seen = f"{record_rate(link) / MB:.1f} MB/s"
        for name, ps, fs in settings:
            p, f = push_rate(runs[0], link, ps), finalise_rate(runs[0], link, fs)
            w = hm(med(runs, lambda r: wall(r, push_rate(r, link, ps), finalise_rate(r, link, fs), ps.turn)))
            print(f"| {lname} | {seen} | {name} | {p / MB:.1f} | {f / MB:.1f} | {w} | {bench} |")

    print("\n## The fixed costs' share at a 7 MB/s uplink (one stream fills it), defaults\n")
    print("| c0 (s) | q (s) | api (s) | push MB/s | finalise MB/s | wall | one set's push, link idle | finalise, link idle |")
    print("|---|---|---|---|---|---|---|---|")
    for c0 in (2.0, 5.0, 15.0):
        for q in (0.1, 0.3):
            link = Link(U=7e6, r=7e6, c0=c0, q=q, api=q * 1.5)
            pb, ps = push_time(group, link, DEFAULT)
            fb, fs = finalise_time(pub, link, DEFAULT)
            p = push_rate(runs[0], link, DEFAULT)
            w = hm(med(runs, lambda r: wall(r, push_rate(r, link, DEFAULT), finalise_rate(r, link, DEFAULT))))
            print(f"| {c0:g} | {q:g} | {q * 1.5:g} | {p / MB:.2f} | {fb / fs / MB:.2f} | {w} | "
                  f"{hm(ps - pb / link.U)} | {hm(fs - fb / link.U)} |")


if __name__ == "__main__":
    main(sys.argv[1])
