"""An official Wikidata JSON dump (a release) as chunk files for the pipeline.

`download-dump` fetches `wikidata-{release}-all.json.bz2` from dumps.wikimedia.org (or a
mirror) into DUMP_DIR, resumably, over up to 3 connections (Wikimedia's cap per IP; see
MAX_CONNECTIONS), and checks it against the release's md5sums. With
WIKIDATA_DUMP_FORMAT=gz it fetches `wikidata-{release}-all.json.gz` instead (see DECODERS).

`split-dump` decompresses it (bz2 with lbzip2, on all cores; gz with igzip, pigz or gzip)
and cuts it into chunks of CHUNK_ENTITIES entities, each reshaped into the row the
pipeline's process step reads (`id, labels, descriptions, aliases, sitelinks, claims`, the
last five JSON strings, as in philippesaade/wikidata) and routed as it is written: its
scholarly works (see scholarly.py) to `routed/chunk_{N}.parquet` in the scholarly set's
data directory, the rest to the same in ROOT_DATA_DIR, with a line per chunk in the
manifest (MANIFEST) giving both parts. Each part's claims are written as its claims
partition file (`routed/claims/chunk_{N}.parquet`, see claims_build), before the part,
which then has no claims column; a part whose claims do not all fit claims.avsc keeps
them as JSON, for the pipeline to decode as before. Each part's entities are written as its
entities file (`routed/entities/chunk_{N}.parquet`, see entities_build) too, and the part
has no `entity` column. A chunk's manifest line is written
once all its files are in place, and a rerun skips the chunks already in the manifest.
Once every chunk is written, the split ends by routing the release (see `route`). It can
run alongside download-dump (`just download-split`): it follows the download as it is
written, and ends once it is complete and checked (see `_feed`).

The reshaping (see `entity_row`) keeps every field of the dump; only these change shape:

- labels, descriptions: {lang: {language, value}} -> {lang: value}, and aliases
  {lang: [{language, value}]} -> {lang: [value]}, the `language` repeating the key (an
  entry where it does not is counted in the manifest's `mismatched_terms`)
- a snak's datavalue {value, type} -> `datavalue` (the value) and `datavalue_type`
- the entity's own fields (type, ns, title, pageid, lastrevid, modified) -> the `entity`
  column, as JSON

Everything else (snaktype, hashes, statement ids and types, qualifiers-order, references
with their hash and snaks-order, sitelink badges, item values' entity-type and numeric-id)
is kept as it is; the philippesaade copy had dropped all of it.

`route-release` numbers each set's parts from 0 and writes each set's manifest (see
`route`); the split runs it itself, so it is left to do only after an interruption (or for
a split made before the split routed its chunks, whose chunks it routes). The pipeline
reads a release's chunks only once they are routed.
"""

import hashlib
import http.client
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path

import orjson
import polars as pl
import pyarrow as pa
from tqdm import tqdm
from wikidata_reshape import entity_rows

from . import claims_build, entities_build
from .partitioning import fused
from .config import DUMP_DIR, DUMP_FORMAT, OTHER_WORK_DIR, RELEASE, ROOT_DATA_DIR, SCHOLAR
from .scholarly import is_scholarly

DUMPS_URL = "https://dumps.wikimedia.org/wikidatawiki/entities"
# Wikimedia refuses (403) urllib's default User-Agent; its policy asks for one naming the
# client and how to reach its maintainer
USER_AGENT = "wikidata-pq/0.1 (https://github.com/lmmx/wikidata-pq)"


def _open(url: str, *, method: str = "GET", headers: dict | None = None, timeout: int = 60):
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
    return urllib.request.urlopen(request, timeout=timeout)
# Entities per chunk file (the philippesaade files have 10,000 rows each)
CHUNK_ENTITIES = 10_000
# The most decompressed bytes handed to the workers and not yet written, bounding memory
# (early entities are large: about 50 kB each in 20260928's first chunk)
MAX_PENDING_BYTES = 4 * 1024**3
# The chunk files' zstd level, for the split's chunks and routing's parts: level 3 takes a
# quarter of level 9's CPU time at 1.1x its size (on the dump's first chunk, 498 MB of JSON:
# 1.4 s and 55 MB against 5.5 s and 50 MB; level 1: 0.9 s, 58 MB). Each chunk is written
# twice (split, route), so level 9 cost about 8 CPU s more per 500 MB of JSON: about 5
# CPU hours over a dump of ~1 TB of JSON
ZSTD_LEVEL = 3
MANIFEST = ROOT_DATA_DIR / "manifest.jsonl"
SPLIT_DONE = ROOT_DATA_DIR / "split.done"
# Routing: the split's own manifest is kept as SPLIT_MANIFEST once MANIFEST lists the
# routed chunks; ROUTE_LOG has a line per routed chunk; ROUTE_DONE ends it, in both sets
SPLIT_MANIFEST = ROOT_DATA_DIR / "manifest.split.jsonl"
ROUTE_LOG = ROOT_DATA_DIR / "route.jsonl"
ROUTE_DONE = ROOT_DATA_DIR / "route.done"
ROUTED = "routed"
# Each part's claims partition file, written by the split (claims_build), in `claims/`
# beside the part: `routed/claims/chunk_{N}.parquet` until routing moves it with its part
CLAIMS = claims_build.CLAIMS_DIR
# Each part's entities file, written by the split (entities_build), likewise in `entities/`
ENTITIES = entities_build.ENTITIES_DIR
# The tables the split writes beside each part as a file each (the term tables, with their
# indexes, are partitioning.fused's)
BUILT = (CLAIMS, ENTITIES)
# Chunks routed at once (each holds a whole chunk in memory, up to about 1 GB of JSON)
ROUTE_WORKERS = 8


# Each format's decoders, the first installed one used. lbzip2 decodes bz2 on every core,
# and that is most of the split's CPU time: 160-210 CPU s per GB of bz2 on a 4-core CI runner.
# gz decodes on one core, in 5-8 CPU s per GB with igzip (ISA-L, `apt install isal`),
# 15-21 with pigz and 25-30 with gzip; the gz dump is 1.5 times the bz2's size.
DECODERS = {
    "bz2": [["lbzip2", "-dc"]],
    "gz": [["igzip", "-dc"], ["pigz", "-dc"], ["gzip", "-dc"]],
}


def dump_name(release: str, fmt: str = DUMP_FORMAT) -> str:
    return f"wikidata-{release}-all.json.{fmt}"


def dump_path(release: str, fmt: str = DUMP_FORMAT) -> Path:
    return DUMP_DIR / dump_name(release, fmt)


def decoder(fmt: str) -> list[str]:
    """The command decompressing a dump in `fmt` from stdin to stdout (see DECODERS)."""
    for tool, *args in DECODERS[fmt]:
        if path := shutil.which(tool):
            return [path, *args]
    if fmt == "bz2":
        raise SystemExit("lbzip2 decompresses the dump on all cores: apt install lbzip2")
    raise SystemExit("igzip, pigz or gzip decompresses the gz dump: apt install isal")


def latest_release(base: str = DUMPS_URL) -> str:
    """The newest release directory holding a full JSON dump (in DUMP_FORMAT)."""
    with _open(f"{base}/") as r:
        dates = sorted(set(re.findall(r'href="(\d{8})/"', r.read().decode())), reverse=True)
    for date in dates:
        with _open(f"{base}/{date}/") as r:
            if dump_name(date) in r.read().decode():
                return date
    raise SystemExit(f"No {dump_name('*')} under {base}/")


def _require_release() -> str:
    if not RELEASE:
        raise SystemExit("Set WIKIDATA_RELEASE to a dump date, e.g. WIKIDATA_RELEASE=20260928")
    if SCHOLAR:
        raise SystemExit("The dump is downloaded, split and routed without WIKIDATA_SCHOLAR")
    return RELEASE


def download(base: str | None = None) -> Path:
    """Download the release's dump (in DUMP_FORMAT) to DUMP_DIR (resuming a partial
    download), checked against the release's md5sums, over WIKIDATA_DUMP_CONNECTIONS
    connections (MAX_CONNECTIONS, 3, by default; see _fetch). `base` is a mirror's
    .../wikidatawiki/entities URL (WIKIDATA_DUMPS_URL; dumps.wikimedia.org by default); the
    md5sums always come from dumps.wikimedia.org."""
    release = _require_release()
    connections = _connections()
    base = (base or os.environ.get("WIKIDATA_DUMPS_URL") or DUMPS_URL).rstrip("/")
    name = dump_name(release)
    dst = dump_path(release)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _open(f"{DUMPS_URL}/{release}/wikidata-{release}-md5sums.txt") as r:
        sums = dict(line.split()[::-1] for line in r.read().decode().splitlines() if line.strip())
    if name not in sums:
        raise SystemExit(f"{name} is not in release {release}'s md5sums: {sorted(sums)}")
    md5 = sums[name]
    if dst.exists():
        print(f"{dst} exists; checking it", flush=True)
        if _md5(dst) == md5:
            return dst
        raise SystemExit(f"{dst} does not match its md5 {md5}; delete it to download again")
    part = dst.with_suffix(dst.suffix + ".part")
    url = f"{base}/{release}/{name}"
    with _open(url, method="HEAD") as r:
        total = int(r.headers["Content-Length"])
    done = part.stat().st_size if part.exists() else 0
    print(f"{name} from {urllib.parse.urlsplit(url).hostname}, {connections} connection(s)",
          flush=True)
    # (every bar is closed on every path: see split)
    with tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=name) as bar:
        if done < total:
            _fetch(url, part, done, total, connections, bar)
    if _md5(part) != md5:
        # Moved aside, so a split following the download stops (see _feed)
        part.replace(bad := part.with_suffix(".bad"))
        raise SystemExit(f"{part} does not match its md5 {md5} (moved to {bad}): download again")
    part.replace(dst)
    print(f"Wrote {dst}", flush=True)
    return dst


# dumps.wikimedia.org: "Downloads are also rate limited and capped at 3 connections per-IP"
# (https://dumps.wikimedia.org/, and "Clients that ... try to evade these limits may be
# blocked"). Its rate limit is per connection, about 5 MB/s each, so the dump is fetched over
# WIKIDATA_DUMP_CONNECTIONS connections at once: 3 by default, and never more. The cap counts
# every connection from the IP, so a second download from the same IP takes from it
MAX_CONNECTIONS = 3
# The dump is fetched in pieces of PIECE bytes, each one range request on one of the (kept
# alive) connections, and the pieces are appended to the .part in order: the .part is always
# the dump's first bytes, so a split follows it (see _feed) and a rerun resumes from its size,
# as with one connection. Pieces up to WINDOW per connection past the .part's end are fetched
# ahead and held in memory until their turn: at most 192 MB with 3 connections
PIECE = 32 << 20
WINDOW = 2
# Seconds to wait before retrying a failed request (doubling to at most RETRY_MAX), or after a
# 429 or 503 without a Retry-After
RETRY = 2
RETRY_MAX = 60


def _connections() -> int:
    n = int(os.environ.get("WIKIDATA_DUMP_CONNECTIONS") or MAX_CONNECTIONS)
    if not 1 <= n <= MAX_CONNECTIONS:
        raise SystemExit(
            f"WIKIDATA_DUMP_CONNECTIONS is 1 to {MAX_CONNECTIONS}, not {n}: dumps.wikimedia.org"
            f" caps downloads at {MAX_CONNECTIONS} connections per IP"
        )
    return n


class _Wait(Exception):
    """The server asked for a pause (429 or 503) of `seconds`."""

    def __init__(self, seconds: float):
        self.seconds = seconds


class _Ranges:
    """One kept-alive connection fetching byte ranges of `url` (through the HTTPS proxy, if
    the environment sets one for its host), following redirects."""

    def __init__(self, url: str):
        self.url = url
        self.conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        u = urllib.parse.urlsplit(self.url)
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        proxy = urllib.request.getproxies().get(u.scheme)
        if proxy and not urllib.request.proxy_bypass(u.hostname):
            p = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
            conn = cls(p.hostname, p.port, timeout=120)  # TLS to the host, tunnelled
            conn.set_tunnel(u.hostname, u.port)
            return conn
        return cls(u.hostname, u.port, timeout=120)

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def get(self, start: int, end: int) -> bytes:
        """Bytes `start` to `end` (exclusive), retrying until they come."""
        wait = RETRY
        while True:
            try:
                return self._get(start, end)
            except _Wait as e:
                self.close()
                print(f"\nAsked to wait {e.seconds:.0f}s at {start:,} bytes", file=sys.stderr,
                      flush=True)
                time.sleep(e.seconds)
            except (OSError, http.client.HTTPException) as e:
                self.close()
                print(f"\n{e!r}; retrying {start:,} bytes in {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                wait = min(2 * wait, RETRY_MAX)

    def _get(self, start: int, end: int) -> bytes:
        for _ in range(10):
            if self.conn is None:
                self.conn = self._connect()
            u = urllib.parse.urlsplit(self.url)
            headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end - 1}"}
            self.conn.request("GET", u.path + (f"?{u.query}" if u.query else ""), headers=headers)
            r = self.conn.getresponse()
            if r.status in (301, 302, 303, 307, 308) and (to := r.getheader("Location")):
                r.read()
                self.close()
                self.url = urllib.parse.urljoin(self.url, to)
                continue
            if r.status in (429, 503):
                r.read()
                after = r.getheader("Retry-After") or ""
                raise _Wait(float(after) if after.isdigit() else RETRY_MAX)
            if r.status != 206:
                raise SystemExit(f"{self.url} ignored the range request (status {r.status})")
            data = r.read()
            if len(data) != end - start:
                raise http.client.IncompleteRead(data, end - start - len(data))
            return data
        raise SystemExit(f"{self.url}: too many redirects")


def _fetch(url: str, part: Path, done: int, total: int, connections: int, bar) -> None:
    """Append bytes `done` to `total` of `url` to `part` (which holds the first `done`), in
    order, fetched in PIECE-sized ranges over `connections` connections at once."""
    cond = threading.Condition()
    pieces: dict[int, bytes] = {}
    at = {"next": done, "written": done}
    failed: list[BaseException] = []
    ahead = WINDOW * connections * PIECE

    def fetcher() -> None:
        ranges = _Ranges(url)
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: failed or at["next"] >= total
                                  or at["next"] < at["written"] + ahead)
                    if failed or at["next"] >= total:
                        return
                    start, end = at["next"], min(at["next"] + PIECE, total)
                    at["next"] = end
                data = ranges.get(start, end)
                with cond:
                    pieces[start] = data
                    cond.notify_all()
        except BaseException as e:
            with cond:
                failed.append(e)
                cond.notify_all()
        finally:
            ranges.close()

    threads = [threading.Thread(target=fetcher, daemon=True) for _ in range(connections)]
    for t in threads:
        t.start()
    with part.open("ab") as out:
        while at["written"] < total:
            with cond:
                cond.wait_for(lambda: failed or at["written"] in pieces)
                if failed:
                    raise failed[0]
                data = pieces.pop(at["written"])
            out.write(data)
            out.flush()
            with cond:
                at["written"] += len(data)
                cond.notify_all()
            bar.update(len(data))
    for t in threads:
        t.join()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f, tqdm(
        total=path.stat().st_size, unit="B", unit_scale=True, desc=f"md5 {path.name}"
    ) as bar:
        while block := f.read(16 << 20):
            h.update(block)
            bar.update(len(block))
    return h.hexdigest()


def _map(value: object) -> dict:
    """A map, which Wikibase writes as [] when empty"""
    return value if isinstance(value, dict) else {}


def _snak(snak: dict) -> dict:
    """A snak with its datavalue's value in place of the datavalue, and the datavalue's
    type as `datavalue_type`; every other field as it is."""
    out = {k: v for k, v in snak.items() if k != "datavalue"}
    if "datavalue" in snak:
        out["datavalue"] = snak["datavalue"].get("value")
        out["datavalue_type"] = snak["datavalue"].get("type")
    return out


def _snaks(by_property: object) -> dict:
    return {p: [_snak(s) for s in snaks] for p, snaks in _map(by_property).items()}


def _statement(statement: dict) -> dict:
    out = dict(statement)
    out["mainsnak"] = _snak(statement["mainsnak"])
    if "qualifiers" in statement:
        out["qualifiers"] = _snaks(statement["qualifiers"])
    if "references" in statement:
        out["references"] = [
            {**ref, "snaks": _snaks(ref.get("snaks"))} for ref in statement["references"]
        ]
    return out


def _json(value: object) -> str:
    return orjson.dumps(value).decode()


def _terms(by_language: object, mismatched: list[int]) -> dict:
    """{lang: {language, value}} as {lang: value}; `language` repeats the key, and an
    entry where it does not (or has other fields) is counted in `mismatched`."""
    out = {}
    for lang, term in _map(by_language).items():
        if term.get("language") != lang or len(term) != 2:
            mismatched[0] += 1
        out[lang] = term["value"]
    return out


def _alias_terms(by_language: object, mismatched: list[int]) -> dict:
    out = {}
    for lang, terms in _map(by_language).items():
        for term in terms:
            if term.get("language") != lang or len(term) != 2:
                mismatched[0] += 1
        out[lang] = [term["value"] for term in terms]
    return out


# The entity's own fields, beside its terms, sitelinks and claims (type, ns, title,
# pageid, lastrevid, modified in the 2026 dumps), kept as JSON in the `entity` column
ENTITY_PARTS = {"id", "labels", "descriptions", "aliases", "sitelinks", "claims"}


def entity_row(entity: dict, mismatched: list[int], fields: dict) -> tuple[str, ...]:
    """One dump entity as the pipeline's source row (see the module docstring). The names
    of the entity's own fields and of its sitelinks' fields are added to `fields`, so the
    manifest shows any the pipeline does not expect."""
    fields["entity"].update(k for k in entity if k not in ENTITY_PARTS)
    for link in _map(entity.get("sitelinks")).values():
        fields["sitelink"].update(link)
    return (
        entity["id"],
        _json(_terms(entity.get("labels"), mismatched)),
        _json(_terms(entity.get("descriptions"), mismatched)),
        _json(_alias_terms(entity.get("aliases"), mismatched)),
        _json(_map(entity.get("sitelinks"))),
        _json({p: [_statement(s) for s in ss] for p, ss in _map(entity.get("claims")).items()}),
        _json({k: v for k, v in entity.items() if k not in ENTITY_PARTS}),
    )


COLUMNS = ["id", "labels", "descriptions", "aliases", "sitelinks", "claims", "entity"]


def reshape(lines: list[bytes]) -> tuple[pl.DataFrame, int, dict]:
    frame, mismatched, fields, _ = _reshape(lines, build=False)
    return frame, mismatched, fields


def _reshape(lines: list[bytes], build: bool = True) -> tuple[pl.DataFrame, int, dict, dict[str, pa.Array]]:
    """A chunk's dump lines as its rows (`entity_row`'s, column by column), the count of
    mismatched terms, and the fields seen (see `entity_row`). The reshape is compiled
    (reshape/src/lib.rs: each line scanned once, its row copied from it in runs and
    rewritten only where orjson writes it differently), its columns' buffers handed over
    without a copy; a line it does not expect (a field of the wrong type, say) is
    reshaped by entity_row.

    The claims are built from the lines in the same scan where they can be (see
    claims_build.reshape_built), and are then among the arrays (claims_build.BUILT, with
    each row's set), the claims column's cells empty; otherwise the chunk is reshaped
    with its claims as text, as before (and always without `build`: `reshape`'s rows)."""
    built = claims_build.reshape_built(lines) if build else None
    if built is not None:
        (columns, mismatched, entity_fields, sitelink_fields, left), built = built
    else:
        columns, mismatched, entity_fields, sitelink_fields, left = entity_rows(lines)
    fields = {"entity": set(entity_fields), "sitelink": set(sitelink_fields)}
    arrays = {
        name: pa.Array.from_buffers(
            pa.large_string(), len(lines), [None, pa.py_buffer(offsets), pa.py_buffer(data)]
        )
        for name, (data, offsets) in zip(COLUMNS, columns)
    }
    frame = pl.from_arrow(pa.table(arrays))
    if left:
        counted = [0]
        rows = [entity_row(orjson.loads(lines[i]), counted, fields) for i in left]
        mismatched += counted[0]
        rows = pl.DataFrame(rows, schema={c: pl.String for c in COLUMNS}, orient="row")
        frame = frame.with_columns(frame[c].scatter(left, rows[c]) for c in COLUMNS)
    # The columns as large strings, for the builders (claims_build, fused): the reshape's
    # own buffers, unless entity_row filled some in
    if left:
        arrays = {c: fused.large_strings(frame.get_column(c)) for c in COLUMNS}
    if built is not None:  # (no line was left to entity_row: see reshape_built)
        arrays[claims_build.BUILT] = built
    return frame, mismatched, fields, arrays


def write_chunk(index: int, lines: list[bytes], data_dir: Path, scholar_dir: Path) -> dict:
    """Reshape a chunk's dump lines and write its two parts (see `route_parts`). Returns
    its manifest entry, with each part's (`main`, `scholar`) and the parts' bytes."""
    frame, mismatched, fields, arrays = _reshape(lines)
    entry = {
        "chunk": index,
        "file": f"chunk_{index}.parquet",
        "rows": frame.height,
        "first": frame["id"][0],
        "last": frame["id"][-1],
        "mismatched_terms": mismatched,
        "fields": {part: sorted(names) for part, names in fields.items()},
    }
    parts = route_parts(frame, entry, data_dir, scholar_dir, arrays=arrays)
    entry["bytes"] = sum(part["bytes"] for part in parts.values() if part)
    return {**entry, **parts}


# How long the split waits, at the end of what the download has written, to look again;
# and how often it says it is waiting
FOLLOW_POLL = 1.0
FOLLOW_NOTICE = 300


class DownloadFailed(Exception):
    """The download the split was following failed its md5 check."""


def _feed(path: Path, out, fed: list[int], failed: list[BaseException]) -> None:
    """Copy the dump at `path` into `out` (the decoder's input), counting the bytes in
    fed[0]. While download-dump is still writing it (as `path`.part), follow the part as it
    grows: the dump ends once the part is renamed to `path` (after its md5 check) and
    every byte of it is copied, and the download failed if the part is gone (moved aside,
    see `download`). So the split can run alongside the download."""
    part = path.with_suffix(path.suffix + ".part")
    try:
        while not (src := path if path.exists() else part if part.exists() else None):
            if not fed[1:]:
                print(f"Waiting for download-dump to write {part}", flush=True)
                fed.append(0)
            time.sleep(FOLLOW_POLL)
        with src.open("rb") as f, out:
            since = time.monotonic()
            while True:
                if block := f.read(8 << 20):
                    out.write(block)
                    fed[0] += len(block)
                    since = time.monotonic()
                elif path.exists() and fed[0] >= path.stat().st_size:
                    return
                elif not part.exists() and not path.exists():
                    raise DownloadFailed(f"{part} failed its md5 check")
                else:
                    if time.monotonic() - since > FOLLOW_NOTICE:
                        print(f"\nWaiting for download-dump to write {part} past {fed[0]:,} bytes",
                              flush=True)
                        since = time.monotonic()
                    time.sleep(FOLLOW_POLL)
    except BrokenPipeError:
        pass  # the decoder stopped early: the split is ending
    except BaseException as e:
        failed.append(e)


def _dump_lines(path: Path, fed: list[int]):
    """The dump's entity lines (without the trailing comma), decompressed (by the decoder
    for its suffix, see DECODERS) from the dump at `path`, or from its download as it is
    written (see `_feed`, which counts the bytes given to the decoder in fed[0])."""
    cmd = decoder(path.suffix.removeprefix("."))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=16 << 20)
    failed: list[BaseException] = []
    feeder = threading.Thread(target=_feed, args=(path, proc.stdin, fed, failed), daemon=True)
    feeder.start()
    try:
        for line in proc.stdout:
            line = line.rstrip(b",\r\n")
            if line.startswith(b"{"):
                yield line
    finally:
        proc.stdout.close()
        code = proc.wait()
        feeder.join()
        if failed:
            raise failed[0]
        if code not in (0, -13):  # -13: closed early (SIGPIPE)
            raise RuntimeError(f"{Path(cmd[0]).name} exited with {code}")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def _read_manifest(path: Path = MANIFEST) -> dict[int, dict]:
    return {e["chunk"]: e for e in _read_jsonl(path)}


def split(workers: int | None = None) -> None:
    """Cut the release's dump into chunk files (see the module docstring)."""
    release = _require_release()
    path = dump_path(release)
    if SPLIT_DONE.exists():
        print(f"{SPLIT_DONE} exists: the dump is already split", flush=True)
        return
    sets = _sets()
    for set_dir in sets.values():
        (set_dir / ROUTED).mkdir(parents=True, exist_ok=True)
    done = _read_manifest()
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    print(
        f"Splitting {path} into chunks of {CHUNK_ENTITIES:,} entities with {workers} "
        f"workers ({len(done):,} chunks already written)",
        flush=True,
    )
    pending = {}  # future -> bytes
    pending_bytes = 0
    entities = 0
    # (the dump's size is known here only once it is downloaded: see _feed)
    size = path.stat().st_size if path.exists() else None
    bar = tqdm(total=size, unit="B", unit_scale=True, desc="dump (compressed)")
    fed = [0]
    # The bar is closed on every path, an error's too: one left open is closed by tqdm's
    # __del__ as the interpreter shuts down, and that crashed it (a segfault, exit 139,
    # after the error's traceback) whenever a worker's error ended the split
    try:
        # Threads: a chunk's reshape (compiled), test and writes let go of the GIL, so the
        # workers share the lines read here rather than each taking a copy of its own
        with ThreadPoolExecutor(workers) as pool, MANIFEST.open("ab") as manifest:

            def collect(block: bool) -> None:
                nonlocal pending_bytes
                if not pending:
                    return
                timeout = None if block else 0
                finished, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=timeout)
                for future in finished:
                    pending_bytes -= pending.pop(future)
                    manifest.write(orjson.dumps(future.result()) + b"\n")
                    manifest.flush()

            index, batch, batch_bytes = 0, [], 0
            for line in _dump_lines(path, fed):
                batch.append(line)
                batch_bytes += len(line)
                if len(batch) < CHUNK_ENTITIES:
                    continue
                entities += len(batch)
                if index not in done:
                    while pending and pending_bytes + batch_bytes > MAX_PENDING_BYTES:
                        collect(block=True)
                    pending[pool.submit(write_chunk, index, batch, *sets.values())] = batch_bytes
                    pending_bytes += batch_bytes
                    collect(block=False)
                index += 1
                batch, batch_bytes = [], 0
                bar.set_postfix(chunks=index, entities=f"{entities:,}")
                bar.update(fed[0] - bar.n)
            if batch:
                entities += len(batch)
                if index not in done:
                    pending[pool.submit(write_chunk, index, batch, *sets.values())] = batch_bytes
                index += 1
            while pending:
                collect(block=True)
        bar.update(fed[0] - bar.n)
    except DownloadFailed as e:
        # Its chunks came from a download that failed its check: split again from nothing
        MANIFEST.unlink(missing_ok=True)
        for set_dir in sets.values():
            for f in (set_dir / ROUTED).glob("chunk_*"):
                f.unlink()
            for d in (*BUILT, *fused.FUSED):
                shutil.rmtree(set_dir / ROUTED / d, ignore_errors=True)
        raise SystemExit(f"{e}: the chunks split from it are deleted; download it again") from e
    finally:
        bar.close()
    SPLIT_DONE.write_text(f"{index} chunks, {entities} entities\n")
    print(f"Wrote {index:,} chunks ({entities:,} entities) to {ROOT_DATA_DIR}", flush=True)
    route()


def _sets() -> dict[str, Path]:
    """Each set's data directory: the main set's and the scholarly set's."""
    assert OTHER_WORK_DIR is not None
    return {"main": ROOT_DATA_DIR, "scholar": OTHER_WORK_DIR / "data"}


def route_chunk(entry: dict, data_dir: Path, scholar_dir: Path) -> dict:
    """Route a chunk file written by a split made before the split routed its chunks (see
    `route_parts`), the chunk itself left as it is."""
    frame = pl.read_parquet(data_dir / entry["file"])
    return {"chunk": entry["chunk"], **route_parts(frame, entry, data_dir, scholar_dir)}


def route_parts(
    frame: pl.DataFrame, entry: dict, data_dir: Path, scholar_dir: Path, *, arrays: dict[str, pa.Array] | None = None
) -> dict:
    """Write a chunk's scholarly works and the rest to `routed/chunk_{N}.parquet` in
    scholar_dir and data_dir; an empty part is not written. Returns each part's manifest
    entry (None if empty).

    Given the chunk's `claims` text (the split's, as large strings), a part's claims are
    written as its claims partition file (`routed/claims/chunk_{N}.parquet`, see
    claims_build), before the part itself, which then has no claims column: the pipeline
    places that file rather than decoding the claims (process.py). A part whose claims do
    not all fit is written with them as JSON, as before, and its entry says why
    (`claims_json`). Likewise each part's labels, descriptions, aliases and links: each
    table built and written as one file, its row groups by key (`routed/{table}/chunk_{N}.parquet`
    and its index beside it, see partitioning.fused), and the part has none of their four
    columns; or, if any of their cells does not fit, all four kept as JSON (`terms_json`).
    And each part's entities file (`routed/entities/chunk_{N}.parquet`, see entities_build),
    the part then without its `entity` column.

    Claims built from the chunk's lines (claims_build.BUILT among the arrays) are written
    as each part's claims file as they are, each row's set the one their scan gave it."""
    built = None if arrays is None else arrays.get(claims_build.BUILT)
    claims = None if arrays is None else arrays["claims"]
    if built is not None:
        scholarly = built.scholarly
    else:
        scholarly = frame.select(is_scholarly(pl.col("claims"))).to_series()
    out: dict = {}
    for name, mask, set_dir in (("scholar", scholarly, scholar_dir), ("main", ~scholarly, data_dir)):
        dst = set_dir / ROUTED / entry["file"]
        claims_dst = set_dir / ROUTED / CLAIMS / entry["file"]
        entities_dst = set_dir / ROUTED / ENTITIES / entry["file"]
        if not mask.any():
            for f in (dst, claims_dst, entities_dst):
                f.unlink(missing_ok=True)
            for t in fused.FUSED:
                (set_dir / ROUTED / t / entry["file"]).unlink(missing_ok=True)
                fused.index_path(set_dir / ROUTED / t, entry["file"]).unlink(missing_ok=True)
            out[name] = None
            continue
        why = "not built"
        if built is not None:
            why = claims_build.write_built(frame.get_column("id").filter(mask), built.claims[name], claims_dst)
        elif claims is not None:
            # The part's claims read from the chunk's text where they are, not copied
            rows = None if mask.all() else pa.array(mask.arg_true().cast(pl.Int64).to_numpy())
            ids = frame.get_column("id").filter(mask)
            why = claims_build.write_claims(ids, claims, rows, claims_dst)
        terms, terms_why = None, "not built"
        if arrays is not None:
            texts = {t: arrays[c] for t, c in fused.FUSED.items()}
            rows = None if mask.all() else pa.array(mask.arg_true().cast(pl.Int64).to_numpy())
            terms, terms_why = fused.write_one_file(arrays["id"], texts, rows, set_dir / ROUTED, entry["file"])
        entities_why = "not built"
        if arrays is not None:  # (the split's: not a chunk routed after an older split)
            entities_why = entities_build.write_entities(frame.select("id", "entity").filter(mask), entities_dst)
        dropped = [*(["claims"] if why is None else []), *(fused.FUSED.values() if terms is not None else []),
                   *(["entity"] if entities_why is None else [])]
        part = frame.drop(dropped).filter(mask)
        for f, w in ((claims_dst, why), (entities_dst, entities_why)):
            if w is not None:
                f.unlink(missing_ok=True)  # one left by an earlier attempt
        tmp = dst.with_suffix(".tmp")
        part.write_parquet(tmp, compression="zstd", compression_level=ZSTD_LEVEL)
        tmp.replace(dst)
        out[name] = {
            "rows": part.height,
            "bytes": dst.stat().st_size,
            "first": part["id"][0],
            "last": part["id"][-1],
            "fields": entry["fields"],  # the whole chunk's
        }
        if why is None:
            out[name]["claims_bytes"] = claims_dst.stat().st_size
        elif claims is not None:
            out[name]["claims_json"] = why
        if terms is not None:
            out[name]["terms_bytes"] = {str(t): n for t, n in terms.items()}
        elif arrays is not None:
            out[name]["terms_json"] = terms_why
        if entities_why is None:
            out[name]["entities_bytes"] = entities_dst.stat().st_size
        elif arrays is not None:
            out[name]["entities_json"] = entities_why
        del part
    return out


def _set_manifest(results: list[dict], name: str) -> list[dict]:
    """A set's manifest from the route log: its non-empty parts, numbered from 0 in the
    order of the chunks they came from (`split_chunk`)."""
    parts = [(r["chunk"], r[name]) for r in sorted(results, key=lambda r: r["chunk"]) if r[name]]
    return [
        {"chunk": i, "file": f"chunk_{i}.parquet", "split_chunk": c, **part}
        for i, (c, part) in enumerate(parts)
    ]


def _write_manifest(path: Path, entries: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(b"".join(orjson.dumps(e) + b"\n" for e in entries))
    tmp.replace(path)


def _move_routed(set_dir: Path, entries: list[dict]) -> None:
    """Move a set's routed parts to their numbers in the set's manifest (a part already
    moved is skipped, so this resumes)."""
    for e in entries:
        for table in BUILT:  # (each file moved on its own: any may already be)
            built = set_dir / ROUTED / table / f"chunk_{e['split_chunk']}.parquet"
            if built.exists():
                (set_dir / table).mkdir(exist_ok=True)
                built.replace(set_dir / table / e["file"])
        split_name = f"chunk_{e['split_chunk']}.parquet"
        for t in fused.FUSED:  # its file, then its index (the mark the table is complete)
            routed_t = set_dir / ROUTED / t
            if (routed_t / split_name).exists():
                (set_dir / t).mkdir(exist_ok=True)
                (routed_t / split_name).replace(set_dir / t / e["file"])
            if fused.index_path(routed_t, split_name).exists():
                (set_dir / t).mkdir(exist_ok=True)
                fused.index_path(routed_t, split_name).replace(fused.index_path(set_dir / t, e["file"]))
        src = set_dir / ROUTED / f"chunk_{e['split_chunk']}.parquet"
        if src.exists():
            src.replace(set_dir / e["file"])
    built_dirs = [set_dir / ROUTED / d for d in (*BUILT, *fused.FUSED)]
    for tmp in [f for d in (set_dir / ROUTED, *built_dirs) for f in d.glob("*.tmp")]:
        tmp.unlink()  # a worker's, interrupted
    for d in built_dirs:
        for work in d.glob("*.rowgroups"):
            shutil.rmtree(work)  # claims_build.RowGroups', interrupted
        if d.exists():
            d.rmdir()
    (set_dir / ROUTED).rmdir()


def route(workers: int = ROUTE_WORKERS) -> None:
    """Move the release's scholarly works into the scholarly set's chunks (see the module
    docstring). A chunk the split routed as it wrote it has its two parts in `routed/` in
    the two sets' data directories and in its manifest entry already. A chunk file (from a
    split made before the split routed its chunks) has its two parts written there, logged
    in ROUTE_LOG, and only then is the chunk deleted, so an interrupted run resumes from
    the log: a logged chunk is not routed again. Once every chunk is routed, each set's
    manifest numbers its non-empty parts from 0 (the pipeline groups chunks by
    consecutive numbers), the split's own manifest is kept as SPLIT_MANIFEST, the parts
    are moved to their numbers, and each set gets `split.done` and ROUTE_DONE."""
    _require_release()
    if ROUTE_DONE.exists():
        print(f"{ROUTE_DONE} exists: the release is already routed", flush=True)
        return
    if not SPLIT_DONE.exists():
        raise SystemExit(f"{SPLIT_DONE} is missing: run split-dump first")
    sets = _sets()
    for set_dir in sets.values():
        (set_dir / ROUTED).mkdir(parents=True, exist_ok=True)
    split_entries = _read_manifest(SPLIT_MANIFEST if SPLIT_MANIFEST.exists() else MANIFEST)
    logged = {r["chunk"]: r for r in _read_jsonl(ROUTE_LOG)}
    # The chunks the split routed as it wrote them (each part in its manifest entry)
    logged |= {
        c: {k: e[k] for k in ("chunk", *sets)} for c, e in split_entries.items() if "main" in e
    }
    todo = [e for c, e in sorted(split_entries.items()) if c not in logged]
    print(
        f"Routing {len(todo):,} of {len(split_entries):,} chunks into {sets['main']} and "
        f"{sets['scholar']} with {workers} workers",
        flush=True,
    )
    with ProcessPoolExecutor(workers) as pool, ROUTE_LOG.open("ab") as log:
        futures = {
            pool.submit(route_chunk, e, ROOT_DATA_DIR, sets["scholar"]): e for e in todo
        }
        # (every bar is closed on every path: see split)
        with tqdm(total=len(futures), desc="route", unit="chunk") as bar:
            for future in as_completed(futures):
                if future.exception():
                    pool.shutdown(cancel_futures=True)
                result = future.result()
                log.write(orjson.dumps(result) + b"\n")
                log.flush()
                os.fsync(log.fileno())
                logged[result["chunk"]] = result
                (ROOT_DATA_DIR / futures[future]["file"]).unlink()
                bar.update()
    if len(logged) != len(split_entries):
        raise RuntimeError(f"Routed {len(logged):,} of {len(split_entries):,} chunks")
    results = list(logged.values())
    manifests = {name: _set_manifest(results, name) for name in sets}
    # Until the manifests switch, a chunk file in data_dir is a split chunk (one logged
    # just before an interruption is deleted here); after, it is a moved part
    if not SPLIT_MANIFEST.exists():
        for e in split_entries.values():
            (ROOT_DATA_DIR / e["file"]).unlink(missing_ok=True)
        MANIFEST.replace(SPLIT_MANIFEST)
    for name, set_dir in sets.items():
        _write_manifest(set_dir / MANIFEST.name, manifests[name])
        _move_routed(set_dir, manifests[name])
    summary = ", ".join(
        f"{name}: {sum(e['rows'] for e in m):,} entities in {len(m):,} chunks"
        for name, m in manifests.items()
    )
    (sets["scholar"] / SPLIT_DONE.name).write_text(SPLIT_DONE.read_text())
    (sets["scholar"] / ROUTE_DONE.name).write_text(summary + "\n")
    ROUTE_DONE.write_text(summary + "\n")
    print(f"Routed: {summary}", flush=True)


# The fields the pipeline reads from a release's entities and sitelinks (process.py's
# ENTITY_SCHEMA and SITELINK_SCHEMA); a field outside them would be dropped there
EXPECTED_FIELDS = {
    "entity": {"type", "datatype", "ns", "title", "pageid", "lastrevid", "modified"},
    "sitelink": {"site", "title", "badges"},
}


def split_manifest() -> pl.DataFrame:
    """The split's manifest (chunk, file, rows, bytes, ...), once the split is complete;
    halts if any chunk's entities or sitelinks have a field the pipeline would drop."""
    if not SPLIT_DONE.exists():
        raise SystemExit(f"{SPLIT_DONE} is missing: run split-dump first")
    if not ROUTE_DONE.exists():
        raise SystemExit(f"{ROUTE_DONE} is missing: run route-release first")
    entries = sorted(_read_manifest().values(), key=lambda e: e["chunk"])
    unexpected = {
        part: sorted({f for e in entries for f in e.get("fields", {}).get(part, [])} - known)
        for part, known in EXPECTED_FIELDS.items()
    }
    if any(unexpected.values()):
        raise SystemExit(f"Fields the pipeline would drop (add them to process.py): {unexpected}")
    return pl.DataFrame(
        [{**{k: e[k] for k in ("chunk", "file", "rows", "bytes")}, "claims_bytes": e.get("claims_bytes"),
          "entities_bytes": e.get("entities_bytes"),
          "terms_bytes": sum(e["terms_bytes"].values()) if "terms_bytes" in e else None, "terms": "terms_bytes" in e}
         for e in entries],
        schema={"chunk": pl.Int64, "file": pl.String, "rows": pl.Int64, "bytes": pl.Int64, "claims_bytes": pl.Int64,
                "entities_bytes": pl.Int64, "terms_bytes": pl.Int64, "terms": pl.Boolean},
    )


def chunk_sizes() -> pl.LazyFrame:
    """Each chunk's bytes, as pull.prefetch._expected_chunk_sizes gives a source repo's: its
    file's, and its claims, entities and term tables' files' where the split built them."""
    size = (pl.col("bytes") + pl.col("claims_bytes").fill_null(0) + pl.col("entities_bytes").fill_null(0)
            + pl.col("terms_bytes").fill_null(0))
    return split_manifest().lazy().select(
        "chunk", size.alias("size"), (size / 1024**3).alias("size_gb")
    )


def check_chunk(chunk_idx: int, state_dir: Path) -> None:
    """The pull step for a release: its chunk file is already local (split from the dump),
    so check its size against the manifest and mark it pulled."""
    from .state import Step, get_chunk_state, update_state

    state = get_chunk_state(state_dir, chunk_idx)
    if state.is_empty() or state["step"].max() > Step.PULL:
        return
    entry = split_manifest().filter(pl.col("chunk") == chunk_idx).row(0, named=True)
    path = ROOT_DATA_DIR / entry["file"]
    if not path.exists() or path.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"{path} is missing or not {entry['bytes']:,} bytes: split again")
    # A part whose claims or entities the split built: their files too (see route_parts)
    for table in BUILT:
        built, size = ROOT_DATA_DIR / table / entry["file"], entry[f"{table}_bytes"]
        if size is not None and not (built.exists() and built.stat().st_size == size):
            raise RuntimeError(f"{built} is missing or not {size:,} bytes: split again")
    if entry["terms"]:  # its built term tables: each one's index, and its file if it has rows
        for t in fused.FUSED:
            if not fused.index_path(ROOT_DATA_DIR / t, entry["file"]).exists():
                raise RuntimeError(f"{t}/{entry['file']}'s built table is missing: split again")
    update_state(Path(entry["file"]), Step.PULL, state_dir)


def run_download() -> None:
    download()


def run_split() -> None:
    split()


def run_route() -> None:
    route()


def run_latest() -> None:
    print(latest_release())
