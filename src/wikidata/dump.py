"""An official Wikidata JSON dump (a release) as chunk files for the pipeline.

`download-dump` fetches `wikidata-{release}-all.json.bz2` from dumps.wikimedia.org (or a
mirror) into DUMP_DIR, resumably, and checks it against the release's md5sums.

`split-dump` decompresses it with lbzip2 (all cores) and cuts it into chunks of
CHUNK_ENTITIES entities, each reshaped into the row the pipeline's process step reads
(`id, labels, descriptions, aliases, sitelinks, claims`, the last five JSON strings, as in
philippesaade/wikidata) and routed as it is written: its scholarly works (see
scholarly.py) to `routed/chunk_{N}.parquet` in the scholarly set's data directory, the rest
to the same in ROOT_DATA_DIR, with a line per chunk in the manifest (MANIFEST) giving both
parts. A rerun skips the chunks already in the manifest. Once every chunk is written, the
split ends by routing the release (see `route`). It can run alongside download-dump (`just
download-split`): it follows the download as it is written, and ends once it is complete
and checked (see `_feed`).

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
import os
import re
import shutil
import subprocess
import sys
import threading
import time
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

from .config import DUMP_DIR, OTHER_WORK_DIR, RELEASE, ROOT_DATA_DIR, SCHOLAR
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
# Chunks routed at once (each holds a whole chunk in memory, up to about 1 GB of JSON)
ROUTE_WORKERS = 8


def dump_name(release: str) -> str:
    return f"wikidata-{release}-all.json.bz2"


def dump_path(release: str) -> Path:
    return DUMP_DIR / dump_name(release)


def latest_release(base: str = DUMPS_URL) -> str:
    """The newest release directory holding a full JSON dump (bz2)."""
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
    """Download the release's bz2 dump to DUMP_DIR (resuming a partial download), checked
    against the release's md5sums. `base` is a mirror's .../wikidatawiki/entities URL
    (WIKIDATA_DUMPS_URL; dumps.wikimedia.org by default)."""
    release = _require_release()
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
    bar = tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=name)
    while done < total:
        try:
            with _open(url, headers={"Range": f"bytes={done}-"}, timeout=120) as r, part.open(
                "ab"
            ) as out:
                if r.status != 206:
                    raise SystemExit(f"{url} ignored the range request (status {r.status})")
                while block := r.read(8 << 20):
                    out.write(block)
                    done += len(block)
                    bar.update(len(block))
        except (OSError, urllib.error.URLError) as e:
            print(f"\n{e}; resuming at {done:,} bytes", file=sys.stderr, flush=True)
    bar.close()
    if _md5(part) != md5:
        # Moved aside, so a split following the download stops (see _feed)
        part.replace(bad := part.with_suffix(".bad"))
        raise SystemExit(f"{part} does not match its md5 {md5} (moved to {bad}): download again")
    part.replace(dst)
    print(f"Wrote {dst}", flush=True)
    return dst


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
    """A chunk's dump lines as its rows (`entity_row`'s, column by column), the count of
    mismatched terms, and the fields seen (see `entity_row`). The reshape is compiled
    (reshape/src/lib.rs: each line parsed once and its row written as orjson writes it);
    a line it does not expect (a field of the wrong type, say) is reshaped by entity_row."""
    columns, mismatched, entity_fields, sitelink_fields, left = entity_rows(lines)
    fields = {"entity": set(entity_fields), "sitelink": set(sitelink_fields)}
    frame = pl.from_arrow(
        pa.table({
            name: pa.Array.from_buffers(
                pa.large_string(), len(lines), [None, pa.py_buffer(offsets), pa.py_buffer(data)]
            )
            for name, (data, offsets) in zip(COLUMNS, columns)
        })
    )
    if left:
        counted = [0]
        rows = [entity_row(orjson.loads(lines[i]), counted, fields) for i in left]
        mismatched += counted[0]
        rows = pl.DataFrame(rows, schema={c: pl.String for c in COLUMNS}, orient="row")
        frame = frame.with_columns(frame[c].scatter(left, rows[c]) for c in COLUMNS)
    return frame, mismatched, fields


def write_chunk(index: int, lines: list[bytes], data_dir: Path, scholar_dir: Path) -> dict:
    """Reshape a chunk's dump lines and write its two parts (see `route_parts`). Returns
    its manifest entry, with each part's (`main`, `scholar`) and the parts' bytes."""
    frame, mismatched, fields = reshape(lines)
    entry = {
        "chunk": index,
        "file": f"chunk_{index}.parquet",
        "rows": frame.height,
        "first": frame["id"][0],
        "last": frame["id"][-1],
        "mismatched_terms": mismatched,
        "fields": {part: sorted(names) for part, names in fields.items()},
    }
    parts = route_parts(frame, entry, data_dir, scholar_dir)
    entry["bytes"] = sum(part["bytes"] for part in parts.values() if part)
    return {**entry, **parts}


# How long the split waits, at the end of what the download has written, to look again;
# and how often it says it is waiting
FOLLOW_POLL = 1.0
FOLLOW_NOTICE = 300


class DownloadFailed(Exception):
    """The download the split was following failed its md5 check."""


def _feed(path: Path, out, fed: list[int], failed: list[BaseException]) -> None:
    """Copy the dump at `path` into `out` (lbzip2's input), counting the bytes in fed[0].
    While download-dump is still writing it (as `path`.part), follow the part as it
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
        pass  # lbzip2 stopped early: the split is ending
    except BaseException as e:
        failed.append(e)


def _dump_lines(path: Path, fed: list[int]):
    """The dump's entity lines (without the trailing comma), decompressed by lbzip2 from
    the dump at `path`, or from its download as it is written (see `_feed`, which counts
    the bytes given to lbzip2 in fed[0])."""
    tool = shutil.which("lbzip2")
    if not tool:
        raise SystemExit("lbzip2 decompresses the dump on all cores: apt install lbzip2")
    proc = subprocess.Popen(
        [tool, "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=16 << 20
    )
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
            raise RuntimeError(f"lbzip2 exited with {code}")


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
    except DownloadFailed as e:
        # Its chunks came from a download that failed its check: split again from nothing
        MANIFEST.unlink(missing_ok=True)
        for set_dir in sets.values():
            for f in (set_dir / ROUTED).glob("chunk_*"):
                f.unlink()
        raise SystemExit(f"{e}: the chunks split from it are deleted; download it again") from e
    bar.update(fed[0] - bar.n)
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


def route_parts(frame: pl.DataFrame, entry: dict, data_dir: Path, scholar_dir: Path) -> dict:
    """Write a chunk's scholarly works and the rest to `routed/chunk_{N}.parquet` in
    scholar_dir and data_dir; an empty part is not written. Returns each part's manifest
    entry (None if empty)."""
    scholarly = frame.select(is_scholarly(pl.col("claims"))).to_series()
    out: dict = {}
    parts = {"scholar": (frame.filter(scholarly), scholar_dir), "main": (frame.filter(~scholarly), data_dir)}
    for name, (part, set_dir) in parts.items():
        dst = set_dir / ROUTED / entry["file"]
        if part.is_empty():
            dst.unlink(missing_ok=True)
            out[name] = None
            continue
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
        src = set_dir / ROUTED / f"chunk_{e['split_chunk']}.parquet"
        if src.exists():
            src.replace(set_dir / e["file"])
    for tmp in (set_dir / ROUTED).glob("*.tmp"):  # a worker's, interrupted
        tmp.unlink()
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="route", unit="chunk"):
            if future.exception():
                pool.shutdown(cancel_futures=True)
            result = future.result()
            log.write(orjson.dumps(result) + b"\n")
            log.flush()
            os.fsync(log.fileno())
            logged[result["chunk"]] = result
            (ROOT_DATA_DIR / futures[future]["file"]).unlink()
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
        [{k: e[k] for k in ("chunk", "file", "rows", "bytes")} for e in entries]
    )


def chunk_sizes() -> pl.LazyFrame:
    """Each chunk's bytes, as pull.prefetch._expected_chunk_sizes gives a source repo's."""
    return split_manifest().lazy().select(
        "chunk", pl.col("bytes").alias("size"), (pl.col("bytes") / 1024**3).alias("size_gb")
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
    update_state(Path(entry["file"]), Step.PULL, state_dir)


def run_download() -> None:
    download()


def run_split() -> None:
    split()


def run_route() -> None:
    route()


def run_latest() -> None:
    print(latest_release())
