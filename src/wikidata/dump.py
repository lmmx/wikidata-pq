"""An official Wikidata JSON dump (a release) as chunk files for the pipeline.

`download-dump` fetches `wikidata-{release}-all.json.bz2` from dumps.wikimedia.org (or a
mirror) into DUMP_DIR, resumably, and checks it against the release's md5sums.

`split-dump` decompresses it with lbzip2 (all cores) and cuts it into chunks of
CHUNK_ENTITIES entities, each reshaped into the row the pipeline's process step reads
(`id, labels, descriptions, aliases, sitelinks, claims`, the last five JSON strings, as in
philippesaade/wikidata) and written as ROOT_DATA_DIR/chunk_{N}.parquet, with a line per
chunk in the manifest (MANIFEST). A rerun skips the chunks already in the manifest.

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
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import orjson
import polars as pl
from tqdm import tqdm

from .config import DUMP_DIR, RELEASE, ROOT_DATA_DIR

DUMPS_URL = "https://dumps.wikimedia.org/wikidatawiki/entities"
# Entities per chunk file (the philippesaade files have 10,000 rows each)
CHUNK_ENTITIES = 10_000
# The most decompressed bytes handed to the workers and not yet written, bounding memory
# (early entities are large: about 50 kB each in 20260928's first chunk)
MAX_PENDING_BYTES = 4 * 1024**3
ZSTD_LEVEL = 9  # about the bz2's own size (level 3: 1.1x, level 19: 0.9x at 30x the time)
MANIFEST = ROOT_DATA_DIR / "manifest.jsonl"
SPLIT_DONE = ROOT_DATA_DIR / "split.done"


def dump_name(release: str) -> str:
    return f"wikidata-{release}-all.json.bz2"


def dump_path(release: str) -> Path:
    return DUMP_DIR / dump_name(release)


def latest_release(base: str = DUMPS_URL) -> str:
    """The newest release directory holding a full JSON dump (bz2)."""
    with urllib.request.urlopen(f"{base}/", timeout=60) as r:
        dates = sorted(set(re.findall(r'href="(\d{8})/"', r.read().decode())), reverse=True)
    for date in dates:
        with urllib.request.urlopen(f"{base}/{date}/", timeout=60) as r:
            if dump_name(date) in r.read().decode():
                return date
    raise SystemExit(f"No {dump_name('*')} under {base}/")


def _require_release() -> str:
    if not RELEASE:
        raise SystemExit("Set WIKIDATA_RELEASE to a dump date, e.g. WIKIDATA_RELEASE=20260928")
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
    with urllib.request.urlopen(f"{DUMPS_URL}/{release}/wikidata-{release}-md5sums.txt") as r:
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
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        total = int(r.headers["Content-Length"])
    done = part.stat().st_size if part.exists() else 0
    bar = tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=name)
    while done < total:
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(request, timeout=120) as r, part.open("ab") as out:
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
        raise SystemExit(f"{part} does not match its md5 {md5}; delete it to download again")
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


def write_chunk(index: int, lines: list[bytes], out_dir: Path) -> dict:
    """Reshape a chunk's dump lines and write them as out_dir/chunk_{index}.parquet."""
    mismatched = [0]
    fields = {"entity": set(), "sitelink": set()}
    rows = [entity_row(orjson.loads(line), mismatched, fields) for line in lines]
    frame = pl.DataFrame(rows, schema={c: pl.String for c in COLUMNS}, orient="row")
    path = out_dir / f"chunk_{index}.parquet"
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp, compression="zstd", compression_level=ZSTD_LEVEL)
    tmp.replace(path)
    return {
        "chunk": index,
        "file": path.name,
        "rows": frame.height,
        "bytes": path.stat().st_size,
        "first": rows[0][0],
        "last": rows[-1][0],
        "mismatched_terms": mismatched[0],
        "fields": {part: sorted(names) for part, names in fields.items()},
    }


def _dump_lines(source):
    """The dump's entity lines (without the trailing comma), decompressed by lbzip2 from
    the open file `source` (whose offset then shows how far it has read)."""
    tool = shutil.which("lbzip2")
    if not tool:
        raise SystemExit("lbzip2 decompresses the dump on all cores: apt install lbzip2")
    proc = subprocess.Popen([tool, "-dc"], stdin=source, stdout=subprocess.PIPE, bufsize=16 << 20)
    try:
        for line in proc.stdout:
            line = line.rstrip(b",\r\n")
            if line.startswith(b"{"):
                yield line
    finally:
        proc.stdout.close()
        if proc.wait() not in (0, -13):  # -13: closed early (SIGPIPE)
            raise RuntimeError(f"lbzip2 exited with {proc.returncode}")


def _read_manifest() -> dict[int, dict]:
    if not MANIFEST.exists():
        return {}
    entries = (orjson.loads(line) for line in MANIFEST.read_bytes().splitlines() if line)
    return {e["chunk"]: e for e in entries}


def split(workers: int | None = None) -> None:
    """Cut the release's dump into chunk files (see the module docstring)."""
    release = _require_release()
    path = dump_path(release)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run download-dump first")
    if SPLIT_DONE.exists():
        print(f"{SPLIT_DONE} exists: the dump is already split", flush=True)
        return
    ROOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    bar = tqdm(total=path.stat().st_size, unit="B", unit_scale=True, desc="dump (compressed)")
    source = path.open("rb")
    with source, ProcessPoolExecutor(workers) as pool, MANIFEST.open("ab") as manifest:

        def collect(block: bool) -> None:
            nonlocal pending_bytes
            if not pending:
                return
            finished, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=None if block else 0)
            for future in finished:
                pending_bytes -= pending.pop(future)
                manifest.write(orjson.dumps(future.result()) + b"\n")
                manifest.flush()

        index, batch, batch_bytes = 0, [], 0
        for line in _dump_lines(source):
            batch.append(line)
            batch_bytes += len(line)
            if len(batch) < CHUNK_ENTITIES:
                continue
            entities += len(batch)
            if index not in done:
                while pending and pending_bytes + batch_bytes > MAX_PENDING_BYTES:
                    collect(block=True)
                pending[pool.submit(write_chunk, index, batch, ROOT_DATA_DIR)] = batch_bytes
                pending_bytes += batch_bytes
                collect(block=False)
            index += 1
            batch, batch_bytes = [], 0
            bar.set_postfix(chunks=index, entities=f"{entities:,}")
            bar.update(os.lseek(source.fileno(), 0, os.SEEK_CUR) - bar.n)
        if batch:
            entities += len(batch)
            if index not in done:
                pending[pool.submit(write_chunk, index, batch, ROOT_DATA_DIR)] = batch_bytes
            index += 1
        while pending:
            collect(block=True)
    bar.update(bar.total - bar.n)
    bar.close()
    SPLIT_DONE.write_text(f"{index} chunks, {entities} entities\n")
    print(f"Wrote {index:,} chunks ({entities:,} entities) to {ROOT_DATA_DIR}", flush=True)


def run_download() -> None:
    download()


def run_split() -> None:
    split()


def run_latest() -> None:
    print(latest_release())
