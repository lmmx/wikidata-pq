"""A smaller real dump: byte ranges of an official Wikidata bz2 dump, cut to whole bz2
blocks and whole entity lines, spliced into one valid bz2 stream.

Two shapes (bench/full/README.md):

- **prefix** (`--prefix-gb N`): the dump's first N GB. The dump is 8 shards back to back
  (docs/journal/2026-10-26-progressive-finalise.md), so a prefix of a few GB is shard 0's
  head only: its early, large entities, not the whole dump's mix.
- **windows** (`--windows K --window-mb W`): K windows of W MB, evenly spaced across the
  file (the i-th centred at (i + 0.5)/K of it), as docs/journal/2026-10-27-compute-frontier.md
  took its real-data figures. Closer to the dump's mix: every shard, early and late ids.

**How it is cut.** The dump is one bz2 stream (lbzip2's, `BZh9`), its blocks bit-aligned.
Each range's block starts are found by their 48-bit magic at any bit alignment, and each
block between two starts is kept as it is, bit for bit: the dump's own compressed blocks,
so the decoder decodes what it decodes at a release. Only a range's first and last few
blocks are decoded, to cut its text at line ends: the partial line at its start is
dropped, and the text before the first line end of its last blocks is kept. Those ends
are re-encoded (Python's bz2, level 9: about 1% of the output) and spliced in. The output
starts with `[` and ends with `]` as the dump does, and its stream CRC is the blocks'
combined CRC, so the decoder checks every block and the whole stream.

**Where the bytes come from:** dumps.wikimedia.org by HTTP range requests (at most 3
connections, Wikimedia's cap an IP; 2 by default), or a local copy of the dump
(`--from-file`). The output is written as it is made, to OUT.part, renamed to OUT at the
end: the split follows a growing .part (dump._feed), so this can run as the split's
"download", the bytes arriving, spliced and decoded at once (run.py's streaming mode).
`--log` gets a line a piece received (its time and bytes), for the throughput over time.

Usage: splice.py OUT.json.bz2 --release R (--prefix-gb N | --windows K --window-mb W)
         [--from-file DUMP] [--connections C] [--log NET.jsonl] [--meta META.json]
"""

from __future__ import annotations

import argparse
import bz2
import collections
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DUMPS_URL = "https://dumps.wikimedia.org/wikidatawiki/entities"
USER_AGENT = "wikidata-pq/0.1 (https://github.com/lmmx/wikidata-pq)"  # (dump.USER_AGENT)
MAGIC = 0x314159265359  # a block's start (pi)
EOS = 0x177245385090  # the stream's end (sqrt(pi))
MAGIC_B = MAGIC.to_bytes(6, "big")
EOS_B = EOS.to_bytes(6, "big")
PIECE = 16 << 20  # (bytes a range request)
HOLD = 4  # (a range's last blocks held back, decoded to cut its text at a line end)


def dump_url(release: str) -> str:
    return f"{DUMPS_URL}/{release}/wikidata-{release}-all.json.bz2"


def _request(url: str, method: str = "GET", headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def remote_info(release: str) -> dict:
    """The dump's size, Last-Modified and ETag (pinning the bytes the ranges are of), and
    its published md5 and sha1."""
    with _request(dump_url(release), "HEAD") as r:
        info = {"url": dump_url(release), "bytes": int(r.headers["Content-Length"]),
                "last_modified": r.headers.get("Last-Modified"), "etag": r.headers.get("ETag")}
    name = f"wikidata-{release}-all.json.bz2"
    for algo in ("md5", "sha1"):
        try:
            with _request(f"{DUMPS_URL}/{release}/wikidata-{release}-{algo}sums.txt") as r:
                sums = dict(line.split()[::-1] for line in r.read().decode().splitlines() if line.strip())
            info[algo] = sums.get(name)
        except OSError:
            info[algo] = None
    return info


def _get(url: str, a: int, b: int, etag: str | None) -> bytes:
    """Bytes [a, b) of `url`, retried; refused if the file changed (its ETag)."""
    delay = 2.0
    for attempt in range(8):
        try:
            with _request(url, headers={"Range": f"bytes={a}-{b - 1}"}) as r:
                if r.status != 206:
                    raise OSError(f"range request answered {r.status}")
                if etag and r.headers.get("ETag") not in (None, etag):
                    raise SystemExit(f"{url} changed while being read (ETag {r.headers.get('ETag')} != {etag})")
                data = r.read()
            if len(data) != b - a:
                raise OSError(f"short read: {len(data)} of {b - a}")
            return data
        except SystemExit:
            raise
        except OSError as e:
            if attempt == 7:
                raise
            print(f"[splice] bytes {a}-{b}: {e}; retrying in {delay:.0f} s", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise AssertionError


def pieces(ranges: list[tuple[int, int]], url: str | None, path: Path | None, connections: int,
           etag: str | None, log) -> "collections.abc.Iterator[tuple[int, bytes]]":
    """(range index, bytes) for every piece of every range, in order: read from `path`, or
    fetched from `url` over `connections` connections, a few pieces ahead."""
    tasks = [(i, a, min(a + PIECE, b)) for i, (s, b) in enumerate(ranges) for a in range(s, b, PIECE)]
    got = 0
    t0 = time.time()
    if path is not None:
        with open(path, "rb") as f:
            for i, a, b in tasks:
                f.seek(a)
                yield i, f.read(b - a)
        return
    with ThreadPoolExecutor(connections) as pool:
        ahead: collections.deque = collections.deque()
        it = iter(tasks)
        for t in it:
            ahead.append((t, pool.submit(_get, url, t[1], t[2], etag)))
            if len(ahead) >= 2 * connections:
                break
        while ahead:
            (i, a, b), fut = ahead.popleft()
            data = fut.result()
            if (t := next(it, None)) is not None:
                ahead.append((t, pool.submit(_get, url, t[1], t[2], etag)))
            got += len(data)
            if log:
                log.write(json.dumps({"t": time.time(), "bytes": len(data), "received": got,
                                      "since_start_s": round(time.time() - t0, 3)}) + "\n")
                log.flush()
            yield i, data


# -- bits
def find(buf: bytes, pattern: bytes, from_bit: int = 0) -> list[int]:
    """Every bit offset in `buf`, from `from_bit`, at which the 48-bit `pattern` starts,
    at any alignment. At bit 8j + r, the pattern's bits [8 - r, 48 - r) fill bytes j + 1
    to j + 5 whole, so each alignment is a byte search for those 5 bytes (memmem), each
    hit checked whole."""
    value, n, out = int.from_bytes(pattern, "big"), len(buf), []
    for r in range(8):
        inner = ((value >> r) & ((1 << 40) - 1)).to_bytes(5, "big")
        i = max(1, (from_bit - r) // 8 + 1)
        while (i := buf.find(inner, i)) != -1:
            k = 8 * (i - 1) + r
            if k + 48 > 8 * n:
                break
            if k >= from_bit and bits(buf, k, k + 48) == value:
                out.append(k)
            i += 1
    return sorted(out)


def bits(buf: bytes, p: int, q: int) -> int:
    """Bits [p, q) of `buf` as an integer."""
    lo, hi = p // 8, (q + 7) // 8
    return (int.from_bytes(buf[lo:hi], "big") >> (hi * 8 - q)) & ((1 << (q - p)) - 1)


class BitWriter:
    def __init__(self, f):
        self.f, self.acc, self.n = f, 0, 0

    def write(self, value: int, nbits: int) -> None:
        self.acc = (self.acc << nbits) | value
        self.n += nbits
        if self.n >= 8:
            rem = self.n % 8
            self.f.write((self.acc >> rem).to_bytes(self.n // 8, "big"))
            self.acc &= (1 << rem) - 1
            self.n = rem

    def close(self) -> None:
        if self.n:
            self.f.write((self.acc << (8 - self.n)).to_bytes(1, "big"))
            self.acc = self.n = 0


def _crc(value: int, nbits: int) -> int:
    return (value >> (nbits - 80)) & 0xFFFFFFFF  # (the 32 bits after the block's magic)


def decode_block(value: int, nbits: int) -> bytes:
    """One block's text: the block alone as a stream, decoded (its CRC checked)."""
    f = io.BytesIO()
    w = BitWriter(f)
    w.write(int.from_bytes(b"BZh9", "big"), 32)
    w.write(value, nbits)
    w.write(EOS, 48)
    w.write(_crc(value, nbits), 32)
    w.close()
    return bz2.decompress(f.getvalue())


def encode_blocks(text: bytes) -> list[tuple[int, int]]:
    """`text` compressed (level 9), as its stream's blocks."""
    if not text:
        return []
    s = bz2.compress(text, 9)
    starts = find(s, MAGIC_B, 32)
    end = find(s, EOS_B, starts[-1] + 48)[-1]
    return [(bits(s, p, q), q - p) for p, q in zip(starts, [*starts[1:], end])]


def blocks(chunks, start: int):
    """The whole blocks of a byte range (its pieces `chunks`, from file offset `start`),
    as (bits, nbits), in order; the partial blocks at its ends dropped."""
    buf, scanned, prev = b"", 0, None  # (bit offsets in buf)
    for data in chunks:
        buf += data
        for m in find(buf, MAGIC_B, scanned):
            if prev is not None:
                yield bits(buf, prev, m), m - prev
            prev = m
        scanned = max(scanned, 8 * len(buf) - 47)
        cut = (prev if prev is not None else scanned) // 8
        buf, scanned = buf[cut:], scanned - 8 * cut
        prev = None if prev is None else prev - 8 * cut


class Splicer:
    """The output stream: a header, blocks (kept or re-encoded), the end mark and the
    combined CRC; its bytes hashed as they are written."""

    def __init__(self, f):
        self.f, self.sha = f, hashlib.sha256()
        self.crc, self.kept, self.kept_bits, self.encoded, self.encoded_bits, self.bytes = 0, 0, 0, 0, 0, 0
        self.text_encoded = 0
        self.w = BitWriter(self)
        self.w.write(int.from_bytes(b"BZh9", "big"), 32)

    def write(self, b: bytes) -> None:  # (the BitWriter's sink)
        self.f.write(b)
        self.sha.update(b)
        self.bytes += len(b)

    def block(self, value: int, nbits: int, kept: bool = True) -> None:
        self.w.write(value, nbits)
        self.crc = (((self.crc << 1) | (self.crc >> 31)) & 0xFFFFFFFF) ^ _crc(value, nbits)
        if kept:
            self.kept, self.kept_bits = self.kept + 1, self.kept_bits + nbits
        else:
            self.encoded, self.encoded_bits = self.encoded + 1, self.encoded_bits + nbits

    def text(self, data: bytes) -> None:
        self.text_encoded += len(data)
        for v, n in encode_blocks(data):
            self.block(v, n, kept=False)

    def close(self) -> None:
        self.w.write(EOS, 48)
        self.w.write(self.crc, 32)
        self.w.close()
        self.f.flush()


def splice(out: Path, ranges: list[tuple[int, int]], url: str | None, path: Path | None,
           connections: int = 2, etag: str | None = None, log=None) -> dict:
    """The ranges spliced into `out` (written as out.part, renamed at the end)."""
    part = out.with_name(out.name + ".part")
    t0 = time.time()
    with open(part, "wb") as f:
        sp = Splicer(f)
        stream = pieces(ranges, url, path, connections, etag, log)
        pending: dict = {"i": None}

        def of_range(i: int):
            """The pieces of range i (the stream's, in order)."""
            if pending["i"] is not None:
                if pending["i"][0] != i:
                    return
                yield pending["i"][1]
                pending["i"] = None
            for j, data in stream:
                if j != i:
                    pending["i"] = (j, data)
                    return
                yield data

        for i, (a, b) in enumerate(ranges):
            first, last = i == 0, i == len(ranges) - 1
            it = blocks(of_range(i), a)
            if a > 0:  # (the range starts inside a line: drop it)
                if first:
                    sp.text(b"[\n")
                head = b""
                for blk in it:
                    head += decode_block(*blk)
                    if b"\n" in head:
                        break
                else:
                    raise SystemExit(f"range {a}-{b}: no line end in its blocks; take larger windows")
                sp.text(head[head.index(b"\n") + 1:])
            held: collections.deque = collections.deque()
            for blk in it:
                held.append(blk)
                if len(held) > HOLD:
                    sp.block(*held.popleft())
                    f.flush()
            tail = b"".join(decode_block(*blk) for blk in held)
            if (end := tail.rfind(b"\n")) < 0:
                raise SystemExit(f"range {a}-{b}: no line end in its last blocks; take larger windows")
            tail = tail[:end + 1]
            if last and tail.endswith(b",\n"):  # (the dump's last line, then its `]`)
                tail = tail[:-2] + b"\n]\n"
            sp.text(tail)
            f.flush()
        sp.close()
    part.replace(out)
    return {"bytes": sp.bytes, "sha256": sp.sha.hexdigest(), "blocks_kept": sp.kept,
            "blocks_kept_bytes": sp.kept_bits // 8, "blocks_reencoded": sp.encoded,
            "blocks_reencoded_bytes": sp.encoded_bits // 8, "text_reencoded_bytes": sp.text_encoded,
            "ranges": [list(r) for r in ranges], "range_bytes": sum(b - a for a, b in ranges),
            "wall_s": round(time.time() - t0, 2)}


def plan(size: int, prefix_gb: float | None, windows: int | None, window_mb: float | None) -> tuple[str, list]:
    """The label and the byte ranges of a prefix or of evenly spaced windows."""
    if prefix_gb:
        n = int(prefix_gb * 1e9)
        assert n < size - (2 << 20), "a prefix short of the whole dump's end"
        return f"prefix{prefix_gb:g}g", [(0, n)]
    assert windows and window_mb, "--prefix-gb, or --windows with --window-mb"
    w = int(window_mb * 1e6)
    assert w >= 1 << 20, "windows of at least 1 MB (a few whole blocks each, beside the cut ends)"
    starts = [max(0, int((i + 0.5) * size / windows - w / 2)) for i in range(windows)]
    ranges = [(s, min(s + w, size - (2 << 20))) for s in starts]
    assert all(b <= a2 for (_, b), (a2, _) in zip(ranges, ranges[1:])), "windows overlapping"
    return f"windows{windows}x{window_mb:g}m", ranges


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--release", required=True)
    ap.add_argument("--prefix-gb", type=float)
    ap.add_argument("--windows", type=int)
    ap.add_argument("--window-mb", type=float)
    ap.add_argument("--from-file", type=Path)
    ap.add_argument("--connections", type=int, default=int(os.environ.get("WIKIDATA_DUMP_CONNECTIONS") or 2))
    ap.add_argument("--log", type=Path)
    ap.add_argument("--meta", type=Path)
    a = ap.parse_args(argv)
    if not 1 <= a.connections <= 3:
        raise SystemExit("--connections: 1 to 3 (Wikimedia's cap is 3 an IP)")
    info = ({"path": str(a.from_file), "bytes": a.from_file.stat().st_size} if a.from_file
            else remote_info(a.release))
    label, ranges = plan(info["bytes"], a.prefix_gb, a.windows, a.window_mb)
    log = open(a.log, "a") if a.log else None
    print(f"[splice] {label}: {len(ranges)} range(s), {sum(b - s for s, b in ranges) / 1e9:.3f} GB of "
          f"{info['bytes'] / 1e9:.1f} GB, from {info.get('url') or info['path']}", flush=True)
    res = splice(a.out, ranges, None if a.from_file else info["url"], a.from_file, a.connections,
                 info.get("etag"), log)
    meta = {"label": label, "release": a.release, "source": info, **res,
            "connections": None if a.from_file else a.connections}
    print(f"[splice] wrote {a.out}: {res['bytes'] / 1e9:.3f} GB, {res['blocks_kept']} blocks kept, "
          f"{res['blocks_reencoded']} re-encoded ({res['blocks_reencoded_bytes'] / 1e6:.1f} MB), "
          f"in {res['wall_s']} s", flush=True)
    if a.meta:
        a.meta.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
