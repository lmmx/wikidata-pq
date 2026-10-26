"""The split's sweep reader (WIKIDATA_SPLIT_SWEEPS=N, off by default): the dump's lines
in rising id order, for a frontier (docs/journal/2026-10-26-progressive-finalise.md, PR 2).

Wikimedia's `wikidata-*-all.json.bz2` is one bz2 stream holding the dump's N shards (8)
back to back, each a sweep over the whole id range in page-id order: read in byte order,
the ids reset N times. This reads the shards as N sweeps at once, a decoder each, and
gives the split its lines merged so that the *frontier*, the least of the sweeps'
running maxima (their largest Q id read so far), is as high as it can be: the next line
always comes from the sweep whose running maximum is least (the lowest-numbered of
those equal). The split cuts its chunks from the merged lines as it did from the serial
ones, so every chunk is a narrow id range across the shards. Only the order of the
lines changes: every line is read exactly once, whatever the dump's layout.

**The cuts** (`plan`, once a dump, kept in the split's state as PLAN beside its
manifest, so a rerun takes the same ones): sweep k starts at the first line of the k-th
shard present, found exactly on the local file. bz2 blocks start at any bit, so the
file is read by byte range and each block found by its magic (reshape's bz2_blocks)
and decoded alone (rewrapped as a stream of its own, as bzip2recover writes them, and
checked by libbz2: its CRC). A probe at a byte offset decodes the first block after it
and takes the shard (dump.shard_of, Wikibase's rule mod N) of the first line that
starts in it; PROBES_PER_SWEEP * N probes at even offsets bracket each change of shard,
bisection narrows it to RESOLUTION bytes, and the blocks there are decoded in order to
find the first line of the new shard: its block and its offset in the block's text. A
dump whose probes are not in shard order is refused (unset the flag). N = 1 is one
sweep over the whole dump: the serial order, through this reader.

**The decoders.** Sweep k reads its byte range, from its first block up to the next
sweep's (or the stream's end), in segments of about SEGMENT_BYTES of bz2, each built in
memory and decoded whole by its own `lbzip2 -dc -n1`, its text then passed on a
TEXT_PIECE at a time. At most DECODERS segments (the cores) are decoded at once over all
the sweeps, as many threads as today's one `lbzip2` takes (8 decoders at once on 4 cores
took 1.7 times the CPU), and a sweep holds its slot only while its segment decodes. A segment is fed to its
decoder as a stream of its own, without shifting a bit of it: a stream header, a tiny
block of DUMMIES (whose length in bits puts the segment's first block at the bit of its
byte where it is in the file), the file's bytes as they are, and an end-of-stream mark
with the segment's stream CRC, `fold` of its blocks' CRCs. The dummy block's text is
dropped from the decoder's output.

**The seams.** Sweep k's text is its blocks' text from its first line on; the first
bytes of its first block, up to that line, end sweep k-1 (decoded alone at the start,
from that block), so the line straddling the cut is joined exactly and no line is read
twice. Within a sweep, lines straddle blocks and segments as the decoder's output runs
on.

**Integrity**, as today's one `lbzip2` checks it: each block's CRC (by its decoder);
each segment's stream CRC (by its decoder), so the blocks a decoder read are exactly
the ones found, in order; and, once every sweep has ended, the stream CRC of the whole
dump recomputed from every block's CRC in file order (`combine`), against the dump's
own (its last 80 bits). A block missed, repeated or misplaced between sweeps, a
corrupt block or a spurious block magic all halt the split. The md5 is download-dump's,
as before.

**Restart.** Each chunk's manifest line holds every sweep's position after it (`state`):
the segment its next line is in (its first block's bit, and the CRCs folded before it),
the bytes of that segment's text before the line, and the sweep's running maximum; or
the sweep's end, with its blocks' CRCs folded. A rerun resumes each sweep from that
segment (at most SEGMENT_BYTES of bz2 decoded again, none from byte 0) and the merge
from the running maxima, so the lines that follow, and so the chunks, are the same.
"""

import bz2
import hashlib
import heapq
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import orjson
from wikidata_reshape import bz2_blocks

# A block's 48-bit magic, and the stream's end-of-stream mark (both may start at any bit)
BLOCK = 0x314159265359
EOS = 0x177245385090
MASK48 = (1 << 48) - 1
# bz2_blocks reports a block whose first WINDOW bytes it was given (reshape/src/bz2.rs)
WINDOW = 16
# A sweep's blocks go to a decoder of their own in segments of about this much bz2 (about
# 6.5 MB of text at a dump's ratio), each decoded whole: a restart decodes at most one
# segment a sweep again, and a sweep holds at most one segment's text beside its queue
SEGMENT_BYTES = 512 << 10
# Bytes of bz2 read at once
PIECE = 4 << 20
# A segment's text goes to the merge in pieces of TEXT_PIECE, a sweep's queue holding at
# most LOOKAHEAD of them. (Segments of 2 MB, their whole text queued 2 ahead, held about
# 100 MB a sweep: the split's peak RSS was 1.3-1.4 GB over byte order's on the bench's
# 8-shard dump. A decoder's text streamed as it came, its sweep letting its slot go while
# its queue was full, kept the memory down but doubled the decoders' CPU: a decoder whose
# output is not read still decodes into its own buffers, so 8 decoded at once again.)
TEXT_PIECE = 1 << 20
LOOKAHEAD = 4
# Segments decoded at once, over every sweep: as many as the cores (as many threads as
# today's one lbzip2 takes). The 8 decoders at once on 4 cores took 1.7 times the CPU of
# the same 8 run 4 at a time (the bench's 8-shard dump, 81 MB of bz2: 56 s against 33 s)
DECODERS = os.cpu_count() or 4
# The cuts: probes a sweep at even offsets, and bisection to this many bytes
PROBES_PER_SWEEP = 4
RESOLUTION = 128 << 10
PROBE_BYTES = 2 << 20
# The sweeps' cuts, in the split's state (dump.ROOT_DATA_DIR)
PLAN = "sweeps.json"
# The bytes a line's id is looked for in
ID_BYTES = 256


def _rotl(c: int, n: int) -> int:
    n %= 32
    return ((c << n) | (c >> (32 - n))) & 0xFFFFFFFF if n else c


def fold(c: int, crc: int) -> int:
    """bz2's stream CRC `c` with one more block's CRC: rotated left a bit, then XORed."""
    return _rotl(c, 1) ^ crc


def combine(a: int, n: int, b: int) -> int:
    """The stream CRC of blocks A then B, given A's (`a`), B's own from 0 (`b`) and B's
    count (`n`): `fold` is linear, so A's is rotated past B's blocks."""
    return _rotl(a, n) ^ b


def shard_of(entity_id: str, factor: int) -> int:
    """The dump shard of an entity: Wikibase's DumpGenerator::idMatchesShard,
    `hexdec(substr(sha1($id), 0, 8)) % shardingFactor` (dump.shard_of for 8)."""
    return int.from_bytes(hashlib.sha1(entity_id.encode()).digest()[:4], "big") % factor


def line_id(line: bytes) -> str | None:
    """An entity line's id (`{"type":"item","id":"Q42",...`), None if its first ID_BYTES
    have none."""
    i = line.find(b'"id":"', 0, ID_BYTES)
    if i < 0:
        return None
    j = line.find(b'"', i + 6, i + 6 + 64)
    return line[i + 6:j].decode() if j > 0 else None


def line_q(line: bytes) -> int:
    """An entity line's Q number, -1 if it is not a Q id (a property), as the merge orders
    by it (a line it cannot read is ordered as -1: the order is never correctness)."""
    i = line.find(b'"id":"', 0, ID_BYTES)
    if i < 0 or line[i + 6:i + 7] != b"Q":
        return -1
    j = line.find(b'"', i + 7, i + 7 + 24)
    try:
        return int(line[i + 7:j])
    except ValueError:
        return -1


class _Dummy:
    """A tiny block whose bits number `bits` (its residue mod 8 is what it is for), its
    CRC and its text."""

    def __init__(self, value: int, bits: int, crc: int, text: bytes):
        self.value, self.bits, self.crc, self.text = value, bits, crc, text


def _make_dummies() -> dict[int, _Dummy]:
    """For each residue mod 8, the first of a run of tiny texts whose one compressed block
    is that many bits mod 8 long, as libbz2 writes it (checked by decoding it back)."""
    out: dict[int, _Dummy] = {}
    n = 0
    while len(out) < 8:
        n += 1
        text = hashlib.sha256(b"%d" % n).hexdigest()[: 1 + n % 47].encode()
        s = bz2.compress(text, 9)
        x, total = int.from_bytes(s, "big"), len(s) * 8
        for pad in range(8):
            if (x >> (pad + 32)) & MASK48 == EOS and not x & ((1 << pad) - 1):
                break
        else:
            continue
        bits = total - pad - 80 - 32
        if bits % 8 in out:
            continue
        d = _Dummy((x >> (pad + 80)) & ((1 << bits) - 1), bits, (x >> pad) & 0xFFFFFFFF, text)
        alone = _pad(((int.from_bytes(b"BZh9", "big") << bits | d.value) << 80) | EOS << 32 | fold(0, d.crc),
                     32 + bits + 80)
        if bz2.decompress(alone) != text:
            continue
        out[bits % 8] = d
    return out


def _pad(value: int, bits: int) -> bytes:
    pad = -bits % 8
    return (value << pad).to_bytes((bits + pad) // 8, "big")


_DUMMIES: dict[int, _Dummy] = {}


def dummies() -> dict[int, _Dummy]:
    if not _DUMMIES:
        _DUMMIES.update(_make_dummies())
    return _DUMMIES


def _head(level: bytes, d: _Dummy, first: bytes, r: int) -> bytes:
    """A stream's first bytes: its header and the dummy block `d` (32 + d.bits ≡ r bits
    mod 8), merged into the file's byte `first` whose low 8 - r bits are where a block
    starts (its top r bits, the block before's, replaced by the dummy's last)."""
    prefix = int.from_bytes(b"BZh" + level, "big") << d.bits | d.value
    return _pad(prefix << (8 - r) | (first[0] & ((1 << (8 - r)) - 1)), 32 + d.bits + 8 - r)


def _tail(t: int, last: int, crc: int) -> bytes:
    """A stream's last bytes: the top `t` bits of the file's byte `last` (where the
    stream's last block ends), then the end-of-stream mark and the stream's CRC, padded."""
    return _pad((last >> (8 - t)) << 80 | EOS << 32 | crc, t + 80)


def stream(level: bytes, buf: bytes, at: int, start: int, end: int, crcs: list[int]) -> tuple[bytes, int]:
    """The blocks from bit `start` to bit `end` of the file (the bytes of `buf` from byte
    `at` on hold them) as a stream of their own, and the length of the dummy block's text
    that its decoded text starts with."""
    d = dummies()[start % 8]
    c = fold(0, d.crc)
    for crc in crcs:
        c = fold(c, crc)
    s, e = start // 8 - at, end // 8 - at
    return (_head(level, d, buf[s:s + 1], start % 8) + buf[s + 1:e]
            + _tail(end % 8, buf[e] if end % 8 else 0, c)), len(d.text)


class _File:
    """The dump, read by offset: its bz2 level, size and the stream's end (the end mark's
    bit, and the stream's CRC after it)."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        head = self.read(0, 4)
        if head[:3] != b"BZh" or not head[3:4].isdigit():
            raise SystemExit(f"{path} is not a bz2 stream: WIKIDATA_SPLIT_SWEEPS reads only bz2")
        self.level = head[3:4]
        tail = self.read(max(0, self.size - 16), 16)
        x = int.from_bytes(tail, "big")
        for pad in range(8):
            if (x >> (pad + 32)) & MASK48 == EOS and not x & ((1 << pad) - 1):
                self.eos = self.size * 8 - pad - 80
                self.crc = (x >> pad) & 0xFFFFFFFF
                break
        else:
            raise SystemExit(f"{path} does not end with a bz2 end-of-stream mark: is it whole?")

    def read(self, at: int, n: int) -> bytes:
        return os.pread(self.fd, n, at)

    def close(self) -> None:
        os.close(self.fd)

    def blocks(self, byte: int, n: int = PROBE_BYTES) -> tuple[bytes, list[tuple[int, int]]]:
        """The bytes from `byte`, and the blocks found in them (absolute bits, CRCs)."""
        buf = self.read(byte, n)
        return buf, [(byte * 8 + b, c) for b, c in bz2_blocks(buf) if byte * 8 + b < self.eos]

    def block_after(self, byte: int) -> tuple[int, int] | None:
        """The first block at or after `byte` (its bit and CRC), None if none is."""
        while byte * 8 < self.eos:
            _, hits = self.blocks(byte)
            if hits:
                return hits[0]
            byte += PROBE_BYTES - WINDOW
        return None

    def block_text(self, bit: int, crc: int) -> tuple[int, int | None, bytes]:
        """The block starting at `bit` (CRC `crc`) decoded alone: where it ends (the next
        block's bit, or the stream's end), that block's CRC (None at the end), its text."""
        n = PROBE_BYTES
        while True:
            buf, hits = self.blocks(bit // 8, n)
            after = [h for h in hits if h[0] > bit]
            if after or bit // 8 + n >= self.size:
                end, end_crc = after[0] if after else (self.eos, None)
                data, drop = stream(self.level, buf, bit // 8, bit, end, [crc])
                try:
                    text = bz2.decompress(data)[drop:]
                except (OSError, EOFError) as e:
                    raise RuntimeError(f"{self.path}: the block at bit {bit:,} does not decode ({e})") from e
                return end, end_crc, text
            n *= 2


class _Line:
    """Where an entity line starts: its block (bit, end, CRC), its offset in the block's
    text, and its id."""

    def __init__(self, bit: int, end: int, crc: int, offset: int, eid: str):
        self.bit, self.end, self.crc, self.offset, self.id = bit, end, crc, offset, eid

    def json(self, shard: int) -> dict:
        return {"shard": shard, "id": self.id, "bit": self.bit, "end": self.end, "crc": self.crc, "skip": self.offset}


def _line_starts(f: _File, block: tuple[int, int]):
    """The entity lines that start in the blocks from `block` (its bit and CRC) on, in
    order, each a _Line: from the first line that starts in it (the line its text begins
    in may have started in the block before)."""
    texts = []  # (bit, end, CRC, text, the next block's CRC) of each block decoded

    def decode(i: int) -> bool:
        while len(texts) <= i:
            bit, crc = block if not texts else (texts[-1][1], texts[-1][4])
            if crc is None:
                return False  # (the stream's end)
            end, end_crc, text = f.block_text(bit, crc)
            texts.append((bit, end, crc, text, end_crc))
        return True

    def ahead(i: int, p: int, n: int) -> bytes:
        out = texts[i][3][p:p + n]
        while len(out) < n and decode(i + 1):
            i += 1
            out += texts[i][3][:n - len(out)]
        return out

    i = 0
    if not decode(i):
        return
    while (nl := texts[i][3].find(b"\n")) < 0:  # (a block inside one line: none starts in it)
        i += 1
        if not decode(i):
            return
    p = nl + 1
    while True:
        text = texts[i][3]
        if p >= len(text):  # (the line starts the next block's text)
            i, p = i + 1, 0
            if not decode(i):
                return
            continue
        start = ahead(i, p, ID_BYTES)
        if start[:1] == b"{" and (eid := line_id(start)):
            yield _Line(texts[i][0], texts[i][1], texts[i][2], p, eid)
        while (q := text.find(b"\n", p)) < 0:  # (the line ends in a later block)
            i, p = i + 1, 0
            if not decode(i):
                return
            text = texts[i][3]
        p = q + 1


def plan(path: Path, factor: int) -> dict:
    """The sweeps' cuts in the dump at `path` (see the module docstring): for each sweep,
    its shard, its first line's id, block (bit, end, CRC) and offset in the block's text;
    and the dump's size, level and stream CRC."""
    f = _File(path)
    t = time.perf_counter()
    try:
        first = f.block_after(0)
        if first is None or first[0] != 32:
            raise SystemExit(f"{path}: its first block is not after its stream header")
        end, end_crc, _ = f.block_text(*first)
        sweeps = [{"shard": None, "id": None, "bit": 32, "end": end, "crc": first[1], "skip": 0}]
        probes = 0
        if factor > 1:
            cache: dict[int, _Line | None] = {}

            def at(x: int) -> _Line | None:
                nonlocal probes
                if x not in cache:
                    probes += 1
                    b = f.block_after(x)
                    cache[x] = None if b is None else next(_line_starts(f, b), None)
                return cache[x]

            floor = (32, -1)  # (where the last cut is: a line before it is the shards' before)

            def shard(line: _Line) -> int:
                return -1 if (line.bit, line.offset) < floor else shard_of(line.id, factor)

            def refuse(why: str):
                raise SystemExit(f"{path} is not {factor} shards one after another ({why}): unset "
                                 "WIKIDATA_SPLIT_SWEEPS to split it in byte order")

            xs = sorted({i * f.size // (PROBES_PER_SWEEP * factor) for i in range(PROBES_PER_SWEEP * factor)})
            head = at(0)
            if head is None:
                refuse("no entity line")
            a, lo = shard(head), 0
            sweeps[0] |= {"shard": a, "id": head.id}
            for x in xs[1:]:
                r = at(x)
                if r is None:
                    continue
                if 0 <= shard(r) < a:
                    refuse(f"shard {shard(r)} at byte {x:,} after shard {a}")
                while shard(r) > a:
                    hi = x
                    while hi - lo > RESOLUTION:
                        mid = (lo + hi) // 2
                        m = at(mid)
                        if m is None or shard(m) > a:
                            hi = mid
                        elif 0 <= shard(m) < a:
                            refuse(f"shard {shard(m)} at byte {mid:,} after shard {a}")
                        else:
                            lo = mid
                    b = f.block_after(lo)
                    for line in _line_starts(f, b):
                        if shard(line) > a:
                            break
                        if 0 <= shard(line) < a:
                            refuse(f"shard {shard(line)} ({line.id}) after shard {a}")
                    else:
                        refuse(f"no line of a shard after {a} past byte {lo:,}")
                    a = shard(line)
                    if line.bit == sweeps[-1]["bit"] and len(sweeps) == 1:
                        # (shard a's first line in the stream's first block: the shards before it, all
                        # in that block, are read with it)
                        sweeps[0] |= {"shard": a, "id": line.id}
                    elif line.bit == sweeps[-1]["bit"]:
                        # (the shard before it starts in the same block: it is read with the sweep
                        # before, so that each sweep starts in a block of its own)
                        sweeps[-1] = line.json(a)
                    else:
                        sweeps.append(line.json(a))
                    lo, floor = line.bit // 8, (line.bit, line.offset)
                lo = x
        return {"version": 1, "factor": factor, "dump": path.name, "bytes": f.size, "level": f.level.decode(),
                "eos": f.eos, "crc": f.crc, "probes": probes, "seconds": round(time.perf_counter() - t, 2),
                "sweeps": sweeps}
    finally:
        f.close()


def load_plan(state_dir: Path, path: Path, factor: int) -> dict:
    """The sweeps' cuts kept in the split's state (PLAN), or found now and kept there
    first (written whole, then moved into place), for a dump of the same size and factor."""
    p = state_dir / PLAN
    if p.exists():
        kept = orjson.loads(p.read_bytes())
        size = path.stat().st_size
        if kept["factor"] != factor or kept["bytes"] != size:
            raise SystemExit(f"{p} is for a dump of {kept['bytes']:,} bytes in {kept['factor']} shards, not "
                             f"{size:,} in {factor}: this split was begun on another dump or factor")
        return kept
    made = plan(path, factor)
    tmp = p.with_suffix(".tmp")
    with tmp.open("wb") as out:
        out.write(orjson.dumps(made, option=orjson.OPT_INDENT_2))
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(p)
    return made


class _Stopped(Exception):
    pass


class _Sweep:
    """One sweep's decoding, in a thread of its own: each segment built in memory
    (`_segments`) and decoded whole by its own `lbzip2 -dc -n1` (`_decode`) while the sweep
    holds one of the reader's decoder slots (`slots`, as many as the cores: more decoders
    than cores at once cost about 1.7 times the CPU), then, the slot let go, its text put in
    `text` a piece at a time as (segment, offset, bytes). `segment` is (its first block's bit, the sweep's CRCs folded before it, their
    count, the bytes of its text skipped). It ends with ("end", the sweep's CRCs folded,
    their count)."""

    def __init__(self, k: int, path: Path, level: bytes, size: int, start: dict, end: int, seam: bytes,
                 lbzip2: str, stop: threading.Event, slots: threading.Semaphore):
        self.k, self.path, self.level, self.size, self.end, self.seam = k, path, level, size, end, seam
        self.start, self.lbzip2, self.stop, self.slots = start, lbzip2, stop, slots
        self.text: queue.Queue = queue.Queue(LOOKAHEAD)
        self.live: set[subprocess.Popen] = set()
        self.fed = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"sweep{k}")
        self.thread.start()

    def _put(self, item) -> None:
        while True:
            try:
                self.text.put(item, timeout=0.2)
                return
            except queue.Full:
                if self.stop.is_set():
                    raise _Stopped from None

    def _segments(self, fd: int):
        """The sweep's segments, from its start: each (its info, its stream, the length of
        its dummy block's text), and at the end ("end", the sweep's CRCs folded, their
        count). A segment ends at the first block SEGMENT_BYTES or more past its first
        (or at the sweep's end), so a resumed sweep's are the same as before."""
        pos, crcs, n, skip = self.start["bit"], self.start["fold"], self.start["blocks"], self.start["at"]
        end, stop_at = self.end, min(self.size, self.end // 8 + WINDOW)
        buf, at = b"", pos // 8  # (the file's bytes from byte `at`, from the segment's first)
        read_at, last = at, pos - 1
        seg, info, limit = [], (pos, crcs, n, skip), pos + SEGMENT_BYTES * 8
        while read_at < stop_at:
            piece = os.pread(fd, min(PIECE, stop_at - read_at), read_at)
            if not piece:
                raise RuntimeError(f"{self.path} ended at byte {read_at:,}, before sweep {self.k}'s end")
            scan = max(0, len(buf) - WINDOW)  # (only what is new, and the bytes a block there needs)
            buf += piece
            read_at += len(piece)
            for b, crc in bz2_blocks(buf[scan:]):
                m = (at + scan) * 8 + b
                if m <= last:
                    continue
                if m >= end:
                    break
                if last < pos and m != pos:
                    raise RuntimeError(f"sweep {self.k}: no block at bit {pos:,} (found {m:,}): the cuts are not this "
                                       "dump's")
                last = m
                if m >= limit:
                    data, dummy = stream(self.level, buf, at, info[0], m, seg)
                    yield info, data, dummy
                    buf, scan, at = buf[m // 8 - at:], scan - (m // 8 - at), m // 8
                    seg, info, limit = [], (m, crcs, n, 0), m + SEGMENT_BYTES * 8
                seg.append(crc)
                crcs = fold(crcs, crc)
                n += 1
        data, dummy = stream(self.level, buf, at, info[0], end, seg)
        yield info, data, dummy
        yield "end", crcs, n

    def _acquire(self) -> None:
        while not self.slots.acquire(timeout=0.2):
            if self.stop.is_set():
                raise _Stopped

    def _decode(self, info: tuple, data: bytes, dummy: int) -> int:
        """A segment's stream decoded whole by its own `lbzip2 -n1`, holding a decoder slot,
        then its text put in `text` a piece at a time from its first line (`info`'s skip,
        past the dummy block's text), the slot let go: so at most DECODERS decode at once,
        and a sweep waiting for the merge never holds a slot. Returns its text's length."""
        fd = os.memfd_create(f"sweep{self.k}")  # (the stream as the decoder's input file)
        try:
            view, done = memoryview(data), 0
            while done < len(data):
                done += os.write(fd, view[done:])
            os.lseek(fd, 0, os.SEEK_SET)
            self._acquire()
            held = True
            try:
                if self.stop.is_set():
                    raise _Stopped
                proc = subprocess.Popen([self.lbzip2, "-dc", "-n1"], stdin=fd, stdout=subprocess.PIPE)
                self.live.add(proc)
                os.close(fd)
                fd = -1
                text = proc.stdout.read()
                proc.stdout.close()
                proc.wait()
                self.live.discard(proc)
            finally:
                if held:
                    self.slots.release()
            drop, read, offset = dummy + info[3], len(text), info[3]
            for at in range(min(drop, len(text)), len(text), TEXT_PIECE):
                piece = text[at:at + TEXT_PIECE]
                self._put((info, offset, piece))
                offset += len(piece)
            drop = max(0, drop - len(text))
            del text
        finally:
            if fd >= 0:
                os.close(fd)
        if proc.returncode:
            if self.stop.is_set():
                raise _Stopped
            raise RuntimeError(f"lbzip2 exited with {proc.returncode} on sweep {self.k}'s blocks: the dump is "
                               "corrupt (a block's or the segment's CRC)")
        self._drop = drop
        return read - dummy

    def _run(self) -> None:
        try:
            fd = os.open(self.path, os.O_RDONLY)
            try:
                drop, last, size = 0, None, 0  # (the bytes still to skip; the last segment, its text's length)
                for info, data, dummy in self._segments(fd):
                    if info == "end":
                        crcs, n = data, dummy
                        break
                    if drop:
                        raise RuntimeError(f"sweep {self.k}: a position past its segment's text")
                    try:
                        size = self._decode(info, data, dummy)
                    except RuntimeError as e:
                        raise RuntimeError(f"{e} (the segment from bit {info[0]:,})") from None
                    self.fed += len(data)
                    drop, last = self._drop, info
            finally:
                os.close(fd)
            if self.seam and last is not None:  # (the next sweep's first block, to its first line)
                if drop > len(self.seam):
                    raise RuntimeError(f"sweep {self.k}: a position past its end")
                if len(self.seam) > drop:
                    self._put((last, size + drop, self.seam[drop:]))
            self._put(("end", crcs, n))
        except _Stopped:
            pass
        except BaseException as e:
            try:
                self._put(("error", e))
            except _Stopped:
                pass

    def close(self) -> None:
        for proc in list(self.live):
            proc.kill()
        while self.thread.is_alive():
            try:  # (unblocking it, were it waiting to put)
                while True:
                    self.text.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=0.1)


class _Lines:
    """A sweep's entity lines (without the trailing comma), in order, each with its Q
    number (`line_q`), and where the sweep is after the last one taken (`pos`)."""

    def __init__(self, sweep: _Sweep | None, start: dict):
        self.sweep = sweep
        self.max = start["max"]
        self.done = start.get("done", False)
        self.final = (start["fold"], start["blocks"]) if self.done else None
        self.pos = (start.get("bit"), start["fold"], start["blocks"], start.get("at"))
        self.info, self.lines, self.qs, self.ends, self.i = None, [], [], [], 0
        self.partial = b""

    def _fill(self) -> bool:
        item = self.sweep.text.get()
        if item[0] == "error":
            raise item[1]
        if item[0] == "end":
            if self.partial.strip(b",\r\n").startswith(b"{"):
                raise RuntimeError(f"sweep {self.sweep.k} ended inside a line")
            self.final = item[1:]
            return False
        info, offset, data = item
        parts = data.split(b"\n")
        # (each part but the last ends with a newline, at these offsets)
        ends, e = [], offset
        for p in parts[:-1]:
            e += len(p) + 1
            ends.append(e)
        if parts[:-1]:
            parts[0] = self.partial + parts[0]
            self.partial = parts[-1]
        else:
            self.partial += parts[-1]
        self.info, self.lines, self.qs, self.ends, self.i = info, [], [], [], 0
        for p, e in zip(parts[:-1], ends):
            line = p.rstrip(b",\r")
            if line.startswith(b"{"):
                self.lines.append(line)
                self.qs.append(line_q(line))
                self.ends.append(e)
        return True

    def next(self) -> tuple[bytes, int] | None:
        while self.i == len(self.lines):
            if self.done or not self._fill():
                self.done = True
                return None
        i = self.i
        self.i += 1
        self.pos = (self.info[0], self.info[1], self.info[2], self.ends[i])
        return self.lines[i], self.qs[i]

    def state(self) -> dict:
        if self.done:
            return {"done": True, "fold": self.final[0], "blocks": self.final[1], "max": self.max}
        bit, crcs, n, at = self.pos
        return {"bit": bit, "fold": crcs, "blocks": n, "at": at, "max": self.max}


class Reader:
    """The dump's entity lines as synchronised sweeps (see the module docstring), from the
    start or from the sweeps' positions in the last line of the split's manifest
    (`done`, in chunk order from 0). `lines()` gives them; `state()` every sweep's
    position after the last line it gave, for that line's chunk."""

    def __init__(self, path: Path, factor: int, state_dir: Path, done: dict[int, dict]):
        lbzip2 = shutil.which("lbzip2")
        if not lbzip2:
            raise SystemExit("lbzip2 decodes the sweeps: apt install lbzip2")
        if any("sweeps" not in e for e in done.values()):
            raise SystemExit("This split was begun without WIKIDATA_SPLIT_SWEEPS (its manifest's lines have no "
                             "sweeps' positions): finish it without, or delete its chunks to split again")
        self.plan = load_plan(state_dir, path, factor)
        cuts = self.plan["sweeps"]
        last = done[max(done)]["sweeps"] if done else None
        if last is not None and len(last) != len(cuts):
            raise SystemExit(f"The split's manifest has {len(last)} sweeps, its plan {len(cuts)}")
        dummies()  # (made once, before the sweeps' threads take them)
        f = _File(path)
        try:
            if (f.eos, f.crc) != (self.plan["eos"], self.plan["crc"]):
                raise SystemExit(f"{path} ends with the stream CRC {f.crc:08x}, its cuts' {self.plan['crc']:08x}: the "
                                 "dump changed since this split began")
            # Each sweep's end: the first bytes of the next one's first block, to its first line
            seams = [f.block_text(c["bit"], c["crc"])[2][:c["skip"]] if c["skip"] else b"" for c in cuts[1:]] + [b""]
        finally:
            f.close()
        self.chunks = len(done)
        self.entities = sum(e["rows"] for e in done.values())
        self.stop = threading.Event()
        slots = threading.Semaphore(DECODERS)
        self.sweeps: list[_Lines] = []
        ends = [c["bit"] for c in cuts[1:]] + [self.plan["eos"]]
        for k, c in enumerate(cuts):
            start = last[k] if last else {"bit": c["bit"], "fold": 0, "blocks": 0, "at": c["skip"], "max": -1}
            sweep = None
            if not start.get("done"):
                sweep = _Sweep(k, path, self.plan["level"].encode(), self.plan["bytes"], start, ends[k], seams[k],
                               lbzip2, self.stop, slots)
            self.sweeps.append(_Lines(sweep, start))

    @property
    def fed(self) -> int:
        return sum(s.sweep.fed for s in self.sweeps if s.sweep)

    def lines(self):
        """The entity lines, merged: each from the sweep whose running maximum is least (see
        the module docstring). Once every sweep has ended, the stream CRC is checked."""
        try:
            heap = [(s.max, k) for k, s in enumerate(self.sweeps) if not s.done]
            heapq.heapify(heap)
            while heap:
                r, k = heapq.heappop(heap)
                s = self.sweeps[k]
                while (got := s.next()) is not None:
                    line, q = got
                    if q > r:
                        r = s.max = q
                    yield line
                    if heap and (r, k) > heap[0]:
                        heapq.heappush(heap, (r, k))
                        break
            self.check()
        finally:
            self.close()

    def state(self) -> list[dict]:
        return [s.state() for s in self.sweeps]

    def check(self) -> None:
        """The dump's stream CRC, from every block's CRC in file order (each sweep's own,
        combined), against the dump's."""
        c = 0
        for s in self.sweeps:
            crcs, n = s.final
            c = combine(c, n, crcs)
        if c != self.plan["crc"]:
            raise RuntimeError(f"The sweeps' blocks give the stream CRC {c:08x}, the dump's is "
                               f"{self.plan['crc']:08x}: a block was missed, repeated or corrupt")

    def close(self) -> None:
        self.stop.set()
        for s in self.sweeps:
            if s.sweep is not None:
                s.sweep.close()
