"""Xet's chunking, hashing and upload-time deduplication, ported from xet-core v1.6.0
(github.com/huggingface/xet-core, tag v1.6.0, de71453), for measuring offline how much of
a release's upload Xet would deduplicate against the release before. Nothing here talks
to the Hub.

- **Chunks** (`chunk_ends`): the content-defined chunker in xetchunk.c, a C port of
  `xet_data/src/deduplication/chunking.rs` (64 KiB target, 8 KiB min, 128 KiB max, the
  gearhash crate's table and rolling hash). Built on first use with the system's cc, to
  the temporary directory (or XETCHUNK_SO).
- **Chunk hashes** (`chunk_hash`): BLAKE3 keyed with DATA_KEY
  (`xet_core_structures/src/merklehash/data_hash.rs:272-276, 294-297`).
- **File and xorb hashes** (`file_hash`, `xorb_hash`): the aggregated merkle hash
  (`merklehash/aggregated_hashes.rs`): nodes merged in groups cut where a hash % 4 == 0
  (at least 2, at most 9 a group), each group hashed as lines of "{hex} : {size}\\n" by
  BLAKE3 keyed with INTERNAL_NODE_HASH (`data_hash.rs:278-281, 320-323`); the file hash
  is that root keyed again with a zero salt (`aggregated_hashes.rs:183-195`).
- **Deduplication** (`Uploader`): `FileDeduper::process_chunks`
  (`xet_data/src/deduplication/file_deduplication.rs:80-298`) per file, against the
  shards it knows: a dedup range is a run of consecutive chunks of one xorb
  (`metadata_shard/shard_format.rs:673-757`), the file's own pending xorb is searched
  too (`dedup_query_against_local_data`), and the fragmentation guard
  (`defrag_prevention.rs`; `xet_runtime/src/config/groups/deduplication.rs`: 128
  ranges, 8 chunks a range, hysteresis 0.5) may refuse a short range. New chunks go into
  the file's xorb, cut at 64 MiB or 8,192 chunks (`xorb_object/constants.rs:17, 22`).
  What the shards know is the scenario's (`known`): every chunk of the release before
  (its shards in the local shard cache: the same machine, within the cache's 3 weeks),
  or only what global dedup queries fetch (`GlobalIndex`).

The hash helpers need the `blake3` package (`uv run --with blake3`).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

TARGET_CHUNK_SIZE = 64 * 1024  # xorb_object/constants.rs:5
MINIMUM_CHUNK_DIVISOR = 8  # :10
MAXIMUM_CHUNK_MULTIPLIER = 2  # :14
MAX_XORB_BYTES = 64 * 1024 * 1024  # :17
MAX_XORB_CHUNKS = 8 * 1024  # :22
MIN_CHUNK = TARGET_CHUNK_SIZE // MINIMUM_CHUNK_DIVISOR
MAX_CHUNK = TARGET_CHUNK_SIZE * MAXIMUM_CHUNK_MULTIPLIER

# merklehash/data_hash.rs:272-281 ("as generated from random.org")
DATA_KEY = bytes([102, 151, 245, 119, 91, 149, 80, 222, 49, 53, 203, 172, 165, 151, 24, 28, 157, 228, 33, 16,
                  155, 235, 43, 88, 180, 208, 176, 75, 147, 173, 242, 41])
INTERNAL_NODE_KEY = bytes([1, 126, 197, 199, 165, 71, 41, 150, 253, 148, 102, 102, 180, 138, 2, 230, 93, 221, 83,
                           111, 55, 199, 109, 210, 248, 99, 82, 230, 74, 83, 113, 63])
BRANCHING = 4  # aggregated_hashes.rs:3 (groups of 2 to 2 * 4 + 1)

# metadata_shard/constants.rs:8: a chunk is global-dedup eligible when hash % 1024 == 0
GLOBAL_DEDUP_MODULUS = 1024
# processing/constants.rs:6: at least 256 chunks (~16 MiB) between a file's global queries
MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES = 256
# processing/constants.rs:31: a file is read and chunked 8 MiB at a time; each block's chunks
# are one process_chunks call (file_upload_session.rs:155-183)
INGESTION_BLOCK_SIZE = 8 * 1024 * 1024
# xet_runtime/src/config/groups/deduplication.rs:7, 19, 25
NRANGES_IN_ESTIMATOR, MIN_N_CHUNKS_PER_RANGE, HYSTERESIS = 128, 8.0, 0.5


# -- chunking ----------------------------------------------------------------------------

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        c = HERE / "xetchunk.c"
        tag = hashlib.sha256(c.read_bytes()).hexdigest()[:12]
        so = Path(os.environ.get("XETCHUNK_SO") or Path(tempfile.gettempdir()) / f"xetchunk-{tag}.so")
        if not so.exists():
            tmp = so.with_suffix(f".{os.getpid()}.tmp")
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", str(tmp), str(c)], check=True)
            tmp.replace(so)
        lib = ctypes.CDLL(str(so))
        lib.xet_chunk.restype = ctypes.c_size_t
        lib.xet_chunk.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        _LIB = lib
    return _LIB


def chunk_ends(data: bytes, target: int = TARGET_CHUNK_SIZE) -> list[int]:
    """Each chunk's end offset in `data`, as Xet's chunker cuts it (the last is len(data);
    none for empty data)."""
    if not data:
        return []
    cap = len(data) // (target // 8) + 2
    ends = (ctypes.c_size_t * cap)()
    k = _lib().xet_chunk(data, len(data), target, ends, cap)
    return list(ends[:k])


# -- hashes ------------------------------------------------------------------------------

def _blake3():
    import blake3

    return blake3.blake3


def chunk_hash(data: bytes | memoryview) -> bytes:
    """compute_data_hash: BLAKE3 keyed with DATA_KEY, as 32 raw bytes."""
    return _blake3()(data, key=DATA_KEY).digest()


def chunks_of(data: bytes) -> list[tuple[bytes, int]]:
    """(hash, size) of each chunk of `data`, in order."""
    b3, mv, out, s = _blake3(), memoryview(data), [], 0
    for e in chunk_ends(data):
        out.append((b3(mv[s:e], key=DATA_KEY).digest(), e - s))
        s = e
    return out


def hexh(h: bytes) -> str:
    """DataHash::hex: the four little-endian u64 words, each as 16 hex digits."""
    return "".join(f"{int.from_bytes(h[i:i + 8], 'little'):016x}" for i in range(0, 32, 8))


def from_hex(s: str) -> bytes:
    return b"".join(int(s[i:i + 16], 16).to_bytes(8, "little") for i in range(0, 64, 16))


def h_mod(h: bytes, m: int) -> int:
    """DataHash % m: its last u64 word (data_hash.rs:123-125)."""
    return int.from_bytes(h[24:32], "little") % m


def _merged(group: list[tuple[bytes, int]]) -> tuple[bytes, int]:
    buf = "".join(f"{hexh(h)} : {s}\n" for h, s in group).encode()
    return _blake3()(buf, key=INTERNAL_NODE_KEY).digest(), sum(s for _, s in group)


def _next_cut(hv: list[tuple[bytes, int]], start: int) -> int:
    n = len(hv) - start
    if n <= 2:
        return n
    end = min(2 * BRANCHING + 1, n)
    for i in range(2, end):
        if h_mod(hv[start + i][0], BRANCHING) == 0:
            return i + 1
    return end


def aggregated_hash(chunks: list[tuple[bytes, int]]) -> bytes:
    if not chunks:
        return bytes(32)
    hv = list(chunks)
    while len(hv) > 1:
        out, i = [], 0
        while i < len(hv):
            c = _next_cut(hv, i)
            out.append(_merged(hv[i:i + c]))
            i += c
        hv = out
    return hv[0][0]


def xorb_hash(chunks: list[tuple[bytes, int]]) -> bytes:
    return aggregated_hash(chunks)


def file_hash(chunks: list[tuple[bytes, int]], salt: bytes = bytes(32)) -> bytes:
    if not chunks:
        return bytes(32)
    return _blake3()(aggregated_hash(chunks), key=salt).digest()


# -- deduplication -----------------------------------------------------------------------

@dataclass
class Xorb:
    chunks: list[tuple[bytes, int]]


@dataclass
class Metrics:
    """The DeduplicationMetrics hf_xet keeps (dedup_metrics.rs), per file or summed:
    `deduped` counts every matched range, the refused ones too (`refused`); `new` is what
    goes into new xorbs, those refused chunks included. So the bytes uploaded are `new`,
    and deduplicated are deduped - refused."""
    total: int = 0
    deduped: int = 0
    refused: int = 0
    new: int = 0
    global_queries: int = 0
    by_global: int = 0

    def add(self, o: "Metrics") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))


class Index:
    """The chunk lookup of a set of shards: chunk hash -> (xorb, index in it), at most 8
    places a hash (shard_format.rs:747)."""

    def __init__(self) -> None:
        self.xorbs: list[Xorb] = []
        self.where: dict[bytes, list[tuple[int, int]]] = {}

    def add_xorb(self, x: Xorb) -> int:
        xi = len(self.xorbs)
        self.xorbs.append(x)
        for i, (h, _) in enumerate(x.chunks):
            places = self.where.setdefault(h, [])
            if len(places) < 8:
                places.append((xi, i))
        return xi

    def query(self, hashes: list[bytes], start: int) -> tuple[int, int, int, int] | None:
        """chunk_hash_dedup_query: the longest run of hashes[start:] matching consecutive
        chunks of one xorb from where the first is found, as (n chunks, bytes, the xorb's
        key, chunk start)."""
        for xi, ci in self.where.get(hashes[start], ()):
            xc = self.xorbs[xi].chunks
            n, b = 0, 0
            while start + n < len(hashes) and ci + n < len(xc) and xc[ci + n][0] == hashes[start + n]:
                b += xc[ci + n][1]
                n += 1
            if n:
                return n, b, id(self.xorbs[xi]), ci
        return None


class Defrag:
    """DefragPrevention (defrag_prevention.rs), per file."""

    def __init__(self) -> None:
        self.last: list[int] = []
        self.total = 0
        self.low = True

    def inc_last(self, n: int) -> None:
        if self.last:
            self.last[-1] += n
            self.total += n

    def add_range(self, n: int) -> None:
        self.last.append(n)
        self.total += n
        if len(self.last) > NRANGES_IN_ESTIMATOR:
            self.total -= self.last.pop(0)

    def allow(self, n: int) -> bool:
        if len(self.last) < NRANGES_IN_ESTIMATOR:
            return True
        cpr = self.total / len(self.last)
        target = MIN_N_CHUNKS_PER_RANGE * HYSTERESIS if self.low else MIN_N_CHUNKS_PER_RANGE
        if cpr < target:
            if n < cpr:
                self.low = False
                return False
        else:
            self.low = True
        return True


class GlobalIndex:
    """The Hub's global dedup index, as the client sees it (file_deduplication.rs:156-185;
    shard_interface/native.rs:87-103). The Hub indexes each uploaded shard's eligible
    chunks: a file's first chunk, and every chunk whose hash % 1024 == 0
    (streaming_shard.rs:457-500). A client queries a chunk it can't yet deduplicate when it
    is eligible (its file's first, or hash % 1024 == 0) and at least 256 chunks after its
    file's last query; a hit returns a shard, whose xorbs then deduplicate. `respond` is
    what that shard holds, which the client doesn't decide: "xorb" (only the xorb holding
    the chunk) or "session" (every xorb of the upload session, the commit, that uploaded
    it)."""

    def __init__(self, respond: str = "xorb") -> None:
        self.respond = respond
        self.sessions: list[list[Xorb]] = []
        self.by_chunk: dict[bytes, tuple[int, int]] = {}

    def add_session(self, xorbs: list[Xorb], file_firsts: set[bytes]) -> None:
        si = len(self.sessions)
        self.sessions.append(list(xorbs))
        for xi, x in enumerate(xorbs):
            for h, _ in x.chunks:
                if h_mod(h, GLOBAL_DEDUP_MODULUS) == 0 or h in file_firsts:
                    self.by_chunk.setdefault(h, (si, xi))

    def lookup(self, h: bytes) -> list[Xorb]:
        if h not in self.by_chunk:
            return []
        si, xi = self.by_chunk[h]
        return list(self.sessions[si]) if self.respond == "session" else [self.sessions[si][xi]]


class Uploader:
    """One upload session's deduplication (FileUploadSession), file by file: its own new
    xorbs (the session shard), and the shards it knows (`known`: the local shard cache's,
    and any a global dedup query fetched). `upload` returns the file's Metrics; `xorbs`
    and `firsts` are what the session's shard registers (its xorbs, its files' first
    chunks), for the uploads after it."""

    def __init__(self, known: Index | None = None, global_index: GlobalIndex | None = None) -> None:
        self.known = known if known is not None else Index()
        self.session = Index()
        self.global_index = global_index
        self.fetched: set[int] = set()
        self.xorbs: list[Xorb] = []
        self.firsts: set[bytes] = set()

    def upload(self, data: bytes) -> Metrics:
        return self.upload_chunks(chunks_of(data))

    def upload_chunks(self, chunks: list[tuple[bytes, int]]) -> Metrics:
        m = Metrics()
        d = Defrag()
        hashes = [h for h, _ in chunks]
        if hashes:
            self.firsts.add(hashes[0])
        pending: list[tuple[bytes, int]] = []
        pending_at: dict[bytes, int] = {}
        pending_bytes = 0
        MARK = -1  # (MerkleHash::marker(): the file's xorb not yet cut)
        last: list[int] | None = None  # the file's last entry: [xorb key, chunk start, chunk end]
        next_global = 0

        def cut() -> None:
            nonlocal pending, pending_at, pending_bytes
            x = Xorb(pending)
            self.session.add_xorb(x)
            self.xorbs.append(x)
            if last is not None and last[0] == MARK:  # (cut_new_xorb fills in the marker entries)
                last[0] = id(x)
            pending, pending_at, pending_bytes = [], {}, 0

        # Each ingestion block's chunks are one process_chunks call (the chunker's own
        # buffering moves a block's last partial chunk into the next; taken here by bytes)
        blocks, s0, acc = [], 0, 0
        for i, (_, size) in enumerate(chunks):
            acc += size
            if acc >= INGESTION_BLOCK_SIZE:
                blocks.append((s0, i + 1))
                s0, acc = i + 1, 0
        if s0 < len(chunks):
            blocks.append((s0, len(chunks)))

        for b0, b1 in blocks:
            sub = hashes[:b1]
            found: dict[int, tuple[int, int, int, int]] = {}  # start -> (n, bytes, xorb key, chunk start)
            for first_pass in (True, False):
                i, new_shards = b0, False
                while i < b1:
                    if i in found:
                        i += found[i][0]
                        continue
                    r = None
                    for idx in (self.session, self.known):
                        r = idx.query(sub, i)
                        if r:
                            break
                    if r:
                        if not first_pass:
                            m.by_global += r[1]
                        found[i] = r
                        i += r[0]
                        continue
                    if (first_pass and self.global_index is not None and i >= next_global
                            and (i == 0 or h_mod(hashes[i], GLOBAL_DEDUP_MODULUS) == 0)):
                        m.global_queries += 1
                        next_global = i + MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES
                        for x in self.global_index.lookup(hashes[i]):
                            if id(x) not in self.fetched:
                                self.fetched.add(id(x))
                                self.known.add_xorb(x)
                                new_shards = True
                    i += 1
                if not new_shards:
                    break
            i = b0
            while i < b1:
                q = found.pop(i, None)
                if q is None and hashes[i] in pending_at:  # dedup_query_against_local_data
                    base = pending_at[hashes[i]]
                    n, nb = 1, pending[base][1]
                    while i + n < b1 and pending_at.get(hashes[i + n]) == base + n:
                        nb += pending[base + n][1]
                        n += 1
                    q = (n, nb, MARK, base)
                if q is not None:
                    n, nb, key, start = q
                    m.deduped += nb
                    m.total += nb
                    continues = last is not None and last[0] == key and last[2] == start
                    if continues or d.allow(n):
                        if continues:
                            d.inc_last(n)
                            last[2] = start + n
                        else:
                            d.add_range(n)
                            last = [key, start, start + n]
                        i += n
                        continue
                    m.refused += nb
                # a new chunk
                h, size = chunks[i]
                m.total += size
                m.new += size
                if pending_bytes + size > MAX_XORB_BYTES or len(pending) + 1 > MAX_XORB_CHUNKS:
                    cut()
                if last is not None and last[0] == MARK and last[2] == len(pending):
                    d.inc_last(1)
                    last[2] += 1
                else:
                    d.add_range(1)
                    last = [MARK, len(pending), len(pending) + 1]
                pending_at[h] = len(pending)
                pending.append((h, size))
                pending_bytes += size
                i += 1
        if pending:
            cut()
        return m
