"""Byte-range probes of a Wikidata JSON dump's bz2 file (stdlib only).

A bz2 file is one or more streams (`BZh` + level, byte-aligned), each a run of blocks whose
48-bit magic (0x314159265359) is bit-aligned, ended by a 48-bit end-of-stream magic
(0x177245385090), a CRC and padding to a byte. A block can be decoded on its own: copy the
bits from its magic to the end into a new stream after a `BZh9` header (as bzip2recover
does). `probe(offset)` reads a little of the file at `offset`, finds the first block, and
decodes a few blocks, returning the entity ids of the whole lines in them, in order.
"""

import bz2
import re
import sys
import urllib.request

UA = "wikidata-pq/0.1 (https://github.com/lmmx/wikidata-pq)"
URL = "https://dumps.wikimedia.org/wikidatawiki/entities/{r}/wikidata-{r}-all.json.bz2"
BLOCK = 0x314159265359
EOS = 0x177245385090
ID = re.compile(rb'^\{"type":"(\w+)",(?:"datatype":"[^"]*",)?"id":"([A-Z])(\d+)"')
STREAM = re.compile(rb"BZh[1-9]1AY&SY")

READS = [0]  # bytes read


def fetch(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise SystemExit(f"range request ignored: {r.status}")
        data = r.read()
    READS[0] += len(data)
    return data


def bit_positions(buf: bytes, magic: int, limit: int | None = None) -> list[int]:
    """Bit offsets in `buf` where the 48-bit `magic` starts (any alignment), ascending."""
    n = len(buf) * 8
    x = int.from_bytes(buf, "big")
    hits = []
    for s in range(8):
        # y is buf shifted left by s bits in one more byte: buf's bit b is y's bit b + 8 - s
        y = (x << s).to_bytes(len(buf) + 1, "big")
        pat = magic.to_bytes(6, "big")
        i = y.find(pat)
        while i != -1:
            b = i * 8 - 8 + s  # y's bit 8i is buf's bit 8i - (8 - s)
            if 0 <= b and b + 48 <= n:
                hits.append(b)
            if limit and len(hits) > 4 * limit:
                break
            i = y.find(pat, i + 1)
    return sorted(set(hits))


def from_bit(buf: bytes, bit: int) -> bytes:
    """`buf` from bit `bit` on, left-aligned into bytes (the last byte zero-padded)."""
    n = len(buf) * 8
    x = int.from_bytes(buf, "big") & ((1 << (n - bit)) - 1)
    pad = (-(n - bit)) % 8
    return (x << pad).to_bytes((n - bit + pad) // 8, "big")


def decode_blocks(buf: bytes, bit: int, want: int = 1 << 20) -> bytes:
    """Decode the blocks from the block magic at `bit` on, until `want` bytes come out or
    the data ends (an incomplete last block yields nothing)."""
    data = b"BZh9" + from_bit(buf, bit)
    d = bz2.BZ2Decompressor()
    out = []
    got = 0
    step = 1 << 16
    for i in range(0, len(data), step):
        try:
            o = d.decompress(data[i:i + step])
        except (OSError, EOFError):
            break
        out.append(o)
        got += len(o)
        if got >= want or d.eof:
            break
    return b"".join(out)


def ids(text: bytes) -> list[tuple[str, str, int]]:
    """(type, letter, number) of each whole line of `text` (the first and last lines,
    which may be cut, are dropped)."""
    lines = text.split(b"\n")[1:-1]
    out = []
    for line in lines:
        m = ID.match(line)
        if m:
            out.append((m.group(1).decode(), m.group(2).decode(), int(m.group(3))))
    return out


def probe(url: str, offset: int, read: int = 1 << 20, want: int = 1 << 20) -> dict:
    """The ids decoded from the first block at or after `offset`, with what the read found:
    the block's bit offset in the file, any stream headers and end-of-stream marks."""
    buf = fetch(url, offset, read)
    blocks = bit_positions(buf, BLOCK, limit=8)
    eos = bit_positions(buf, EOS, limit=8)
    streams = [offset + m.start() for m in STREAM.finditer(buf)]
    res = {"offset": offset, "streams": streams, "eos": [offset * 8 + b for b in eos][:8],
           "blocks_in_read": len(blocks)}
    if not blocks:
        return {**res, "ids": [], "block_bit": None}
    text = decode_blocks(buf, blocks[0], want)
    return {**res, "block_bit": offset * 8 + blocks[0], "text_bytes": len(text), "ids": ids(text)}


if __name__ == "__main__":
    r = sys.argv[1]
    off = int(sys.argv[2])
    p = probe(URL.format(r=r), off)
    print({k: v for k, v in p.items() if k != "ids"}, len(p["ids"]), p["ids"][:5], p["ids"][-5:])
