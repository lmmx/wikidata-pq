"""What a hashing-only pass costs beside the decode (indicative: the shared machine).

On a bz2 file of whole blocks (bzip2recover's, concatenated) and its decoded text:
- the decode alone, `lbzip2 -dc -n1` to /dev/null, its CPU time (the children's clock);
- over the text, in memory: the lines found (memchr, `bytes.split`), each line's id read
  from its head (the split's own regex, bz2probe.ID) and its xxh3-64 (and, for scale,
  blake2b-64 and the whole text's xxh3 as one buffer: the hash's floor).

Each 3 rounds; the ratios to the decode are the evidence, not the absolute times.

    uv run --no-project --with xxhash python -I hash_bench.py all.bz2 all.json
"""

import hashlib
import os
import re
import subprocess
import sys
import time

import xxhash

ID = re.compile(rb'^\{"type":"(\w+)",(?:"datatype":"[^"]*",)?"id":"([A-Z])(\d+)"')


def cpu_children() -> float:
    t = os.times()
    return t.children_user + t.children_system


def main() -> None:
    bz, js = sys.argv[1], sys.argv[2]
    mb_bz = os.path.getsize(bz) / 1e6
    text = open(js, "rb").read()
    mb_txt = len(text) / 1e6
    print(f"bz2 {mb_bz:.1f} MB, text {mb_txt:.1f} MB (×{mb_txt / mb_bz:.1f})")
    for _ in range(3):
        c0 = cpu_children()
        subprocess.run(["lbzip2", "-dc", "-n1", bz], stdout=subprocess.DEVNULL, check=True)
        d = cpu_children() - c0
        t0 = time.process_time()
        lines = text.split(b"\n")
        t1 = time.process_time()
        n = 0
        for line in lines:
            if line.startswith(b"{"):
                m = ID.match(line)
                h = xxhash.xxh3_64_intdigest(line.rstrip(b","))
                n += 1
        t2 = time.process_time()
        for line in lines:
            hashlib.blake2b(line, digest_size=8).digest()
        t3 = time.process_time()
        xxhash.xxh3_64_intdigest(text)
        t4 = time.process_time()
        print(f"decode {d:.3f}s ({d / mb_bz * 1000:.1f} CPU-s a GB of bz2) | lines {t1 - t0:.3f}s | "
              f"id + xxh3 a line {t2 - t1:.3f}s ({n} lines) | blake2b a line {t3 - t2:.3f}s | xxh3 whole {t4 - t3:.4f}s | "
              f"lines+id+xxh3 over decode: {(t2 - t0) / d:.3f}")
        del lines


if __name__ == "__main__":
    main()
