"""Checks the port (xet.py, xetchunk.c) against xet-core's reference vectors
(test_vectors.json, from vectors.py) and, given files, against hf_xet's own file hashes.

    uv run --no-project --with blake3 python -I test_xet.py
    uv run --no-project --with blake3 --with hf_xet==1.6.0 python -I test_xet.py FILE...

hf_xet.hash_files is "pure local computation, no authentication or server connection"
(xet_data/src/processing/data_client.rs:160-173): the same chunker and file hash an upload
computes, nothing sent. A matching file hash checks every chunk boundary and every chunk
hash of the file at once. Its cache and logs are pointed at a temporary directory.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xet  # noqa: E402

V = json.loads((Path(__file__).resolve().parent / "test_vectors.json").read_text())
M64 = (1 << 64) - 1


def splitmix64_data(n: int, seed: int) -> bytes:
    """create_random_data (chunking.rs tests): SplitMix64 words, little-endian."""
    out, state = bytearray(), seed
    while len(out) < n:
        state = (state + 0x9E3779B97F4A7C15) & M64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out += (z ^ (z >> 31)).to_bytes(8, "little")
    return bytes(out[:n])


def triggering(pattern: list[int], n: int, padding: int) -> bytes:
    """get_triggering_base_data (chunking.rs tests)."""
    data = bytearray(pattern) + bytes(padding)
    while len(data) < n:
        take = min(n - len(data), len(data))
        data += data[:take]
    return bytes(data)


def check_vectors() -> int:
    c, fails = V["chunker"], 0
    r = c["random_1mb"]
    data = splitmix64_data(r["n"], r["seed"])
    assert all(data[i] == b for i, b in r["samples"]), "SplitMix64 data differs from xet-core's"
    fails += xet.chunk_ends(data) != r["boundaries"]
    k = c["const_1mb"]
    fails += xet.chunk_ends(bytes([k["byte"]]) * k["n"]) != k["boundaries"]
    h = c["hitting"]
    for case in h["cases"]:
        data = triggering(h["pattern"], h["n"], case["padding"])
        assert data[11111] == case["sample_at_11111"], "triggering data differs from xet-core's"
        fails += xet.chunk_ends(data) != case["boundaries"]
    n_chunk = 2 + len(h["cases"])
    for case in V["merkle"]:
        nodes = [(xet.from_hex(hx), s) for hx, s in case["nodes"]]
        fails += xet.hexh(xet.xorb_hash(nodes)) != case["xorb_hash"]
        for f in case["file_hashes"]:
            fails += xet.hexh(xet.file_hash(nodes, xet.from_hex(f["salt"]))) != f["hash"]
    fails += xet.hexh(bytes(V["hex"]["raw"])) != V["hex"]["hex"]
    n_merkle = sum(1 + len(c["file_hashes"]) for c in V["merkle"])
    print(f"xet-core vectors: {n_chunk} chunker, {n_merkle} merkle, 1 hex: {fails} failed")
    return fails


def check_files(paths: list[str]) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(HF_HOME=tmp, HF_XET_CACHE=tmp, HF_XET_LOG_DEST=tmp + os.sep, HF_HUB_OFFLINE="1")
        import hf_xet

        theirs = hf_xet.hash_files([str(Path(p).resolve()) for p in paths])
    fails, nbytes = 0, 0
    for p, t in zip(paths, theirs):
        data = Path(p).read_bytes()
        nbytes += len(data)
        ours = xet.hexh(xet.file_hash(xet.chunks_of(data)))
        if ours != t.hash or len(data) != t.file_size:
            fails += 1
            print(f"MISMATCH {p}: ours {ours}, hf_xet {t.hash}")
    from importlib.metadata import version

    print(f"hf_xet {version('hf_xet')} hash_files: {len(paths)} files, {nbytes / 1e6:.1f} MB: "
          f"{fails} differ")
    return fails


if __name__ == "__main__":
    bad = check_vectors()
    if sys.argv[1:]:
        bad += check_files(sys.argv[1:])
    sys.exit(1 if bad else 0)
