"""xet-core's own reference vectors, extracted from its test source into test_vectors.json
(stdlib only). Run once against a checkout of huggingface/xet-core at v1.6.0:

    git clone --depth 1 --branch v1.6.0 https://github.com/huggingface/xet-core.git
    python3 -I vectors.py xet-core test_vectors.json

- **The chunker** (`xet_data/src/deduplication/chunking.rs`, its tests):
  `test_correctness_1mb_random_data` (1,000,000 bytes of SplitMix64 from seed 0, three
  sample bytes and the chunk boundaries), `test_correctness_1mb_const_data`, and
  `test_correctness_100kb_hitting_data` (a 128-byte pattern known to trigger a boundary,
  padded with 0 to 128 + i bytes and repeated to 65,536 bytes, for i in 0..128: a sample
  byte and the boundaries of each). All at the default 64 KiB target.
- **The merkle hashes** (`xet_core_structures/src/merklehash/aggregated_hashes.rs`,
  `test_correctness`: "intended to be used as a reference to ensure that other
  implementations or ports of these functions produce the correct hashes"): node lists,
  their xorb hash, and their file hash under three salts.
- **The hex form** (`merklehash/data_hash.rs`, `test_hash_hex_string_endianness`).
"""

import ast
import json
import re
import sys
from pathlib import Path


def rust_literal(text: str):
    return ast.literal_eval(text.replace("vec![", "["))


def block_after(src: str, marker: str, open_: str = "vec![") -> str:
    """The bracketed literal starting at the first `open_` after `marker`."""
    i = src.index(open_, src.index(marker) + len(marker) - len(open_))
    depth, j = 0, i + len(open_) - 1
    for j in range(i + len(open_) - 1, len(src)):
        depth += {"[": 1, "]": -1}.get(src[j], 0)
        if depth == 0:
            break
    return src[i:j + 1]


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    chunking = (root / "xet_data/src/deduplication/chunking.rs").read_text()
    agg = (root / "xet_core_structures/src/merklehash/aggregated_hashes.rs").read_text()
    dh = (root / "xet_core_structures/src/merklehash/data_hash.rs").read_text()

    rnd = chunking[chunking.index("fn test_correctness_1mb_random_data"):]
    samples = [tuple(map(int, m)) for m in re.findall(r"assert_eq!\(data\[(\d+)\], (\d+)\);", rnd[:2000])]
    random_1mb = {"n": 1000000, "seed": 0, "samples": samples,
                  "boundaries": rust_literal(block_after(rnd, "assert_eq!(\n            chunk_boundaries"))}
    const = chunking[chunking.index("fn test_correctness_1mb_const_data"):]
    const_1mb = {"n": 1000000, "byte": int(re.search(r"vec!\[(\d+)u8; 1000000\]", const).group(1)),
                 "boundaries": rust_literal(block_after(const, "assert_eq!(chunk_boundaries"))}
    hit = chunking[chunking.index("fn get_triggering_base_data"):]
    pattern = rust_literal(block_after(hit, "let mut data = vec!["))
    hitting = []
    for i in range(128):
        sample = int(re.search(rf"data_sample_at_11111\[{i}\] = (\d+);", chunking).group(1))
        cb = rust_literal(re.search(rf"ref_cb\[{i}\] = (vec!\[[\d, ]+\]);", chunking).group(1))
        hitting.append({"padding": i, "sample_at_11111": sample, "boundaries": cb})
    ref = rust_literal(block_after(agg[agg.index("fn test_correctness"):], "let reference = vec!["))
    merkle = [{"nodes": [list(x) for x in nodes], "xorb_hash": cas,
               "file_hashes": [{"salt": a, "hash": b} for a, b in files]} for nodes, cas, files in ref]
    e = dh[dh.index("fn test_hash_hex_string_endianness"):]
    hexform = {"raw": rust_literal(block_after(e, "let hash_raw: [u8; 32] = [", "[")),
               "hex": re.search(r'expected_hex_string = "([0-9a-f]{64})"', e).group(1)}
    out.write_text(json.dumps({
        "source": "huggingface/xet-core v1.6.0 (de71453): xet_data/src/deduplication/chunking.rs, "
                  "xet_core_structures/src/merklehash/aggregated_hashes.rs, data_hash.rs (tests)",
        "chunker": {"random_1mb": random_1mb, "const_1mb": const_1mb,
                    "hitting": {"pattern": pattern, "n": 65536, "cases": hitting}},
        "merkle": merkle, "hex": hexform}, indent=1))
    print(f"{out}: {len(hitting)} hitting cases, {len(merkle)} merkle cases")


if __name__ == "__main__":
    main()
