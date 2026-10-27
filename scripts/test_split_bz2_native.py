"""split-dump with the native bz2 decoder (WIKIDATA_BZ2_DECODER=native) against lbzip2's:
every file the split writes (chunks, built tables, manifests, in both sets) the same byte
for byte; the same after a split killed partway and run again, with either decoder before
and after the kill (the split restarts from its manifest, decoding the dump from its start
and skipping the chunks it has); and a corrupted dump failing the split with either.

    .venv/bin/python scripts/test_split_bz2_native.py dump-slice.json.bz2

(bench/dumpslice.py makes a slice.) The split following its download, with the native
decoder: WIKIDATA_BZ2_DECODER=native scripts/test_split_follow.py.
"""

import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RELEASE = "20261012"
NAME = f"wikidata-{RELEASE}-all.json.bz2"


def start(work: Path, decoder: str, dump: Path) -> subprocess.Popen:
    d = work / "releases" / RELEASE / "dump"
    if not d.exists():
        d.mkdir(parents=True)
        os.link(dump, d / NAME)
    env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "WIKIDATA_BZ2_DECODER": decoder}
    env.pop("WIKIDATA_DUMP_FORMAT", None)
    err = open(work / "split.err", "a")
    return subprocess.Popen([sys.executable, "-c", "from wikidata import dump; dump.split()"], cwd=work, env=env,
                            stdout=subprocess.DEVNULL, stderr=err)


def files(work: Path) -> dict[str, str]:
    """Every file the split wrote (the dump aside), by its path: its sha256"""
    root = work / "releases"
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.parent.name != "dump"}


def killed_run(work: Path, first: str, then: str, dump: Path, after_lines: int) -> int:
    """A split by `first`, killed once its manifest has `after_lines` lines, then run again
    by `then` to its end. Returns the manifest's lines at the kill."""
    manifest = work / "releases" / RELEASE / "data" / "manifest.split.jsonl"
    p = start(work, first, dump)
    while p.poll() is None:
        if manifest.exists() and len(manifest.read_bytes().splitlines()) >= after_lines:
            p.send_signal(signal.SIGKILL)
            break
        time.sleep(0.05)
    p.wait()
    lines = len(manifest.read_bytes().splitlines()) if manifest.exists() else 0
    assert not (work / "releases" / RELEASE / "data" / "split.done").exists(), "the split ended before the kill"
    assert start(work, then, dump).wait() == 0, (work / "split.err").read_text()[-3000:]
    return lines


def main(dump: Path) -> int:
    failed = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        tmp = Path(tmp)
        t = time.perf_counter()
        assert start(tmp / "lbzip2", "lbzip2", dump).wait() == 0, (tmp / "lbzip2" / "split.err").read_text()[-3000:]
        want = files(tmp / "lbzip2")
        print(f"lbzip2: {len(want)} files in {time.perf_counter() - t:.1f}s", flush=True)
        t = time.perf_counter()
        assert start(tmp / "native", "native", dump).wait() == 0, (tmp / "native" / "split.err").read_text()[-3000:]
        got = files(tmp / "native")
        ok = got == want
        failed += not ok
        print(f"native: {len(got)} files in {time.perf_counter() - t:.1f}s, "
              f"{'byte for byte the same' if ok else 'DIFFERENT: ' + str(sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))[:10])}",
              flush=True)
        for first, then in (("native", "native"), ("lbzip2", "native"), ("native", "lbzip2")):
            work = tmp / f"killed-{first}-{then}"
            lines = killed_run(work, first, then, dump, 2)
            got = files(work)
            ok = got == want
            failed += not ok
            print(f"{first} killed after {lines} chunks, run again by {then}: "
                  f"{'byte for byte the same' if ok else 'DIFFERENT'}", flush=True)
        # a corrupted dump: a byte flipped in the middle (a block's Huffman-coded data)
        bad = tmp / "bad.json.bz2"
        data = bytearray(dump.read_bytes())
        data[len(data) // 2] ^= 0x10
        bad.write_bytes(data)
        for decoder in ("lbzip2", "native"):
            work = tmp / f"bad-{decoder}"
            code = start(work, decoder, bad).wait()
            err = (work / "split.err").read_text()
            said = err.strip().splitlines()[-1] if err.strip() else ""
            ok = code != 0 and not (work / "releases" / RELEASE / "data" / "split.done").exists()
            failed += not ok
            print(f"corrupted dump, {decoder}: exit {code}{'' if ok else ' FAILED'}: {said[:200]}", flush=True)
    print("FAILED" if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
