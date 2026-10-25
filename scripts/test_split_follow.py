"""split-dump following download-dump, against a stand-in for dumps.wikimedia.org (a local
server with Range requests, throttled): the split run alongside the download, started
before it, and with the download killed partway and run again, each giving the same chunk
files and manifests as a split of the downloaded dump; and a download failing its md5
check stops the split, its chunks deleted.

    .venv/bin/python scripts/test_split_follow.py dump-slice.json.bz2 [bytes per second]

(bench/dumpslice.py makes a slice; at the default 5 MB/s its 77 MB take about 15 s.)
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl

RELEASE = "20261012"
NAME = f"wikidata-{RELEASE}-all.json.bz2"


def serve(data: bytes, md5: str, rate: float) -> str:
    """A server for the release's dump and md5sums; returns its base URL"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def body(self) -> tuple[bytes | None, bool]:
            if self.path == f"/{RELEASE}/wikidata-{RELEASE}-md5sums.txt":
                return f"{md5}  {NAME}\n".encode(), False
            return (data, True) if self.path == f"/{RELEASE}/{NAME}" else (None, False)

        def do_HEAD(self):
            body, _ = self.body()
            if body is None:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()

        def do_GET(self):
            body, ranged = self.body()
            if body is None:
                return self.send_error(404)
            start = 0
            if ranged and (r := self.headers.get("Range")):
                start = int(r.split("=")[1].split("-")[0])
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            step = 1 << 18
            try:
                for i in range(start, len(body), step):
                    self.wfile.write(body[i : i + step])
                    if ranged:
                        time.sleep(step / rate)
            except (BrokenPipeError, ConnectionResetError):
                pass  # a download killed partway

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def step(work: Path, name: str, base: str) -> subprocess.Popen:
    code = {
        "download": f"from wikidata import dump; dump.DUMPS_URL = {base!r}; dump.download({base!r})",
        "split": "from wikidata import dump; dump.split()",
    }[name]
    env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
    err = open(work / f"{name}.err", "a")
    return subprocess.Popen([sys.executable, "-c", code], cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=err)


def state(work: Path) -> dict:
    """Each set's chunk files (as frames) and manifest (its bytes aside)"""
    out = {}
    for s in (RELEASE, f"{RELEASE}-scholar"):
        data = work / "releases" / s / "data"
        manifest = [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines()]
        out[s] = (
            {p.name: pl.read_parquet(p) for p in sorted(data.glob("chunk_*.parquet"))},
            [{k: v for k, v in e.items() if k != "bytes"} for e in manifest],
        )
    return out


def same(a: dict, b: dict) -> bool:
    return all(
        a[s][1] == b[s][1] and a[s][0].keys() == b[s][0].keys() and all(a[s][0][n].equals(b[s][0][n]) for n in a[s][0])
        for s in a
    )


def main(slice_path: Path, rate: float) -> int:
    data = slice_path.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    good, wrong = serve(data, md5, rate), serve(data, "0" * 32, rate * 4)
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "reference"
        (ref / "releases" / RELEASE / "dump").mkdir(parents=True)
        os.link(slice_path, ref / "releases" / RELEASE / "dump" / NAME)
        assert step(ref, "split", good).wait() == 0, (ref / "split.err").read_text()
        want = state(ref)
        for case in ("sequential", "together", "split first", "download killed"):
            work = Path(tmp) / case.replace(" ", "-")
            work.mkdir()
            t = time.perf_counter()
            if case == "sequential":
                assert step(work, "download", good).wait() == 0
                split = step(work, "split", good)
            else:
                split = step(work, "split", good)
                time.sleep(3 if case == "split first" else 0)
                download = step(work, "download", good)
                if case == "download killed":
                    time.sleep(rate and len(data) / rate / 2)
                    download.send_signal(signal.SIGKILL)
                    download.wait()
                    download = step(work, "download", good)
                assert download.wait() == 0, (work / "download.err").read_text()[-2000:]
            assert split.wait() == 0, (work / "split.err").read_text()[-2000:]
            ok = same(want, state(work))
            failed += not ok
            print(f"{case}: {time.perf_counter() - t:.1f}s, {'the same chunks' if ok else 'DIFFERENT chunks'}", flush=True)
        work = Path(tmp) / "md5-failed"
        work.mkdir()
        split, download = step(work, "split", wrong), step(work, "download", wrong)
        download_code, split_code = download.wait(), split.wait()
        data_dir = work / "releases" / RELEASE / "data"
        left = (data_dir / "manifest.jsonl").exists() or any(data_dir.glob("routed/chunk_*"))
        ok = download_code != 0 and split_code != 0 and not left
        failed += not ok
        print(f"md5 failed: download exit {download_code}, split exit {split_code}, chunks {'left' if left else 'deleted'}"
              f"{'' if ok else ' FAILED'}")
    print("FAILED" if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), float(sys.argv[2]) if len(sys.argv) > 2 else 5e6))
