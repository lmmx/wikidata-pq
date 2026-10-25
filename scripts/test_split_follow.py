"""split-dump following download-dump, against a stand-in for dumps.wikimedia.org (a local
server with Range requests and kept-alive connections, each throttled, as Wikimedia's are):
the split run alongside the download, started before it, and with the download killed
partway and run again, each giving the same chunk files and manifests as a split of the
downloaded dump; and a download failing its md5 check stops the split, its chunks deleted.
The download runs over 3 connections (its default, in pieces of PIECE bytes), and over one
in the case "one connection"; the server counts the connections open at once, which must
reach 3 in the first and never pass it, and a 4th connection asked for is refused.

    .venv/bin/python scripts/test_split_follow.py dump-slice.json.bz2 [bytes per second]

(bench/dumpslice.py makes a slice; at the default 5 MB/s its 77 MB take about 15 s.) A
`.json.gz` slice is split as the gz dump (WIKIDATA_DUMP_FORMAT=gz).
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
FORMAT = "gz" if sys.argv[1:] and sys.argv[1].endswith(".gz") else "bz2"
NAME = f"wikidata-{RELEASE}-all.json.{FORMAT}"


# The download's pieces here (its PIECE is 32 MiB), so a slice is many of them
PIECE = 1 << 20
# Connections open at once to the dump's servers: now, and the most since the last reset
OPEN = {"now": 0, "peak": 0}
OPEN_LOCK = threading.Lock()


def serve(data: bytes, md5: str, rate: float) -> str:
    """A server for the release's dump and md5sums, sending each connection at most `rate`
    bytes per second; returns its base URL"""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # connections kept alive between requests

        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            with OPEN_LOCK:
                OPEN["now"] += 1
                OPEN["peak"] = max(OPEN["peak"], OPEN["now"])

        def finish(self):
            with OPEN_LOCK:
                OPEN["now"] -= 1
            super().finish()

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
            start, end = 0, len(body)
            if ranged and (r := self.headers.get("Range")):
                first, last = r.split("=")[1].split("-")
                start, end = int(first), min(int(last) + 1, len(body)) if last else len(body)
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Length", str(end - start))
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{end - 1}/{len(body)}")
            self.end_headers()
            step = 1 << 18
            try:
                for i in range(start, end, step):
                    self.wfile.write(body[i : min(i + step, end)])
                    if ranged:
                        time.sleep(min(step, end - i) / rate)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True  # a download killed partway

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def step(work: Path, name: str, base: str, connections: int | None = None) -> subprocess.Popen:
    code = {
        "download": f"from wikidata import dump; dump.DUMPS_URL = {base!r}; dump.PIECE = {PIECE};"
                    f" dump.download({base!r})",
        "split": "from wikidata import dump; dump.split()",
    }[name]
    env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "WIKIDATA_DUMP_FORMAT": FORMAT, "NO_PROXY": "127.0.0.1",
           "no_proxy": "127.0.0.1"}
    if connections:
        env["WIKIDATA_DUMP_CONNECTIONS"] = str(connections)
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
        for case in ("sequential", "together", "split first", "download killed", "one connection"):
            work = Path(tmp) / case.replace(" ", "-")
            work.mkdir()
            connections = 1 if case == "one connection" else 3
            with OPEN_LOCK:
                OPEN["peak"] = 0
            t = time.perf_counter()
            if case == "sequential":
                download = step(work, "download", good)
                assert download.wait() == 0, (work / "download.err").read_text()[-2000:]
                took = time.perf_counter() - t
                split = step(work, "split", good)
            else:
                split = step(work, "split", good)
                time.sleep(3 if case == "split first" else 0)
                download = step(work, "download", good, connections)
                if case == "download killed":
                    time.sleep(rate and len(data) / rate / connections / 2)
                    download.send_signal(signal.SIGKILL)
                    download.wait()
                    part = work / "releases" / RELEASE / "dump" / f"{NAME}.part"
                    killed_at = part.stat().st_size if part.exists() else 0
                    download = step(work, "download", good)
                assert download.wait() == 0, (work / "download.err").read_text()[-2000:]
                took = time.perf_counter() - t
            assert split.wait() == 0, (work / "split.err").read_text()[-2000:]
            ok = same(want, state(work))
            peak = OPEN["peak"]
            # the dump's own connections, plus at most one at a time for its md5sums and HEAD
            ok_open = peak <= 3 and (peak >= connections)
            failed += not (ok and ok_open)
            killed = f", killed at {killed_at:,} of {len(data):,} bytes" if case == "download killed" else ""
            print(f"{case}: {time.perf_counter() - t:.1f}s (download {took:.1f}s{killed}),"
                  f" {'the same chunks' if ok else 'DIFFERENT chunks'},"
                  f" at most {peak} connection(s) open{'' if ok_open else ' FAILED'}", flush=True)
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
        work = Path(tmp) / "four-connections"
        work.mkdir()
        code = step(work, "download", good, 4).wait()
        ok = code != 0 and "1 to 3" in (work / "download.err").read_text()
        failed += not ok
        print(f"four connections: download exit {code}{'' if ok else ' FAILED'}")
    print("FAILED" if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), float(sys.argv[2]) if len(sys.argv) > 2 else 5e6))
