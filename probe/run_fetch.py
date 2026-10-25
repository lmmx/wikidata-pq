"""Throwaway: the branch's dump._fetch over a long window, from a runner."""
import hashlib, os, sys, time, types
from pathlib import Path
sys.modules["wikidata_reshape"] = types.SimpleNamespace(entity_rows=None)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import importlib.util
import wikidata  # noqa: the package, for its config and scholarly modules
spec = importlib.util.spec_from_file_location("wikidata.dump", Path(__file__).parent / "dump_impl.py")
dump = importlib.util.module_from_spec(spec); sys.modules["wikidata.dump"] = dump; spec.loader.exec_module(dump)
from tqdm import tqdm
base, n, length = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3]))
url = f"{base}/20261019/wikidata-20261019-all.json.bz2"
out = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "part"
out.unlink(missing_ok=True)
start = 30_000_000_000
t = time.perf_counter()
dump._fetch(url, out, start, start + length, n, tqdm(disable=True))
s = time.perf_counter() - t
line = f"impl {base} conns={n} bytes={out.stat().st_size} secs={s:.1f} MBps={length/1e6/s:.1f} md5={hashlib.md5(out.read_bytes()).hexdigest()}"
print(line, flush=True)
open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(line + "\n")
out.unlink()
