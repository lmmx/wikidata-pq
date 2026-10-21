"""Byte comparison of partition outputs (out/ and audit/, by relative path) between runs:
every new run and every base run against base round 1 of the same chunk; for a chunk
whose base runs all failed, new runs against new round 1. Prints one line per comparison."""
import hashlib, json
from pathlib import Path
def files(run: Path) -> dict[str, str]:
    return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("out", "audit") for p in sorted((run / d).rglob("*")) if p.is_file()}
runs = [json.loads(l) for l in open("bench.jsonl")]
ok = {(r["chunk"], r["variant"], r["round"]) for r in runs if "error" not in r["result"]}
for chunk in dict.fromkeys(r["chunk"] for r in runs):
    ref_v = "base" if (chunk, "base", 1) in ok else "new"
    ref = files(Path("runs") / chunk / ref_v / "r1")
    for v in ("base", "new"):
        for rnd in (1, 2, 3):
            if (v, rnd) == (ref_v, 1): continue
            if (chunk, v, rnd) not in ok:
                print(f"{chunk:4s} {v:4s} r{rnd}: failed run"); continue
            got = files(Path("runs") / chunk / v / f"r{rnd}")
            diff = sorted(k for k in ref.keys() | got.keys() if ref.get(k) != got.get(k))
            print(f"{chunk:4s} {v:4s} r{rnd} vs {ref_v} r1: {len(got)} files, {'identical' if not diff else 'DIFFER ' + str(diff[:5])}")
