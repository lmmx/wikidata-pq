"""A new claims field found mid-run, and the release schema widened for it (bench 17.9.0):
run apart from the timed run, after the misfit chunk (ab.run_once), so none of its figures
take this in.

A release's dump can hold a claims field the stored schema lacks (a new datavalue member,
say). Before src/wikidata/release_schema.py such a chunk halted the run. Now the field is
added to the release schema (a version), the split's fast path follows it, a group's
files of an earlier version are written in its latest when merged, and finalise reads
every file as the frozen version. This stage measures each of those costs on the
sample's first chunk file (`c0`), turned back into its dump lines (dumpslice.py), its
first entity's first statement given a new mainsnak field (FIELD) and, in a directory of
its own, each step in a process of its own (its CPU, wall and peak RSS its own):

- **discovery,** the pipeline's process step on the part holding the field, as the split
  left it (as JSON: no version has it yet): its claims fail the latest version's check,
  are inferred again for their schema, the field is merged into the log as version 1,
  and the claims written in it: `evolve_discover_cpu`, `_wall`, `_rss_mb`;
- **known,** the same part's process step once the field is in the release schema (a
  part the split left as JSON, as before the split followed it): `evolve_known_cpu`,
  `_wall`, `_rss_mb`; discovery's overhead is the difference;
- **the split, following,** dump.write_chunk on the chunk's lines with the field in the
  release schema (the part built by the fast path: `evolve_split_built_cpu`, `_wall`), and
  with none (the part left as JSON for the pipeline: `evolve_split_json_cpu`, `_wall`),
  and `evolve_claims_json`, the parts left as JSON once the release schema has the field
  (0 expected);
- **widening at a merge,** the part's claims file of version 0 written in version 1
  (ReleaseSchema.upgrade_file, as a group's merge writes a chunk's file of an earlier
  version): `evolve_upgrade_cpu`, `_wall`, `_rss_mb`, and its bytes, the disk it takes
  until the group is merged (`evolve_upgrade_mb`, beside the file's own `evolve_file_mb`);
- **widening on read,** the claims file of version 0 read whole as the sort reads a key
  (sort_by_id._read_file), its derived leaves put back if it was written without them
  (bench 17.14.0, claims_derived), and conformed to version 1 (`evolve_read_conform_cpu`,
  `_rss_mb`), against the read alone (`evolve_read_cpu`, `_rss_mb`).

Parity (`evolve_parity_ok`): the field's value is in the part's claims file as injected;
the version 0 file written in version 1 has the rows of the part processed in version 1
(its schema and values); the part the split built in version 1 has the rows genson's
inference gives it in version 1; and the read and conformed rows are the file's in
version 1. A commit without a release schema (one before it) halts at the field: its
discovery is recorded as `evolve_halts`, and nothing else is run.

Usage: pipeline.py --evolve <src> <out.json> <chunk file>
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import orjson

from dumpslice import dump_escaped, orjson_line

FIELD = "zz_evolve"
VALUE = "found mid-run"

STEP = r"""
import json, os, resource, sys, time
sys.path.insert(0, {src!r})
from pathlib import Path
step, args = sys.argv[1], json.loads(sys.argv[2])
cpu = lambda: sum(getattr(resource.getrusage(resource.RUSAGE_SELF), f) for f in ("ru_utime", "ru_stime"))
import polars as pl, pyarrow as pa, pyarrow.parquet as pq
from wikidata import dump, process
out = {{}}
# A claims file's rows with any derived leaves it was written without put back, as finalise
# reads it (src/wikidata/claims_derived.py, bench 17.14.0; a commit before it has none)
def restored(rows, path):
    try:
        from wikidata import claims_derived, release_schema
    except ImportError:
        return rows
    schema = pq.read_schema(path)
    if not claims_derived.dropped(schema):
        return rows
    full = release_schema.CLAIMS.file_schema(release_schema.version_of(schema))
    return claims_derived.restore(rows.replace_schema_metadata(schema.metadata), full).replace_schema_metadata()
if step == "process":
    Path(args["state"]).mkdir(parents=True, exist_ok=True)
    t, c = time.perf_counter(), cpu()
    try:
        process.process(data_dir=Path(args["data"]), output_dir=Path(args["output"]), repo_id="unused",
                        state_dir=Path(args["state"]), chunk_idx=0)
    except SystemExit as e:
        out["halted"] = str(e)[:300]
    out |= {{"wall": time.perf_counter() - t, "cpu": cpu() - c}}
elif step == "split":
    lines = Path(args["lines"]).read_bytes().split(b"\n")  # (a dump line holds no newline)
    t, c = time.perf_counter(), cpu()
    entry = dump.write_chunk(0, lines, Path(args["main"]), Path(args["scholar"]))
    out |= {{"wall": time.perf_counter() - t, "cpu": cpu() - c,
            "claims_json": sum(1 for k in ("main", "scholar") if entry.get(k) and entry[k].get("claims_json"))}}
elif step == "upgrade":
    from wikidata import release_schema
    t, c = time.perf_counter(), cpu()
    release_schema.CLAIMS.upgrade_file(Path(args["src"]), Path(args["dst"]), 1)
    out |= {{"wall": time.perf_counter() - t, "cpu": cpu() - c}}
elif step in ("read", "read_conform"):
    from wikidata import sort_by_id
    t, c = time.perf_counter(), cpu()
    rows = sort_by_id._read_file(Path(args["src"]))
    if step == "read_conform":
        from wikidata import release_schema
        rows = restored(rows, Path(args["src"]))
        rows = release_schema.CLAIMS.conform(rows, 1)
    out |= {{"wall": time.perf_counter() - t, "cpu": cpu() - c, "rows": rows.num_rows}}
    out["rss_mb"] = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) // 1024
    if args.get("check"):  # (after its peak is taken: the rows to compare are read here too)
        want = restored(pq.read_table(args["check"]), Path(args["check"])).replace_schema_metadata()
        out["same"] = rows.equals(want) and rows.schema.equals(want.schema)
# (this process's own peak: VmHWM, not ru_maxrss, which a child takes from its parent at the fork)
out.setdefault("rss_mb", next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) // 1024)
Path(args["out"]).write_text(json.dumps(out))
"""


def evolve_lines(path: Path) -> tuple[list[bytes], list[bytes], str]:
    """The chunk file's rows as the dump's lines, and the same with its first entity with
    claims given FIELD in its first statement's mainsnak; and that entity's id."""
    import pyarrow.parquet as pq

    plain, lines, found = [], [], None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            line = orjson_line(row)
            plain.append(dump_escaped(line))
            if found is None:
                entity = orjson.loads(line)
                statements = [s for ss in entity.get("claims", {}).values() for s in ss]
                if statements:
                    statements[0]["mainsnak"][FIELD] = VALUE
                    line, found = orjson.dumps(entity), entity["id"]
            lines.append(dump_escaped(line))
    if found is None:
        raise SystemExit(f"{path}: no entity with claims to give a new field")
    return plain, lines, found


def evolve_stage(src: Path, out: Path, path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plain, lines, entity = evolve_lines(path)
        for name, ls in (("plain", plain), ("lines", lines)):
            (tmp / f"{name}.lines").write_bytes(b"\n".join(ls))
        del plain, lines
        env = {**os.environ, "WIKIDATA_RELEASE": "20261012", "HF_HUB_OFFLINE": "1"}
        env.pop("WIKIDATA_SCHOLAR", None)

        def step(name: str, cwd: Path, extra: dict | None = None, **args) -> dict:
            cwd.mkdir(parents=True, exist_ok=True)
            res = cwd / f"{name}.json"
            subprocess.run([sys.executable, "-c", STEP.format(src=str(src)), name,
                            json.dumps({**args, "out": str(res)})], cwd=cwd, env=env | (extra or {}), check=True)
            return json.loads(res.read_text())

        def split(cwd: Path, which: str = "lines", extra: dict | None = None) -> dict:
            return step("split", cwd, extra, lines=str(tmp / f"{which}.lines"), main=str(cwd / "main"),
                        scholar=str(cwd / "scholar"))

        def process(cwd: Path) -> dict:
            # (the part of the main set: c0's entities are all the main set's)
            return step("process", cwd, data=str(cwd / "main" / "routed"), output=str(cwd / "results"),
                        state=str(cwd / "state"))

        def seeded(cwd: Path) -> Path:
            d = cwd / "releases" / "20261012" / "schema"
            d.mkdir(parents=True)
            (d / "claims.json").write_text((tmp / "discover" / "releases" / "20261012" / "schema" / "claims.json").read_text())
            return cwd

        claims = lambda cwd: cwd / "results" / "claims" / "chunk_0.parquet"  # noqa: E731
        # Discovery: the split with no version leaves the part as JSON, and its process step
        # adds the field as version 1
        a = tmp / "discover"
        res = {f"evolve_split_json_{k}": round(v, 2) for k, v in split(a).items() if k in ("cpu", "wall")}
        d = process(a)
        if "halted" in d:  # (a commit before the release schema)
            res |= {"evolve_halts": True, "evolve_discover_wall": round(d["wall"], 2)}
            out.write_text(json.dumps(res))
            print(f"evolve: the field halts the run ({d['halted'][:120]})", flush=True)
            return
        res |= {"evolve_halts": False, "evolve_discover_cpu": round(d["cpu"], 2),
                "evolve_discover_wall": round(d["wall"], 2), "evolve_discover_rss_mb": d["rss_mb"]}
        # The split with the field in the release schema: the part built by the fast path
        b = seeded(tmp / "built")
        built = split(b)
        res |= {f"evolve_split_built_{k}": round(v, 2) for k, v in built.items() if k in ("cpu", "wall")}
        res["evolve_claims_json"] = built["claims_json"]
        # Known: the field in the release schema, the part as JSON (the split not caught up)
        c = seeded(tmp / "known")
        split(c, extra={"WIKIDATA_SPLIT_CLAIMS": "json"})
        k = process(c)
        res |= {"evolve_known_cpu": round(k["cpu"], 2), "evolve_known_wall": round(k["wall"], 2),
                "evolve_known_rss_mb": k["rss_mb"]}
        # A chunk without the field processed before it was found (version 0), and the same
        # chunk in version 1 (built by the split, the field in the release schema)
        z = tmp / "v0"
        split(z, "plain")
        process(z)
        y = seeded(tmp / "v1")
        split(y, "plain")
        process(y)
        # Its version 0 file written in version 1 (a merge's widening), and read as version 1
        # (finalise's), each in the release schema of `discover`
        upgraded = tmp / "upgraded.parquet"
        u = step("upgrade", a, src=str(claims(z)), dst=str(upgraded))
        res |= {"evolve_upgrade_cpu": round(u["cpu"], 2), "evolve_upgrade_wall": round(u["wall"], 2),
                "evolve_upgrade_rss_mb": u["rss_mb"], "evolve_upgrade_mb": round(upgraded.stat().st_size / 2**20, 2),
                "evolve_file_mb": round(claims(z).stat().st_size / 2**20, 2)}
        r = step("read", a, src=str(claims(z)))
        rc = step("read_conform", a, src=str(claims(z)), check=str(claims(y)))
        res |= {"evolve_read_cpu": round(r["cpu"], 3), "evolve_read_rss_mb": r["rss_mb"],
                "evolve_read_conform_cpu": round(rc["cpu"], 3), "evolve_read_conform_rss_mb": rc["rss_mb"]}

        import polars as pl
        import pyarrow.parquet as pq

        v1 = pq.read_table(claims(a))
        native = pq.read_table(claims(y))
        up = pq.read_table(upgraded)
        found = pl.from_arrow(v1.select(["id", FIELD])).filter(pl.col("id") == entity)[FIELD].to_list()
        parity = {
            "value": VALUE in found,
            "upgraded": up.equals(native) and up.schema.equals(native.schema, check_metadata=True),
            "built": pq.read_table(b / "main" / "routed" / "claims" / "chunk_0.parquet").equals(v1),
            "conformed": bool(rc.get("same")),
            "no JSON once known": built["claims_json"] == 0,
        }
        res["evolve_parity_ok"] = all(parity.values())
        if not res["evolve_parity_ok"]:
            res["evolve_parity"] = parity
    out.write_text(json.dumps(res))
    print(f"evolve ({entity} given {FIELD}): discovery {res['evolve_discover_cpu']} CPU-s against "
          f"{res['evolve_known_cpu']} known; a version 0 file written in version 1 {res['evolve_upgrade_cpu']} CPU-s, "
          f"{res['evolve_upgrade_rss_mb']} MB; read and conformed {res['evolve_read_conform_cpu']} CPU-s against "
          f"{res['evolve_read_cpu']} read; split built {res['evolve_split_built_cpu']} CPU-s, as JSON "
          f"{res['evolve_split_json_cpu']}; parity {'passed' if res['evolve_parity_ok'] else 'FAILED'}", flush=True)
