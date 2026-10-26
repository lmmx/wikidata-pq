"""A chunk whose claims misfit claims.avsc, as a release's chunks can (bench 17.3.0): run
apart from the timed run, after it (ab.run_once), so none of its figures take this in.

The sample's chunks all fit claims.avsc, which was inferred from them, so the timed run
never takes the path a misfit chunk takes. A release's dump has them: in 124 chunks of
10,000 entities read from five places in wikidata-20261020-all.json.bz2, 10 had a
claims member claims.avsc lacked, each a reference snak's globe-coordinate
(docs/journal/2026-10-26-claims-avsc-complete.md). Such a chunk is reshaped twice in the
split (the raw path's scan, up to the misfit, then the text path), its part's claims are
written as JSON, and the pipeline decodes them with genson's inference (the fallback)
rather than placing the claims file the split built.

Here the sample's first chunk file (`c0`, the dump's first 10,000 entities) is turned back
into its dump lines (dumpslice.py), one reference of its first entity with references
given a globe-coordinate snak (MISFIT: one from the dump, Q618385's P625), and then, in
this process, in a directory of its own:

- **the split's work on the chunk** (dump.write_chunk: its reshape, its claims, terms and
  entities built, its parts written), timed: `misfit_split_cpu` (this process's CPU, its
  threads' with it) and `misfit_split_wall`;
- **the pipeline's process step on the misfit's part** (process.process, as a pipeline
  worker runs it, on the part as the split routed it), timed: `misfit_process_cpu`,
  `misfit_process_wall`;
- their sums, `misfit_chunk_cpu` and `misfit_chunk_wall`; and `misfit_claims_json`, the
  parts whose claims the split left as JSON.

Parity (`misfit_parity_ok`): the part's claims file must be the one genson's inference
makes of the part's claims (normalise_claims_direct with claims.avsc set aside: what a
misfit part is published as), in its rows, order, schema (with its metadata) and row
groups' rows. A part left as JSON was decoded by inference, so it holds by construction
and is not run again.

Usage: pipeline.py --misfit <src> <out.json> <chunk file>
"""

import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import orjson

from dumpslice import dump_escaped, orjson_line

# A reference snak's globe-coordinate, as the dump writes it (wikidata-20261020, Q618385's
# P625 in a reference of its P276): the shape that made each real-dump misfit
MISFIT = {"snaktype": "value", "property": "P625",
          "datavalue": {"value": {"latitude": 49.88866430638875, "longitude": 10.885455104509411, "altitude": None,
                                  "precision": 1e-08, "globe": "http://www.wikidata.org/entity/Q2"},
                        "type": "globecoordinate"},
          "datatype": "globe-coordinate"}


def misfit_lines(path: Path) -> tuple[list[bytes], str, int]:
    """The chunk file's rows as the dump's lines, the first entity with a reference given
    MISFIT in its first reference without a P625; and that entity's id and line."""
    import pyarrow.parquet as pq

    lines, misfit = [], None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            line = orjson_line(row)
            if misfit is None:
                entity = orjson.loads(line)
                refs = (r for ss in entity["claims"].values() for s in ss for r in s.get("references", [])
                        if "P625" not in r["snaks"])
                if (ref := next(refs, None)) is not None:
                    ref["snaks"]["P625"] = [MISFIT]
                    ref["snaks-order"] = [*ref.get("snaks-order", []), "P625"]
                    line, misfit, at = orjson.dumps(entity), entity["id"], len(lines)
            lines.append(dump_escaped(line))
    if misfit is None:
        raise SystemExit(f"{path}: no entity with a reference to give a misfit")
    return lines, misfit, at


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def same_partition(a: Path, b: Path) -> bool:
    """Two claims partition files the same in their rows, their order, their schema (with
    its metadata) and their row groups' rows (as pipeline.split_parity compares them)."""
    import pyarrow.parquet as pq

    fa, fb = pq.ParquetFile(a), pq.ParquetFile(b)
    groups = lambda f: [f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)]  # noqa: E731
    ta, tb = fa.read(), fb.read()
    return ta.schema.equals(tb.schema, check_metadata=True) and ta.equals(tb) and groups(fa) == groups(fb)


def misfit_stage(src: Path, out: Path, path: Path) -> None:
    lines, misfit, at = misfit_lines(path)
    sys.path.insert(0, str(src))
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        import polars as pl

        from wikidata import dump, process
        from wikidata.scholarly import is_scholarly

        sets = {"main": Path(tmp) / "main", "scholar": Path(tmp) / "scholar"}
        t, c = time.perf_counter(), cpu()
        entry = dump.write_chunk(0, lines, sets["main"], sets["scholar"])
        res = {"misfit_split_wall": round(time.perf_counter() - t, 2), "misfit_split_cpu": round(cpu() - c, 1)}
        res["misfit_claims_json"] = sum(1 for k in sets if entry.get(k) and entry[k].get("claims_json"))
        # The misfit's part, as the split routed it, through the pipeline's process step
        name = "scholar" if dump.reshape([lines[at]])[0].select(is_scholarly(pl.col("claims"))).item() else "main"
        part = entry[name]
        data, output, state = sets[name] / dump.ROUTED, Path(tmp) / "results", Path(tmp) / "state"
        state.mkdir()
        left_json = bool(part.get("claims_json"))
        t, c = time.perf_counter(), cpu()
        process.process(data_dir=data, output_dir=output, repo_id="unused", state_dir=state, chunk_idx=0)
        res["misfit_process_wall"] = round(time.perf_counter() - t, 2)
        res["misfit_process_cpu"] = round(cpu() - c, 1)
        res["misfit_chunk_wall"] = round(res["misfit_split_wall"] + res["misfit_process_wall"], 2)
        res["misfit_chunk_cpu"] = round(res["misfit_split_cpu"] + res["misfit_process_cpu"], 1)
        # Parity: the part's claims as genson's inference gives them (the fallback)
        if left_json:
            res["misfit_parity_ok"] = True
        else:
            frame = dump.reshape(lines)[0]
            frame = frame.filter(is_scholarly(pl.col("claims")) == (name == "scholar")).select("id", "claims")
            source = Path(tmp) / "fallback" / "chunk_0.parquet"
            source.parent.mkdir()
            frame.write_parquet(source)
            process.CLAIMS_AVRO = None  # (genson infers, as for a part that misfits claims.avsc)
            made = process.normalise_claims_direct(source, source.parent / "claims" / "chunk_0.parquet",
                                                   source.parent / "lookup.parquet", source.parent / "q.parquet",
                                                   frame.height)
            res["misfit_parity_ok"] = same_partition(made, output / "claims" / "chunk_0.parquet")
        os.chdir("/")
    print(f"misfit chunk ({misfit} given a reference's coordinate): split {res['misfit_split_cpu']} CPU-s, "
          f"process {res['misfit_process_cpu']} CPU-s, {res['misfit_claims_json']} parts' claims as JSON, "
          f"parity {'passed' if res['misfit_parity_ok'] else 'FAILED'}", flush=True)
    out.write_text(json.dumps(res))
