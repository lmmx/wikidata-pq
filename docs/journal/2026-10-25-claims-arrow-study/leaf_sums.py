"""Leaf checksums of genson's claims output (the claims call on a slice), named as the
prototype names them, for comparing the two."""
import sys
from pathlib import Path
import pyarrow as pa, pyarrow.compute as pc
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_side_lib import claims_call

def walk(arr, path, out):
    t = arr.type
    nn = len(arr) - arr.null_count
    if pa.types.is_large_list(t) or pa.types.is_list(t):
        child = arr.values if arr.offset == 0 else arr.flatten()
        ct = t.value_type
        if pa.types.is_struct(ct) and [f.name for f in ct] == ["key", "value"]:
            out.append((f"{path}{{}}", len(arr), nn, 0))
            walk(child.field("key"), f"{path}{{}}.key", out)
            walk(child.field("value"), f"{path}{{}}.value", out)
        else:
            out.append((f"{path}[]", len(arr), nn, 0))
            walk(child, f"{path}[]", out)
    elif pa.types.is_struct(t):
        out.append((f"{path}<>", len(arr), nn, 0))
        for i, f in enumerate(t):
            walk(arr.field(i), f"{path}.{f.name}", out)
    elif pa.types.is_large_string(t) or pa.types.is_string(t):
        b = pc.sum(pc.binary_length(arr)).as_py() or 0
        out.append((path, len(arr), nn, b))
    elif pa.types.is_integer(t):
        out.append((path, len(arr), nn, pc.sum(arr).as_py() or 0))
    elif pa.types.is_floating(t):
        out.append((path, len(arr), nn, 0))
    elif pa.types.is_null(t):
        pass
    else:
        raise TypeError(t)

t = claims_call(Path(sys.argv[1])).combine_chunks()
out = []
walk(t.column("claims").chunk(0), "claims", out)
for p, n, nn, x in out:
    print(f"SUM\t{p}\t{n}\t{nn}\t{x}")
