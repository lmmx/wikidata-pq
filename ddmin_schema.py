# ddmin_schema.py
"""Delta debugging to find minimal row range that breaks map inference."""
import subprocess
import tempfile
import json
from pathlib import Path
import polars as pl

SRC = Path("data/huggingface_hub/philippesaade/wikidata/data/chunk_0-00057-of-00546.parquet")

def test_range(df: pl.DataFrame, start: int, end: int) -> bool:
    """Returns True if map inference works, False if broken."""
    if start >= end:
        return True  # Empty range passes
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in df.slice(start, end - start).iter_rows():
            f.write(row[0] + "\n")
        tmp_path = f.name
    
    result = subprocess.run(
        [
            "genson-cli", "--ndjson",
            "--wrap-root", "claims",
            "--map-threshold", "0",
            "--unify-maps",
            "--force-parent-type", "mainsnak:record",
            "--force-scalar-promotion", "datavalue,precision,latitude",
            "--no-unify", "qualifiers",
            "--max-builders", "1000",
            tmp_path,
        ],
        capture_output=True,
        text=True,
    )
    
    try:
        parsed = json.loads(result.stdout)
        claims_schema = parsed.get("properties", {}).get("claims", {})
        
        if "properties" in claims_schema:
            props = claims_schema["properties"]
            has_p_keys = any(k.startswith("P") and k[1:].isdigit() for k in props.keys())
            if has_p_keys:
                return False
        
        if "additionalProperties" in claims_schema:
            return True
        
        return "additionalProperties" in result.stdout
        
    except json.JSONDecodeError:
        return False


def ddmin_range(df: pl.DataFrame, start: int, end: int) -> tuple[int, int]:
    """Find minimal [start, end) range that still fails."""
    assert not test_range(df, start, end), "Initial range must fail"
    
    size = end - start
    if size <= 1:
        return start, end
    
    # Try shrinking from the left
    best_start, best_end = start, end
    
    # Binary search for latest start that still fails
    lo, hi = start, end - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        print(f"  Trying start={mid}, end={best_end}... ", end="", flush=True)
        if not test_range(df, mid, best_end):
            print("FAILS")
            lo = mid
            best_start = mid
        else:
            print("passes")
            hi = mid - 1
    
    # Binary search for earliest end that still fails
    lo, hi = best_start + 1, end
    while lo < hi:
        mid = (lo + hi) // 2
        print(f"  Trying start={best_start}, end={mid}... ", end="", flush=True)
        if not test_range(df, best_start, mid):
            print("FAILS")
            hi = mid
            best_end = mid
        else:
            print("passes")
            lo = mid + 1
    
    return best_start, best_end


def ddmin_subset(df: pl.DataFrame, indices: list[int]) -> list[int]:
    """Generalized delta debugging - find minimal subset of row indices that fails."""
    
    def test_indices(idx_list: list[int]) -> bool:
        if not idx_list:
            return True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in idx_list:
                f.write(df.row(i)[0] + "\n")
            tmp_path = f.name
        
        result = subprocess.run(
            [
                "genson-cli", "--ndjson",
                "--wrap-root", "claims",
                "--map-threshold", "0",
                "--unify-maps",
                "--force-parent-type", "mainsnak:record",
                "--force-scalar-promotion", "datavalue,precision,latitude",
                "--no-unify", "qualifiers",
                "--max-builders", "1000",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        
        try:
            parsed = json.loads(result.stdout)
            claims_schema = parsed.get("properties", {}).get("claims", {})
            if "properties" in claims_schema:
                props = claims_schema["properties"]
                if any(k.startswith("P") and k[1:].isdigit() for k in props.keys()):
                    return False
            return "additionalProperties" in claims_schema or "additionalProperties" in result.stdout
        except json.JSONDecodeError:
            return False
    
    assert not test_indices(indices), "Initial set must fail"
    
    n = 2
    while len(indices) >= 2:
        chunk_size = max(1, len(indices) // n)
        progress = False
        
        # Try removing each chunk
        for i in range(0, len(indices), chunk_size):
            chunk_end = min(i + chunk_size, len(indices))
            complement = indices[:i] + indices[chunk_end:]
            
            if complement and not test_indices(complement):
                print(f"  Reduced to {len(complement)} rows (removed indices {i}:{chunk_end})")
                indices = complement
                n = max(n - 1, 2)
                progress = True
                break
        
        if not progress:
            if n >= len(indices):
                break
            n = min(n * 2, len(indices))
    
    return indices


def main():
    df = pl.read_parquet(SRC, columns=["claims"])
    total = len(df)
    print(f"Total rows: {total}")
    
    print(f"\nVerifying full file fails... ", end="", flush=True)
    if test_range(df, 0, total):
        print("PASSES - no bug?")
        return
    print("FAILS\n")
    
    # Phase 1: Find minimal contiguous range
    print("=== Phase 1: Finding minimal contiguous range ===")
    min_start, min_end = ddmin_range(df, 0, total)
    print(f"\nMinimal contiguous range: [{min_start}, {min_end}) = {min_end - min_start} rows")
    
    # Phase 2: Try to find even smaller non-contiguous subset
    if min_end - min_start > 1:
        print("\n=== Phase 2: Finding minimal subset within range ===")
        indices = list(range(min_start, min_end))
        minimal_indices = ddmin_subset(df, indices)
        print(f"\nMinimal failing subset: {len(minimal_indices)} rows")
        print(f"Row indices: {minimal_indices}")
        
        # Save minimal failing rows
        with open("minimal_fail.jsonl", "w") as f:
            for i in minimal_indices:
                f.write(df.row(i)[0] + "\n")
        print(f"\nSaved to minimal_fail.jsonl")
    else:
        print(f"\nSingle row failure at index {min_start}")
        with open("minimal_fail.jsonl", "w") as f:
            f.write(df.row(min_start)[0] + "\n")
        print("Saved to minimal_fail.jsonl")

if __name__ == "__main__":
    main()