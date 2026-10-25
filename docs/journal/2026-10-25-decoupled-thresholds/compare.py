"""Two layouts (layout.py) compared: fingerprints, schemas, bytes and row groups by table."""
import json, sys, statistics as st
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for run in ("first", "x3"):
    A, B = a[run], b[run]
    assert set(A) == set(B), set(A) ^ set(B)
    fp = [k for k in A if A[k]["fp"] != B[k]["fp"]]
    sch = [k for k in A if A[k]["schema"] != B[k]["schema"]]
    same_bytes = sum(1 for k in A if [f["sha"] for f in A[k]["files"]] == [f["sha"] for f in B[k]["files"]])
    nfiles = lambda X: sum(len(v["files"]) for v in X.values())
    rg_same = sum(1 for k in A if [f["rgs"] for f in A[k]["files"]] == [f["rgs"] for f in B[k]["files"]])
    print(f"{run}: {len(A)} keys; ordered fingerprint differs: {len(fp)}; schema differs: {len(sch)}; "
          f"files {nfiles(A)} -> {nfiles(B)}; keys byte-identical {same_bytes}; keys with the same row groups {rg_same}")
    tables = sorted({k.split('/')[0] for k in A})
    for t in tables:
        ks = [k for k in A if k.startswith(t + '/')]
        fa = [f for k in ks for f in A[k]["files"]]; fb = [f for k in ks for f in B[k]["files"]]
        ra = [len(f["rgs"]) for f in fa]; rb = [len(f["rgs"]) for f in fb]
        diff = [k for k in ks if [f["rgs"] for f in A[k]["files"]] != [f["rgs"] for f in B[k]["files"]]]
        print(f"  {t}: keys {len(ks)}, files {len(fa)} -> {len(fb)}, row groups {sum(ra)} -> {sum(rb)}, "
              f"bytes {sum(f['bytes'] for f in fa):,} -> {sum(f['bytes'] for f in fb):,}, keys whose row groups changed {len(diff)} {diff[:4]}")
    for k in ("claims/all", "entities/all", "labels/en"):
        print(f"  {k}: before {[(f['bytes'], f['rgs']) for f in A[k]['files']]}")
        print(f"  {k}: after  {[(f['bytes'], f['rgs']) for f in B[k]['files']]}")
