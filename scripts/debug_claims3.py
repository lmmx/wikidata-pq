#!/usr/bin/env python3
"""Deep exploration of claims language structure."""
import polars as pl
from pathlib import Path

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")

# Get a small sample to work with interactively
base = (
    pl.scan_parquet(claims_file)
    .select(pl.col("claims").explode().struct.unnest())
    .drop("key")
    .explode("value")
    .unnest("value")
    .unnest("mainsnak")
    .head(100)
    .collect()
)

print("=== Datatype distribution in sample ===")
print(base.group_by("datatype").len().sort("len", descending=True))

print("\n" + "="*80)
print("=== Datavalue schema (what's inside the struct) ===")
print(base.get_column("datavalue").dtype)

# Let's look at one wikibase-item claim in detail
print("\n" + "="*80)
print("=== WIKIBASE-ITEM example ===")
wb_item = base.filter(pl.col("datatype") == "wikibase-item").head(1)
print(f"\nProperty: {wb_item['property'][0]}")
print(f"\nProperty-labels (first 5):")
prop_labels = wb_item["property-labels"][0]
for i, item in enumerate(prop_labels[:5]):
    print(f"  {item}")

print(f"\nDatavalue struct fields:")
dv = wb_item["datavalue"][0]
print(f"  id: {dv['id']}")
print(f"  labels (first 5):")
if dv["labels"]:
    for i, item in enumerate(dv["labels"][:5]):
        print(f"    {item}")

# Check if there's a STRING type claim for comparison
print("\n" + "="*80)
print("=== STRING/EXTERNAL-ID example (if present) ===")
str_claim = base.filter(pl.col("datatype").is_in(["string", "external-id"])).head(1)
if len(str_claim) > 0:
    print(f"\nProperty: {str_claim['property'][0]}")
    print(f"Datatype: {str_claim['datatype'][0]}")
    dv = str_claim["datavalue"][0]
    print(f"datavalue__string: {dv['datavalue__string']}")
    print(f"labels: {dv['labels']}")
else:
    print("No string/external-id claims in sample")

# Check quantity type
print("\n" + "="*80)
print("=== QUANTITY example (if present) ===")
qty_claim = base.filter(pl.col("datatype") == "quantity").head(1)
if len(qty_claim) > 0:
    print(f"\nProperty: {qty_claim['property'][0]}")
    dv = qty_claim["datavalue"][0]
    print(f"amount: {dv['amount']}")
    print(f"unit: {dv['unit']}")
    print(f"unit-labels (first 5):")
    if dv["unit-labels"]:
        for i, item in enumerate(dv["unit-labels"][:5]):
            print(f"  {item}")
else:
    print("No quantity claims in sample")

# Check time type
print("\n" + "="*80)
print("=== TIME example (if present) ===")
time_claim = base.filter(pl.col("datatype") == "time").head(1)
if len(time_claim) > 0:
    print(f"\nProperty: {time_claim['property'][0]}")
    dv = time_claim["datavalue"][0]
    print(f"time: {dv['time']}")
    print(f"labels: {dv['labels']}")
else:
    print("No time claims in sample")

print("\n" + "="*80)
print("=== THE MATCHING PROBLEM ===")
print("""
For a wikibase-item claim:
- property-labels has languages: [en, fr, de, pl, zh, ...]  (~150)
- datavalue.labels has languages: [en, fr, pl]  (subset)

If we partition ONLY on property-labels language, the 'de' partition gets:
  property_label = "ist ein(e)"  (German)
  datavalue.labels = [{"pl","wieś"}, {"en","village"}, ...]  (NO GERMAN!)

This is JUNK. The German speaker sees a German property name but the 
datavalue description is in Polish/English/whatever.

The OLD CODE filtered: keep row only if property-label-lang IN datavalue.labels.keys()
""")

# Demonstrate the mismatch
print("\n=== Demonstrating the mismatch ===")
wb_sample = base.filter(pl.col("datatype") == "wikibase-item").head(1)
if len(wb_sample) > 0:
    prop_langs = {item["key"] for item in wb_sample["property-labels"][0]}
    dv_langs = {item["key"] for item in wb_sample["datavalue"][0]["labels"]} if wb_sample["datavalue"][0]["labels"] else set()
    
    print(f"Property-label languages ({len(prop_langs)}): {sorted(prop_langs)[:10]}...")
    print(f"Datavalue label languages ({len(dv_langs)}): {sorted(dv_langs)[:10]}...")
    print(f"\nLanguages in property but NOT in datavalue: {sorted(prop_langs - dv_langs)[:10]}...")
    print(f"  ^ These would be JUNK rows if we don't filter!")