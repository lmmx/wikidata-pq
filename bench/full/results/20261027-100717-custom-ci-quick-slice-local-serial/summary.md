# Compute benchmark: 20261027-100717-custom-ci-quick-slice-local-serial

- **Result:** ok; both sets finalised: {'main': True, 'scholar': True}
- **Dump:** release 20261019, custom-ci-quick-slice: 0.08 GB of bz2, 1.08 GB of JSON; the whole dump's md5 None, sha1 None
- **Source:** local (no network on the timed path); orchestration: serial; cards: False
- **Machine:** Intel(R) Xeon(R) Processor @ 2.80GHz, 4 CPUs, 16.87 GB RAM, ext4 on /dev/vda (model unknown), Ubuntu 24.04.5 LTS, kernel 6.18.44-fc-v165
- **Code:** f55a5f8-dirty (f55a5f82d3eae1957cb2163aabf4b0c00c9aea8e), wikidata 0.1.0, wikidata-reshape 0.1.0, polars 2.0.0, pyarrow 25.0.1, pq-glue 0.3.0, polars-genson 1.6.0, orjson 3.12.0, numpy 2.5.3, huggingface-hub 1.32.0, hf-xet 1.6.0
- **Decoder:** ['/usr/bin/lbzip2', '-dc']; WIKIDATA_BZ2_DECODER=None, WIKIDATA_BZ2_THREADS=None, WIKIDATA_SPLIT_WRITER_THREADS=0

## Measured

- **Wall, end to end:** 97 s; **CPU:** 210 s (2.16 cores on average)
- **Time to the first processed chunk** (the split's first chunk written): 21.0 s; each set's first part ready for its run: {'main': 21.0, 'scholar': 21.0} s (±1 s, the monitor's interval)
- **Peak RSS, every process at once:** 2797 MB (the largest process 2785 MB); the machine's memory in use at its peak 3936 MB (least available 12937 MB, other work included)
- **Disk:** peak 0.19 GB taken (free space's fall), work tree peak 0.22 GB; the device read 0.01 GB and wrote 0.19 GB; the stages' own I/O read 0.01 GB and wrote 0.49 GB
- **Network:** received 0.00 GB, sent 0.00 GB (the machine's, every process)
- **Other work on the machine:** 4.9 CPU s besides the run's own (0.05 cores on average; mean load 3.04). Near 0, the run had the machine to itself

| stage | start s | wall s | CPU s | cores | other CPU s | peak RSS MB (tree) | read GB | written GB | machine: user / sys / iowait / idle | bound |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| split | 0 | 23.6 | 82.6 | 3.50 | 1.1 | 2797 | 0.01 | 0.16 | 78% / 9% / 0% / 11% | CPU, every core |
| run_scholar | 24 | 3.8 | 6.2 | 1.64 | 1.6 | 380 | 0.00 | 0.04 | 31% / 8% / 0% / 59% | CPU on 1.6 of 4 cores (serial parts, or waiting on another stage) |
| run_main | 28 | 9.1 | 11.2 | 1.23 | 1.7 | 465 | 0.00 | 0.11 | 19% / 13% / 1% / 64% | CPU on 1.2 of 4 cores (serial parts, or waiting on another stage) |
| finalise | 37 | 60.6 | 110.5 | 1.82 | 3.7 | 2632 | 0.00 | 0.19 | 30% / 16% / 1% / 52% | CPU on 1.8 of 4 cores (serial parts, or waiting on another stage) |

| finalise run | start s | wall s | CPU s | largest process MB |
|---|--:|--:|--:|--:|
| finalise_run1_main | 37 | 28.8 | 51.0 | 1821 |
| finalise_run2_scholar | 65 | 26.4 | 43.6 | 2289 |
| finalise_run3_main | 91 | 5.7 | 14.7 | 219 |

**claims_labels** (its build, compaction and sort, within finalise's runs): main 3.4 s wall, 11.1 s CPU; scholar 2.0 s wall, 6.0 s CPU

<details><summary>finalise's steps and the group closes (wall, CPU incl. workers)</summary>

| step | calls | wall s | CPU s |
|---|--:|--:|--:|
| main/build_claims_labels | 1 | 3.2 | 10.9 |
| main/close_group | 1 | 6.7 | 7.5 |
| main/collect_refs_stage | 2 | 0.2 | 0.5 |
| main/compact_table/aliases | 2 | 0.9 | 0.8 |
| main/compact_table/claims | 2 | 0.1 | 0.1 |
| main/compact_table/claims_labels | 1 | 0.0 | 0.0 |
| main/compact_table/descriptions | 2 | 1.1 | 1.3 |
| main/compact_table/entities | 2 | 0.1 | 0.2 |
| main/compact_table/labels | 2 | 0.7 | 0.8 |
| main/compact_table/links | 2 | 1.6 | 1.6 |
| main/sort_table/aliases | 2 | 1.0 | 2.8 |
| main/sort_table/claims | 2 | 13.1 | 19.9 |
| main/sort_table/claims_labels | 1 | 0.1 | 0.1 |
| main/sort_table/descriptions | 2 | 1.1 | 3.2 |
| main/sort_table/entities | 2 | 0.2 | 0.3 |
| main/sort_table/labels | 2 | 2.2 | 7.6 |
| main/sort_table/links | 2 | 2.1 | 6.1 |
| scholar/build_claims_labels | 1 | 1.9 | 5.9 |
| scholar/close_group | 1 | 0.9 | 0.9 |
| scholar/collect_refs_stage | 1 | 0.5 | 1.5 |
| scholar/compact_table/aliases | 1 | 0.1 | 0.2 |
| scholar/compact_table/claims | 1 | 0.1 | 0.1 |
| scholar/compact_table/claims_labels | 1 | 0.0 | 0.0 |
| scholar/compact_table/descriptions | 1 | 0.2 | 0.2 |
| scholar/compact_table/entities | 1 | 0.0 | 0.0 |
| scholar/compact_table/labels | 1 | 0.1 | 0.2 |
| scholar/compact_table/links | 1 | 0.0 | 0.0 |
| scholar/sort_table/aliases | 1 | 0.1 | 0.2 |
| scholar/sort_table/claims | 1 | 18.6 | 23.8 |
| scholar/sort_table/claims_labels | 1 | 0.1 | 0.1 |
| scholar/sort_table/descriptions | 1 | 0.7 | 2.3 |
| scholar/sort_table/entities | 1 | 0.1 | 0.1 |
| scholar/sort_table/labels | 1 | 1.4 | 5.0 |
| scholar/sort_table/links | 1 | 0.1 | 0.1 |

</details>

## Transfers a release would make (logged by the stand-in Hub, not timed)

| way | set/kind | calls | files | GB |
|---|---|--:|--:|--:|
| upload | main/commit | 16 | 2842 | 0.07 |
| upload | main/file | 6 | 6 | 0.00 |
| upload | main/group | 6 | 2324 | 0.07 |
| upload | scholar/commit | 8 | 606 | 0.03 |
| upload | scholar/file | 6 | 6 | 0.00 |
| upload | scholar/group | 6 | 176 | 0.03 |
| download | main/snapshot | 6 | 2324 | 0.07 |
| download | scholar/snapshot | 6 | 176 | 0.03 |

Uploads in all: 0.20 GB (at 7 MB/s, the one rate on record: 0.0 h; at 60 MB/s: 0.0 h; modelled, not measured).
The stand-in's own cost on the timed path (the Hub-side sha256 it computes): 0.9 s CPU over 0.20 GB.
Linked files changed after placing: 0 of 3448 checked (0 copied, not linked). Published digest: 8dffb86c22cb7c473f6de801ba2d2a8a119c75773b489deff9c631fb4527e046

## Per GB of bz2, against the CI bench's model

0.0792 GB of bz2; JSON 13.69x its bz2 bytes (the CI bench's slice 13.69x, #194's 12 windows of the real dump 17.46x).

| stage | CPU s / GB | CPU / 4 cores, s / GB (measured) | model s / GB (CI bench) | measured / model | wall s / GB | measured / model, a JSON byte |
|---|--:|--:|--:|--:|--:|--:|
| split | 1042.1 | 260.5 | 116.0 | 2.25 | 298.0 | 2.25 |
| pipeline | 219.8 | 55.0 | 55.1 | 1.0 | 162.6 | 1.0 |
| finalise | 1394.7 | 348.7 | 77.1 | 4.52 | 765.2 | 4.52 |

The model: bench 17.12.0's release model at master 8f7720b (6 runs, AMD EPYC 7763, 4 cores): a stage's sample CPU time scaled to the release and spread over 4 cores. Its figures are modelled; this run's are measured. A JSON byte's: the model's CPU over its slice's JSON (13.69x its bz2 bytes).

**Modelled** (this run's rates, linear in bz2 bytes, to the release's 103.5 GB): wall 35.3 h, CPU 76.4 h, peak disk ~246 GB (2.38x the bz2). this scaled run's figures times the release's bz2 bytes over its own: linear in bytes, which a sample of the dump's mix only approximates; modelled, not measured.
