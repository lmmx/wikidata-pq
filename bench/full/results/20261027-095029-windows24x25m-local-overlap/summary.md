# Compute benchmark: 20261027-095029-windows24x25m-local-overlap

- **Result:** ok; both sets finalised: {'main': True, 'scholar': True}
- **Dump:** release 20261019, windows24x25m: 0.60 GB of bz2, 11.19 GB of JSON; the whole dump's md5 cfec3f8ab5fd8f6f0b70f58c590a02da, sha1 a05b8b809ab80576da6d838c67138ddf401055e6
- **Source:** local (no network on the timed path); orchestration: overlap; cards: False
- **Machine:** Intel(R) Xeon(R) Processor @ 2.80GHz, 4 CPUs, 16.87 GB RAM, ext4 on /dev/vda (model unknown), Ubuntu 24.04.5 LTS, kernel 6.18.44-fc-v165
- **Code:** f55a5f8-dirty (f55a5f82d3eae1957cb2163aabf4b0c00c9aea8e), wikidata 0.1.0, wikidata-reshape 0.1.0, polars 2.0.0, pyarrow 25.0.1, pq-glue 0.3.0, polars-genson 1.6.0, orjson 3.12.0, numpy 2.5.3, huggingface-hub 1.32.0, hf-xet 1.6.0
- **Decoder:** ['/usr/bin/lbzip2', '-dc']; WIKIDATA_BZ2_DECODER=None, WIKIDATA_BZ2_THREADS=None, WIKIDATA_SPLIT_WRITER_THREADS=0

## Measured

- **Wall, end to end:** 482 s; **CPU:** 1426 s (2.96 cores on average)
- **Time to the first processed chunk** (the split's first chunk written): 9.0 s; each set's first part ready for its run: {'main': 9.0, 'scholar': 9.0} s (±1 s, the monitor's interval)
- **Peak RSS, every process at once:** 12185 MB (the largest process 11849 MB); the machine's memory in use at its peak 12822 MB (least available 4051 MB, other work included)
- **Disk:** peak 1.39 GB taken (free space's fall), work tree peak 1.91 GB; the device read 0.62 GB and wrote 2.21 GB; the stages' own I/O read 0.61 GB and wrote 2.95 GB
- **Network:** received 0.00 GB, sent 0.01 GB (the machine's, every process)
- **Other work on the machine:** 26.4 CPU s besides the run's own (0.05 cores on average; mean load 8.94). Near 0, the run had the machine to itself

| stage | start s | wall s | CPU s | cores | other CPU s | peak RSS MB (tree) | read GB | written GB | machine: user / sys / iowait / idle | bound |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| split | 0 | 217.3 | 806.9 | 3.71 | 41.3 | 2270 | 0.61 | 1.01 | 91% / 6% / 0% / 1% | CPU, every core |
| run_scholar | 0 | 240.4 | 19.6 | 0.08 | 860.5 | 537 | 0.00 | 0.35 | 85% / 7% / 0% / 7% | CPU on 0.1 of 4 cores, the machine full (other stages, or other work) |
| run_main | 0 | 237.8 | 36.2 | 0.15 | 839.9 | 719 | 0.00 | 0.50 | 85% / 7% / 0% / 6% | CPU on 0.2 of 4 cores, the machine full (other stages, or other work) |
| finalise | 238 | 237.7 | 555.7 | 2.34 | 14.5 | 12185 | 0.01 | 1.10 | 34% / 26% / 0% / 38% | CPU on 2.3 of 4 cores (serial parts, or waiting on another stage) |
| tail_run_scholar | 476 | 1.0 | 1.4 | 1.31 | 2.3 | 51 | 0.00 | 0.00 | 39% / 6% / 0% / 53% | CPU on 1.3 of 4 cores (serial parts, or waiting on another stage) |
| tail_run_main | 477 | 1.1 | 1.4 | 1.32 | 2.9 | 165 | 0.00 | 0.00 | 30% / 6% / 0% / 63% | CPU on 1.3 of 4 cores (serial parts, or waiting on another stage) |
| tail_finalise | 478 | 4.1 | 5.5 | 1.34 | 0.5 | 190 | 0.00 | 0.00 | 29% / 6% / 0% / 64% | CPU on 1.3 of 4 cores (serial parts, or waiting on another stage) |

(Overlapped stages share the machine: their machine-wide split is the window's, not theirs alone.)

| finalise run | start s | wall s | CPU s | largest process MB |
|---|--:|--:|--:|--:|
| finalise_run1_main | 239 | 140.4 | 248.0 | 11324 |
| finalise_run2_scholar | 379 | 91.0 | 288.4 | 2943 |
| finalise_run3_main | 469 | 6.5 | 18.0 | 371 |
| tail_finalise_run1_main | 479 | 1.1 | 1.4 | 144 |
| tail_finalise_run2_scholar | 480 | 1.1 | 1.4 | 144 |
| tail_finalise_run3_main | 481 | 1.0 | 1.3 | 144 |

**claims_labels** (its build, compaction and sort, within finalise's runs): main 4.5 s wall, 15.0 s CPU; scholar 3.1 s wall, 10.2 s CPU

<details><summary>finalise's steps and the group closes (wall, CPU incl. workers)</summary>

| step | calls | wall s | CPU s |
|---|--:|--:|--:|
| main/build_claims_labels | 3 | 4.4 | 14.9 |
| main/close_group | 1 | 18.8 | 20.5 |
| main/collect_refs_stage | 4 | 0.9 | 2.0 |
| main/compact_table/aliases | 4 | 1.5 | 1.8 |
| main/compact_table/claims | 4 | 0.9 | 0.9 |
| main/compact_table/claims_labels | 3 | 0.0 | 0.0 |
| main/compact_table/descriptions | 4 | 0.9 | 0.8 |
| main/compact_table/entities | 4 | 0.2 | 0.2 |
| main/compact_table/labels | 4 | 1.1 | 1.9 |
| main/compact_table/links | 4 | 1.0 | 0.9 |
| main/sort_table/aliases | 4 | 1.9 | 6.4 |
| main/sort_table/claims | 4 | 113.7 | 183.2 |
| main/sort_table/claims_labels | 3 | 0.1 | 0.1 |
| main/sort_table/descriptions | 4 | 4.8 | 17.4 |
| main/sort_table/entities | 4 | 1.8 | 2.4 |
| main/sort_table/labels | 4 | 5.0 | 18.1 |
| main/sort_table/links | 4 | 2.2 | 6.7 |
| scholar/build_claims_labels | 2 | 3.0 | 10.1 |
| scholar/close_group | 1 | 21.5 | 5.9 |
| scholar/collect_refs_stage | 2 | 1.8 | 4.8 |
| scholar/compact_table/aliases | 2 | 0.2 | 0.2 |
| scholar/compact_table/claims | 2 | 0.8 | 0.8 |
| scholar/compact_table/claims_labels | 2 | 0.0 | 0.0 |
| scholar/compact_table/descriptions | 2 | 0.3 | 0.3 |
| scholar/compact_table/entities | 2 | 0.0 | 0.0 |
| scholar/compact_table/labels | 2 | 0.3 | 0.6 |
| scholar/compact_table/links | 2 | 0.2 | 0.3 |
| scholar/sort_table/aliases | 2 | 0.1 | 0.4 |
| scholar/sort_table/claims | 2 | 74.7 | 244.4 |
| scholar/sort_table/claims_labels | 2 | 0.1 | 0.1 |
| scholar/sort_table/descriptions | 2 | 3.5 | 12.9 |
| scholar/sort_table/entities | 2 | 0.5 | 0.5 |
| scholar/sort_table/labels | 2 | 2.6 | 9.2 |
| scholar/sort_table/links | 2 | 0.1 | 0.2 |

</details>

## Transfers a release would make (logged by the stand-in Hub, not timed)

| way | set/kind | calls | files | GB |
|---|---|--:|--:|--:|
| upload | main/commit | 14 | 2453 | 0.42 |
| upload | main/file | 6 | 6 | 0.00 |
| upload | main/group | 6 | 1982 | 0.45 |
| upload | scholar/commit | 7 | 542 | 0.33 |
| upload | scholar/file | 6 | 6 | 0.00 |
| upload | scholar/group | 6 | 305 | 0.34 |
| download | main/snapshot | 6 | 1982 | 0.45 |
| download | scholar/snapshot | 6 | 305 | 0.34 |

Uploads in all: 1.53 GB (at 7 MB/s, the one rate on record: 0.1 h; at 60 MB/s: 0.0 h; modelled, not measured).
The stand-in's own cost on the timed path (the Hub-side sha256 it computes): 5.3 s CPU over 1.53 GB.
Linked files changed after placing: 0 of 2995 checked (0 copied, not linked). Published digest: 3ad0c4aad5deeb7c6450d506038d34a1daace866f4c54d4e68ce65b4a75e3d48

## Per GB of bz2, against the CI bench's model

0.5988 GB of bz2; JSON 18.68x its bz2 bytes (the CI bench's slice 13.69x, #194's 12 windows of the real dump 17.46x).

| stage | CPU s / GB | CPU / 4 cores, s / GB (measured) | model s / GB (CI bench) | measured / model | wall s / GB | measured / model, a JSON byte |
|---|--:|--:|--:|--:|--:|--:|
| split | 1347.5 | 336.9 | 116.0 | 2.9 | 362.9 | 2.13 |
| pipeline | 93.1 | 23.3 | 55.1 | 0.42 | – | 0.31 |
| finalise | 937.1 | 234.3 | 77.1 | 3.04 | – | 2.23 |

The model: bench 17.12.0's release model at master 8f7720b (6 runs, AMD EPYC 7763, 4 cores): a stage's sample CPU time scaled to the release and spread over 4 cores. Its figures are modelled; this run's are measured. A JSON byte's: the model's CPU over its slice's JSON (13.69x its bz2 bytes).

**Modelled** (this run's rates, linear in bz2 bytes, to the release's 103.5 GB): wall 23.1 h, CPU 68.5 h, peak disk ~241 GB (2.33x the bz2). this scaled run's figures times the release's bz2 bytes over its own: linear in bytes, which a sample of the dump's mix only approximates; modelled, not measured.
