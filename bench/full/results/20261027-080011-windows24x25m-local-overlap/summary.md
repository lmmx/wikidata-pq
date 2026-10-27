# Compute benchmark: 20261027-080011-windows24x25m-local-overlap

- **Result:** ok; both sets finalised: {'main': True, 'scholar': True}
- **Dump:** release 20261019, windows24x25m: 0.60 GB of bz2, 11.19 GB of JSON; the whole dump's md5 cfec3f8ab5fd8f6f0b70f58c590a02da, sha1 a05b8b809ab80576da6d838c67138ddf401055e6
- **Source:** local (no network on the timed path); orchestration: overlap; cards: False
- **Machine:** Intel(R) Xeon(R) Processor @ 2.80GHz, 4 CPUs, 16.87 GB RAM, ext4 on /dev/vda (model unknown), Ubuntu 24.04.5 LTS, kernel 6.18.44-fc-v165
- **Code:** 95fdbf7-dirty (95fdbf7e894aa6184a79050bac3f5320ef9660a1), wikidata 0.1.0, wikidata-reshape 0.1.0, polars 2.0.0, pyarrow 25.0.1, pq-glue 0.3.0, polars-genson 1.6.0, orjson 3.12.0, numpy 2.5.3, huggingface-hub 1.32.0, hf-xet 1.6.0
- **Decoder:** ['/usr/bin/lbzip2', '-dc']; WIKIDATA_BZ2_DECODER=None, WIKIDATA_BZ2_THREADS=None, WIKIDATA_SPLIT_WRITER_THREADS=None

## Measured

- **Wall, end to end:** 445 s; **CPU:** 1311 s (2.95 cores on average)
- **Time to the first processed chunk** (the split's first chunk written): None s; each set's first part ready for its run: {'main': None, 'scholar': None} s (±1 s, the monitor's interval)
- **Peak RSS, every process at once:** 12071 MB (the largest process 11735 MB); the machine's memory in use at its peak 12667 MB (least available 4206 MB, other work included)
- **Disk:** peak 1.33 GB taken (free space's fall), work tree peak 1.70 GB; the device read 0.30 GB and wrote 2.29 GB; the stages' own I/O read 0.29 GB and wrote 2.94 GB
- **Network:** received 0.00 GB, sent 0.01 GB (the machine's, every process)
- **Other work on the machine:** 27.6 CPU s besides the run's own (0.06 cores on average; mean load 9.57). Near 0, the run had the machine to itself

| stage | start s | wall s | CPU s | cores | other CPU s | peak RSS MB (tree) | read GB | written GB | machine: user / sys / iowait / idle | bound |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| split | 0 | 189.2 | 700.0 | 3.70 | 38.9 | 2402 | 0.29 | 1.00 | 91% / 7% / 0% / 1% | CPU, every core |
| run_scholar | 0 | 212.4 | 19.9 | 0.09 | 753.0 | 537 | 0.00 | 0.35 | 84% / 8% / 0% / 7% | CPU on 0.1 of 4 cores, the machine full (other stages, or other work) |
| run_main | 0 | 209.8 | 36.8 | 0.18 | 731.9 | 872 | 0.00 | 0.49 | 84% / 8% / 0% / 7% | CPU on 0.2 of 4 cores, the machine full (other stages, or other work) |
| finalise | 210 | 228.4 | 546.0 | 2.39 | 17.2 | 12071 | 0.00 | 1.09 | 34% / 28% / 0% / 37% | CPU on 2.4 of 4 cores (serial parts, or waiting on another stage) |
| tail_run_scholar | 438 | 1.0 | 1.4 | 1.33 | 0.9 | 55 | 0.00 | 0.00 | 26% / 5% / 0% / 68% | CPU on 1.3 of 4 cores (serial parts, or waiting on another stage) |
| tail_run_main | 440 | 1.1 | 1.5 | 1.29 | 1.4 | 46 | 0.00 | 0.00 | 29% / 6% / 0% / 64% | CPU on 1.3 of 4 cores (serial parts, or waiting on another stage) |
| tail_finalise | 441 | 4.1 | 5.4 | 1.32 | 0.6 | 192 | 0.00 | 0.00 | 30% / 5% / 0% / 64% | CPU on 1.3 of 4 cores (serial parts, or waiting on another stage) |

(Overlapped stages share the machine: their machine-wide split is the window's, not theirs alone.)

| finalise run | start s | wall s | CPU s | largest process MB |
|---|--:|--:|--:|--:|
| finalise_run1_main | 211 | 130.6 | 239.2 | 11291 |
| finalise_run2_scholar | 341 | 90.0 | 286.2 | 2932 |
| finalise_run3_main | 431 | 7.1 | 19.1 | 349 |
| tail_finalise_run1_main | 441 | 1.1 | 1.4 | 145 |
| tail_finalise_run2_scholar | 442 | 1.0 | 1.3 | 144 |
| tail_finalise_run3_main | 443 | 1.1 | 1.4 | 144 |

**claims_labels** (its build, compaction and sort, within finalise's runs): main 4.7 s wall, 15.7 s CPU; scholar 3.3 s wall, 10.7 s CPU

<details><summary>finalise's steps and the group closes (wall, CPU incl. workers)</summary>

| step | calls | wall s | CPU s |
|---|--:|--:|--:|
| main/build_claims_labels | 3 | 4.5 | 15.5 |
| main/close_group | 1 | 20.1 | 21.9 |
| main/collect_refs_stage | 4 | 0.8 | 1.8 |
| main/compact_table/aliases | 4 | 1.4 | 1.7 |
| main/compact_table/claims | 4 | 0.9 | 0.9 |
| main/compact_table/claims_labels | 3 | 0.0 | 0.0 |
| main/compact_table/descriptions | 4 | 1.3 | 0.8 |
| main/compact_table/entities | 4 | 0.1 | 0.2 |
| main/compact_table/labels | 4 | 0.8 | 1.5 |
| main/compact_table/links | 4 | 0.7 | 0.6 |
| main/sort_table/aliases | 4 | 1.8 | 5.9 |
| main/sort_table/claims | 4 | 105.1 | 177.3 |
| main/sort_table/claims_labels | 3 | 0.1 | 0.1 |
| main/sort_table/descriptions | 4 | 5.4 | 19.5 |
| main/sort_table/entities | 4 | 1.8 | 2.3 |
| main/sort_table/labels | 4 | 4.4 | 16.3 |
| main/sort_table/links | 4 | 1.8 | 5.7 |
| scholar/build_claims_labels | 2 | 3.1 | 10.6 |
| scholar/close_group | 1 | 21.8 | 6.8 |
| scholar/collect_refs_stage | 2 | 1.5 | 4.2 |
| scholar/compact_table/aliases | 2 | 0.2 | 0.3 |
| scholar/compact_table/claims | 2 | 0.8 | 0.8 |
| scholar/compact_table/claims_labels | 2 | 0.0 | 0.0 |
| scholar/compact_table/descriptions | 2 | 0.5 | 0.4 |
| scholar/compact_table/entities | 2 | 0.0 | 0.0 |
| scholar/compact_table/labels | 2 | 0.5 | 0.7 |
| scholar/compact_table/links | 2 | 0.2 | 0.2 |
| scholar/sort_table/aliases | 2 | 0.2 | 0.4 |
| scholar/sort_table/claims | 2 | 74.1 | 244.6 |
| scholar/sort_table/claims_labels | 2 | 0.1 | 0.1 |
| scholar/sort_table/descriptions | 2 | 3.3 | 12.2 |
| scholar/sort_table/entities | 2 | 0.6 | 0.6 |
| scholar/sort_table/labels | 2 | 2.3 | 7.5 |
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
The stand-in's own cost on the timed path (the Hub-side sha256 it computes): 4.9 s CPU over 1.53 GB.
Linked files changed after placing: 0 of 2995 checked (0 copied, not linked). Published digest: 3ad0c4aad5deeb7c6450d506038d34a1daace866f4c54d4e68ce65b4a75e3d48

## Per GB of bz2, against the CI bench's model

0.5988 GB of bz2; JSON 18.68x its bz2 bytes (the CI bench's slice 13.69x, #194's 12 windows of the real dump 17.46x).

| stage | CPU s / GB | CPU / 4 cores, s / GB (measured) | model s / GB (CI bench) | measured / model | wall s / GB | measured / model, a JSON byte |
|---|--:|--:|--:|--:|--:|--:|
| split | 1169.1 | 292.3 | 116.0 | 2.52 | 316.0 | 1.85 |
| pipeline | 94.7 | 23.7 | 55.1 | 0.43 | – | 0.32 |
| finalise | 920.9 | 230.2 | 77.1 | 2.99 | – | 2.19 |

The model: bench 17.12.0's release model at master 8f7720b (6 runs, AMD EPYC 7763, 4 cores): a stage's sample CPU time scaled to the release and spread over 4 cores. Its figures are modelled; this run's are measured. A JSON byte's: the model's CPU over its slice's JSON (13.69x its bz2 bytes).

**Modelled** (this run's rates, linear in bz2 bytes, to the release's 103.5 GB): wall 21.4 h, CPU 62.9 h, peak disk ~230 GB (2.23x the bz2). this scaled run's figures times the release's bz2 bytes over its own: linear in bytes, which a sample of the dump's mix only approximates; modelled, not measured.
