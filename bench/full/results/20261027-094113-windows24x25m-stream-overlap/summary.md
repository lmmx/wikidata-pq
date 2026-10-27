# Compute benchmark: 20261027-094113-windows24x25m-stream-overlap

- **Result:** ok; both sets finalised: {'main': True, 'scholar': True}
- **Dump:** release 20261019, windows24x25m: 0.60 GB of bz2, 11.19 GB of JSON; the whole dump's md5 cfec3f8ab5fd8f6f0b70f58c590a02da, sha1 a05b8b809ab80576da6d838c67138ddf401055e6
- **Source:** stream (fetched as the run goes, the split following it); orchestration: overlap; cards: False
- **Machine:** Intel(R) Xeon(R) Processor @ 2.80GHz, 4 CPUs, 16.87 GB RAM, ext4 on /dev/vda (model unknown), Ubuntu 24.04.5 LTS, kernel 6.18.44-fc-v165
- **Code:** f55a5f8-dirty (f55a5f82d3eae1957cb2163aabf4b0c00c9aea8e), wikidata 0.1.0, wikidata-reshape 0.1.0, polars 2.0.0, pyarrow 25.0.1, pq-glue 0.3.0, polars-genson 1.6.0, orjson 3.12.0, numpy 2.5.3, huggingface-hub 1.32.0, hf-xet 1.6.0
- **Decoder:** ['/usr/bin/lbzip2', '-dc']; WIKIDATA_BZ2_DECODER=None, WIKIDATA_BZ2_THREADS=None, WIKIDATA_SPLIT_WRITER_THREADS=0

## Measured

- **Wall, end to end:** 518 s; **CPU:** 1504 s (2.90 cores on average)
- **Time to the first processed chunk** (the split's first chunk written): 12.0 s; each set's first part ready for its run: {'main': 12.0, 'scholar': 12.0} s (±1 s, the monitor's interval)
- **Peak RSS, every process at once:** 12131 MB (the largest process 11795 MB); the machine's memory in use at its peak 12922 MB (least available 3951 MB, other work included)
- **Disk:** peak 1.37 GB taken (free space's fall), work tree peak 1.70 GB; the device read 0.04 GB and wrote 2.82 GB; the stages' own I/O read 0.04 GB and wrote 3.54 GB
- **Network:** received 0.61 GB, sent 0.01 GB (the machine's, every process)
- **Other work on the machine:** 42.2 CPU s besides the run's own (0.08 cores on average; mean load 8.56). Near 0, the run had the machine to itself

| stage | start s | wall s | CPU s | cores | other CPU s | peak RSS MB (tree) | read GB | written GB | machine: user / sys / iowait / idle | bound |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| download | 0 | 86.7 | 36.4 | 0.42 | 289.4 | 175 | 0.00 | 0.60 | 87% / 7% / 0% / 5% | CPU on 0.4 of 4 cores, the machine full (other stages, or other work) |
| split | 0 | 237.7 | 823.6 | 3.47 | 88.7 | 2343 | 0.00 | 1.00 | 90% / 6% / 0% / 3% | CPU, every core |
| run_scholar | 0 | 262.9 | 19.9 | 0.08 | 928.8 | 538 | 0.00 | 0.35 | 84% / 6% / 0% / 8% | CPU on 0.1 of 4 cores, the machine full (other stages, or other work) |
| run_main | 0 | 260.2 | 37.1 | 0.14 | 906.0 | 530 | 0.00 | 0.49 | 84% / 6% / 0% / 8% | CPU on 0.1 of 4 cores, the machine full (other stages, or other work) |
| finalise | 260 | 249.8 | 577.5 | 2.31 | 18.0 | 12131 | 0.04 | 1.09 | 34% / 25% / 0% / 39% | CPU on 2.3 of 4 cores (serial parts, or waiting on another stage) |
| tail_run_scholar | 510 | 1.4 | 1.8 | 1.23 | 0.7 | 74 | 0.00 | 0.00 | 26% / 5% / 0% / 68% | CPU on 1.2 of 4 cores (serial parts, or waiting on another stage) |
| tail_run_main | 512 | 1.2 | 1.5 | 1.25 | 2.6 | 165 | 0.00 | 0.00 | 29% / 5% / 0% / 65% | CPU on 1.3 of 4 cores (serial parts, or waiting on another stage) |
| tail_finalise | 513 | 5.3 | 6.1 | 1.15 | 0.8 | 320 | 0.00 | 0.00 | 26% / 5% / 0% / 68% | CPU on 1.2 of 4 cores (serial parts, or waiting on another stage) |

(Overlapped stages share the machine: their machine-wide split is the window's, not theirs alone.)

| finalise run | start s | wall s | CPU s | largest process MB |
|---|--:|--:|--:|--:|
| finalise_run1_main | 261 | 142.2 | 247.5 | 11325 |
| finalise_run2_scholar | 403 | 99.7 | 308.1 | 2934 |
| finalise_run3_main | 502 | 7.4 | 20.5 | 370 |
| tail_finalise_run1_main | 514 | 1.2 | 1.5 | 145 |
| tail_finalise_run2_scholar | 515 | 1.4 | 1.7 | 144 |
| tail_finalise_run3_main | 517 | 1.2 | 1.5 | 145 |

**claims_labels** (its build, compaction and sort, within finalise's runs): main 4.9 s wall, 16.7 s CPU; scholar 3.4 s wall, 11.3 s CPU

<details><summary>finalise's steps and the group closes (wall, CPU incl. workers)</summary>

| step | calls | wall s | CPU s |
|---|--:|--:|--:|
| main/build_claims_labels | 3 | 4.8 | 16.6 |
| main/close_group | 1 | 20.3 | 21.8 |
| main/collect_refs_stage | 4 | 0.8 | 2.1 |
| main/compact_table/aliases | 4 | 1.7 | 1.9 |
| main/compact_table/claims | 4 | 1.0 | 1.0 |
| main/compact_table/claims_labels | 3 | 0.0 | 0.0 |
| main/compact_table/descriptions | 4 | 0.9 | 0.8 |
| main/compact_table/entities | 4 | 0.1 | 0.1 |
| main/compact_table/labels | 4 | 1.1 | 1.9 |
| main/compact_table/links | 4 | 1.3 | 1.2 |
| main/sort_table/aliases | 4 | 2.3 | 7.1 |
| main/sort_table/claims | 4 | 114.0 | 178.8 |
| main/sort_table/claims_labels | 3 | 0.1 | 0.1 |
| main/sort_table/descriptions | 4 | 5.3 | 18.1 |
| main/sort_table/entities | 4 | 1.8 | 2.2 |
| main/sort_table/labels | 4 | 5.2 | 19.4 |
| main/sort_table/links | 4 | 2.1 | 7.1 |
| scholar/build_claims_labels | 2 | 3.3 | 11.1 |
| scholar/close_group | 1 | 23.3 | 6.3 |
| scholar/collect_refs_stage | 2 | 2.0 | 4.4 |
| scholar/compact_table/aliases | 2 | 0.2 | 0.3 |
| scholar/compact_table/claims | 2 | 0.9 | 0.9 |
| scholar/compact_table/claims_labels | 2 | 0.0 | 0.0 |
| scholar/compact_table/descriptions | 2 | 0.3 | 0.3 |
| scholar/compact_table/entities | 2 | 0.1 | 0.1 |
| scholar/compact_table/labels | 2 | 0.2 | 0.5 |
| scholar/compact_table/links | 2 | 0.2 | 0.3 |
| scholar/sort_table/aliases | 2 | 0.2 | 0.5 |
| scholar/sort_table/claims | 2 | 82.2 | 262.4 |
| scholar/sort_table/claims_labels | 2 | 0.1 | 0.1 |
| scholar/sort_table/descriptions | 2 | 3.7 | 13.6 |
| scholar/sort_table/entities | 2 | 0.6 | 0.6 |
| scholar/sort_table/labels | 2 | 2.7 | 9.2 |
| scholar/sort_table/links | 2 | 0.2 | 0.3 |

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
The stand-in's own cost on the timed path (the Hub-side sha256 it computes): 5.5 s CPU over 1.53 GB.
Linked files changed after placing: 0 of 2995 checked (0 copied, not linked). Published digest: 3ad0c4aad5deeb7c6450d506038d34a1daace866f4c54d4e68ce65b4a75e3d48

## Streaming

- received 0.60 GB in 86.74 s: 6.9 MB/s on average; 10-s rates p10/p50/p90 {'p10': 4.33, 'p50': 7.43, 'p90': 9.11}
- the download ended at 86.8 s, the split at 237.7 s (150.9 s after it)
- the first chunk was processed at 12.0 s, before the download ended; by its end the split had written 24 chunks and numbered 24 main and 23 scholarly parts for the runs

## Per GB of bz2, against the CI bench's model

0.5988 GB of bz2; JSON 18.68x its bz2 bytes (the CI bench's slice 13.69x, #194's 12 windows of the real dump 17.46x).

| stage | CPU s / GB | CPU / 4 cores, s / GB (measured) | model s / GB (CI bench) | measured / model | wall s / GB | measured / model, a JSON byte |
|---|--:|--:|--:|--:|--:|--:|
| split | 1375.5 | 343.9 | 116.0 | 2.96 | 396.9 | 2.17 |
| pipeline | 95.2 | 23.8 | 55.1 | 0.43 | – | 0.32 |
| finalise | 974.6 | 243.6 | 77.1 | 3.16 | – | 2.32 |

The model: bench 17.12.0's release model at master 8f7720b (6 runs, AMD EPYC 7763, 4 cores): a stage's sample CPU time scaled to the release and spread over 4 cores. Its figures are modelled; this run's are measured. A JSON byte's: the model's CPU over its slice's JSON (13.69x its bz2 bytes).

**Modelled** (this run's rates, linear in bz2 bytes, to the release's 103.5 GB): wall 24.9 h, CPU 72.2 h, peak disk ~237 GB (2.29x the bz2). this scaled run's figures times the release's bz2 bytes over its own: linear in bytes, which a sample of the dump's mix only approximates; modelled, not measured.
