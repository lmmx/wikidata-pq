# Compute benchmark: 20261027-074539-windows24x25m-local-serial

- **Result:** ok; both sets finalised: {'main': True, 'scholar': True}
- **Dump:** release 20261019, windows24x25m: 0.60 GB of bz2, 11.19 GB of JSON; the whole dump's md5 cfec3f8ab5fd8f6f0b70f58c590a02da, sha1 a05b8b809ab80576da6d838c67138ddf401055e6
- **Source:** local (no network on the timed path); orchestration: serial; cards: False
- **Machine:** Intel(R) Xeon(R) Processor @ 2.80GHz, 4 CPUs, 16.87 GB RAM, ext4 on /dev/vda (model unknown), Ubuntu 24.04.5 LTS, kernel 6.18.44-fc-v165
- **Code:** 95fdbf7-dirty (95fdbf7e894aa6184a79050bac3f5320ef9660a1), wikidata 0.1.0, wikidata-reshape 0.1.0, polars 2.0.0, pyarrow 25.0.1, pq-glue 0.3.0, polars-genson 1.6.0, orjson 3.12.0, numpy 2.5.3, huggingface-hub 1.32.0, hf-xet 1.6.0
- **Decoder:** ['/usr/bin/lbzip2', '-dc']; WIKIDATA_BZ2_DECODER=None, WIKIDATA_BZ2_THREADS=None, WIKIDATA_SPLIT_WRITER_THREADS=None

## Measured

- **Wall, end to end:** 415 s; **CPU:** 1195 s (2.88 cores on average)
- **Time to the first processed chunk** (the split's first chunk written): None s; each set's first part ready for its run: {'main': None, 'scholar': None} s (±1 s, the monitor's interval)
- **Peak RSS, every process at once:** 12155 MB (the largest process 11818 MB); the machine's memory in use at its peak 12897 MB (least available 3976 MB, other work included)
- **Disk:** peak 1.33 GB taken (free space's fall), work tree peak 1.70 GB; the device read 0.19 GB and wrote 2.22 GB; the stages' own I/O read 0.19 GB and wrote 2.93 GB
- **Network:** received 0.00 GB, sent 0.00 GB (the machine's, every process)
- **Other work on the machine:** 13.5 CPU s besides the run's own (0.03 cores on average; mean load 4.85). Near 0, the run had the machine to itself

| stage | start s | wall s | CPU s | cores | other CPU s | peak RSS MB (tree) | read GB | written GB | machine: user / sys / iowait / idle | bound |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| split | 0 | 165.1 | 638.1 | 3.86 | 4.0 | 2409 | 0.17 | 1.00 | 92% / 6% / 0% / 1% | CPU, every core |
| run_scholar | 165 | 10.1 | 18.1 | 1.78 | 1.9 | 1244 | 0.00 | 0.35 | 34% / 11% / 1% / 52% | CPU on 1.8 of 4 cores (serial parts, or waiting on another stage) |
| run_main | 175 | 23.6 | 33.2 | 1.41 | 2.7 | 1232 | 0.00 | 0.49 | 26% / 10% / 1% / 62% | CPU on 1.4 of 4 cores (serial parts, or waiting on another stage) |
| finalise | 199 | 216.1 | 505.2 | 2.34 | 8.7 | 12155 | 0.02 | 1.10 | 34% / 25% / 0% / 39% | CPU on 2.3 of 4 cores (serial parts, or waiting on another stage) |

| finalise run | start s | wall s | CPU s | largest process MB |
|---|--:|--:|--:|--:|
| finalise_run1_main | 200 | 126.5 | 224.7 | 11299 |
| finalise_run2_scholar | 326 | 83.1 | 262.3 | 2987 |
| finalise_run3_main | 409 | 5.8 | 16.8 | 373 |

**claims_labels** (its build, compaction and sort, within finalise's runs): main 4.1 s wall, 13.8 s CPU; scholar 2.9 s wall, 9.7 s CPU

<details><summary>finalise's steps and the group closes (wall, CPU incl. workers)</summary>

| step | calls | wall s | CPU s |
|---|--:|--:|--:|
| main/build_claims_labels | 1 | 3.9 | 13.7 |
| main/close_group | 1 | 18.0 | 19.7 |
| main/collect_refs_stage | 2 | 0.7 | 1.5 |
| main/compact_table/aliases | 2 | 1.6 | 1.9 |
| main/compact_table/claims | 2 | 1.0 | 0.9 |
| main/compact_table/claims_labels | 1 | 0.0 | 0.0 |
| main/compact_table/descriptions | 2 | 1.1 | 0.7 |
| main/compact_table/entities | 2 | 0.1 | 0.1 |
| main/compact_table/labels | 2 | 0.8 | 1.5 |
| main/compact_table/links | 2 | 0.8 | 0.7 |
| main/sort_table/aliases | 2 | 1.9 | 6.4 |
| main/sort_table/claims | 2 | 103.3 | 168.2 |
| main/sort_table/claims_labels | 1 | 0.1 | 0.1 |
| main/sort_table/descriptions | 2 | 4.2 | 15.2 |
| main/sort_table/entities | 2 | 1.2 | 1.4 |
| main/sort_table/labels | 2 | 4.4 | 15.7 |
| main/sort_table/links | 2 | 1.7 | 5.6 |
| scholar/build_claims_labels | 1 | 2.8 | 9.6 |
| scholar/close_group | 1 | 5.1 | 6.1 |
| scholar/collect_refs_stage | 1 | 1.8 | 3.8 |
| scholar/compact_table/aliases | 1 | 0.2 | 0.2 |
| scholar/compact_table/claims | 1 | 0.9 | 0.8 |
| scholar/compact_table/claims_labels | 1 | 0.0 | 0.0 |
| scholar/compact_table/descriptions | 1 | 0.3 | 0.3 |
| scholar/compact_table/entities | 1 | 0.0 | 0.0 |
| scholar/compact_table/labels | 1 | 0.2 | 0.5 |
| scholar/compact_table/links | 1 | 0.2 | 0.2 |
| scholar/sort_table/aliases | 1 | 0.2 | 0.4 |
| scholar/sort_table/claims | 1 | 67.9 | 222.1 |
| scholar/sort_table/claims_labels | 1 | 0.1 | 0.1 |
| scholar/sort_table/descriptions | 1 | 3.3 | 12.6 |
| scholar/sort_table/entities | 1 | 0.4 | 0.5 |
| scholar/sort_table/labels | 1 | 2.2 | 8.0 |
| scholar/sort_table/links | 1 | 0.1 | 0.2 |

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
The stand-in's own cost on the timed path (the Hub-side sha256 it computes): 4.7 s CPU over 1.53 GB.
Linked files changed after placing: 0 of 2995 checked (0 copied, not linked). Published digest: 3ad0c4aad5deeb7c6450d506038d34a1daace866f4c54d4e68ce65b4a75e3d48

## Per GB of bz2, against the CI bench's model

0.5988 GB of bz2; JSON 18.68x its bz2 bytes (the CI bench's slice 13.69x, #194's 12 windows of the real dump 17.46x).

| stage | CPU s / GB | CPU / 4 cores, s / GB (measured) | model s / GB (CI bench) | measured / model | wall s / GB | measured / model, a JSON byte |
|---|--:|--:|--:|--:|--:|--:|
| split | 1065.7 | 266.4 | 116.0 | 2.3 | 275.8 | 1.68 |
| pipeline | 85.7 | 21.4 | 55.1 | 0.39 | 56.3 | 0.29 |
| finalise | 843.6 | 210.9 | 77.1 | 2.74 | 360.8 | 2.0 |

The model: bench 17.12.0's release model at master 8f7720b (6 runs, AMD EPYC 7763, 4 cores): a stage's sample CPU time scaled to the release and spread over 4 cores. Its figures are modelled; this run's are measured. A JSON byte's: the model's CPU over its slice's JSON (13.69x its bz2 bytes).

**Modelled** (this run's rates, linear in bz2 bytes, to the release's 103.5 GB): wall 19.9 h, CPU 57.4 h, peak disk ~230 GB (2.23x the bz2). this scaled run's figures times the release's bz2 bytes over its own: linear in bytes, which a sample of the dump's mix only approximates; modelled, not measured.
