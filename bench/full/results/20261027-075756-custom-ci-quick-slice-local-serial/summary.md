# Compute benchmark: 20261027-075756-custom-ci-quick-slice-local-serial

- **Result:** ok; both sets finalised: {'main': True, 'scholar': True}
- **Dump:** release 20261019, custom-ci-quick-slice: 0.08 GB of bz2, 1.08 GB of JSON; the whole dump's md5 None, sha1 None
- **Source:** local (no network on the timed path); orchestration: serial; cards: False
- **Machine:** Intel(R) Xeon(R) Processor @ 2.80GHz, 4 CPUs, 16.87 GB RAM, ext4 on /dev/vda (model unknown), Ubuntu 24.04.5 LTS, kernel 6.18.44-fc-v165
- **Code:** 95fdbf7-dirty (95fdbf7e894aa6184a79050bac3f5320ef9660a1), wikidata 0.1.0, wikidata-reshape 0.1.0, polars 2.0.0, pyarrow 25.0.1, pq-glue 0.3.0, polars-genson 1.6.0, orjson 3.12.0, numpy 2.5.3, huggingface-hub 1.32.0, hf-xet 1.6.0
- **Decoder:** ['/usr/bin/lbzip2', '-dc']; WIKIDATA_BZ2_DECODER=None, WIKIDATA_BZ2_THREADS=None, WIKIDATA_SPLIT_WRITER_THREADS=None

## Measured

- **Wall, end to end:** 85 s; **CPU:** 185 s (2.17 cores on average)
- **Time to the first processed chunk** (the split's first chunk written): None s; each set's first part ready for its run: {'main': None, 'scholar': None} s (±1 s, the monitor's interval)
- **Peak RSS, every process at once:** 2783 MB (the largest process 2771 MB); the machine's memory in use at its peak 3913 MB (least available 12960 MB, other work included)
- **Disk:** peak 0.19 GB taken (free space's fall), work tree peak 0.25 GB; the device read 0.00 GB and wrote 0.26 GB; the stages' own I/O read 0.00 GB and wrote 0.48 GB
- **Network:** received 0.00 GB, sent 0.00 GB (the machine's, every process)
- **Other work on the machine:** 5.1 CPU s besides the run's own (0.06 cores on average; mean load 2.5). Near 0, the run had the machine to itself

| stage | start s | wall s | CPU s | cores | other CPU s | peak RSS MB (tree) | read GB | written GB | machine: user / sys / iowait / idle | bound |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| split | 0 | 19.1 | 67.1 | 3.52 | 2.0 | 2783 | 0.00 | 0.16 | 77% / 10% / 0% / 12% | CPU, every core |
| run_scholar | 19 | 2.8 | 5.1 | 1.79 | 0.3 | 520 | 0.00 | 0.04 | 36% / 9% / 0% / 54% | CPU on 1.8 of 4 cores (serial parts, or waiting on another stage) |
| run_main | 22 | 7.5 | 9.1 | 1.21 | 1.6 | 295 | 0.00 | 0.11 | 22% / 12% / 1% / 65% | CPU on 1.2 of 4 cores (serial parts, or waiting on another stage) |
| finalise | 30 | 55.5 | 103.6 | 1.86 | 4.0 | 2609 | 0.00 | 0.18 | 32% / 16% / 1% / 50% | CPU on 1.9 of 4 cores (serial parts, or waiting on another stage) |

| finalise run | start s | wall s | CPU s | largest process MB |
|---|--:|--:|--:|--:|
| finalise_run1_main | 30 | 28.2 | 50.4 | 1798 |
| finalise_run2_scholar | 57 | 22.7 | 37.4 | 2289 |
| finalise_run3_main | 79 | 5.5 | 14.4 | 219 |

**claims_labels** (its build, compaction and sort, within finalise's runs): main 3.5 s wall, 11.3 s CPU; scholar 2.0 s wall, 5.7 s CPU

<details><summary>finalise's steps and the group closes (wall, CPU incl. workers)</summary>

| step | calls | wall s | CPU s |
|---|--:|--:|--:|
| main/build_claims_labels | 1 | 3.3 | 11.2 |
| main/close_group | 1 | 5.4 | 5.7 |
| main/collect_refs_stage | 2 | 0.3 | 0.6 |
| main/compact_table/aliases | 2 | 0.8 | 0.8 |
| main/compact_table/claims | 2 | 0.2 | 0.1 |
| main/compact_table/claims_labels | 1 | 0.0 | 0.0 |
| main/compact_table/descriptions | 2 | 1.4 | 1.6 |
| main/compact_table/entities | 2 | 0.1 | 0.1 |
| main/compact_table/labels | 2 | 0.7 | 0.8 |
| main/compact_table/links | 2 | 1.6 | 1.6 |
| main/sort_table/aliases | 2 | 1.1 | 3.1 |
| main/sort_table/claims | 2 | 12.1 | 18.7 |
| main/sort_table/claims_labels | 1 | 0.1 | 0.1 |
| main/sort_table/descriptions | 2 | 1.1 | 3.1 |
| main/sort_table/entities | 2 | 0.2 | 0.2 |
| main/sort_table/labels | 2 | 2.5 | 8.3 |
| main/sort_table/links | 2 | 2.0 | 6.0 |
| scholar/build_claims_labels | 1 | 1.9 | 5.5 |
| scholar/close_group | 1 | 0.6 | 0.7 |
| scholar/collect_refs_stage | 1 | 0.4 | 1.4 |
| scholar/compact_table/aliases | 1 | 0.1 | 0.2 |
| scholar/compact_table/claims | 1 | 0.1 | 0.1 |
| scholar/compact_table/claims_labels | 1 | 0.0 | 0.0 |
| scholar/compact_table/descriptions | 1 | 0.1 | 0.2 |
| scholar/compact_table/entities | 1 | 0.0 | 0.0 |
| scholar/compact_table/labels | 1 | 0.1 | 0.2 |
| scholar/compact_table/links | 1 | 0.0 | 0.0 |
| scholar/sort_table/aliases | 1 | 0.1 | 0.2 |
| scholar/sort_table/claims | 1 | 15.5 | 20.2 |
| scholar/sort_table/claims_labels | 1 | 0.1 | 0.1 |
| scholar/sort_table/descriptions | 1 | 0.5 | 1.7 |
| scholar/sort_table/entities | 1 | 0.1 | 0.1 |
| scholar/sort_table/labels | 1 | 1.1 | 4.2 |
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
| split | 847.2 | 211.8 | 116.0 | 1.83 | 241.0 | 1.83 |
| pipeline | 179.0 | 44.7 | 55.1 | 0.81 | 130.6 | 0.81 |
| finalise | 1307.0 | 326.7 | 77.1 | 4.24 | 700.9 | 4.24 |

The model: bench 17.12.0's release model at master 8f7720b (6 runs, AMD EPYC 7763, 4 cores): a stage's sample CPU time scaled to the release and spread over 4 cores. Its figures are modelled; this run's are measured. A JSON byte's: the model's CPU over its slice's JSON (13.69x its bz2 bytes).

**Modelled** (this run's rates, linear in bz2 bytes, to the release's 103.5 GB): wall 30.9 h, CPU 67.1 h, peak disk ~244 GB (2.36x the bz2). this scaled run's figures times the release's bz2 bytes over its own: linear in bytes, which a sample of the dump's mix only approximates; modelled, not measured.
