# Records

One entry per run, appended by the Benchmark workflow (`bench/record.py`) to
`records.jsonl`, of which this file is a view: a push to master in the quick tier (the
new commit against the previous one), or a run of the full tier from the Actions tab.
Each entry carries the version of the harness that measured it (`bench/pyproject.toml`,
changes in `bench/CHANGELOG.md`): entries compare only within a major version. Speed is
the commit before's mean wall time over the commit after's, over the rounds.

## bench 17.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-27 | 17.16.0 | `dae8080` | `0c34b26` | 0.99× ± 0.05 (+1.4%) | 2.60 / 2.63 | 1.360 / 1.457 | 877 / 882 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40537433787) |
| 2026-10-27 | 17.15.0 | `0c34b26` | `8c186e2` | 1.01× ± 0.03 (-0.7%) | 2.93 / 2.90 | 1.338 / 1.456 | 772 / 904 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40531861806) |
| 2026-10-27 | 17.15.0 | `8c186e2` | `69d3d1e` | 1.00× ± 0.03 (+0.3%) | 2.90 / 2.91 | 1.349 / 1.421 | 940 / 779 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40528532689) |
| 2026-10-27 | 17.15.0 | `acd31e9` | `7fbc7eb` | 1.00× ± 0.06 (-0.5%) | 2.31 / 2.30 | 1.587 / 1.506 | 800 / 998 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40518670645) |
| 2026-10-27 | 17.15.0 | `7fbc7eb` | `dddbcde` | 0.98× ± 0.04 (+1.7%) | 2.73 / 2.75 | 1.435 / 1.532 | 858 / 815 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40518462023) |
| 2026-10-27 | 17.14.0 | `dddbcde` | `661ebbe` | 1.00× ± 0.02 (+0.3%) | 2.76 / 2.78 | 1.436 / 1.437 | 835 / 771 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40515818789) |
| 2026-10-27 | 17.14.0 | `661ebbe` | `71b5e76` | 0.99× ± 0.02 (+1.5%) | 2.84 / 2.88 | 1.427 / 1.351 | 914 / 891 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40497402341) |
| 2026-10-27 | 17.14.0 | `0695edf` | `9392714` | 1.05× ± 0.07 (-4.7%) | 2.68 / 2.58 | 1.432 / 1.455 | 770 / 770 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40484890305) |
| 2026-10-27 | 17.12.0 | `e80525d` | `95fdbf7` | 1.00× ± 0.02 (+0.1%) | 2.80 / 2.81 | 1.361 / 1.318 | 909 / 778 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40470713657) |
| 2026-10-26 | 17.12.0 | `8f7720b` | `f2da3c5` | 1.04× ± 0.03 (-4.0%) | 3.01 / 2.90 | 1.396 / 1.444 | 782 / 790 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40397547381) |
| 2026-10-26 | 17.12.0 | `98fc81c` | `673d13e` | 1.01× ± 0.05 (-1.0%) | 2.93 / 2.91 | 1.349 / 1.292 | 922 / 778 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40392153975) |
| 2026-10-26 | 17.9.0 | `b110f78` | `0de6ecf` | 0.99× ± 0.01 (+0.5%) | 2.67 / 2.68 | 1.383 / 1.346 | 910 / 930 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40319084239) |
| 2026-10-26 | 17.8.0 | `0de6ecf` | `fd03cd0` | 1.02× ± 0.04 (-2.3%) | 2.69 / 2.65 | 1.288 / 1.470 | 782 / 896 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40315907822) |
| 2026-10-26 | 17.6.0 | `fd03cd0` | `61f69e4` | 1.01× ± 0.01 (-0.9%) | 3.05 / 3.02 | 1.573 / 1.557 | 799 / 779 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40313704070) |
| 2026-10-26 | 17.6.0 | `61f69e4` | `831f225` | 1.01× ± 0.02 (-1.0%) | 2.87 / 2.85 | 1.568 / 1.621 | 786 / 876 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40307840727) |
| 2026-10-26 | 17.5.0 | `831f225` | `a470038` | 1.00× ± 0.01 (-0.5%) | 2.53 / 2.51 | 1.587 / 1.680 | 894 / 778 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40301225186) |
| 2026-10-26 | 17.4.0 | `296adf4` | `805bca3` | 1.01× ± 0.01 (-1.3%) | 3.02 / 2.98 | 1.558 / 1.639 | 922 / 787 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40276919447) |
| 2026-10-26 | 17.3.0 | `26e9259` | `718e028` | 1.00× ± 0.02 (-0.3%) | 2.86 / 2.86 | 1.511 / 1.478 | 943 / 784 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40265661727) |
| 2026-10-26 | 17.2.0 | `718e028` | `4423c97` | 0.98× ± 0.01 (+1.8%) | 2.59 / 2.65 | 1.595 / 1.520 | 916 / 873 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40259387706) |
| 2026-10-26 | 17.2.0 | `8eb5b6a` | `c45c0cd` | 0.99× ± 0.02 (+1.1%) | 3.01 / 3.04 | 1.544 / 1.623 | 916 / 785 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40255914769) |
| 2026-10-26 | 17.1.0 | `77d3559` | `a557cf9` | 1.04× ± 0.05 (-3.8%) | 2.80 / 2.71 | 1.588 / 1.547 | 865 / 956 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40248366974) |
| 2026-10-25 | 17.0.0 | `366df70` | `a9424a6` | 0.98× ± 0.03 (+1.6%) | 2.57 / 2.62 | 1.524 / 1.547 | 938 / 877 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40241193254) |
| 2026-10-25 | 17.0.0 | `a9424a6` | `62ea815` | 1.02× ± 0.01 (-1.8%) | 2.79 / 2.74 | 1.615 / 1.631 | 856 / 813 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40240440807) |
| 2026-10-25 | 17.0.0 | `62ea815` | `30195a6` | 1.00× ± 0.01 (+0.3%) | 2.52 / 2.53 | 1.611 / 1.598 | 914 / 906 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40239341404) |

## bench 16.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 16.0.0 | `30195a6` | `a4d4937` | 1.01× ± 0.02 (-1.0%) | 2.95 / 2.91 | 1.595 / 1.609 | 733 / 738 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40238078280) |
| 2026-10-25 | 16.0.0 | `a4d4937` | `4947d6e` | 0.98× ± 0.02 (+1.8%) | 2.67 / 2.71 | 1.675 / 1.667 | 751 / 725 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40236129513) |

## bench 15.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 15.2.0 | `3046fdb` | `1de065b` | 1.15× ± 0.03 (-12.7%) | 3.18 / 2.77 | 1.605 / 1.604 | 657 / 652 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40232702757) |
| 2026-10-25 | 15.2.0 | `1b51d17` | `5ef2147` | 1.01× ± 0.02 (-0.6%) | 2.92 / 2.90 | 1.574 / 1.627 | 798 / 648 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40231350383) |
| 2026-10-25 | 15.1.0 | `5d3b86f` | `63b6fd3` | 0.99× ± 0.01 (+1.1%) | 3.10 / 3.12 | 1.566 / 1.607 | 925 / 751 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40229648716) |
| 2026-10-25 | 15.1.0 | `63b6fd3` | `cf5b768` | 0.99× ± 0.03 (+1.2%) | 3.03 / 3.05 | 1.575 / 1.650 | 808 / 918 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40227984080) |
| 2026-10-25 | 15.0.0 | `cf5b768` | `6d76418` | 1.01× ± 0.01 (-1.0%) | 3.26 / 3.22 | 1.601 / 1.623 | 794 / 709 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40227015820) |

## bench 14.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 14.1.0 | `1bc7caf` | `b989521` | 1.03× ± 0.01 (-2.9%) | 3.41 / 3.31 | 1.774 / 1.630 | 704 / 686 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40224516921) |
| 2026-10-25 | 14.1.0 | `b989521` | `b44e497` | 0.99× ± 0.03 (+1.4%) | 3.15 / 3.19 | 1.730 / 1.716 | 795 / 674 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40224430766) |
| 2026-10-25 | 14.0.0 | `0a574cf` | `f806305` | 1.16× ± 0.02 (-14.0%) | 3.71 / 3.19 | 2.066 / 1.673 | 638 / 734 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40223102466) |

## bench 13.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 13.0.0 | `f806305` | `0fa6f68` | 0.99× ± 0.02 (+0.5%) | 3.99 / 4.02 | 1.964 / 2.126 | 722 / 634 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40222381819) |
| 2026-10-25 | 13.0.0 | `0fa6f68` | `5df0c40` | 1.01× ± 0.02 (-0.9%) | 3.87 / 3.83 | 1.925 / 2.050 | 698 / 707 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40221433555) |
| 2026-10-25 | 13.0.0 | `5df0c40` | `79438b1` | 0.99× ± 0.01 (+1.4%) | 3.35 / 3.39 | 2.098 / 2.142 | 655 / 764 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40221195177) |
| 2026-10-25 | 13.0.0 | `79438b1` | `2954123` | 1.04× ± 0.04 (-3.5%) | 3.33 / 3.22 | 2.205 / 2.157 | 860 / 829 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40218127750) |
| 2026-10-25 | 13.0.0 | `e8e4aa8` | `6f58fcc` | 0.99× ± 0.04 (+0.9%) | 3.40 / 3.43 | 2.101 / 2.155 | 863 / 975 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40217424469) |
| 2026-10-25 | 13.0.0 | `c654e92` | `435f1b2` | 1.01× ± 0.08 (-1.0%) | 3.59 / 3.55 | 2.163 / 2.098 | 994 / 978 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40216647264) |

## bench 12.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 12.0.0 | `435f1b2` | `cec1474` | 1.23× ± 0.38 (-18.6%) | 6.00 / 3.77 | 2.102 / 2.093 | 974 / 788 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40216640719) |
| 2026-10-25 | 12.0.0 | `cec1474` | `fa959e4` | 0.97× ± 0.02 (+3.2%) | 3.61 / 3.71 | 2.076 / 2.099 | 1,016 / 786 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40216100365) |
| 2026-10-25 | 12.0.0 | `fa959e4` | `cdda6ea` | 1.90× ± 0.02 (-47.3%) | 6.44 / 3.40 | 3.709 / 2.070 | 1,064 / 1,187 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40213201862) |
| 2026-10-25 | 12.0.0 | `cdda6ea` | `82c56d8` | 1.00× ± 0.01 (-0.1%) | 7.34 / 7.32 | 3.612 / 3.514 | 831 / 1,033 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40213183812) |

## bench 11.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 11.0.0 | `44f4e0e` | `b90dda2` | 1.01× ± 0.04 (-0.9%) | 6.25 / 6.22 | 3.920 / 3.884 | 1,114 / 743 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40210586346) |
| 2026-10-25 | 11.0.0 | `b90dda2` | `1fd5927` | 0.98× ± 0.03 (+1.9%) | 6.15 / 6.25 | 3.933 / 3.977 | 1,063 / 1,302 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40209552181) |

## bench 10.x, full tier (66 chunks, 6 workers, groups of 22)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 10.0.0 | `3861ad3` | `4e58c15` | 1.00× ± 0.03 (-0.2%) | 43.75 / 43.64 | 6.563 / 6.560 | 7,329 / 7,646 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40208487315) |

## bench 10.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 10.0.0 | `4e58c15` | `266273e` | 1.01× ± 0.03 (-0.5%) | 6.29 / 6.27 | 3.799 / 3.936 | 1,098 / 1,066 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40207699100) |
| 2026-10-25 | 10.0.0 | `a0ca3d9` | `fde78af` | 1.01× ± 0.02 (-0.7%) | 6.64 / 6.61 | 3.958 / 3.861 | 807 / 780 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40206414788) |

## bench 9.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 9.0.0 | `fde78af` | `308cbb2` | 0.99× ± 0.01 (+0.6%) | 7.04 / 7.08 | 3.870 / 3.825 | 818 / 1,010 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40204271780) |

## bench 8.x, full tier (66 chunks, 6 workers, groups of 22)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 8.0.0 | `a90c9de` | `b7eed31` | 1.01× ± 0.02 (-1.1%) | 43.27 / 42.89 | 6.488 / 6.385 | 10,197 / 7,588 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40201101316) |

## bench 8.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 8.0.0 | `308cbb2` | `de06271` | 1.01× ± 0.04 (-0.9%) | 6.75 / 6.71 | 3.873 / 3.924 | 755 / 990 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40203274264) |
| 2026-10-25 | 8.0.0 | `b7eed31` | `45de9b7` | 1.01× ± 0.01 (-0.6%) | 7.19 / 7.14 | 3.756 / 3.887 | 917 / 884 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40200992975) |
| 2026-10-25 | 8.0.0 | `3b4c00a` | `d8f2826` | 0.98× ± 0.01 (+2.4%) | 7.43 / 7.61 | 3.907 / 3.906 | 1,126 / 902 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40197257138) |
| 2026-10-25 | 8.0.0 | `7bc3153` | `0a37a6a` | 1.00× ± 0.01 (+0.2%) | 6.55 / 6.56 | 3.962 / 3.888 | 1,264 / 1,173 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40195296884) |

## bench 7.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 7.5.0 | `481bc85` | `3ec7865` | 0.99× ± 0.02 (+0.6%) | 7.17 / 7.21 | 3.912 / 3.912 | 1,159 / 1,420 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40192784487) |
| 2026-10-25 | 7.5.0 | `3ec7865` | `9423325` | 1.00× ± 0.02 (+0.2%) | 5.50 / 5.52 | 4.109 / 3.867 | 1,309 / 1,172 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40191869751) |
| 2026-10-25 | 7.5.0 | `6f3dd32` | `a4da462` | 1.00× ± 0.00 (+0.4%) | 6.14 / 6.17 | 4.035 / 3.896 | 1,202 / 1,292 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40190399578) |
| 2026-10-25 | 7.4.0 | `a4da462` | `7ad8018` | 1.10× ± 0.01 (-9.4%) | 7.27 / 6.60 | 3.700 / 3.926 | 1,490 / 1,266 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40187859214) |
| 2026-10-25 | 7.4.0 | `cb5d772` | `b6e67bd` | 1.02× ± 0.02 (-1.9%) | 8.00 / 7.85 | 3.827 / 3.829 | 1,370 / 1,156 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40183647277) |
| 2026-10-25 | 7.3.0 | `f862038` | `f6ef584` | 1.00× ± 0.01 (-0.3%) | 7.94 / 7.91 | 3.742 / 3.746 | 2,002 / 1,162 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40181917894) |
| 2026-10-25 | 7.2.0 | `f6ef584` | `59afb84` | 1.01× ± 0.01 (-0.9%) | 7.41 / 7.36 | 3.877 / 3.914 | 2,056 / 1,974 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40179951878) |
| 2026-10-25 | 7.1.0 | `59afb84` | `7a89053` | 1.05× ± 0.02 (-4.5%) | 7.93 / 7.57 | 3.843 / 3.751 | 2,032 / 2,088 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40178991975) |
| 2026-10-25 | 7.1.0 | `97acff2` | `7e737e8` | 0.99× ± 0.00 (+1.5%) | 8.17 / 8.29 | 3.817 / 3.768 | 1,814 / 1,874 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40177912576) |
| 2026-10-25 | 7.1.0 | `7e737e8` | `b569fc6` | 1.01× ± 0.02 (-1.1%) | 7.08 / 6.99 | 3.900 / 3.780 | 1,556 / 1,482 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40175206886) |
| 2026-10-25 | 7.1.0 | `b569fc6` | `991f590` | 1.00× ± 0.01 (+0.0%) | 7.54 / 7.53 | 3.818 / 3.763 | 1,739 / 1,334 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40174240752) |
| 2026-10-24 | 7.0.0 | `991f590` | `fa253c9` | 0.99× ± 0.01 (+0.7%) | 8.24 / 8.29 | 3.752 / 3.864 | 1,508 / 1,644 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40135762023) |
| 2026-10-24 | 7.0.0 | `fa253c9` | `126aa8f` | 1.00× ± 0.00 (+0.2%) | 8.08 / 8.09 | 3.813 / 3.785 | 1,979 / 1,438 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40132622985) |
| 2026-10-24 | 7.0.0 | `126aa8f` | `45720f9` | 1.00× ± 0.00 (-0.1%) | 8.30 / 8.29 | 3.803 / 3.666 | 1,756 / 1,752 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40131996904) |

## bench 6.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 6.0.0 | `45720f9` | `0f19623` | 1.00× ± 0.07 (+0.4%) | 7.18 / 7.17 | 3.795 / 3.881 | 1,398 / 1,787 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40130147657) |
| 2026-10-24 | 6.0.0 | `0f19623` | `dab80c4` | 1.00× ± 0.01 (+0.3%) | 7.11 / 7.10 | 3.782 / 3.842 | 1,350 / 1,926 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40129416272) |

## bench 5.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 5.1.1 | `dab80c4` | `95d2f5c` | 1.00× ± 0.04 (-0.2%) | 7.17 / 7.17 | 3.794 / 3.827 | 1,440 / 1,366 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40127333635) |
| 2026-10-24 | 5.1.1 | `95d2f5c` | `70f5b47` | 0.99× ± 0.02 (+0.8%) | 8.58 / 8.65 | 3.737 / 3.832 | 1,602 / 1,343 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40126642243) |
| 2026-10-24 | 5.1.0 | `70f5b47` | `8bfd829` | 0.99× ± 0.03 (+0.6%) | 8.00 / 8.05 | 3.683 / 3.723 | 1,540 / 1,306 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40122028375) |
| 2026-10-24 | 5.0.0 | `8bfd829` | `3b90a63` | 1.03× ± 0.03 (-2.9%) | 7.37 / 7.18 | 3.842 / 3.832 | 1,703 / 1,664 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40120876473) |

## bench 4.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 4.0.0 | `3b90a63` | `af59222` | 1.24× ± 0.02 (-19.5%) | 9.70 / 7.80 | 2.851 / 3.787 | 1,699 / 1,220 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40111163945) |
| 2026-10-24 | 4.0.0 | `cae7177` | `271ab02` | 1.01× ± 0.03 (-0.8%) | 9.30 / 9.21 | 2.869 / 2.803 | 1,560 / 1,256 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40108708080) |
| 2026-10-24 | 4.0.0 | `271ab02` | `6967411` | 1.04× ± 0.01 (-4.1%) | 10.08 / 9.67 | 2.897 / 2.857 | 1,204 / 1,415 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40106987244) |
| 2026-10-24 | 4.0.0 | `7782985` | `7ca5e52` | 1.02× ± 0.01 (-1.9%) | 8.94 / 8.77 | 2.814 / 2.843 | 1,413 / 1,301 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40105312335) |
| 2026-10-24 | 4.0.0 | `7ca5e52` | `368de70` | 0.98× ± 0.04 (+1.7%) | 9.40 / 9.52 | 2.894 / 2.846 | 1,226 / 1,434 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40104668921) |
| 2026-10-24 | 4.0.0 | `7a2aa55` | `c2ca552` | 0.99× ± 0.02 (+0.9%) | 8.94 / 8.99 | 2.881 / 2.920 | 1,974 / 1,894 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40104482820) |
| 2026-10-24 | 4.0.0 | `0241ebe` | `7a2aa55` | 1.00× ± 0.02 (+0.4%) | 10.29 / 10.33 | 2.860 / 2.832 | 1,619 / 1,257 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40104517210) |
| 2026-10-24 | 4.0.0 | `c2ca552` | `673ab81` | 0.99× ± 0.01 (+1.3%) | 10.21 / 10.34 | 2.869 / 2.830 | 1,900 / 1,649 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40102088109) |

## bench 3.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 3.1.1 | `673ab81` | `a8631ad` | 0.99× ± 0.02 (+1.1%) | 10.20 / 10.30 | 2.814 / 2.882 | 238 / 237 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40101890228) |
| 2026-10-24 | 3.1.1 | `a8631ad` | `7e68724` | 1.01× ± 0.02 (-1.0%) | 10.16 / 10.08 | 2.824 / 2.860 | 318 / 500 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40101145633) |
| 2026-10-24 | 3.1.1 | `7e68724` | `afb582c` | 1.06× ± 0.01 (-6.0%) | 10.19 / 9.57 | 3.475 / 2.841 | 396 / 346 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40101120346) |
| 2026-10-24 | 3.1.1 | `afb582c` | `cefeffa` | 1.02× ± 0.02 (-1.5%) | 10.61 / 10.46 | 3.629 / 3.491 | 335 / 333 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40101050485) |
| 2026-10-24 | 3.1.1 | `cefeffa` | `3a7e534` | 1.01× ± 0.01 (-1.1%) | 10.37 / 10.26 | 3.427 / 3.457 | 377 / 368 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40091759224) |
| 2026-10-24 | 3.1.1 | `3a7e534` | `0fd935e` | 1.02× ± 0.01 (-1.5%) | 10.01 / 9.86 | 3.383 / 3.654 | 460 / 475 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40090366848) |
| 2026-10-24 | 3.1.0 | `0fd935e` | `5c7dc7c` | 1.23× ± 0.04 (-18.9%) | 10.43 / 8.49 | 3.687 / 3.624 | 260 / 278 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40089388802) |
| 2026-10-24 | 3.1.0 | `5c7dc7c` | `62a6a3a` | 0.99× ± 0.01 (+1.4%) | 12.37 / 12.53 | 3.549 / 3.465 | 268 / 340 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40088644942) |
| 2026-10-24 | 3.1.0 | `62a6a3a` | `617d095` | 0.98× ± 0.02 (+1.9%) | 12.85 / 13.08 | 3.563 / 3.471 | 276 / 435 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40087944254) |
| 2026-10-24 | 3.1.0 | `617d095` | `0c480da` | 0.99× ± 0.02 (+1.2%) | 12.67 / 12.81 | 3.570 / 3.408 | 253 / 289 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40087925245) |
| 2026-10-24 | 3.0.1 | `0c480da` | `d97935f` | 1.00× ± 0.01 (-0.3%) | 11.89 / 11.87 | 3.598 / 3.736 | 660 / 623 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40075754602) |
| 2026-10-24 | 3.0.0 | `d97935f` | `c4a090f` | 0.94× ± 0.06 (+6.4%) | 11.40 / 12.15 | 3.695 / 3.663 | 504 / 491 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40054339707) |

## bench 2.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 2.1.1 | `c4a090f` | `fb3cc80` | 1.19× ± 0.01 (-16.1%) | 15.85 / 13.30 | 3.719 / 3.356 | 198 / 194 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40052912989) |
| 2026-10-24 | 2.1.1 | `fb3cc80` | `3abf40b` | 1.04× ± 0.06 (-3.8%) | 15.63 / 15.06 | 3.703 / 3.683 | 196 / 196 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40050433365) |
| 2026-10-24 | 2.1.0 | `3abf40b` | `9374bda` | 0.98× ± 0.02 (+2.0%) | 14.42 / 14.68 | 3.663 / 3.660 | 252 / 254 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40046630308) |
| 2026-10-24 | 2.0.1 | `9374bda` | `01333f6` | 1.00× ± 0.01 (-0.3%) | 13.02 / 12.98 | 3.741 / 3.686 | 240 / 239 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40045725437) |
| 2026-10-24 | 2.0.0 | `01333f6` | `059b583` | 0.99× ± 0.01 (+0.6%) | 13.62 / 13.69 | 3.615 / 3.758 | 332 / 295 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40045274880) |

## bench 1.x, full tier (66 chunks, 6 workers, groups of 22)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 1.1.0 | `c69dcac` | `2d25be6` | 1.87× ± 0.11 (-46.5%) | 205.90 / 109.28 | 12.924 / 12.618 | 3,225 / 1,945 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40042031069) |
| 2026-10-23 | 1.1.0 | `6090f4f` | `c69dcac` | 1.54× ± 0.01 (-35.2%) | 103.35 / 66.96 | 12.590 / 7.034 | 1,786 / 1,575 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40042033399) |

## bench 1.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 1.1.1 | `059b583` | `a955439` | 1.02× ± 0.02 (-1.7%) | 13.87 / 13.63 | 3.672 / 3.942 | 372 / 365 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40043511839) |
| 2026-10-24 | 1.1.0 | `a955439` | `7c5aeba` | 1.00× ± 0.01 (+0.3%) | 11.95 / 11.97 | 3.817 / 3.761 | 459 / 196 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40043147066) |
| 2026-10-23 | 1.0.0 | `7c5aeba` | `68fdc4b` | 1.02× ± 0.02 (-1.7%) | 15.69 / 15.43 | 3.641 / 3.632 | 197 / 196 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40041103525) |
| 2026-10-23 | 1.0.0 | `68fdc4b` | `89a9bb4` | 1.01× ± 0.01 (-0.6%) | 13.53 / 13.44 | 3.570 / 3.804 | 396 / 378 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40040615022) |
| 2026-10-23 | 1.0.0 | `89a9bb4` | `8172d2c` | 0.95× (+4.9%) | 10.23 / 10.73 | 3.819 / 3.952 | 381 / 389 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40039429402) |
| 2026-10-23 | 1.0.0 | `8172d2c` | `28dc50e` | 0.99× (+1.3%) | 15.25 / 15.45 | 3.562 / 3.690 | 195 / 199 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40038870040) |
| 2026-10-23 | 1.0.0 | `7ad8c9b` | `68a1d41` | 0.98× (+2.5%) | 15.39 / 15.78 | 3.663 / 3.551 | 204 / 206 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40032386680) |
