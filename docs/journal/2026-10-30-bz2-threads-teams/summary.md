# The native bz2 decoder's threads through the split (03cb647992)

- **Machine:** Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz; 20 logical CPUs on 10 physical cores (L3 20480K); 125.7 GB. `auto` is 4 decoder threads here.
- **Held constant (each run's log):** [18] parse workers, [18] writer threads; every split from nothing.
- **Workload:** bench-full's run (`--release 20261019 --windows 24 --window-mb 25`); 8 rounds, each arm once a round, the order rotated and reversed (`arm_order`).
- **Paired:** each round's ratio to `auto` in the same round, the mean and its 95% interval; *lower* or *higher* only when the interval excludes no change.

## Release-shaped (bench-full run)

`run.py run --release 20261019 --windows 24 --window-mb 25`

| threads | wall (s), median | CPU-s | peak RSS (MB) | split wall (s) | split CPU-s | wall vs auto | CPU vs auto | split wall vs auto | split CPU vs auto | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| auto (4) | 103 | 639 | 21198 | 25 | 239 | control | control | control | control | 8 |
| 8 (8) | 100 | 683 | 21094 | 24 | 280 | -2.0% ±0.6% (lower) | +6.7% ±0.4% (higher) | -4.2% ±0.5% (lower) | +16.9% ±0.3% (higher) | 8 |

- **Parity:** 16 of 16 runs ok, their published files' digest the first run's.
