# 2026-11-15 · The bz2 decoder cap under mibz2 0.3.4's workers: no arm passes, the cap stays at 4

**Question (Louis, 15 November).** mibz2 0.3.4 (lmmx/mibz2#425) moved its team threshold to 10
threads. On the i9, a decoder cap of 6, 8 or 9 now means one block per thread with Cells, not
the teams the cap of 4 was last tested against
([2026-10-30-bz2-threads-teams.md](2026-10-30-bz2-threads-teams.md)). Does raising
`BZ2_THREADS_CAP` now pay?

**Rule:** posted on #97 before the data (comment 6898604750). It reused the 30 October gates
unchanged:
- total wall lower by at least 2.0%, with its 95% interval wholly below zero;
- CPU up by at most 10%;
- peak RSS up by at most 5%;
- parity.

The CPU-seconds spent per wall-second saved were reported, not gated.

**Run:**
- `just submit bench-bz2-cap` (#232) at harness `26916b867d`, mibz2 0.3.4, on the i9-10900K
  (10 physical cores, 20 MiB L3; the topology the rule requires).
- 8 rounds, arms auto (4), 6, 8 and 9, release-shaped (`run.py run --release 20261019
  --windows 24 --window-mb 25`).
- 18 parse workers and 18 writer threads held constant.
- Evidence: `bench-bz2-cap/20261115T174946Z` on the `evidence` branch at `045dbbd715`. Its
  SHA-256, `12e3e2afb57e9afe81a7943e77f3aca6f948ef68f4a8e74cd77ff17b9f1b9f27`, was checked
  against the #97 post (6899350837).

## Result

Each change is paired with auto in the same round: the mean, with its 95% interval.

| cap | plan | wall | CPU | peak RSS | split wall | split CPU | CPU-s per wall-s saved | verdict |
|---|---|--:|--:|--:|--:|--:|--:|---|
| auto (4) | Workers(4) | 89 s | 579 CPU-s | 20.8 GB | 19 s | 210 CPU-s | control | – |
| 6 | Workers(6) | −0.0% ±0.5% | +6.0% ±0.5% | +0.4% ±0.8% | +4.0% ±0.9% | +15.1% ±0.6% | 816 | fails gate 1 |
| 8 | Workers(8) | **+1.0% ±0.7% (slower)** | +9.4% ±0.5% | +0.3% ±0.9% | +7.0% ±0.7% | +24.3% ±0.6% | none saved | fails gate 1; slower |
| 9 | Workers(9) | **+1.2% ±0.6% (slower)** | +10.4% ±0.5% | +0.4% ±1.0% | +7.8% ±1.0% | +26.8% ±0.7% | none saved | fails gates 1 and 2; slower |

Parity: 32 of 32 runs ok.

**Verdict, under the rule:**
- No arm passes, and the cap stays at 4.
- Caps 8 and 9 are recorded as slower: their wall intervals lie wholly above zero.
- Cap 6 is no faster, and it costs 6% more CPU.

## What it says

- **More decoder threads now cost wall time, not just CPU.**
  - On 30 October, a cap of 8 (teams) took 2.0% off the wall for +6.7% CPU, about 21 CPU-s per
    wall-s saved.
  - Today 8 adds 1.0% to the wall.
  - The split itself is 4.0–7.8% slower at 6–9. The decoder threads compete with the 18 parse
    workers and writer threads for the same 10 cores, and with 0.3.4's faster decoder at 4
    threads there is no decode wait left for the extra threads to remove.
- **Inferred, not measured:** at 4 threads the decoder already keeps the split fed. The
  bottleneck is downstream (parsing and writing), so the decoder's own scaling is irrelevant here.
- **The mibz2 0.3.4 bump (#227) doesn't depend on this.** It runs at cap 4, as before, and is
  judged on its own bench.

## Not tested

- Caps above 9, which would reach teams.
- Any other split of the 20 logical CPUs between parse workers and the decoder.
- A smaller machine. On CI's 2-core runners the cap doesn't bind.
