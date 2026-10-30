# The decoder's cap and mibz2's cooperative plan (pre-registered; the user's decision, 2026-10-30)

**The question:** does the release-shaped run end sooner with the split's native bz2 decoder
on 8 threads than on the default cap? On the i9, the default is 4. With mibz2 0.2.1, 8
threads there means 4 teams of 2: the cooperative plan.

**Why now:**
- `dump.bz2_threads()` gives one thread per physical core, at most 4 (#212). The cap was
  measured with mibz2 0.1.x workers (`2026-10-28-bz2-threads.md`). There, every thread past
  4 added a live table that thrashed the L3: the decoder alone at 20 threads ran half as
  fast as at 4.
- mibz2 0.2.0 added teams: 2 threads share one block's table. It plans them only at 6 or
  more threads, on 6 or more physical cores. So under the cap the release has never run
  them.
- 4 teams keep 4 tables live, as 4 workers do, on twice the threads. That is a different
  trade-off from the one the cap was fitted to.
- The decoder alone, on the i9: mibz2's default at 8 threads ran 1.80× (1.76–1.84) 0.1.6's
  at 8, and 1.248× its 4 workers (mibz2 `2026-10-29-cooperative-decoder.md`). That is the
  decoder on a machine to itself, not the split.

**What this tests, and what it doesn't:**
- It tests whether that decoder gain survives the split's overlap with the rest of the
  release, on the real pipeline.
- It is not a test of the decoder alone. A faster decode doesn't justify raising the cap if
  the run as a whole is not faster.
- This entry changes no default. The cap stays at 4 unless the gate below is met. Any
  change would then be its own PR, with its own bench.

## The design

- **Machine:** the i9-10900K: 10 cores, 20 CPUs with SMT, 20 MiB L3. The harness records the
  topology.
- **Harness:** `bench/bz2threads.py --full`, unchanged. It runs `bench/full/run.py run`:
  - the split, both sets' runs and finalise;
  - overlapped, as a release runs;
  - from the cached dump, each run into a fresh work directory;
  - the network and the Hub off the timed path.
- **Arms:**
  - `auto`: `WIKIDATA_BZ2_THREADS` unset, the default; 4 on the i9;
  - `8`: `WIKIDATA_BZ2_THREADS=8`.
  - Nothing else changes between arms: the parse workers (18) and writer threads (18) stay
    at their defaults, and the harness checks them in each run's log.
- **The plan each arm gets**, derived and not logged. The split logs a thread count, and
  4 teams of 2 also report 8:
  - mibz2 0.2.1's `Plan::for_threads` gives `Workers(4)` at 4 threads, and
    `Teams { teams: 4, k: 2 }` at 8 on 10 physical cores and a 20 MiB L3. The 4 is the L3
    cap: about 0.7 of the L3 in 3.6 MB tables.
  - The run is read as a test of teams only if the recorded topology is that one: 10
    physical cores and 20 MiB L3 at the time of the run.
- **Idle team helpers don't hold CPUs.** They spin at most 2,000 iterations (a few µs) and
  then park (mibz2 `coop.rs`). The other threads of the run only compete with decoder
  threads that are working.
- **Scale and rounds:**
  - `--release 20261019 --windows 24 --window-mb 25`, the earlier release-shaped run's;
  - 8 rounds, each arm once a round, the order rotated and reversed (`arm_order`). Each arm
    holds each position equally often, twice over.
- **On master after #213** (mibz2 0.2.1), in one checkout for both arms.
- **The command:** `just bench-bz2-teams`. Results in `bench-bz2-teams/`.

## The gate (fixed now)

The harness pairs each round's `8` run with its `auto` run. It reports the mean paired
change and its 95% interval (`combine.interval`).

**Met only if all of these hold:**
1. **Primary: total wall** (the release-shaped run, split to finalise) is lower by at least
   **2.0%**, with its 95% interval wholly below zero.
2. **CPU budget:** total CPU-seconds up by at most **10%** (mean paired change).
3. **Memory budget:** peak RSS of every process up by at most **5%** (mean paired change).
   At this scale that is about 1 GB of 21 GB.
4. **Correctness:**
   - every run's published files have the first run's digest;
   - every log shows the native decoder at its arm's count, with no chunk already
     written. The harness exits 1 otherwise.

**The readings:**
- **Met:** a separate PR may propose a new default for machines where teams engage. That
  PR has its own bench, and on CI's 2-core runners its default must not change.
- **Wall interval wholly above zero:** 8 threads is slower in the real pipeline. The cap
  stays, and the result is recorded with its figures.
- **Otherwise** (an interval that includes zero, a gain under 2%, or a budget exceeded):
  not met, and the cap stays.

**Reported, not gated:**
- the split stage's wall and CPU, paired;
- the finalise and run stages' timings, from each run's log;
- each arm's medians;
- CI's correctness run (below).

## A prediction (a hypothesis, not a criterion)

- **The split's wall lower by 5–15%** at 8 threads, from a decode about 1.2× as fast.
- **The total wall lower by 0–4%.** In the earlier run, the split's saving carried almost
  whole into the release's end (8 s of 33 s became 7 s of 106 s). This run's competition
  from the overlapped stages is unknown.

## What it can't establish

- **One machine class.** On machines with fewer than 6 physical cores, teams never engage
  and the cap gives workers either way. That covers CI's runners and most 4-core laptops.
- **A 0.6 GB sample, not a release's dump.** The same limit applied to the cap's own
  measurement.
- **CI checks correctness only.** A "bz2 threads" run at `auto,8` on CI's 2-core runners
  checks the split's parity and logs at 8 threads. There it plans workers, not teams.
  mibz2's own tests and Miri cover the teams' correctness. On the i9, each run's published
  digest covers the application's.
