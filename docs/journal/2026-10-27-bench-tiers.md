# 2026-10-27: The bench's PR tier: a pull request's figures in about 6 minutes, the full suite on master

The Benchmark workflow (`.github/workflows/bench.yml`, `bench/ab.py`) had grown from a
1.4-minute median to 23 minutes in four days, while `bench/CHANGELOG.md` went from
version 2 to 17.14. Each version added a check worth having. Every pull request paid for
all of them, and nothing showed what each one cost. Bench 17.15.0 splits the checks into
two tiers, records each section's wall, and guards the PR tier's latency.

## How it grew

The median of the successful runs, by half-day, oldest first (from 23 Oct):

| half-day | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| median run (min) | 1.4 | 2.4 | 6.4 | 7.1 | 9.5 | 16.9 | 23.2 | 23.0 |

One round of master's run 40497402341 (job 122166254596), measured from its log:

| part | wall |
|---|---|
| setup (checkouts, data, decoders, environments) | ~1.5 min |
| "before" arm | ~7 min: the timed core ~1 min, then ~6 min of untimed sections |
| "after" arm | ~7.5 min, the same |

The untimed sections, after each arm's core and outside its figures:

- the split from gz (`pipeline.py --split-gz`) with its parity;
- `overlap.py` with the emulated 7 and 60 MB/s links, which mostly sleep (finalise's three
  runs over 7 MB/s take about 50 s, done twice an arm); together with the split from gz,
  about 4.5 minutes an arm;
- `misfit.py` and its parity, about 5 s;
- `evolve.py` and its parity, about 35 s;
- `bz2native.py`, about 20 s: the split with `WIKIDATA_BZ2_DECODER=native` against
  lbzip2's, with byte parity.

The rounds run in parallel, so a round's job is the run's wall: about 16 minutes of job
for 2 minutes of timed measurement.

## What moved where

| section | before 17.15.0 | from 17.15.0 |
|---|---|---|
| timed core, parity | every run | every run (the PR tier) |
| `bz2native` | every run | every run (the PR tier); also triggered explicitly by mibz2's `reshape/Cargo.lock` entry differing between the arms |
| `split_gz` | every run | full tier; on a PR touching the split or its dump reading |
| `overlap` and the link | every run | full tier; on a PR touching the stage orchestration or the Hub stand-in's paths |
| `misfit`, `evolve` | every run | full tier; on a PR touching the claims or schema code, or a `polars-genson` bump in `uv.lock` |

- **The full tier** runs on every push to master, nightly (03:17 UTC, not recorded) and
  from the Actions tab by default. It is the suite as it was: each section in each arm's
  run after its core, in the same order. Master's records are as complete as before.
- **On a pull request**, the extras it is relevant to run after both arms' cores, so
  they don't hold up the comment. The comment's full table posts once every round's PR
  tier is in, and the extras update it after. The label `bench:full`, or a line
  `bench: full` in the body, runs them all.
- **Sections that didn't run** read "not run on this PR (full tier on master)", row by
  row, so the table keeps its shape.

## The PR tier's budget

From the reference job: setup ~1.3 min, then for each arm about 1.6 min of core and 20 s
of `bz2native`, and parity ~10 s. That is about 5.5 minutes a job, plus queueing and the
comment. **The target is 6 minutes end to end.**

The finish job takes the median of the last 10 pull request runs' latency from the
Actions API (`bench/latency.py`): the run's start to its last job's "Comment: PR tier"
step. The results store records master only. Over 8 minutes, the comment opens with a
warning. The guard uses the median so that one slow runner does not trip it.

Each section's wall goes into results.json (`durations`), into the comment (a collapsed
table) and into master's records (`bench_duration`). The bench history page charts it,
so the next slow growth shows up as it happens.

**The policy** (bench/README.md, CLAUDE.md "Measuring"): new bench scenarios go in the
full tier by default. Adding work to the PR tier needs an explicit justification in the
PR, with its cost in seconds per round against the 6-minute budget.
