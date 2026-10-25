Scripts for `../2026-10-25-group-row-groups.md`, run with the worktree's venv from its root.

- `deltas.py RUN_DIR keys`, `phase_deltas.py RUN_DIR span`, `proc_cpu.py RUN_DIR span`:
  a ledger run's artifacts (`gh run download`), per-round deltas of results figures,
  per-function phase CPU, and finalise's process counts.
- `build_index.py OUT chunk.parquet ...`: the bench chunks built as the split builds them.
- `mklinks.py BUILT OUT N chunks`: a group of N chunks, the built files hard linked.
- `sim_group.py BUILT OUT N chunks`: the group closed by `push.core._copy_built`.
- `cmp_merge.py OUT`: each key's file copied and merged, compared and timed.
- `fin_costs.py OUT copy,merge`: finalise's per-file costs on each layout.
- `published.py BUILT groupA groupB` (cwd: an empty directory): the sorted files from both
  layouts, by sha256.
- `proto.py OUT copy,arrow,glue`: re-encoding at the close (rejected).
