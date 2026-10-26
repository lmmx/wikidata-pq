# 2026-10-26: One schema per release, evolving while it is built

This entry records a review of the campaign's results and its open levers. That review led
to a closer look at how the published claims schema is kept single and consistent, and to
two decisions, both made on 2026-10-26:
- the schema evolves automatically during a release;
- whole-number coordinates follow one canonical rule.

The queue issue is #97, which has the decision comments.

## Where the campaign stands

Modelled at master, after #170 (L1a) and #175 (L2), a full release with finalise's
transfers takes:

| Link rate | Modelled release |
|---|---|
| 7 MB/s | ~11h49m |
| 30 MB/s | ~7h15m |
| 60 MB/s | ~6h41m |
| 120 MB/s | ~6h35m |

The release has moved from mostly CPU-bound to network-bound at slow links. Three changes
matter most, and they rest on two different kinds of evidence:
- **#170 (finalise commits in the background)** and **#175 (finalise before the scholarly
  uploads end)** take time off the critical path by overlapping work. Their release savings
  are modelled: −31 and −30 min at 7 MB/s.
- **#172 (claims.avsc completed)** removes slow-path work. It is measured on the chunks it
  affects: −39% wall on a misfit chunk.

## Evidence, not permission

The next big levers were first described as needing the user's decision. The user
corrected this: most need **evidence**, not permission.
- **The Hub's real upload rate** comes first, because it decides which levers matter.
  - **Below ~15 MB/s:** the link is the bottleneck, so L1b, cross-run overlap and the
    `statement_id` encoding pay.
  - **Above ~30 MB/s:** the release is ~6.5–7h, and the CPU side is at its floor.
  - **Measuring it without a Hub call:** a past release's group ledger records when each
    group was `merged` and `pushed`. With each group's bytes, that gives the push rate.
  - **A fresh timed upload** is a Hub call. Agents may not make one, so the user would run
    it.
- **L1b (`preupload_lfs_files`)** is not a byte stream into an unfinished file.
  - **How it would work:** it hashes whole files, and nothing is visible until
    `create_commit`. So the design is to preupload each immutable sorted part once it is
    written and checked, and commit only after the existing checks pass. The published
    output is unchanged.
  - **Before it is authorised:** establish its exact semantics from the huggingface_hub
    source, the bandwidth it reaches, and what a rerun does with blobs that were uploaded
    but never committed. The modelled −2h at 7 MB/s is not enough on its own.
- **Cross-run finalise overlap** can be prototyped on bench/link.py's stand-in Hub before
  any recipe decision.
- **The `statement_id` delta encoding**'s PRs can be prepared. Only their releases wait.
- **The claims-last table order** stays configurable. Its default is chosen once the rate
  is known.

## The schema question

### The concern

The schema "was meant to be inferred from the entire dataset and shared by every file". Had
the campaign replaced that with a schema guessed from a preview, so that files earlier and
later in the pipeline could differ?

### What the code and history show

**There is one published schema, and it is enforced, not assumed:**
- **One stored schema.** `claims_schema` (`src/wikidata/claims_schema.py`) is written out
  field by field. Every chunk's claims are conformed to it, with the fields a chunk lacks
  as null, on both paths: the split's direct build (`claims_build.py`) and the inference
  fallback (`process.py`, `schema=claims_schema`).
- **A mismatch stops the run.** On the fallback path genson checks each chunk's inferred
  schema against `claims_schema` before it writes. A field the stored schema lacks, or
  another dtype, halts the run ("Schema mismatch … update its stored schema").
- **Compaction and the sort re-check it.** Every group file and every written output must
  have exactly the same Arrow schema (`compact.py`, `sort_by_id._check_schemas` /
  `_check_written`). The bench's parity hashes each table's schema, metadata included.

**The design predates the campaign.** The first commit in this repository (5ad5f96, 7 Oct)
already conformed claims to a stored schema. Its DESIGN.md says: *"Claims are conformed to
the stored claims schema (fields a chunk lacks are null), so every chunk, group and
uploaded file has the same schema."* Compaction has only ever checked that schemas are
equal. It never merged differing ones. If the stored schema came from a full pass over a
dump, that pass is older than this history.

**Three schemas are involved, and the first answers given blurred them:**
1. **`claims_schema`**, the published schema, is fixed and shared by every file.
2. **`claims.avsc`** only decides whether a chunk can skip inference. It was inferred from
   the bench's 11 chunk files, and since #172 it has been completed against
   `claims_schema`. It never defines the output.
3. **Per-chunk inference** is the fallback for a chunk that does not fit. Before 24 Oct
   every chunk took this path. Even then, inference saw only its own chunk, never the whole
   dump.

**The conform step was replaced, as recalled.** Before #16 (8ce969a) the pipeline read
genson's output back and conformed it with Polars:
`cast({key: pl.Struct(expected_schema)}, cast_options=ScanCastOptions(missing_struct_fields="insert", extra_struct_fields="ignore"))`,
guarded by a DeepDiff check. #16 had genson write in the target schema directly
(process() 41.7 s → 5.3 s on chunk 0). #17 (9c54898) replaced DeepDiff with genson's
`schema_check`. The Python stored schema was kept; `claims.avsc` was added beside it.

### The gaps

1. **The schema cannot evolve during a release.**
   - **What happens:** a dump with a shape `claims_schema` lacks halts the run. Rerunning
     resumes from the ledgers, but only if nothing changed.
   - **Why widening doesn't help:** after `claims_schema` is widened, every chunk and group
     file already written keeps the old schema, some of it already on the build branch.
     Compaction's equal-schema check would then halt.
   - **The result:** there is no pause-patch-resume path, and a mid-release schema change
     means redoing work.
2. **Whole-number coordinates are placed by chunk.** The schema has two siblings for these
   values (`latitude__number` and `latitude__integer`), but where a whole-number coordinate
   lands depends on the path:
   - the direct decode always gives `__number` (47.0);
   - inference gives `__integer` (47) in a part with no float in that field.

   Every file has the same columns, but where the values sit depends on the path and the
   chunk. This predates the campaign: every chunk was exposed to it while every chunk was
   inferred. #172 made it rarer without removing it.

## Decisions

### Automatic schema evolution: auto by default, strict as an option

The goal is snapshot-like acceptance with resumability, not stopping the release to edit
Python. The analogy is accepting a snapshot test.

**The two modes:**
- **Auto (default):** a genuinely new nullable field joins the release schema. Chunks and
  groups already written are upgraded on read, including those on the build branch. The
  dump is not restarted because the schema grew.
- **Strict (explicit):** it halts, reports the proposed changes, and gives an
  accept-and-resume command. No manual edits.

**Why it can work:**
- **An added field changes no values.** It is null in every row already written, so old
  files gain all-null columns. This is what `missing_struct_fields="insert"` did.
- **Forced promotion turns a new type into a new sibling** (`x__string` beside
  `x__number`), which is also an addition. Only shape changes, such as a scalar becoming a
  record or a list, can't be widened.
- **Finalise already reads and rewrites every group file,** so the upgrade can happen on
  read there. "Widening on read" does not mean no rewriting: the old Parquet files stay as
  they are, and finalise writes the published files in the final schema, as it already
  does.

**The safeguards the user set:**
- **No silent drops.** A newly encountered field is never discarded, on the fast decode
  path included. Old files gain nulls, and new rows keep their real values.
- **Determinism.** The final schema and its field order do not depend on discovery order,
  completion order, concurrency or restarts. For example, base fields keep their stored
  order and added fields follow in a canonical order per struct. Discoveries merge under a
  lock, and versions only grow.
- **Additive and lossless only.** An incompatible structural change halts with a helpful
  diagnostic and is never guessed into compatibility. Adding `x__newtype` is generally
  manageable; changing what an existing column means is not, and the release never decides
  that for itself.
- **The fast path stays valid** as the schema evolves, with a safe fallback while it
  catches up.
- **Completed work stays resumable,** including groups already uploaded to the build
  branch.
- **Finalise freezes one release schema.** It validates the complete contents and their
  order, and every published file has exactly that schema. No incomplete group or mixed
  version escapes.
- **The checks stay exact.** Schema validation is not weakened just to allow widening. An
  equality check becomes, for example, "the frozen schema minus fields the release's log
  records as added after this file's version, each nullable". It must catch everything the
  equality check caught.
- **Promotion.** After a successful release, the accepted schema can be promoted into the
  repository as a reviewed snapshot update for later releases.
- **Testing:** kill-and-rerun tests, concurrent-discovery tests, and additions after some
  groups are written and uploaded. The bench measures the cost of discovery and widening,
  including memory and disk, and shows the fast path is not slowed when the dump fits.

The end state is a fixed, universal schema for each published release, with a schema that
evolves automatically while the release is built.

### Rule B for whole-number coordinates

The rule:
- **What:** a globe-coordinate's `latitude`, `longitude` and `precision` go to the canonical
  `__number` column whether the JSON writes `47` or `47.0`.
- **Where:** the same in the direct decoder and the inference fallback.
- **Schema:** the existing schema stays, including the `__integer` siblings, for
  compatibility.

Rule B was chosen over rule A (a JSON integer always to `__integer`, a float always to
`__number`):
- latitude, longitude and a coordinate's precision are numeric quantities, not tagged
  unions of JSON integer and float syntax;
- `47` and `47.0` should not land in different columns because of how they were written;
- B matches nearly all of today's output, since the direct path, ~99% of chunks, already
  gives `__number`.

**Scope.** `precision` is shared with time values, where it is an integer enum (0–14) and
goes to `precision__integer` on every path today. That placement is deterministic already,
so it stays. Applying B to every precision would have moved every time precision in the
dump, so B applies to globe coordinates only.

**How it must hold:**
- Values are preserved exactly. A conversion Int64 → Float64 can't represent exactly halts
  the run rather than being rounded.
- Whole-number, fractional and null cases are tested in each path.
- The value correction is documented apart from byte-identical parity for every unaffected
  value.
- This is a deliberate, narrowly scoped correction of output values, not a licence to
  change numeric handling elsewhere in claims.

## Work started

Two isolated agents, with CI as the performance authority:
- **Canonical coordinates (rule B):** one rule in every path. A wikidata-pq-only
  implementation is preferred, needing no polars-genson release.
- **Schema evolution:** the release schema, its modes, its deterministic ordering,
  widening on read in compaction and the sort, finalise's freeze, and the accept and
  promote commands.

Their results will be recorded in their own journal entries
(`2026-10-26-canonical-coordinates.md`, `2026-10-26-schema-evolution.md`) and in #97.
Releases of polars-genson or pq-glue, if either change needs one, follow the active
authority mode.
