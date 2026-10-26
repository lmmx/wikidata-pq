# 2026-10-26: The release schema, evolving while a release is built

The user's decision (docs/journal/2026-10-26-schema-integrity.md, #97): a release's
claims schema evolves automatically while the release is built, and every published file
of the release still has one exact schema. This entry is the design as built
(`src/wikidata/release_schema.py`, `src/wikidata/schema_evolution.py`), its restart states,
its tests and its costs.

## What happened before

- **The halt.** genson's `schema_check="compatible"` raised for a chunk whose inferred
  schema had a field or dtype `claims_schema` lacked, and the run stopped ("Schema
  mismatch … update its stored schema").
- **No resume after a fix.** Widening `claims_schema` left every chunk and group file
  written before in the old schema, some of them on the build branch. Compaction and the
  sort require one schema per key, so finalise halted.
- **A second stop, found here.** A chunk whose process step stops after it has moved the
  split's built files into place (the entities file, the term tables) could not be run
  again: `dump.check_chunk` checked the split's files again, found them moved, and asked
  for the split to be run again. A halt at the claims step is such a stop, and so is a
  kill there. `check_chunk` now checks a chunk once, when it is first pulled.
- **A third, found by `scripts/test_group_restart.py`** (one run in three here, at its
  "claims glue" kill point). A run killed while a group's merge deleted a glued key's
  partition files, its staged file already in place, halted when run again: "partition
  files missing". The staged file is moved into place only once its rows are checked, so
  a key with its first partition file gone and its staged file in place is merged; the
  rest of its partition files are deleted. The new kill point "merge: mid deletion of a
  glued key's partition files" fails on master and passes here.

## Design

### The release schema and its log

The release schema starts as the stored schema (`claims_schema`, version 0). Its log,
`releases/{release}/schema/claims.json`, holds the versions after it. Each version is a
list of changes:
- **`add`:** a field a struct lacked, at any depth, of any dtype;
- **`type`:** a field the release added as Null (every value seen null), given the dtype a
  later chunk shows.

Nothing else is a change. The log is shared by the split and both sets, and records the
stored schema it was begun from (a digest): code with another stored schema halts on it.
It is written atomically, under an `fcntl` lock. The latest version's Avro schema is
written beside it (`claims.avsc`), to read; the code derives it.

### Modes

- **Auto (the default).** A chunk whose inferred schema does not fit the latest version is
  inferred again for its schema. What it adds is merged into the log as a new version,
  and the chunk is written in it.
- **Strict (`WIKIDATA_SCHEMA_EVOLUTION=strict`).** The run halts with the changes proposed,
  recorded as pending, and names `just accept-schema <release>`. `accept-schema` merges
  every pending proposal into one version, each change against the version as `propose`
  merges a chunk's. Running again resumes.
- **Either mode** halts on a change that is not additive, with the path, both types and
  the chunk. Such a change is:
  - a scalar become a record or a list;
  - another type for a column (an `Int64` where a `String` was, or a `Float64` where an
    `Int64` was: genson does not widen, and nor does this);
  - a value in a field the stored schema has as Null in every row (`altitude`);
  - a map become a record.

  For the last, genson unifies a map's values unless one does not fit the others, so each
  key's value is diffed against the map's to name the change (a statement's `rank` as an
  integer: `claims[].value[].rank: String in the release schema,
  Struct({'rank__integer': Int64, 'rank__string': String}) in the main set's
  chunk_3.parquet (a scalar became a record (in the value of key 'P31')))`).
- **No field is dropped.** The fields genson leaves out by design are unchanged:
  `QUARANTINE_FIELDS` are pruned into the quarantine, and the label maps are extracted.

### Ordering

Each struct of a version has the stored schema's fields of that struct first, in their
stored order. The fields added to it follow, sorted by name, at every depth: inside an
added record too. So a version's schema depends on its fields alone. It does not depend on:
- which chunk found a field;
- the order chunks were processed or finished in;
- concurrency or restarts.

The merge is a join: a field found twice is added once. A field found as Null and then
typed is typed, in either order. Two dtypes of one field halt, in either order. The
version numbers depend on discovery order, but the final schema does not, and versions
only grow. Since sorting a subset keeps its order, version *v*'s schema is the frozen
schema restricted to the fields of the versions up to *v*.

### Discovery (process.evolve_in_schema)

1. The chunk is normalised in the latest version, decoded against its Avro schema where it
   fits, exactly as before (one pass).
2. If genson's check fails, the chunk is normalised again without a target. Its schema is
   read from the output's `genson_avro_schema` (the pruned schema, its label maps
   extracted), and `propose` merges it under the lock.
3. The chunk is normalised in the version returned, decoded against the schema just
   inferred. Its rows fit that schema strictly, so this pass decodes; it does not infer.

A discovering chunk costs three passes, against one for a chunk that fits. A genson change
that returned the inferred schema with the error would save the second pass. It isn't
needed.

### The fast path follows (claims_build.current)

Version *v*'s Avro schema is the stored `claims.avsc` completed against version *v*'s
schema (`schema_evolution.complete_avro`, the completion #172 made, moved from
`scripts/avro_schemas.py`). The split reads the log's version before each part (a `stat`),
compiles that version's plan once, builds the part in it, and tags its file. A part with a
field no version has yet still misfits, and is left as JSON for the pipeline (the safe
fallback). On the test's slice, with the release schema given its final version before the
split, every part was built by the fast path, the new fields among them.

### Files keep their version, and are checked exactly

A claims partition file written in version *v* > 0 is tagged *v* in its Arrow schema's
metadata (`wikidata.schema_version`). An untagged file is version 0, so a release whose
dump fits writes the bytes it wrote before. The tag survives the merge's glue: `pq_glue`
keeps the first input's key-value metadata, and refuses inputs whose metadata differ.

The check that replaces "every file of a key has the first file's schema" is
`ReleaseSchema.check`:
- the file's schema (its metadata aside) is exactly the schema of the version it is
  tagged;
- that version is at most the limit: the latest at a merge, the set's frozen version in
  finalise.

Version *v*'s schema is the frozen schema less the fields added after *v*, with its typed
fields Null. The new check catches everything the equality check caught: a file of any
other schema fails it. It also catches more:
- a key whose files all share some schema other than the release's;
- a file tagged otherwise than its schema;
- a file of a version after the freeze.

Where files of two versions meet, the earlier ones are read as the later, never changed in
place:
- **A group's merge** (`push.core._harmonise`): each chunk's claims file is checked, and
  those of a version before the group's latest are written in it, a row group at a time,
  to the group's work dir, and glued in their place. A group file has one version.
- **Compaction** checks every group file against its version, and glues runs of one
  version only (`compact._runs`).
- **The sort** checks every file and conforms each one's rows to the frozen schema as it
  reads them: a key in memory (`_read_key`), a bucketed key's spilled row groups
  (`_bucket_source`) and its direct reads (`_read_bucket`). The files it writes are in the
  frozen schema, checked by `_check_written` and `_check_sorted` as before.
- **The plan** (`footer_size`) counts only the stored schema's leaves once a release has
  a version. Every version's files have those leaves with the same counts, so the plan,
  and the published files with it, do not depend on which chunks were written before a
  field was found. A field a release adds is left out of the estimate.

The conform (`schema_evolution.conform_table`) inserts each missing field as nulls at any
depth, gives each Null column its later type, and keeps every other buffer. It does not
cast: pyarrow's cast corrupts the claims' nested structs. It handles slices, chunks and the
name of a list's item field. A field the target lacks, or any other type, raises.

### Finalise freezes one schema

A set's finalise freezes the release schema at its latest version the first time it is
asked (compaction's start). The freeze is recorded per set. A field found by that set's
run afterwards halts. Every published file of the set has exactly the frozen schema.

The main set's first finalise runs beside the scholarly run's last merges
(`overlap-release`). So the scholarly set can add a field after the main set froze, and
then publishes a schema with a field the main set's lacks. No main row holds that field,
since the main run had ended. In the test both sets froze at the same version.

### Promotion

`just promote-schema <release>` writes the latest version's changes to
`src/wikidata/schemas/claims.promoted.json`. `claims_schema` applies them at import, each
release's in turn, so the fields one release added are stored fields, in their order, for
the next. It also completes `schemas/claims.avsc` with them, and prints `git diff --stat`
for review. It checks that the promoted changes make the release's final schema.

`claims_schema.py`'s hand-written definitions are not rewritten. Promoting changes the
stored schema's digest, so the promoted release's own log halts on code after the
promotion: promote last.

### Other tables

- **Labels, descriptions and aliases** have stored schemas, but each is a map of a
  language to a string (or a list of strings). A map has no fields to add, so a change to
  one is never additive, and still halts (now saying so).
- **Entities and sitelinks** halt in the split on a field the pipeline does not read
  (`dump._check_fields`). They are records and could take the same mechanism; not done
  here (claims first).

## Restart states

Each state below was killed (`os._exit`, as by SIGKILL, through a `sitecustomize` hook in
every process, so in spawned workers too) and run again. Each time the outcome was
native's published files, byte for byte, in its schema. Native is the same slice with the
release schema given its final version before the split.
- **A chunk worker** after a version is added and before its chunk's claims file is
  written. The version stays; the chunk is processed again in the latest.
- **The run** mid-merge, after a chunk's file of an earlier version was written in the
  group's. The work dir is redone, and the chunk's own files are kept until the glue.
- **The split** once it has built a part in a later version (the split slowed, so the
  runs find a field while it has parts left).
- **Finalise** while the sort reads a file of an earlier version.
- **A strict halt,** accepted and run again; and the `check_chunk` stop above.

## Tests

`scripts/test_schema_evolution.py` runs the split, both runs and both finalises on a
6,000-entity slice of bench-data's r50 and s30. Four fields are injected:
- a mainsnak string field, in two main chunks (found twice);
- a reference snak datavalue's integer field;
- a statement's record field, in a scholarly entity.

Its scenarios:

| Scenario | Result |
|---|---|
| fits | no version, no tag; the files uploaded and published are master's (831f225, with rule B), byte for byte |
| evolved | 3 versions; the first group verified on the stand-in Hub before chunk 2 found anything (its claims untagged); a group of mixed versions merged (1 of its 3 files written in the group's version); every build-branch claims file exactly its version's schema; every published claims file exactly the frozen schema; each injected value published as injected; native's files byte for byte |
| native | every part built by the split's fast path (none as JSON) |
| overlapped, uninterrupted and killed at 4 points | native's files byte for byte |
| strict | halted, proposals pending, accepted, resumed; native's files byte for byte |
| incompatible | a rank as an integer halts with the diagnostic above; no version added |
| concurrent | two processes released by one barrier, six rounds, different and the same fields: one final schema, in the canonical order |
| promote | `claims.promoted.json` and `claims.avsc` make the stored schema the release's final one |

`scripts/test_schema_algebra.py` (seconds) covers the rest:
- **order:** the four proposals in all 24 orders give one schema;
- **strict:** halts, pending, accept, in every order;
- **conflicts:** each kind, naming path, types and chunk;
- **freeze**, and **another stored schema;**
- **check:** each failure mode;
- **conform:** on sliced and chunked arrays, list item names, and its refusals;
- **promotion's order.**

The existing `test_claims_schema`, `test_claims_build` (r50) and `test_merge` pass.

## Measured (local, indicative; CI measures)

Bench 17.7.0's `evolve.py` stage runs on c0 (10,000 entities, 42 MB of claims partition
file), each step in a process of its own:

| Step | CPU | Wall | Peak RSS |
|---|---|---|---|
| process step finding the field (3 passes) | 20.8 s | 6.5 s | 1,735 MB |
| process step, field known, part as JSON | 9.9 s | 3.3 s | 1,494 MB |
| split, field known (part built) | 6.0 s | 4.1 s | |
| split, field unknown (part as JSON) | 3.9 s | 3.9 s | |
| a version 0 claims file written in version 1 (a merge) | 6.2 s | 3.3 s | 843 MB, 42 MB transient disk |
| a version 0 claims file read | 1.47 s | | 967 MB |
| the same read and conformed to version 1 | 1.41 s | | 966 MB |

- **Discovery** costs a chunk about 11 CPU-s more (2.1 times the process step), once per field (a field found by two
  workers at once costs each its passes, but adds one version).
- **Widening at a merge** costs about a chunk's partition-file write, for each chunk of a
  group written before the field was found. It is bounded by a group's chunks a version.
- **Widening on read** is within noise of the read.
- **The fast path's following:** a later part holding the field is built in the split
  (6.0 CPU-s, against 3.9 to leave it as JSON) and only placed by the pipeline, rather
  than decoded there (9.9 CPU-s): about 8 CPU-s saved a part once the split catches up.
- **On master** the field halts the run (`evolve_halts`), so there is no before figure for
  any of these but that.

## Not done, or negative

- **The genson second pass** could be saved by genson returning the inferred schema with
  its error. Not needed, so there is no genson release.
- **The two sets can publish two schemas** if the scholarly set adds a field after the
  main set's early finalise froze (see the freeze above).
- **The plan's estimate leaves out added fields.** A release that adds a large field would
  get buckets and part files a little larger than their targets.
