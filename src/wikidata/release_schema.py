"""A release's claims schema, which grows as the release runs (docs/journal/
2026-10-26-schema-evolution.md; the algebra is schema_evolution.py).

Every published claims file of a set has one exact schema. It starts as the stored one
(claims_schema.claims_schema, version 0) and is the release schema: a log of versions in
SCHEMA_DIR/claims.json, each the one before with the fields a chunk showed that it lacked
(schema_evolution.apply), shared by the split, both sets' runs and their finalises.

- **Discovery.** A chunk whose inferred schema does not fit the latest version is
  inferred again for its schema (process.normalise_claims_direct), and `propose` merges
  what it adds under the log's lock (fcntl): nothing new (another worker added it, or an
  earlier chunk), the latest version; else a new version. Versions only grow. Each
  struct's fields are its stored fields in their stored order, then those added, sorted
  by name, so a version's schema depends on its fields alone, not on which chunk found
  them, in what order the chunks were processed or finished, or restarts.
- **Modes** (WIKIDATA_SCHEMA_EVOLUTION): `auto` (the default) adds each field as it is
  found; `strict` halts the run with the changes proposed, recorded as pending, until
  `just accept-schema <release>` accepts them as a version (`accept`), and the run is
  resumed. In either mode, a change that is not additive (a scalar become a record or a
  list, another type, a value in a column the stored schema has as null in every row)
  halts the run with its path, both types and the chunk that showed it: an existing
  column never changes what it means.
- **Files.** A claims partition file written in version v > 0 is tagged v in its Arrow
  schema's metadata (VERSION_KEY; untagged is version 0, so a release whose dump fits
  writes the bytes it wrote before). A file keeps its version: a chunk, group or
  uploaded file is never rewritten for a later version, but where files of two versions
  meet: a group's chunks (push.core), whose older files are conformed to the group's
  latest when merged, and finalise's reads (compact, sort_by_id), which conform each file
  to the frozen schema as they read it.
- **Checks.** A file's schema must be exactly the schema of the version it is tagged
  (`check`), which is the frozen schema less the fields added after it (and its typed
  fields Null), and its version at most the latest (at a merge) or the frozen (in
  finalise). That catches all the old equality check did (a file of any other schema),
  and a file tagged other than its schema.
- **Freeze.** A set's finalise uses one schema, frozen the first time finalise asks for
  it (`frozen`): the latest version then, recorded per set; its run can add no field
  after (it would halt). Every published file of the set is written in it.
- **Promotion.** After a release, `just promote-schema <release>` writes the latest
  version's additions to schemas/claims.promoted.json (applied to claims_schema at
  import) and completes schemas/claims.avsc with them, for review as a diff: the next
  release starts from them.

Labels, descriptions and aliases have stored schemas too, but each is a map of a
language to a string (or a list of strings): a map has no field to add, so a change to
one is never additive, and still halts (process.normalise_in_schema). Entities and
sitelinks halt in the split on a field the pipeline does not read (dump._check_fields).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from . import claims_derived
from . import schema_evolution as se
from .claims_schema import CLAIMS_AVRO, PROMOTED_PATH, SCHEMAS_DIR, claims_schema, promoted_schema, stored_claims_schema
from .config import RELEASE, RELEASES_DIR, SCHOLAR, WORK_DIR

MODES = ("auto", "strict")
MODE = os.environ.get("WIKIDATA_SCHEMA_EVOLUTION") or "auto"
if MODE not in MODES:
    raise SystemExit(f"WIKIDATA_SCHEMA_EVOLUTION is auto or strict, not {MODE!r}")

# A release's schema log, shared by its split and both sets: in the main set's work dir
SCHEMA_DIR = (RELEASES_DIR / str(RELEASE) if RELEASE else WORK_DIR) / "schema"
VERSION_KEY = b"wikidata.schema_version"
SET = "scholar" if SCHOLAR else "main"


class SchemaHalt(SystemExit):
    """The run halts: a change that is not additive, one strict mode waits on, or a field
    found after the set's schema was frozen."""


def version_of(schema: pa.Schema) -> int:
    """The version a file's Arrow schema is tagged (0 if untagged)."""
    tag = (schema.metadata or {}).get(VERSION_KEY)
    return int(tag) if tag else 0


def tagged(table: pa.Table, version: int) -> pa.Table:
    """`table` tagged as written in `version` (untouched for version 0)."""
    if not version:
        return table
    return table.replace_schema_metadata({**(table.schema.metadata or {}), VERSION_KEY: str(version).encode()})


class ReleaseSchema:
    """One table's release schema: its log, versions, checks and conform."""

    def __init__(self, table: str, base: pl.Schema, base_avro: str | None, write_schema, directory: Path = SCHEMA_DIR):
        self.table, self.base, self.base_avro, self._write_schema = table, base, base_avro, write_schema
        self.dir = Path(directory)
        self._cache: tuple | None = None
        self._schemas: dict[int, pl.Schema] = {}
        self._files: dict[int, pa.Schema] = {}
        self._writes: dict[int, pa.Schema] = {}
        self._leaves: frozenset[str] | None = None
        self._avros: dict[int, str | None] = {}

    @property
    def path(self) -> Path:
        return self.dir / f"{self.table}.json"

    @property
    def base_spec(self) -> str:
        """The stored schema the log was begun from, as a digest of its spec."""
        return hashlib.sha256(json.dumps(se.spec(pl.Struct(self.base))).encode()).hexdigest()

    # The log

    def log(self) -> dict:
        """The log as last written (re-read when its file changes; empty if none)."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return self._fresh()
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        if self._cache is None or self._cache[0] != key:
            log = json.loads(self.path.read_text())
            if log["base"] != self.base_spec:
                raise SchemaHalt(
                    f"{self.path}: the release's {self.table} schema log was begun from another stored schema than "
                    f"this code's ({PROMOTED_PATH.name} or claims_schema changed during the release). Finish the release "
                    "with the code it was begun with, or start it again.")
            self._cache = (key, log)
        return self._cache[1]

    def _fresh(self) -> dict:
        return {"table": self.table, "base": self.base_spec, "versions": [], "pending": [], "frozen": {}}

    @contextmanager
    def _locked(self):
        """The log, read under its lock, written back (atomically) if changed."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{self.table}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._cache = None
                log = self.log()
                before = json.dumps(log, sort_keys=True)
                log = json.loads(before)  # (a copy: the cache is not changed in place)
                yield log
                if json.dumps(log, sort_keys=True) != before:
                    tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_text(json.dumps(log, indent=1) + "\n")
                    tmp.replace(self.path)
                    self._cache = None
                    self._write_avro(log)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def latest(self) -> int:
        return len(self.log()["versions"])

    def changes(self, version: int, log: dict | None = None) -> list[dict]:
        log = log or self.log()
        if version > len(log["versions"]):
            raise RuntimeError(f"[schema] {self.table}: no version {version} (the latest is {len(log['versions'])})")
        return [c for v in log["versions"][:version] for c in v["changes"]]

    def schema(self, version: int, log: dict | None = None) -> pl.Schema:
        """Version `version`'s schema (version 0: the stored schema)."""
        if version not in self._schemas or log is not None:
            s = pl.Schema(se.apply(pl.Struct(self.base), self.changes(version, log)))
            if log is not None:
                return s
            self._schemas[version] = s
        return self._schemas[version]

    def avro(self, version: int) -> str | None:
        """The Avro schema the version's claims are decoded against (the stored
        claims.avsc completed against the version's schema), or None without one."""
        if version not in self._avros:
            if self.base_avro is None or version == 0:
                self._avros[version] = self.base_avro
            else:
                column = self.base.names()[-1]
                self._avros[version] = se.complete_avro(self.base_avro, self.schema(version)[column], column)
        return self._avros[version]

    def _write_avro(self, log: dict) -> None:
        """The latest version's Avro schema beside the log (to read; the code derives it)."""
        v = len(log["versions"])
        if v and self.base_avro is not None:
            column = self.base.names()[-1]
            text = se.complete_avro(self.base_avro, self.schema(v, log)[column], column)
            tmp = self.dir / f"{self.table}.avsc.{os.getpid()}.tmp"
            tmp.write_text(text + "\n")
            tmp.replace(self.dir / f"{self.table}.avsc")

    def write_schema(self, version: int) -> pa.Schema:
        """The Arrow schema the table's rows of `version` are written from (in memory, as
        the writers hold them: Polars' list item names among them)."""
        if version not in self._writes:
            self._writes[version] = self._write_schema(self.schema(version))
        return self._writes[version]

    def file_schema(self, version: int) -> pa.Schema:
        """The Arrow schema of the table's files written in `version`, as read back."""
        if version not in self._files:
            from .partitioning.core import write_table

            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "empty.parquet"
                write_table(self.write_schema(version).empty_table(), p)
                self._files[version] = pq.read_schema(p).remove_metadata()
        return self._files[version]

    # Discovery

    def propose(self, inferred: pl.Schema, example: str, set_name: str = SET) -> int:
        """Merge what a chunk's inferred schema adds to the latest version, under the log's
        lock, and return the version to write the chunk in (see the module's docstring)."""
        with self._locked() as log:
            latest = len(log["versions"])
            current = self.schema(latest, log)
            changes, conflicts = se.diff(pl.Struct(current), pl.Struct(inferred), pl.Struct(self.base))
            if conflicts:
                raise SchemaHalt(self._conflicts(conflicts, example))
            if not changes:
                return latest
            changes = se.merge_changes(changes)
            if set_name in log["frozen"]:
                raise SchemaHalt(
                    f"[schema] {self.table}: {example} has fields the {set_name} set's frozen schema (version "
                    f"{log['frozen'][set_name]}) lacks, and its finalise has begun:\n{self._list(changes)}")
            if MODE == "strict":  # (recorded as the lock is released, then the run halts)
                pending = {json.dumps(c, sort_keys=True) for p in log["pending"] for c in p["changes"]}
                new = [c for c in changes if json.dumps(c, sort_keys=True) not in pending]
                if new:
                    log["pending"].append({"changes": new, "example": example, "at": time.time()})
                halt = (f"[schema] {self.table}: {example} has fields the release schema (version {latest}) lacks "
                        f"(WIKIDATA_SCHEMA_EVOLUTION=strict), proposed:\n{self._list(changes)}\n"
                        f"Accept them with `just accept-schema {RELEASE}`, then run again to resume.")
            else:
                log["versions"].append({"version": latest + 1, "changes": changes, "example": example, "at": time.time()})
                print(f"[schema] {self.table}: version {latest + 1}, from {example}:\n{self._list(changes)}", flush=True)
                return latest + 1
        raise SchemaHalt(halt)

    def accept(self) -> int | None:
        """Accept every pending proposal as one version (strict mode's `accept-schema`):
        their changes merged, each against the version before as `propose` would merge
        them. Returns the new version, or None if nothing is pending."""
        with self._locked() as log:
            if not log["pending"]:
                return None
            latest = len(log["versions"])
            # Each proposed change merged in turn into the latest version, as propose merges
            # a chunk's: a field two proposals add is added once, a field added as null and
            # then typed is typed, and two dtypes of one field halt (whatever their order)
            base = pl.Struct(self.base)
            running = pl.Struct(self.schema(latest, log))
            changes, conflicts = [], []
            for c in (c for p in log["pending"] for c in p["changes"]):
                path, dtype = tuple(c["path"]), se.dtype_of(c["dtype"])
                have = se.at(running, path)
                if have is None:
                    new, bad = [c], []
                else:
                    new, bad = se.diff(have, dtype, base, path)
                conflicts += bad
                if new:
                    running = se.apply(running, new)
                    changes += new
            if conflicts:
                raise SchemaHalt(self._conflicts(conflicts, "the pending proposals"))
            changes = se.merge_changes(changes)
            examples = ", ".join(sorted({p["example"] for p in log["pending"]}))
            log["versions"].append({"version": latest + 1, "changes": changes, "example": examples,
                                    "accepted": True, "at": time.time()})
            log["pending"] = []
            self.schema(latest + 1, log)  # (raises if they do not apply)
            return latest + 1

    def _list(self, changes: list[dict]) -> str:
        return "\n".join(f"  {c['op']} {se.show(c['path'])}: {se.dtype_of(c['dtype'])}" for c in changes)

    def _conflicts(self, conflicts: list[dict], example: str) -> str:
        lines = [f"  {se.show(c['path'])}: {c['old']} in the release schema, {c['new']} in {example} ({c['why']})"
                 for c in conflicts]
        lines = list(dict.fromkeys(lines))[:20]  # (one line a path and change, as a map's keys repeat it)
        return (f"[schema] {self.table}: {example} does not fit the release schema, and the change is not additive "
                "(only a new nullable field is added; an existing column never changes its type):\n" + "\n".join(lines)
                + f"\nThe chunk is {example}: see its claims at these paths. Nothing was written for it.")

    # Finalise

    def frozen(self, set_name: str = SET) -> int:
        """The set's frozen version: the latest when finalise first asks (recorded)."""
        log = self.log()
        if set_name in log["frozen"]:
            return log["frozen"][set_name]
        with self._locked() as log:
            if set_name not in log["frozen"]:
                log["frozen"][set_name] = len(log["versions"])
                if log["pending"]:
                    raise SchemaHalt(f"[schema] {self.table}: proposals pending (strict mode): accept them "
                                     f"(`just accept-schema {RELEASE}`) and finish the run before finalise")
                print(f"[schema] {self.table}: the {set_name} set's schema frozen at version "
                      f"{log['frozen'][set_name]}", flush=True)
            return log["frozen"][set_name]

    # Checks and conform

    def check(self, schema: pa.Schema, what: str, upto: int) -> int:
        """A file's version, its schema (with its metadata) checked to be exactly that
        version's, and its version at most `upto`. A file whose metadata names derived
        leaves it was written without (claims_derived) is checked to be exactly that
        version's schema less those leaves, each a derived leaf of it."""
        v = version_of(schema)
        if v > upto:
            raise RuntimeError(f"[schema] {what} is of {self.table} schema version {v}, after version {upto}")
        if claims_derived.dropped(schema):
            claims_derived.check_form(schema, self.file_schema(v), f"[schema] {what} ({self.table} schema version {v})")
        elif not schema.remove_metadata().equals(self.file_schema(v)):
            raise RuntimeError(f"[schema] {what} has another schema than {self.table} schema version {v}'s")
        return v

    def check_file(self, path: Path, upto: int) -> int:
        return self.check(pq.read_schema(path), str(path), upto)

    def conform(self, table: pa.Table, version: int) -> pa.Table:
        """Rows of a file (any version up to `version`) as `version`'s rows."""
        return se.conform_table(table, self.file_schema(version))

    def upgrade_file(self, src: Path, dst: Path, version: int, keep=None) -> None:
        """Write `src` (a file of an earlier version, or of other derived leaves dropped)
        to `dst` in `version`, tagged with it, and without the derived leaves `keep` (its
        own if None, which it must be without: claims_derived): a row group at a time, its
        leaves put back (claims_derived.restore), conformed, the leaves of `keep` left out
        again and written as the split writes a claims file (claims_build.RowGroups:
        write_table per row group, glued), so its row groups are the source's, and checked
        to be of the version and form."""
        import shutil

        import pq_glue

        from .partitioning.core import write_table

        pf = pq.ParquetFile(src)
        src_schema = pf.schema_arrow
        v_src, form = self.check(src_schema, str(src), version), claims_derived.dropped(src_schema)
        keep = form if keep is None else frozenset(keep)
        if not keep <= form:
            raise RuntimeError(f"[schema] {src}: has derived leaves to be left out ({sorted(keep - form)})")
        work = dst.with_suffix(".rowgroups")
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        files = []
        for g in range(pf.metadata.num_row_groups):
            files.append(work / f"{g}.parquet")
            rows = pf.read_row_group(g).replace_schema_metadata(src_schema.metadata)
            rows = claims_derived.restore(rows, self.file_schema(v_src))
            rows = tagged(se.conform_table(rows, self.write_schema(version)), version)
            write_table(claims_derived.drop(rows, keep), files[-1])
        if not files:
            files.append(work / "0.parquet")
            empty = tagged(self.write_schema(version).empty_table(), version)
            write_table(claims_derived.drop(empty, keep), files[0])
        tmp = dst.with_suffix(".tmp")
        if len(files) == 1:
            files[0].replace(tmp)
        else:
            pq_glue.concat([str(f) for f in files], str(tmp))
        shutil.rmtree(work)
        if pq.read_metadata(tmp).num_rows != pf.metadata.num_rows:
            raise RuntimeError(f"[schema] {dst}: rows differ from {src}'s")
        written = pq.read_schema(tmp)
        if self.check(written, str(dst), version) != version or claims_derived.dropped(written) != keep:
            raise RuntimeError(f"[schema] {dst}: not written in version {version} without {sorted(keep)}")
        tmp.replace(dst)

    def base_leaves(self) -> frozenset[str]:
        """The Parquet leaf paths of a version 0 file: every version's files have them."""
        if self._leaves is None:
            from .partitioning.core import write_table

            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "empty.parquet"
                write_table(self.write_schema(0).empty_table(), p)
                meta = pq.read_metadata(p)
                self._leaves = frozenset(meta.schema.column(i).path for i in range(meta.num_columns))
        return self._leaves

    def summary(self) -> str:
        log = self.log()
        out = [f"{self.table}: {len(log['versions'])} versions beyond the stored schema"
               + (f", frozen {log['frozen']}" if log["frozen"] else "")]
        for v in log["versions"]:
            out.append(f" version {v['version']} (from {v['example']}):\n{self._list(v['changes'])}")
        for p in log["pending"]:
            out.append(f" pending (from {p['example']}):\n{self._list(p['changes'])}")
        return "\n".join(out)


def _claims_write_schema(schema: pl.Schema) -> pa.Schema:
    """The Arrow schema a claims partition file whose claims are in `schema` is written
    from: genson's output (id, claims) laid out as statements (partitioning.claims)."""
    from .partitioning.claims import statements

    genson = pl.DataFrame(schema={"id": pl.String, **schema}).to_arrow(compat_level=pl.CompatLevel.oldest())
    return statements(genson).schema


CLAIMS = ReleaseSchema("claims", claims_schema, CLAIMS_AVRO, _claims_write_schema)


def evolving(table: str) -> ReleaseSchema | None:
    """The release schema of a table whose schema evolves (claims), else None."""
    return CLAIMS if str(table) == "claims" else None


def is_evolving(table: str) -> bool:
    """A table whose files may be of more than one version: its log has a version."""
    rs = evolving(table)
    return rs is not None and rs.latest() > 0


# Promotion


def promote(dst_dir: Path = SCHEMAS_DIR) -> list[dict]:
    """Write the latest version's additions (with any promoted before) to
    dst_dir/claims.promoted.json, and dst_dir/claims.avsc completed with them, so the next
    release starts from them; nothing is written if there are none. A set must have
    been frozen (its finalise begun), and nothing be pending. Returns the changes."""
    rs = CLAIMS
    log = rs.log()
    if not log["frozen"]:
        raise SystemExit(f"[schema] no set of release {RELEASE} has been finalised: promote after the release")
    if log["pending"]:
        raise SystemExit("[schema] proposals are pending: accept or discard them first")
    latest = len(log["versions"])
    changes = rs.changes(latest)
    if not changes:
        print(f"[schema] {rs.table}: release {RELEASE} added nothing to the stored schema", flush=True)
        return []
    # Each release's changes promoted in turn (claims_schema.promoted_schema): the fields of
    # one become stored fields, in their order, for the next
    promoted = json.loads(PROMOTED_PATH.read_text()) if PROMOTED_PATH.exists() else []
    out = [r for r in promoted if r["release"] != str(RELEASE)] + [{"release": str(RELEASE), "changes": changes}]
    if pl.Struct(promoted_schema(stored_claims_schema(), out)) != pl.Struct(rs.schema(latest)):
        raise SystemExit("[schema] the promoted changes do not make the release's schema")
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / PROMOTED_PATH.name).write_text(json.dumps(out, indent=1) + "\n")
    if rs.base_avro is not None:
        column = rs.base.names()[-1]
        (dst_dir / "claims.avsc").write_text(se.complete_avro(rs.base_avro, rs.schema(latest)[column], column) + "\n")
    print(f"[schema] {rs.table}: {len(changes)} changes of release {RELEASE} promoted into {dst_dir} "
          f"({PROMOTED_PATH.name}, claims.avsc): review them with `git diff`", flush=True)
    return changes


def main() -> None:
    """`python -m wikidata.release_schema show|accept|promote` (WIKIDATA_RELEASE set)."""
    if not RELEASE:
        raise SystemExit("WIKIDATA_RELEASE is needed: a release's schema")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "show":
        print(CLAIMS.summary())
    elif cmd == "accept":
        v = CLAIMS.accept()
        if v is None:
            print("[schema] claims: nothing pending")
        else:
            added = CLAIMS.changes(v)[len(CLAIMS.changes(v - 1)):]
            print(f"[schema] claims: accepted as version {v}:\n{CLAIMS._list(added)}\nRun again to resume.")
    elif cmd == "promote":
        promote()
    else:
        raise SystemExit(f"usage: python -m wikidata.release_schema show|accept|promote, not {cmd!r}")


if __name__ == "__main__":
    main()
