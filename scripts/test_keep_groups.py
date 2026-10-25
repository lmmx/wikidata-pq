#!/usr/bin/env python3
"""Test that verified group files are kept for finalise, and read there instead of
downloaded (no Hub: a stand-in HfApi keeps its "build branch" on disk, and compaction's
download is replaced by one that records its calls).

- kept (WIKIDATA_KEEP_GROUP_FILES=1): once verified, a group's staged files are moved to
  the compaction source, recorded with their sha256, the Hub's, and staging emptied; a
  release's claims_labels are not kept (finalise builds its own there)
- compaction's source stage reads them: no download, every file checked against the
  record
- a kept file corrupted (same size), or missing, or a group with no record: the
  download runs, the corrupted file removed first
- not kept (=0): deleted as before, and the download runs
- unset: not kept (the default); auto: kept while the disk has KEEP_GROUP_FILES_MIN_FREE_GB free
- a clean-up interrupted mid-move, rerun from `verified`: every file kept, none lost

Runs in a temporary directory.

Usage: python scripts/test_keep_groups.py
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

os.environ["WIKIDATA_RELEASE"] = "20261012"
os.environ["HF_HUB_OFFLINE"] = "1"

import polars as pl  # noqa: E402

import wikidata.compact as compact  # noqa: E402
import wikidata.push.core as core  # noqa: E402
from wikidata.config import COMPACT_DIR, STAGING_DIR, STATE_DIR, Table  # noqa: E402
from wikidata.push.groups import Group, record_stage  # noqa: E402

TABLES = list(Table)
KEYS = {t: ["all"] for t in Table} | {Table.LABEL: ["en", "de", "zh-hans"], Table.CLAIMS_LABELS: ["en"]}


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


class FakeHub:
    """The HfApi calls close_group makes, against a directory."""

    def __init__(self, root: Path):
        self.root = root

    def _dir(self, repo_id, revision):
        return self.root / repo_id.replace("/", "__") / (revision or "main")

    def create_repo(self, repo_id, **kw):
        self._dir(repo_id, None).mkdir(parents=True, exist_ok=True)

    def list_repo_refs(self, repo_id, **kw):
        d = self.root / repo_id.replace("/", "__")
        return SimpleNamespace(branches=[SimpleNamespace(name=b.name) for b in d.iterdir()], tags=[])

    def create_branch(self, repo_id, branch, **kw):
        self._dir(repo_id, branch).mkdir(parents=True, exist_ok=True)

    def list_repo_files(self, repo_id, revision=None, **kw):
        d = self._dir(repo_id, revision)
        return [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]

    def file_exists(self, repo_id, path, revision=None, **kw):
        return (self._dir(repo_id, revision) / path).exists()

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision=None, **kw):
        dst = self._dir(repo_id, revision) / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path_or_fileobj)

    def upload_folder(self, repo_id, folder_path, allow_patterns, revision=None, **kw):
        for p in sorted(Path(folder_path).glob(allow_patterns)):
            dst = self._dir(repo_id, revision) / p.relative_to(folder_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)

    def get_paths_info(self, repo_id, paths, revision=None, **kw):
        d = self._dir(repo_id, revision)
        return [SimpleNamespace(path=p, size=(d / p).stat().st_size,
                                lfs=SimpleNamespace(sha256=hashlib.sha256((d / p).read_bytes()).hexdigest()))
                for p in paths if (d / p).exists()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        compact_src = COMPACT_DIR / "src"
        repos = {t: f"test/wikidata-{t}" for t in TABLES}
        downloads = []
        compact.download = lambda table, repo_id: downloads.append(table)

        def stage(groups: list[Group]) -> dict[str, str]:
            """Each group's merged files in staging, as merge_group leaves them."""
            want = {}
            for g in groups:
                for t in TABLES:
                    for k in KEYS[t]:
                        p = core.staged_path(t, k, g, STAGING_DIR)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        pl.DataFrame({"id": [f"Q{g.first}-{k}-{i}" for i in range(20)]}).write_parquet(p)
                        want[f"{t}/{k}/{g.name}.parquet"] = sha(p)
            return want

        def fresh() -> None:
            for d in ("releases", "fakehub"):
                shutil.rmtree(d, ignore_errors=True)
            STATE_DIR.mkdir(parents=True)
            downloads.clear()

        def close(g: Group, stage_: str = "merged") -> None:
            record_stage(STATE_DIR, g, "closed")
            core.close_group(g, target_repos=repos, state_dir=STATE_DIR, stage=stage_, api=FakeHub(Path("fakehub")))

        def source() -> None:
            downloads.clear()
            for t in TABLES:
                if core.keeps(t):
                    compact.source_group_files(t, repos[t], STATE_DIR)

        groups = [Group(0, 4, 9), Group(5, 9, 9)]
        fresh()
        core.KEEP_GROUP_FILES = True
        want = {}
        for g in groups:  # staging holds one group at a time
            want |= stage([g])
            close(g)
        kept = {f"{p.relative_to(compact_src)}": sha(p) for p in sorted(compact_src.rglob("*.parquet"))}
        check("kept: every group's files but a release's claims_labels in the compaction source, as staged",
              kept == {k: v for k, v in want.items() if not k.startswith("claims_labels/")})
        check("kept: staging emptied", not any(STAGING_DIR.rglob("*.parquet")))
        verified = core.read_verified(STATE_DIR)
        check("kept: each file recorded with the Hub's sha256",
              all(verified[(g.first, g.last, str(t))][f"{k}/{g.name}.parquet"][1] == want[f"{t}/{k}/{g.name}.parquet"]
                  for g in groups for t in TABLES for k in KEYS[t]))
        source()
        check("compaction reads the kept files: no download", downloads == [])

        first = compact_src / Table.LABEL / "en" / f"{groups[0].name}.parquet"
        data = bytearray(first.read_bytes())
        data[len(data) // 2] ^= 0xFF
        first.write_bytes(bytes(data))
        source()
        check("a kept file corrupted (same size): removed, and the table downloaded",
              downloads == [Table.LABEL] and not first.exists())
        (compact_src / Table.CLAIMS / "all" / f"{groups[1].name}.parquet").unlink()
        source()
        check("a kept file missing: the table downloaded", downloads == [Table.LABEL, Table.CLAIMS])

        fresh()
        for g in groups:
            stage([g])
            close(g)
        everything = [t for t in TABLES if core.keeps(t)]
        lines = (STATE_DIR / core.VERIFIED).read_text().splitlines()
        (STATE_DIR / core.VERIFIED).write_text("\n".join(line for line in lines if json.loads(line)["first"] != 5) + "\n")
        source()
        check("a group with no verified record (closed before it, or elsewhere): every table downloaded",
              downloads == everything)

        fresh()
        core.KEEP_GROUP_FILES = False
        for g in groups:
            stage([g])
            close(g)
        check("not kept (=0): deleted, as before", not compact_src.exists() and not any(STAGING_DIR.rglob("*.parquet")))
        source()
        check("not kept: the download runs", downloads == everything)

        core.KEEP_GROUP_FILES = None
        for free, kept_ in ((1e9, False), (0, True)):
            fresh()
            core.KEEP_GROUP_FILES_MIN_FREE_GB = free
            stage(groups[:1])
            close(groups[0])
            check(f"auto, {free:g} GB to be left free: {'kept' if kept_ else 'not kept'}",
                  any(compact_src.rglob("*.parquet")) == kept_)

        # A clean-up interrupted mid-move: rerun from `verified`, every file kept
        fresh()
        core.KEEP_GROUP_FILES = True
        want = stage(groups[:1])
        move, moved = shutil.move, []

        def failing(src, dst):
            moved.append(src)
            if len(moved) == 2:
                raise KeyboardInterrupt
            return move(src, dst)

        core.shutil.move = failing
        try:
            close(groups[0])
        except KeyboardInterrupt:
            pass
        core.shutil.move = move
        check("interrupted mid-move: one file kept, the rest still staged",
              len(list(compact_src.rglob("*.parquet"))) == 1 and any(STAGING_DIR.rglob("*.parquet")))
        core.close_group(groups[0], target_repos=repos, state_dir=STATE_DIR, stage="verified", api=FakeHub(Path("fakehub")))
        kept = {f"{p.relative_to(compact_src)}": sha(p) for p in sorted(compact_src.rglob("*.parquet"))}
        check("rerun from `verified`: every file kept, as staged",
              kept == {k: v for k, v in want.items() if not k.startswith("claims_labels/")})
        source()
        check("and compaction reads them: no download", downloads == [])
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
