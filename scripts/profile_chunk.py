"""Profile processing of a single already-downloaded chunk, in isolation from the live run.

Copies the chunk's source parquet into a scratch data_dir/output_dir so nothing
touches the real state/ or results/ directories, then profiles process().

Usage: python scripts/profile_chunk.py <chunk_idx>
"""

import cProfile
import pstats
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wikidata.config import REMOTE_REPO_PATH, REPO_ID, chunk_glob  # noqa: E402
from wikidata.process import process  # noqa: E402
from wikidata.pull import _hf_dl_subdir  # noqa: E402

REAL_DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    chunk_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 188

    real_ds_dir = _hf_dl_subdir(REAL_DATA_DIR, repo_id=REPO_ID)
    src_files = sorted((real_ds_dir / REMOTE_REPO_PATH).glob(chunk_glob(chunk_idx)))
    if not src_files:
        print(f"No local source file found for chunk {chunk_idx} under {real_ds_dir / REMOTE_REPO_PATH}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "data"
        output_dir = tmp / "output"
        state_dir = tmp / "state"
        scratch_ds_dir = _hf_dl_subdir(data_dir, repo_id=REPO_ID) / REMOTE_REPO_PATH
        scratch_ds_dir.mkdir(parents=True)
        output_dir.mkdir()
        state_dir.mkdir()

        for f in src_files:
            shutil.copy(f, scratch_ds_dir / f.name)

        profiler = cProfile.Profile()
        profiler.enable()
        process(
            data_dir=data_dir,
            output_dir=output_dir,
            repo_id=REPO_ID,
            state_dir=state_dir,
            chunk_idx=chunk_idx,
        )
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(25)


if __name__ == "__main__":
    main()
