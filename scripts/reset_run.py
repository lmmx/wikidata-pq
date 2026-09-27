"""Reset the pipeline run: delete every output, locally and on the Hub.

############################################################################################
#                                                                                          #
#   NEVER TO BE RUN BY AN AGENT. ONLY BY THE USER, DIRECTLY, BY HAND, IN A TERMINAL.       #
#                                                                                          #
#   Not from Claude Code or any other AI agent, not from a script, a Justfile recipe, CI,  #
#   cron or any other automation. It permanently deletes the Hub datasets.                 #
#                                                                                          #
############################################################################################

Deletes:
- the six Hub dataset repos (permutans/wikidata-{table}), cards and all
- the local state/, results/, staging/, audit/ and quarantine/ directories

Keeps the downloaded source chunks (data/), which don't depend on the pipeline code.

Refuses to run without an interactive terminal, or inside a Claude Code session, and asks
you to type a confirmation phrase before deleting anything.

Run from anywhere: `uv run python scripts/reset_run.py`
"""

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wikidata.config import (  # noqa: E402
    AUDIT_DIR,
    HF_USER,
    OUTPUT_DIR,
    QUARANTINE_DIR,
    REPO_TARGET,
    STAGING_DIR,
    STATE_DIR,
    Table,
)

CONFIRMATION = "delete the wikidata datasets"


def main() -> None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit("Refusing: not an interactive terminal. Run this yourself, by hand.")
    if os.environ.get("CLAUDECODE"):
        sys.exit("Refusing: running inside Claude Code. Run this yourself, by hand.")

    repos = [REPO_TARGET.format(hf_user=HF_USER, tbl=tbl) for tbl in Table]
    dirs = [ROOT / d for d in (STATE_DIR, OUTPUT_DIR, STAGING_DIR, AUDIT_DIR, QUARANTINE_DIR)]

    print("This will permanently delete:\n")
    for repo in repos:
        print(f"  Hub dataset  {repo}")
    for d in dirs:
        print(f"  local dir    {d}{'' if d.exists() else '  (absent)'}")
    print(f'\nType "{CONFIRMATION}" to go ahead:')
    if input("> ").strip() != CONFIRMATION:
        sys.exit("Not confirmed, nothing deleted.")

    api = HfApi()
    for repo in repos:
        api.delete_repo(repo, repo_type="dataset", missing_ok=True)
        print(f"Deleted {repo}")
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
            print(f"Deleted {d}")
    print("Done. The next run starts from chunk 0.")


if __name__ == "__main__":
    main()
