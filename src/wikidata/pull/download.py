from __future__ import annotations

import time
from pathlib import Path

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, IncompleteSnapshotError

# Transient Hub/CDN failures (gateway timeouts, dropped connections) worth retrying;
# anything else (bad repo id, missing file, auth) should fail immediately.
# snapshot_download wraps a transient HfHubHTTPError/httpx error into
# IncompleteSnapshotError when it leaves a partial snapshot, so that's what's actually
# raised on a 504 (confirmed against the huggingface_hub source in .venv).
TRANSIENT_ERRORS = (
    IncompleteSnapshotError,
    HfHubHTTPError,
    httpx.TransportError,
    httpx.TimeoutException,
)
# Unattended overnight runs should ride out hours-long Hub outages rather than die.
RETRY_DELAYS_S = (30, 60, 120, 300, 600, 900) + (1800,) * 5  # ~3h total


def download_files(
    repo_id: str, root_data_dir: Path, allow_patterns: list[str], chunk_idx: int
) -> None:
    for attempt, delay in enumerate((*RETRY_DELAYS_S, None), start=1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(root_data_dir),
                allow_patterns=allow_patterns,
            )
            return
        except TRANSIENT_ERRORS as e:
            if delay is None:
                # Keep state at PULL (reflects 'in progress'); caller can re-run safely.
                raise RuntimeError(
                    f"[pull] Download failed for chunk {chunk_idx} after "
                    f"{attempt} attempts: {e!r}"
                ) from e
            print(
                f"[pull] Chunk {chunk_idx}: transient error on attempt {attempt} "
                f"({e!r}), retrying in {delay}s…"
            )
            time.sleep(delay)
        except Exception as e:
            # Keep state at PULL (reflects 'in progress'); caller can re-run safely.
            raise RuntimeError(
                f"[pull] Download failed for chunk {chunk_idx}: {e!r}"
            ) from e
