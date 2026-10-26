# 2026-10-26: L1b's evidence: what `preupload_lfs_files` does, how it recovers, and what it would save

L1b of `2026-10-25-release-critical-path.md` (#167). Each sorted part would be uploaded
as soon as finalise has written and checked it (`HfApi.preupload_lfs_files`), while
finalise writes the rest. Then it would be published by the same commits as today, once
every check has passed. The published output doesn't change.

It would be new Hub API use, and the user hasn't authorised it. This entry sets out the
evidence the user asked for first:
- the API's exact semantics;
- its interruption and retry behaviour;
- what limits its bandwidth;
- where it would sit in finalise;
- the gain, re-modelled with L1a (#170) and L2 (#175) merged.

**Nothing here was run against the Hub.** The model is
`2026-10-26-l1b-preupload-evidence/model.py`. The measurement the user can run is
`2026-10-26-l1b-preupload-evidence/preupload_measure.py`. It is theirs to run, and no
agent has run it.

## Current State

- **Docs only.** Nothing in `src/`, `reshape/` or `bench/` changed. No Hub call was made.
- **The versions read:**
  - **huggingface_hub 1.32.0 and hf-xet 1.6.0**, as `uv.lock` pins them (lines 1134–1135
    and 1068–1069);
  - **huggingface_hub:** the installed files, checked against the wheel's `RECORD` hashes
    (`hf_api.py`, `_commit_api.py`, `lfs.py`, `utils/_xet.py`, `utils/_http.py`). Below,
    `hub:` is a path in that package;
  - **hf_xet:** its Rust source, from huggingface/xet-core at tag `v1.6.0` (de71453).
    Below, `xet:` is a path in that tree.
- **The findings:**
  1. **Whole, finished files only.** A preupload takes a file's size and first 512 bytes
     when its operation is built. Then it reads and hashes the whole file once. It can't
     take a file still being written. Neither this huggingface_hub nor hf_xet's Python
     binding offers a stream through `preupload_lfs_files`.
  2. **wikidata-pq goes through Xet, not LFS.** The client picks Xet whenever `hf_xet`
    is installed and `HF_HUB_DISABLE_XET` isn't set. It doesn't ask the repo. The lock
    installs `hf_xet` on x86_64 and arm64.
  3. **Nothing is visible until `create_commit`.** A preupload writes content-addressed
     objects to Xet's storage (CAS). Only the commit request changes the repo.
     - `create_commit` skips an operation's upload because of an in-memory flag
       (`_is_uploaded`) on that same Python object. It doesn't ask the server.
     - A fresh object for the same file is uploaded again, but Xet deduplicates it.
  4. **Recovery is by deduplication, not by a ledger.** A rerun re-reads and re-hashes
     each file locally, and sends little:
     - **after a completed preupload:** a new shard only (the shard is the file's
       metadata);
     - **after a kill mid-call:** at most the last ~20 s of uploads.

     Uploaded but never committed data is orphaned. Nothing in the client collects it,
     and the source doesn't say how long the Hub keeps it.
  5. **Bandwidth.** The Xet path ignores `num_threads`.
     - **Default:** 8 files are chunked at once, and up to 64 parallel uploads are
       managed by an adaptive controller (starting at 2).
     - **High-performance mode** (`HF_XET_HIGH_PERFORMANCE=1`): 100 files, and 4–124
       uploads (starting at 16).
     - **In principle,** the limits are:
       - the uplink;
       - how many parallel uploads the controller allows;
       - the client's CPU, shared with the sort;
       - a fixed cost per call.
  6. **It fits finalise with no new ledger stage.**
     - **Where:** one uploader thread, fed by `sort_buckets`' `made()` for each claims part
       and by `write_table`'s `record()` for every other key.
     - **The commits:** `commit_table` reuses each preuploaded operation. The batches,
       `_key_done` and `verify_table` don't change.
     - **The checks:** each part's own checks pass before its preupload. The claims key's
       own checks (`_check_sorted`, `_check_whole`) still gate the commit, but not the
       preupload.
  7. **The gain over L1a and L2 (modelled):**
     - **−1h31m at 7 MB/s, −1h44m at 15 MB/s and −36 min at 30 MB/s;**
     - −4 min at 60 MB/s and −1 min at 120 MB/s.

     It depends on preuploading the claims parts before the key's own checks pass.
     **Strictly after them, it is −1 to −3 min, and not worth building.**
- **The verdict:** the semantics fit L1b, and so does its integrity. Nothing new becomes
  visible before the existing checks pass. Verify stays an independent check. A kill
  leaves at most orphaned uploads, never a published file that is wrong. Three things
  are the user's: the authorisation, the key-level-checks question, and the real
  bandwidth and per-call cost (the script).

## The semantics, with citations

### 1. What `preupload_lfs_files` needs

- **A finished file.**
  - **The operation:** `CommitOperationAdd(path_or_fileobj=str)` builds its `UploadInfo`
    at construction. It takes the file's size from `getsize` and the first 512 bytes
    (`hub:lfs.py:110-115`). The sha256 is lazy: the property reads the whole file on
    first access (`hub:lfs.py:88-95`).
  - **The commit** later sends that size and the sha256 (`hub:_commit_api.py:955-964`).
  - So the operation must be built once the file is final, and the file mustn't change
    until its upload has read it.
- **Does it hash the whole file first? That depends on the path.**
  - **Xet:** no hash before the upload. hf_xet computes the sha256 while it chunks the
    file, in one read pass. The value is then written back onto the operation
    (`hub:_commit_api.py:389-393, 646-678`; `xet:xet_data/src/processing/file_cleaner.rs:241-249`).
    The sha256 goes into the shard as the file's metadata (`metadata_ext`). Skipping it
    would write none (`file_cleaner.rs:30`).
  - **LFS:** yes. Every missing sha256 is computed first, in parallel
    (`hub:_commit_api.py:413, 451-462`). The LFS batch request is keyed by it
    (`hub:lfs.py:185-197`).
- **Not a stream.** hf_xet has a streaming upload, `start_upload_stream`
  (`xet:hf_xet/src/py_upload_commit.rs:289`). huggingface_hub 1.32.0 never calls it. It
  calls only `start_upload_file` and `start_upload_bytes` (`hub:_commit_api.py:671, 673`;
  `hub:_upload_pipeline.py:461, 465`). A file object (`BufferedIOBase`) falls back to
  LFS (`hub:_commit_api.py:395-410`).
- **Xet or LFS: a client-side choice.**
  - **The rule:** `_upload_files` takes Xet when `is_xet_available()` and no operation is
    a file object (`hub:_commit_api.py:395-407`). `is_xet_available` checks only
    `HF_HUB_DISABLE_XET` and whether `hf_xet` imports (`hub:utils/_runtime.py:155-160`).
    It never consults the repo.
  - **wikidata-pq's case:** its lock installs `hf_xet` on x86_64 and arm64 (`uv.lock`
    line 1141). Nothing in the repo sets `HF_HUB_DISABLE_XET`. So every upload today
    (`commit_table`'s `create_commit`, `push_group`'s `upload_folder`) is a Xet upload.
- **What changes between the two paths:**

  | | Xet (wikidata-pq's) | LFS (`HF_HUB_DISABLE_XET=1`, or no `hf_xet`) |
  |---|---|---|
  | hash before upload | none: sha256 in the chunking pass (`_commit_api.py:646-678`) | whole-file sha256 first (`_commit_api.py:413`) |
  | unit sent | xorbs of chunks (~64 KiB target, ≤64 MiB a xorb: `xet:xet_core_structures/src/xorb_object/constants.rs:5, 17`), `POST /v1/xorbs/{hash}` (`xet:xet_client/src/cas_client/remote_client.rs:799`), then a shard `POST /v1/shards` (`remote_client.rs:325`) | the whole file: one PUT, or multipart PUTs to presigned URLs, then a completion POST (`hub:lfs.py:284-297, 371-391`) |
  | deduplication | by chunk: this session, a resumed session, the local shard cache, then the Hub's global index (`xet:xet_data/src/processing/shard_interface/native.rs:164-191, 146-161`) | by whole-file sha256: the batch API returns no actions for an object it has (`hub:_commit_api.py:517-520`; `hub:lfs.py:269-275`) |
  | parallelism | `num_threads` ignored (`_upload_xet_files` has no such parameter: `_commit_api.py:552-561`); 8 files at once, adaptive 1–64 uploads (below) | `num_threads` files at once (default 5: `hub:hf_api.py:5386`; `hf_thread_map`, `_commit_api.py:542-548`); each file's parts one after another (`hub:lfs.py:427-442`) |

- **Small, non-LFS files.**
  - **The mode is the Hub's call.** The client sends each file's path, size and 512-byte
    sample to `POST /api/{type}s/{repo}/preupload/{revision}`, and the Hub answers
    "lfs" or "regular" (`hub:_commit_api.py:747-779`). An empty file is always
    "regular" (`_commit_api.py:784-785`).
  - **`preupload_lfs_files` uploads only "lfs" files** (`hub:hf_api.py:5503`). A
    "regular" file isn't sent until the commit, base64 in its payload
    (`hub:_commit_api.py:945-953`). The commit's limit is 1 GB of that and 25k LFS files
    (`hub:hf_api.py:5122`).
  - **So L1b does nothing for a regular file.** Whether a small `.parquet` part counts as
    one depends on the repo's `.gitattributes`, which the client never sees. `_key_done`
    already handles both kinds (`compact._same`: the LFS sha256, or the git blob sha1;
    `src/wikidata/compact.py:582-587`).

### 2. Visibility, and what `create_commit` does with preuploaded operations

- **Nothing a preupload sends is a change to the repo.** Its requests are:
  - the upload-mode query, `POST .../preupload/{revision}` (`hub:_commit_api.py:763-768`);
  - a Xet write token, `GET /api/{type}s/{repo}/xet-write-token/{revision}`
    (`hub:utils/_xet.py:218-251`);
  - CAS uploads of xorbs and shards, which are content-addressed objects
    (`remote_client.rs:799, 325`).

  The only request that writes the repo's git history is `create_commit`'s
  `POST /api/{type}s/{repo}/commit/{revision}` (`hub:_commit_api.py:1056-1068`).
  `list_repo_tree`, which `_key_done` and `verify_table` read, lists the git tree at the
  revision. So preuploaded but uncommitted parts aren't in it. (The Hub's side is out of
  the source. The script's step 3 checks it.)
- **`create_commit` calls `preupload_lfs_files` itself, with `free_memory=False`**
  (`hub:hf_api.py:5271-5280`).
  - **The skip:** that call drops every operation whose `_is_uploaded` is True
    (`hub:hf_api.py:5481`). It asks the upload mode only for those with none yet
    (`5485-5499`). A list of operations all preuploaded therefore makes no upload and no
    `/preupload` request.
  - **What the commit sends:** each LFS file's path, sha256 and size (`lfsFile`,
    `hub:_commit_api.py:955-964`).
- **How it knows:** only by the flag on the Python object. `preupload_lfs_files` sets
  it after its upload call has returned without error (`hub:hf_api.py:5533-5541`).
  - **A fresh operation for the same file**, as a rerun builds, is new to the client: its
    mode is asked again and it goes through Xet again. Xet then deduplicates its chunks
    (§3).
  - **The no-op removal** (`_remote_oid == _local_oid`, `hub:hf_api.py:5296-5305`) only
    drops a file already committed at the revision, unchanged.
- **Guards:**
  - An operation can't go into two commits (`_is_committed`, `hub:hf_api.py:5236-5242`).
  - The commit POST isn't retried. `_send_commit`'s `retry_on_error` defaults to False
    (`hub:_commit_api.py:1027, 1064-1067`), because a lost response would make a
    duplicate commit.
  - So a failed commit raises, and the rerun resumes it (`_key_done`).
- **`free_memory=True`** (the default) replaces the operation's path with `b""` after
  the upload (`hub:hf_api.py:5540-5541`).
  - That is harmless for the commit, which reads only `upload_info`.
  - But it would make the operation unreadable if anything read it again, so L1b passes
    `free_memory=False`, as `create_commit` does.

### 3. Interruption, retries and orphans (the Xet path)

- **A preupload call is one Xet upload session**, which succeeds or fails as a unit:
  - the call opens `session.new_upload_commit(...)` and blocks until its `with` block
    ends (`hub:_commit_api.py:664-673`);
  - the session's own description is "a single atomic session that succeeds or fails as
    a unit" (`xet:xet_data/src/processing/file_upload_session.rs:40-45`);
  - the shard registers each file's chunks and sha256 at the end. It goes up only once
    every xorb is up (`file_upload_session.rs:369-370, 572-608`).
- **A kill mid-call leaves xorbs in CAS and no shard.**
  - **Nothing points at them.** No file is registered, no commit can name the file, and
    `_is_uploaded` was never set.
  - **They are recorded locally for resume.**
    - **What:** each xorb, once uploaded, goes to a staging shard "so that future sessions
      can resume quickly" (`file_upload_session.rs:418-419`).
    - **How often:** flushed to disk every 20 s, or every 64 xorbs (`native.rs:199-224`;
      `xet:xet_runtime/src/config/groups/data.rs:78, 86`).
    - **Where:** under `HF_XET_CACHE` (`hub:constants.py:341`):
      `<endpoint>/staging/shard-session/xorb_metadata`
      (`xet:xet_data/src/processing/configurations.rs:117-123`; `native.rs:76-78`).
  - **A later session on the same machine** merges them (`native.rs:80-96`) and
    deduplicates against them first (`native.rs:169-175`). So their xorbs aren't sent
    again.
  - **What a kill costs:** the xorbs uploaded since the last flush (≤20 s of uploads), the
    xorbs in flight (≤64 MiB each), and the shard.
- **A kill after a completed call:**
  - **On the same machine:** the session's shards were moved to the local shard cache
    with a 3-week expiry (`native.rs:310-318`;
    `xet:xet_data/src/processing/constants.rs:16`). A rerun's fresh operations
    deduplicate every chunk locally. They send a new shard only, after a local re-read
    and re-hash of the file.
  - **On another machine:** the Hub's global deduplication finds the same xorbs through
    a sample of chunks: each file's first chunk and 1 in 1,024 others
    (`xet:xet_core_structures/src/metadata_shard/constants.rs:8, 20-21`;
    `streaming_shard.rs:459-500`). Queries are on by default and spaced ≥256 chunks
    apart (`xet:xet_runtime/src/config/groups/deduplication.rs:34`; `data.rs:14`).
  - **A xorb the server already has** is acknowledged without a second insert
    (`was_inserted`, `remote_client.rs:868`).
- **The library's own word on it:** `upload_folder`'s Xet pipeline says "Interrupted
  uploads are resumable by simply re-running the same call: already-committed files are
  dropped ... and already-uploaded chunks are deduplicated by the xet storage backend,
  transferring ~0 bytes" (`hub:_upload_pipeline.py:28-30`).
  - **`upload_large_folder`** goes further. It stores `is_uploaded` in a local file and,
    on a rerun, rebuilds operations with the private `_is_uploaded`, `_upload_mode` and
    sha256 set by hand (`hub:_upload_large_folder.py:712-722, 755-767`). It commits them
    with no re-read.
  - **L1b doesn't copy that** (see the design): it relies on private fields, and it
    trusts that the Hub still holds the data.
- **Retries, per request.**
  - **What is retried:** each xorb or shard POST, up to 5 attempts with exponential
    backoff and jitter, from 3 s, each sleep capped at 6 min
    (`xet:xet_runtime/src/config/groups/client.rs:12, 20, 31`;
    `xet:xet_client/src/cas_client/retry_wrapper.rs:42-45`).
  - **What counts as transient:** 5xx, 408, 429, timeouts and connection errors
    (`retry_wrapper.rs:515, 537-541`). The timeouts are 60 s to connect and 300 s without
    data (`client.rs:53, 63`).
  - **The unit lost to a failure** is one xorb (≤64 MiB), sent again whole, never the
    file.
  - **After the last attempt** the call raises. On an exception the session is aborted
    (`xet:hf_xet/src/py_upload_commit.rs:202-209`), and a KeyboardInterrupt aborts the
    process's Xet session (`hub:_commit_api.py:679-681`).
- **Retries on the LFS path**, for completeness:
  - **The part PUTs:** each is retried by `http_backoff` (`hub:lfs.py:439`), re-seeking
    its slice. The defaults are 5 retries, 1 s to 8 s, on 408/429/5xx and on
    timeout/network/protocol errors (`hub:utils/_http.py:451-456, 485-499, 562-570`).
  - **Within a process,** a file's parts go up one after another, then the completion
    POST (`hub:lfs.py:376-391`).
  - **A multipart upload doesn't resume across processes.** The client keeps no part
    state. A rerun gets new upload actions and sends the whole file again, unless the
    completion landed: the batch API then skips the file by its sha256.
- **Orphans and retention.**
  - **The client:** nothing deletes, lists or expires uploaded but uncommitted data, on
    either path. Whether the Hub collects it, and when, isn't in any client source.
  - **For L1b, the orphans come from:**
    - a part uploaded and then not committed: a kill before the commit, rewritten on the
      rerun by a resort;
    - a failed key-level check.

    Each is bounded by the parts' bytes, and in the common case (a rerun commits the same
    bytes) it is the same data the commit then references.
  - **The risk if the Hub dropped uncommitted data** between the preupload and the commit
    (minutes to ~2 h within one finalise run) is a commit that fails. Today's committer
    then stops and the rerun uploads again. It can't publish a wrong file, as verify
    still compares each file's sha256 and size.

### 4. Concurrency and bandwidth

- **The knobs and their defaults** (hf_xet reads its environment when its session is
  created):

  | setting | default | high performance (`HF_XET_HIGH_PERFORMANCE=1` or `HF_XET_HP=1`) | source |
  |---|---|---|---|
  | files chunked at once, `HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION` | 8 | 100 | `xet:.../groups/data.rs:30`; `xet:xet_runtime/src/config/xet_config.rs:159-167` |
  | read block a file, `HF_XET_DATA_INGESTION_BLOCK_SIZE` | 8 MB | 8 MB | `data.rs:44` |
  | parallel xorb/shard uploads, `HF_XET_CLIENT_AC_{INITIAL,MIN,MAX}_UPLOAD_CONCURRENCY` | 2 initial, 1–64 | 16 initial, 4–124 | `client.rs:207, 215, 223`; `xet_config.rs:163-167` |
  | all three at one value, `HF_XET_FIXED_UPLOAD_CONCURRENCY` | unset | | `xet:xet_runtime/src/config/aliases.rs:33-35` |
  | adaptive control, `HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY` | on (target RTT 60 s, max healthy 90 s) | | `client.rs:77, 135, 143` |
  | retries, `HF_XET_CLIENT_RETRY_{MAX_ATTEMPTS,BASE_DELAY,MAX_DURATION}` | 5, 3 s, 6 min | | `client.rs:12, 20, 31` |
  | xorb compression, `HF_XET_XORB_COMPRESSION_POLICY` | auto (lz4 tried, re-tested every 32 xorbs) | | `xet:.../groups/xorb.rs:13, 22` |
  | Xet itself, `HF_HUB_DISABLE_XET` | off (Xet used) | | `hub:constants.py:342`; `hub:utils/_runtime.py:155-160` |
  | `preupload_lfs_files(num_threads=)` | 5, the LFS path only | | `hub:hf_api.py:5386`; `_commit_api.py:413, 542` |

- **The concurrency controller is shared across calls.** It is cached per endpoint for
  each Xet context (`xet:xet_client/src/cas_client/adaptive_concurrency/cache.rs:1-6, 25`).
  huggingface_hub keeps one Xet session a process (`hub:utils/_xet.py:303-313`). So
  successive preupload calls don't start again from 2 uploads.
- **What limits throughput, in principle:**
  1. **The uplink.**
  2. **Parallel uploads × each one's rate.** One TCP stream's rate is bounded by its
     window over the round trip. The controller adds streams while transfers stay
     healthy.
     - Each stream holds a serialised xorb of up to 64 MiB, its permit taken before the
       task starts (`file_upload_session.rs:391`).
     - So the memory bound is ~4 GiB at 64 streams, and ~8 GiB at 124.
  3. **The client's CPU**, all in finalise's parent process: Xet's content-defined
     chunking, BLAKE3 chunk and xorb hashes, the sha256 and the lz4 probe. The sort
     workers hold ~95% of 4 cores during finalise (#167).
     - The bytes are the same as today's commits hash. L1b moves that CPU earlier, under
       the claims sort, rather than adding any.
  4. **A call's fixed cost:** the `/preupload` request, the token, the session's
     finalise, and its shard after its last xorb. With one uploader, the link idles
     during it, unless calls batch many parts (below).
  5. **The Hub's own limits:** 429s are retried, and server throttling isn't in the
     source.
  6. **Deduplication,** which cuts bytes sent and not bytes hashed.
- **Not measured:** the agents can't measure any of these against the Hub. A client-only
  CPU ceiling could be had locally, through hf_xet's in-memory endpoint (`memory://`,
  `xet:xet_data/src/processing/configurations.rs:38-41`), with no network. It wasn't run
  here: the shared machine was busy with other agents' jobs, and the figure would be
  noise.

## Recovery and integrity assessment

| state at a kill | what is on the Hub | the rerun (L1b as designed below) | integrity |
|---|---|---|---|
| a part preuploaded, its table not `written` | CAS objects, no file | the sort resumes from `files.jsonl` and the manifest. Its parts are queued again, their chunks deduplicated (local shard cache) | nothing visible |
| mid preupload call | some xorbs, no shard | the resumed session dedups what was flushed; ≤20 s of uploads and in-flight xorbs re-sent | nothing visible |
| a part rewritten on the rerun (its listed entry no longer matched) | CAS objects of the old bytes, never committed | the new part is uploaded; the old bytes are orphaned | nothing visible |
| a key-level check fails (`_check_sorted`, `_check_whole`) | the key's preuploaded parts | finalise stops, as today | nothing visible; orphans |
| table `written`, commit not sent | CAS objects | `commit_table` builds fresh operations; `create_commit` re-hashes each part locally, dedups, and commits | as today |
| a batch committed, the next not | the first batch's files | `_key_done` skips its keys (unchanged) | as today |
| commit landed, `committed` not recorded | the files | `_key_done` finds every key; no commit | as today |
| verify | unchanged | unchanged | **verify stays independent:** the Hub's sha256 is hf_xet's, computed from the bytes it read (COMPUTE_SHA256), compared with the manifest's, computed by `_entry` at write time |

- **One new check comes free.** After each preupload, the operation's sha256 (hf_xet's)
  must equal the part's manifest sha256. Under L1b, a part changed between its check and
  its upload fails at once, not at verify.
- **What L1b mustn't do:** pass the manifest's sha256 into hf_xet.
  - **Why it's tempting:** `_sha256_arg` would then skip the hash
    (`hub:_commit_api.py:646-649`).
  - **Why not:** the Hub's sha256 would be ours rather than that of the bytes uploaded,
    and verify would compare our hash with itself.
- **The verdict:** sound. The published bytes and commits are today's.
  - **What becomes visible:** nothing, until the same `create_commit` after the same
    `written` stage.
  - **What a crash leaves:** at most orphaned CAS data, never a published file that is
    wrong.
  - **What goes unchecked:** nothing. Every check stays, verify included.
  - **The new exposure:** the Hub's retention of uncommitted data (§3), which can only
    fail a commit, and orphaned bytes on the Hub after kills and failed checks.

## Design sketch (not built)

- **One `Uploader` thread,** a daemon in finalise's process beside the `Committer`. It
  has a queue of `(table, repo_id, key, part path)`.
- **The producers,** each after the part's checks and its listing:
  - **The claims key (the bucket path):** `sort_buckets`' `made()`
    (`src/wikidata/sort_by_id.py:1201-1210`), after its `files.jsonl` append.
    - **Passed by then:** for a part of one bucket, `_check_sort` and `_check_written`
      (`_sort_bucket`, `:1021-1056`). For a part of several, `_check_glued`
      (`_pack_file`, `:1251-1261`) and its rows against its buckets'. Either way, its
      sha256 (`_entry`, `:592-593`).
    - **Not yet:** `_check_sorted` across the key's parts (`:1247`), `_check_whole`
      (`:1289`) and the key's manifest entry (`:1319`).
  - **Every other key** (sorted in memory, and claims_labels' languages): `write_table`'s
    `record()` (`:1311-1313`), after its manifest append. By then the key's
    `_check_written` (`write_parts`, `:683`) and every part's sha256 have passed.
- **The upload loop.** Take everything queued, up to 256 files (the `/preupload` batch,
  `hub:_upload_pipeline.py:66`; `_commit_api.py:52`). Then:
  - **Build each operation** as `_key_operations` does
    (`CommitOperationAdd(path_in_repo=f"{key}/{name}", path_or_fileobj=str(path))`,
    `:1355-1359`).
  - **Call** `api.preupload_lfs_files(repo_id, ops, repo_type="dataset",
    revision=HUB_REVISION, free_memory=False)`.
  - **After it returns, check each operation:** `_is_uploaded` set, its sha256 and size
    equal to the manifest's (or to `files.jsonl`'s). A mismatch raises.
  - **Register the operation** under `(table, path_in_repo)`.
  - **Batching matters.** A call each part makes the per-call cost link time
    (`o` in the model). With the link the bottleneck, a call takes every part waiting,
    so calls coalesce on their own.
- **The commits.**
  - **Before `commit_table(table)`,** the committer drains the uploader for that table:
    it waits for the call in flight and takes the parts still queued itself, so no part
    is uploaded twice at once.
  - **`_key_operations`** takes a registered operation when there is one, else builds a
    fresh one (a rerun). `create_commit` then uploads the fresh ones itself, as today.
  - **The batches are unchanged** (`COMPACT_COMMIT_MAX_ADDS`, `COMPACT_COMMIT_MAX_OPS`,
    `src/wikidata/config.py:244-245`): a key's additions and deletions stay in one
    commit. A batch of preuploaded operations sends only `POST .../commit`.
- **The ledger: no new stage.**
  - **The stages:** `sourced`, `written`, `committed`, `verified` and `done` keep their
    meaning. `_key_done` is unchanged: a preuploaded part isn't in the tree, so its key
    is still to commit.
  - **Optional:** a `preuploaded.jsonl` (part, bytes, hf_xet's sha256), for progress and
    the bench only. Nothing reads it to decide a commit.
  - **Rejected:** rebuilding `_is_uploaded` operations from a ledger, as
    `upload_large_folder` does. It relies on private fields and trusts the Hub's
    retention. And it saves only a local re-hash on a rerun.
- **Errors.** An uploader failure is printed, and the uploader stops. Its parts go to
  `commit_table`, which uploads them as today. So a preupload failure costs time, not
  correctness. A committer failure stops finalise, as today (`Committer`, `:1483-1610`).
- **Unchanged:**
  - **Disk:** parts stay until `done`, and the 64 GB bound is unchanged.
  - **The three finalise processes:** each run's join already waits for its commits.
- **Memory to watch:** the uploader's in-flight xorbs now coincide with the claims'
  bucket sorts, finalise's memory peak (~10–12 GB in three sorts, #167), not with the
  later tables.
  - `HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY=8` caps them at ~0.5 GiB.
  - Each bucket sort's memory budget is read at its start (`parallel.memory_budget`).
- **The bench:** `bench/link.py`'s stand-in `HfApi` would have to serve
  `preupload_lfs_files` and charge the link per part. Then bench 17.x's link stage
  measures L1b on CI, as it measures L1a.
- **Checks first: per part, yes; per key, no.**
  - **Per part:** every part's own checks pass before its preupload.
  - **The claims key's own checks** (`_check_sorted`, the order across parts;
    `_check_whole`, the rows against the sources) can only run after its last part. If
    they must pass before any upload, the claims parts upload at the key's end, which is
    L1a. That is the "strict" row below: −1 to −3 min.
  - **The other ~600-key tables** are per key either way.

## The gain, re-modelled

`model.py` builds on L2's model (`2026-10-26-finalise-before-uploads-end/model.py`, which
keeps bench 17.2.0's).
- **Kept as built:** L2, with the main set's turn held after the split and its first
  finalise beside the scholarly pushes, and L1a's committer with its 64 GB bound.
- **The input:** master's bench record, 26e9259, 6 runs, medians.
- **Under L1b:**
  - each table's published bytes go up part by part, on one uploader, as each part
    becomes eligible;
  - the table's commit follows once it is written and its parts are up.
- **When a part becomes eligible:**
  - **the claims key:** after the bucketing and the plan, the first `b` of its compute,
    then evenly to its end. `b` is about 0.10: bucketing is ~10 of the claims' ~89 min
    (#150, #167);
  - **every other table:** its keys evenly over its compute.
- `o` is a call's fixed cost, as link time. Every figure is modelled.

Release walls, L3 off (the default):

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| L1a + L2 (master) | 12h05m | 8h58m | 7h45m | 7h12m | 7h08m |
| **L1b: per part, b=0.10, o=0** | **10h34m** | **7h14m** | **7h08m** | 7h07m | 7h07m |
| L1b: per part, b=0 (no bucketing gap: the bound) | 10h27m | 7h14m | 7h08m | 7h07m | 7h07m |
| L1b: per part, b=0.20 | 10h41m | 7h23m | 7h08m | 7h07m | 7h07m |
| L1b: per part, b=0.30 | 10h49m | 7h35m | 7h08m | 7h07m | 7h07m |
| L1b: per part, b=0.10, o=5 s a call | 10h36m | 7h19m | 7h09m | 7h08m | 7h07m |
| L1b: per part, b=0.10, o=30 s a call | 10h46m | 7h34m | 7h14m | 7h11m | 7h10m |
| L1b: o=5 s, a part a call (no coalescing) | 11h09m | 7h48m | 7h16m | 7h13m | 7h13m |
| L1b: strict (claims parts at the key's end) | 12h02m | 8h55m | 7h43m | 7h09m | 7h07m |

Savings against L1a + L2:

| | L3 | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|---|
| **L1b: per part, b=0.10, o=0** | **off** | **−91 min** | **−104 min** | **−36 min** | −4 min | −1 min |
| L1b: per part, b=0.10, o=0 | on | −83 min | −106 min | −43 min | −8 min | −1 min |
| L1b: b=0 (the bound) | off | −98 min | −104 min | −36 min | −4 min | −1 min |
| L1b: b=0.30 | off | −76 min | −83 min | −36 min | −4 min | −1 min |
| L1b: o=5 s a call | off | −89 min | −98 min | −36 min | −4 min | −1 min |
| L1b: o=30 s a call | off | −79 min | −83 min | −31 min | −1 min | +2 min |
| L1b: o=5 s, a part a call | off | −56 min | −70 min | −28 min | +2 min | +4 min |
| L1b: strict | off | −3 min | −3 min | −2 min | −2 min | −1 min |

With L3 on, the walls are 11h44m (master) and 10h21m (L1b) at 7 MB/s.

- **Against #167's L1b row:** #167 modelled about −2h over L1a at 7 MB/s. Since then,
  L2 has taken ~24 min of the same idle time, and the model now has the bucketing gap.
  So it is −1h31m.
- **At 7 MB/s L1b reaches the link's floor.** The wall is 10h34m, against #167's
  10h09m of link alone. Only the link's idle time during finalise is left to remove.
- **The largest saving is at 15 MB/s, −1h44m.** There the claims' upload (~46 min a
  set) fits under the claims sort's own compute. Under L1a the CPU idles for most of it,
  in each run.
- **At ≥60 MB/s L1a already hides nearly all of it,** since the claims commit in ~12
  min, under the later tables. L1b is then worth minutes, and a per-call cost over ~5 s
  makes it a loss.
- **The sensitivities that matter:**
  - **strict checks:** they take the saving away;
  - **a part a call:** the per-call cost becomes link time. So the uploader must batch
    (the design's coalescing);
  - **the bucketing gap:** from −98 to −76 min across 0 to 0.30 at 7 MB/s.
- **Not modelled:**
  - the commit's own latency, which is the same under L1a and L1b;
  - the Xet CPU, the same bytes either way, but moved under the claims sort;
  - deduplication between releases, which would cut both L1a's and L1b's link time.

## What still needs the user

1. **The authorisation.** L1b calls `HfApi.preupload_lfs_files` directly, from finalise,
   ahead of `create_commit`.
   - **The HTTP requests are those `create_commit` already makes**
     (`hub:hf_api.py:5271`), and those `upload_folder` already makes ahead of its
     commits in every `push_group` (`hub:hf_api.py:6072-6088`, `_upload_pipeline.py`).
     L1b only moves them earlier, into a thread of their own.
   - **`hub.py` would not change.** The uploader would live in `sort_by_id.py`.
2. **The checks question.** Is a part preuploaded once its own checks pass, before the
   key's `_check_sorted` and `_check_whole`? Nothing becomes visible before those, and
   the commit still waits for them. If not, L1b isn't worth building (the strict row).
3. **A real measurement**, which only the user can run: `preupload_measure.py`, below.
4. **A question for Hugging Face:** how long uncommitted Xet data is kept, and whether
   it counts against the account's storage.

## The measurement (the user's to run; never run by an agent)

`2026-10-26-l1b-preupload-evidence/preupload_measure.py` uses a private scratch dataset
repo of the user's. It:
1. writes N files of fresh random bytes (incompressible, as zstd Parquet is, and new
   each run, so nothing deduplicates);
2. preuploads them `--batch` files a call, each call timed (its wall, MB/s and bytes
   sent);
3. preuploads `--small` 1 MB files one a call, for the per-call cost (`o`);
4. lists the revision's files after the preuploads, which should show none of them;
5. preuploads the same files again from fresh operations, as a rerun would (the bytes
   sent should be small);
6. with `--kill-after S`, SIGKILLs a child mid-call and preuploads its files again,
   showing what a kill costs;
7. commits everything in one `create_commit`, timed, and compares each file's sha256
   and size on the Hub with the local ones, as verify does.

It writes everything to a JSON file. Run it with the defaults, then with
`HF_XET_HIGH_PERFORMANCE=1`, and once with `HF_HUB_DISABLE_XET=1` for the LFS path. It
deletes nothing on the Hub; the user deletes the scratch repo afterwards.

## Open questions

- **The Hub's real upload rate** at a release, and whether `HF_XET_HIGH_PERFORMANCE`
  raises it. That decides between −4 min and −1h44m.
- **A call's fixed cost (`o`)** and the session's finalise tail. These decide how the
  uploader batches.
- **The Hub's retention of uncommitted Xet data,** and whether it counts as storage.
- **Small parts' upload mode** on the dataset repos (the `.gitattributes`). A "regular"
  part gains nothing.
- **The Xet client's CPU a byte,** beside the sort on 4 cores. It is the same bytes as
  today's commits hash, moved under the claims sort. The `memory://` endpoint can
  measure it locally, without the Hub, on a quiet machine.
- **The uploader's memory beside the bucket sorts,** bounded by the upload concurrency.
  The script's peak RSS is one data point.

## Sources

- **huggingface_hub 1.32.0**, installed and checked against its `RECORD`:
  - `hf_api.py`: `create_commit` 5092–5374, `preupload_lfs_files` 5377–5541,
    `upload_folder`'s Xet branch 6072–6088;
  - `_commit_api.py`, `lfs.py`, `_upload_pipeline.py`, `_upload_large_folder.py`;
  - `utils/_xet.py`, `utils/_http.py`, `utils/_runtime.py`, `constants.py`.
- **hf_xet 1.6.0's source,** huggingface/xet-core `v1.6.0` (de71453):
  - `hf_xet/src/py_upload_commit.rs`;
  - `xet_pkg/src/xet_session/upload_commit.rs`;
  - `xet_data/src/processing/{file_upload_session.rs, file_cleaner.rs,
    configurations.rs, constants.rs, shard_interface/native.rs}`;
  - `xet_client/src/cas_client/{remote_client.rs, retry_wrapper.rs,
    adaptive_concurrency/cache.rs}`;
  - `xet_runtime/src/config/{xet_config.rs, aliases.rs, groups/*.rs}`;
  - `xet_core_structures/src/{xorb_object/constants.rs, metadata_shard/constants.rs,
    metadata_shard/streaming_shard.rs}`.
- **wikidata-pq at ea91849:** `src/wikidata/sort_by_id.py`, `compact.py`, `config.py`,
  `push/core.py`, `uv.lock`.
- **The models:**
  - #167's `critical_path.py`;
  - #170's `model.py`;
  - #175's `model.py`, run on the bench-results branch's
    `runs/quick/26e9259e13911801ec8984be45f5b81b281b74bb.json`. It reproduces #175's
    tables.
