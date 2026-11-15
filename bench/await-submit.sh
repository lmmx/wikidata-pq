#!/usr/bin/env bash
# Wait for the coordinator to mark a queue entry ready, run it once, then suspend the machine.
# Ported from lmmx/mibz2's bench/await-submit.sh (bench/README.md, "The workstation queue").
#
#   just submit-when-ready <entry> <branch> [--no-suspend] [--hours N] [--args A,B]
#   (or: bash bench/await-submit.sh <entry> <branch> ...)
#
# An entry is a recipe, or recipe.label when one recipe is queued more than once (bench-i9
# for two pull requests, say: bench-i9.pr230, bench-i9.pr231). The label only names the
# entry: the recipe run is the part before the first dot. --args gives the recipe's
# arguments, separated by commas (bench-i9's pull request: --args 230).
#
# The signal is a tag on origin, named by the entry. The coordinator sets it (through the Run
# signal workflow) only once the run needs no further decision:
#   ready/<entry>  points at the exact commit to run. The script requires that commit to be
#                  on <branch>, checks it out (detached), then runs the recipe once.
#   hold/<entry>   stop without running anything.
# The run is `just submit <recipe> [args]`: the recipe, then its one bundle submitted
# (.just/submit.just). A recipe that submits its own bundles (bench-i9-ladder submits each
# step's, then its own summary) is in self_submitting below and runs as `just <recipe> [args]`:
# under `just submit` it would make several bundles and the submission would refuse them.
#
# It polls every 5 minutes, for at most --hours (default 8; 0 looks once and doesn't wait).
# Whatever the outcome (run passed, run failed, hold, timeout or a refusal), it suspends
# afterwards, unless --no-suspend.
#
# Nothing is written to the home directory itself. The waiter's state goes where the XDG
# Base Directory spec puts state, under ${XDG_STATE_HOME:-~/.local/state}/wikidata-pq/workstation/:
#   logs/await-submit-<entry>-<stamp>.log   everything this waiter prints;
#   ran/<entry>-<sha>                       the once-per-commit markers.
# (mibz2's queue keeps its own under mibz2/: neither sees the other's markers.)
#
# Guarantees:
# - a run happens at most once per entry and commit on this machine: a marker directory
#   under ran/ is made atomically just before the run, so a restarted or second watcher
#   refuses;
# - HOLD is checked once more just before the run;
# - an annotated ready tag is resolved to its commit;
# - a failed check of origin is logged, and retried at the next poll;
# - the exit status is 0 only when the run ran and passed: 1 when it failed, 3 on HOLD,
#   4 on timeout, 5 on a refusal.
# Notices: when a run starts, and when one fails, a LOG is posted on the queue issue (#97);
# the run's bundle is posted by bench/submit.sh, on its pull request or on #97. A notice is
# best effort: one that can't be posted (gh signed out, no network) is logged and the run
# goes on. AWAIT_SUBMIT_NOTICES=0 turns them off.
set -uo pipefail

# Run from a copy, so that checking out another branch can't change this script mid-run.
if [[ "${AWAIT_SUBMIT_COPY:-}" != 1 ]]; then
  tmp=$(mktemp /tmp/await-submit.XXXXXX.sh)
  cp "$0" "$tmp"
  AWAIT_SUBMIT_COPY=1 exec bash "$tmp" "$@"
fi

# The recipes that submit their own bundles: run bare, never under `just submit`.
self_submitting=(bench-i9-ladder)
channel=97

usage="usage: await-submit.sh <entry> <branch> [--no-suspend] [--hours N] [--args A,B]"
entry=${1:?$usage}
branch=${2:?$usage}
shift 2
suspend=1
hours=8
args=()
while (($#)); do
  case $1 in
    --no-suspend) suspend=0 ;;
    --hours) hours=$2; shift ;;
    --args) IFS=, read -r -a args <<<"$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
[[ $entry =~ ^[a-z0-9][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)?$ ]] \
  || { echo "an entry is a recipe or recipe.label (lower case, digits, hyphens): not $entry"; exit 2; }
recipe=${entry%%.*}
bare=0
for r in "${self_submitting[@]}"; do [[ $r == "$recipe" ]] && bare=1; done
if ((bare)); then run=(just "$recipe" "${args[@]}"); else run=(just submit "$recipe" "${args[@]}"); fi

# State, per the XDG Base Directory spec (a relative XDG_STATE_HOME is ignored, as it says)
state_home=${XDG_STATE_HOME:-}
[[ $state_home == /* ]] || state_home=$HOME/.local/state
state=$state_home/wikidata-pq/workstation
mkdir -p "$state/logs" "$state/ran" || { echo "could not make $state"; exit 2; }
log=$state/logs/await-submit-${entry}-$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$log") 2>&1
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# Post a LOG on the queue issue (best effort, at most 60 s: a notice never stops or fails the run)
notice() {
  [[ ${AWAIT_SUBMIT_NOTICES:-1} == 1 ]] || return 0
  local gh=${GH:-gh} repo=${EVIDENCE_REPO:-lmmx/wikidata-pq} body url
  body=$(printf 'FROM: louis via workstation\nTO: coordinator\nRE: %s\nTYPE: LOG\n\n%s\n' \
    "$entry at ${ready:-?}" "$1")
  if url=$(timeout 60 "$gh" api "repos/$repo/issues/$channel/comments" -f "body=$body" --jq .html_url 2>/dev/null); then
    say "notice posted: $url"
  else
    say "notice not posted (gh failed or timed out); the run goes on"
  fi
  return 0
}

finish() {
  say "outcome: $1"
  if ((suspend)); then
    say "suspending in 30 s (Ctrl-C to cancel)"
    sleep 30
    systemctl suspend || say "systemctl suspend failed"
  fi
  exit "${2:-0}"
}

# Print origin's ready/hold refs, with annotated tags resolved to their commits.
# Fails, and says so, when origin can't be read.
query() {
  local out
  if ! out=$(git ls-remote --tags origin "refs/tags/ready/$entry" "refs/tags/ready/$entry^{}" \
               "refs/tags/hold/$entry" 2>&1); then
    say "could not read origin's tags (retrying at the next poll): $out"
    return 1
  fi
  printf '%s\n' "$out"
}
ready_commit() {  # the peeled commit if the tag is annotated, else the tag's own target
  local peeled plain
  peeled=$(awk -v r="refs/tags/ready/$entry^{}" '$2==r {print $1}' <<<"$1")
  plain=$(awk -v r="refs/tags/ready/$entry" '$2==r {print $1}' <<<"$1")
  echo "${peeled:-$plain}"
}

cd "$(git rev-parse --show-toplevel)" || { say "not in a git checkout"; exit 2; }
say "waiting for ready/$entry or hold/$entry on origin (every 5 min, up to ${hours} h); log $log"

deadline=$(( $(date +%s) + hours * 3600 ))
ready=""
while :; do
  if refs=$(query); then
    if grep -q "refs/tags/hold/$entry\$" <<<"$refs"; then
      finish "HOLD: the coordinator put the run on hold; nothing was run" 3
    fi
    ready=$(ready_commit "$refs")
  fi
  [[ -n "$ready" ]] && break
  (( $(date +%s) + 300 <= deadline )) || break
  sleep 300
done
[[ -z "$ready" ]] && finish "TIMEOUT: no ready/$entry after ${hours} h; nothing was run" 4
say "ready/$entry -> $ready"

if [[ -n "$(git status --porcelain)" ]]; then
  git status --short
  finish "REFUSED: the checkout has uncommitted changes; nothing was run" 5
fi
git fetch origin "$branch" || finish "REFUSED: could not fetch $branch" 5
git merge-base --is-ancestor "$ready" "origin/$branch" 2>/dev/null \
  || finish "REFUSED: the ready commit $ready is not on $branch; nothing was run" 5
git checkout --quiet --detach "$ready" || finish "REFUSED: could not check out $ready" 5
head=$(git rev-parse HEAD)
[[ "$head" == "$ready" ]] || finish "REFUSED: HEAD is $head, not the ready commit $ready; nothing was run" 5

# HOLD once more, just before the run.
if refs=$(query) && grep -q "refs/tags/hold/$entry\$" <<<"$refs"; then
  finish "HOLD: the coordinator put the run on hold before it started; nothing was run" 3
fi
# At most once per entry and commit on this machine (mkdir is atomic).
mark=$state/ran/$entry-$ready
if ! mkdir "$mark" 2>/dev/null; then
  finish "REFUSED: $entry at $ready already ran on this machine ($mark); nothing was run" 5
fi

cmd="${run[*]}"
say "running: $cmd"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if ((bare)); then posted="It submits its own bundles"; else posted="Its bundle is posted when it ends"; fi
notice "**Started:** \`$cmd\` at \`$ready\` ($branch), $started, on $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1). $posted (on its pull request, else here); a failure is posted here too."
if "${run[@]}"; then
  finish "RUN DONE: $cmd exited 0" 0
else
  rc=$?
  notice "**Failed:** \`$cmd\` at \`$ready\` exited $rc, started $started, ended $(date -u +%Y-%m-%dT%H:%M:%SZ). The waiter's log on the workstation: \`$log\`. A bundle, if the run made one, is posted separately."
  finish "RUN FAILED: $cmd exited $rc" 1
fi
