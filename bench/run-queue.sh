#!/usr/bin/env bash
# Run the workstation queue: every entry in bench/workstation-queue.txt (as on origin's master)
# whose ready/<entry> tag is set, once each, in order, through bench/await-submit.sh. Then
# suspend, unless --no-suspend. Ported from lmmx/mibz2's bench/run-queue.sh (bench/README.md,
# "The workstation queue").
#
#   just workstation-queue [--no-suspend] [--hours N] [--wait H]
#
# --hours N waits up to N hours for each entry's ready tag (default 0: run what is ready
# now, and skip the rest). An entry that already ran here, or is on HOLD, is skipped.
#
# Work added while the queue runs is run too. After each pass, the queue and the ready tags
# are read again from origin. Another pass runs every entry whose ready commit has not yet
# run on this machine, such as a rerun the coordinator queued after reading an earlier
# bundle. The queue suspends only after a pass that finds nothing new (at most 10 passes).
#
# --wait H keeps the queue open for new work: once nothing is left to run, it checks origin
# every 5 minutes and runs any entry that becomes ready, until H hours pass with nothing new
# (H may be fractional, such as 1.5). Each run restarts that clock. An empty queue waits too,
# so it can be started before an entry is admitted. An entry refused at one commit (a dirty
# checkout, say) is not retried at that commit. Without --wait, nothing waits.
#
# The run's log is logs/workstation-queue-<stamp>.log, beside each entry's own log and the
# once-per-commit markers (ran/), under ${XDG_STATE_HOME:-~/.local/state}/wikidata-pq/workstation/
# (the XDG Base Directory spec's state directory; nothing is written to the home directory
# itself). mibz2's queue keeps its own under mibz2/. Only one workstation queue runs at a time
# on a machine: stop mibz2's before starting this one, and this one before mibz2's.
set -uo pipefail

if [[ "${RUN_QUEUE_COPY:-}" != 1 ]]; then
  tmp=$(mktemp /tmp/run-queue.XXXXXX.sh); cp "$0" "$tmp"
  RUN_QUEUE_COPY=1 exec bash "$tmp" "$@"
fi

suspend=1; hours=0; wait=0
poll=${RUN_QUEUE_POLL:-300}  # seconds between checks under --wait
while (($#)); do
  case $1 in
    --no-suspend) suspend=0 ;;
    --hours) hours=$2; shift ;;
    --wait) wait=$2; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
wait_secs=$(awk -v h="$wait" 'BEGIN { if (h !~ /^[0-9]*\.?[0-9]+$/) exit 1; printf "%d", h * 3600 }') \
  || { echo "--wait takes a number of hours, such as 2 or 1.5"; exit 2; }

# State, per the XDG Base Directory spec (a relative XDG_STATE_HOME is ignored, as it says)
state_home=${XDG_STATE_HOME:-}
[[ $state_home == /* ]] || state_home=$HOME/.local/state
state=$state_home/wikidata-pq/workstation
mkdir -p "$state/logs" || { echo "could not make $state/logs"; exit 2; }
log=$state/logs/workstation-queue-$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$log") 2>&1
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

cd "$(git rev-parse --show-toplevel)" || { say "not in a git checkout"; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { git status --short; say "REFUSED: the checkout has uncommitted changes; nothing was run"; exit 5; }
git fetch --quiet origin master || { say "could not fetch master"; exit 5; }

# Each pass reads the queue and the waiter afresh from origin's master.
# An entry's line: `entry branch [args=A,B] what it measures`; each parsed entry is
# "entry branch args", args `-` when there are none.
read_queue() {
  git fetch --quiet origin master || { say "could not fetch master"; return 1; }
  queue=$(git show origin/master:bench/workstation-queue.txt) || { say "no queue on master"; return 1; }
  git show origin/master:bench/await-submit.sh > "$waiter" || return 1
  # grep exits 1 on a queue of comments only; that is an empty queue, not a failure
  entries=$( { grep -v '^[[:space:]]*#' <<<"$queue" || true; } \
    | awk 'NF>=2 {a = ($3 ~ /^args=./) ? substr($3, 6) : "-"; print $1, $2, a}')
}
# The commit ready/<entry> names, or nothing when it is unset or hold/<entry> is set.
ready_of() {
  local refs
  refs=$(git ls-remote --tags origin "refs/tags/ready/$1" "refs/tags/ready/$1^{}" \
           "refs/tags/hold/$1" 2>/dev/null) || return 0
  grep -q "refs/tags/hold/$1\$" <<<"$refs" && return 0
  awk -v p="refs/tags/ready/$1^{}" -v r="refs/tags/ready/$1" \
      '$2==p {pk=$1} $2==r {pl=$1} END {print (pk ? pk : pl)}' <<<"$refs"
}
# The entries whose ready tag is set, with no hold, at a commit not yet run on this machine
# (no marker in $state/ran, which the waiter makes just before a run), and not already tried
# at that commit by this queue run.
new_entries() {
  local entry branch args ready
  while read -r entry branch args; do
    [[ -z "$entry" ]] && continue
    ready=$(ready_of "$entry")
    [[ -n "$ready" && ! -e $state/ran/$entry-$ready ]] || continue
    [[ -n "${tried[$entry-$ready]:-}" ]] && continue
    echo "$entry $branch $args $ready"
  done <<<"$entries"
}
run_entries() {  # run each "entry branch args [ready]" line of $1 through the waiter
  local entry branch args ready rc what
  while read -r entry branch args ready; do
    [[ -z "$entry" ]] && continue
    [[ -z "$ready" ]] && ready=$(ready_of "$entry")
    [[ -n "$ready" ]] && tried[$entry-$ready]=1
    with=(); [[ $args != - ]] && with=(--args "$args")
    say "== $entry ($branch${with[*]:+, args $args})"
    # stdin from /dev/null, so that nothing the run starts can read this loop's entries
    AWAIT_SUBMIT_COPY=1 bash "$waiter" "$entry" "$branch" --no-suspend --hours "$hours" "${with[@]}" </dev/null
    rc=$?
    case $rc in
      0) what="ran: passed" ;; 1) what="ran: failed" ;; 3) what="on hold: not run" ;;
      4) what="not ready: not run" ;; 5) what="refused: not run" ;; *) what="exit $rc" ;;
    esac
    summary+=("$entry: $what")
    (( rc == 1 )) && worst=1
    (( rc == 0 || rc == 1 )) && ran=1
  done <<<"$1"
}

waiter=$(mktemp /tmp/await-submit.XXXXXX.sh)
read_queue || exit 5
if [[ -z "$entries" ]]; then
  say "the queue is empty: nothing to run"
else
  say "queue:"; sed 's/^/  /' <<<"$entries"
fi

summary=(); worst=0; declare -A tried=()
ran=0
run_entries "$entries"
# Later passes: whatever was queued or re-tagged meanwhile, until a pass finds nothing new.
pass=1
while (( ran && pass < 10 )); do
  ran=0; pass=$((pass + 1))
  read_queue || break
  more=$(new_entries)
  [[ -z "$more" ]] && { say "pass $pass: nothing new on origin"; break; }
  say "pass $pass: new on origin:"; sed 's/^/  /' <<<"$more"
  run_entries "$more"
done

# --wait: keep checking origin until H hours pass with nothing new to run.
if (( wait_secs > 0 )); then
  deadline=$(( $(date +%s) + wait_secs ))
  say "waiting up to $wait h for new work (checking origin every $(( poll / 60 )) min)"
  while (( $(date +%s) + poll <= deadline )); do
    sleep "$poll"
    read_queue || continue
    more=$(new_entries)
    [[ -z "$more" ]] && continue
    say "new on origin:"; sed 's/^/  /' <<<"$more"
    run_entries "$more"
    deadline=$(( $(date +%s) + wait_secs ))
    say "waiting up to $wait h more for new work"
  done
  say "nothing new for $wait h: done waiting"
fi

git checkout --quiet master 2>/dev/null && git merge --quiet --ff-only origin/master 2>/dev/null
say "summary:"; for s in "${summary[@]}"; do say "  $s"; done
if ((suspend)); then
  say "suspending in 30 s (Ctrl-C to cancel)"; sleep 30
  systemctl suspend || say "systemctl suspend failed"
fi
exit $worst
