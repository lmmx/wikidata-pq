#!/usr/bin/env bash
# The mibz2 version ladder on the workstation (`just bench-i9-ladder [STEP...]`; bench/README.md,
# "The workstation bench"): bench/i9.sh once for each step of bench/i9-ladder.tsv, in order,
# each bundle submitted as soon as it is made (bench/submit.sh: archived on the `evidence`
# branch, posted on the step's pull request, else on the queue issue). A step that fails,
# is refused or isn't submitted is recorded and the next step runs. At the end, the table
# of outcomes is printed and this attempt's own bundle submitted on the queue issue.
#   bash bench/i9-ladder.sh               # every step
#   bash bench/i9-ladder.sh 0.2.5 0.2.8   # only these steps (by name), in the file's order
#
# Each step's after arm is named (bench/i9.sh's AFTER), never checked out here, and every
# step is measured by this checkout's harness. Nothing about the decoder is set: each arm
# runs at the pipeline's defaults (the native decoder's cap of 4 threads).
#
# One new directory per attempt, target/bench-i9-ladder/<UTC stamp> (console.log, steps.tsv,
# summary.md, outcome.txt, meta.txt, and each step's log in steps/), bundled on exit.
set -uo pipefail

root=$(git rev-parse --show-toplevel) || exit 1
cd "$root"
steps_file=${LADDER_STEPS:-$root/bench/i9-ladder.tsv}  # (the three overridable only for
i9=${I9:-$root/bench/i9.sh}                            #  the tests' stand-ins)
submit=${SUBMIT:-$root/bench/submit.sh}
gh=${GH:-gh}
die() { echo "bench-i9-ladder: $*" >&2; exit 1; }

# --- the steps, and the ones asked for
names=() prs=() befores=() afters=() vbs=() vas=()
while read -r name pr before after vb va rest; do
  [[ -z $name || $name = \#* ]] && continue
  [[ -n $va && -z $rest ]] || die "$steps_file: a step is: name pr before after mibz2-before mibz2-after ($name)"
  names+=("$name") prs+=("$pr") befores+=("$before") afters+=("$after") vbs+=("$vb") vas+=("$va")
done < "$steps_file"
[ ${#names[@]} -gt 0 ] || die "no steps in $steps_file"
for want in "$@"; do
  printf '%s\n' "${names[@]}" | grep -qxF -- "$want" || die "no step $want (the steps: ${names[*]})"
done
wanted() { local n=$1; shift; [ $# -eq 0 ] && return 0; printf '%s\n' "$@" | grep -qxF -- "$n"; }

# --- refusals, before anything is run: every step would be refused or unsubmittable
"$gh" auth status >/dev/null 2>&1 || die "gh isn't signed in; run gh auth login first (nothing was run)"
git diff --quiet HEAD -- || die "tracked files have uncommitted changes: commit or stash them (the harness is HEAD's)"
git fetch -q origin || die "couldn't fetch origin"

mibz2() {  # the mibz2 version a commit's reshape/Cargo.lock locks, if any
  git show "$1:reshape/Cargo.lock" 2>/dev/null | awk '$0 == "name = \"mibz2\"" {getline; gsub(/version = |"/, ""); print; exit}'
}

# --- this attempt's directory, new every time, and its console log
top="$root/target/bench-i9-ladder"
mkdir -p "$top"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
DIR="$top/$stamp"
mkdir "$DIR" || exit 1
mkdir "$DIR/steps"
exec 3>&1 4>&2
exec > >(tee -a "$DIR/console.log") 2>&1
tee_pid=$!

total=0
for i in "${!names[@]}"; do wanted "${names[$i]}" "$@" && total=$((total + 1)); done
echo "bench-i9-ladder: $total of ${#names[@]} steps, harness $(git rev-parse HEAD), in $DIR"
printf 'step\tmibz2\tbefore\tafter\toutcome\tsubmitted\n' > "$DIR/steps.tsv"
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$DIR/steps.tsv"; echo "bench-i9-ladder: step $1: $5; $6"; }

k=0 last=""
for i in "${!names[@]}"; do
  name=${names[$i]} pr=${prs[$i]} before=${befores[$i]} after=${afters[$i]}
  wanted "$name" "$@" || continue
  k=$((k + 1))
  versions="${vbs[$i]} → ${vas[$i]}"
  echo
  echo "bench-i9-ladder: step $k of $total, $name: mibz2 $versions, before $before, after $after$([ "$pr" = - ] || echo ", pull request #$pr")"
  # (the step's commits are the ones the file says they are)
  vb=$(mibz2 "$before") va=$(mibz2 "$after")
  if [ "$vb $va" != "${vbs[$i]} ${vas[$i]}" ]; then
    record "$name" "$versions" "$before" "$after" "REFUSED: the commits lock mibz2 ${vb:-?} → ${va:-?}" "-"
    continue
  fi
  # (i9.sh names its attempt by the second: never two in one)
  while [ "$(date -u +%Y%m%dT%H%M%SZ)" = "$last" ]; do sleep 0.2; done
  made_before=$(ls "$root"/target/bench-i9/*.tar.gz 2>/dev/null | sort)
  I9_NOTE="the mibz2 version ladder (bench/i9-ladder.sh), attempt $stamp, step $k of $total ($name)" \
    bash "$i9" "$pr" "$before" "$after" 2>&1 | tee "$DIR/steps/$name.log"
  status=${PIPESTATUS[0]}
  last=$(date -u +%Y%m%dT%H%M%SZ)
  made=$(comm -13 <(echo "$made_before") <(ls "$root"/target/bench-i9/*.tar.gz 2>/dev/null | sort) | grep . || true)
  if [ "$(echo "$made" | grep -c .)" != 1 ]; then
    why=$(grep -E '^bench-i9: ' "$DIR/steps/$name.log" | tail -1)
    record "$name" "$versions" "$before" "$after" "REFUSED (exit $status, $(echo "$made" | grep -c .) bundles): ${why:-no message}" "-"
    continue
  fi
  outcome=$(tar -xzOf "$made" "$(basename "$made" .tar.gz)/outcome.txt" 2>/dev/null || true)
  outcome=${outcome:-"(no outcome.txt; exit $status)"}
  if bash "$submit" "$made" 2>&1 | tee -a "$DIR/steps/$name.log"; [ "${PIPESTATUS[0]}" = 0 ]; then
    url=$(grep -E '^submit: (posted|already posted:) ' "$DIR/steps/$name.log" | tail -1 | grep -oE 'https://[^ ]+')
    submitted="yes${url:+: $url}"
  else
    submitted="NO: retry with just submit-bundle $made"
  fi
  record "$name" "$versions" "$before" "$after" "$outcome" "$submitted"
done

# --- the table, and this attempt's outcome
bad=$(awk -F'\t' 'NR > 1 && ($5 != "COMPLETE" || $6 !~ /^yes/) {print $1}' "$DIR/steps.tsv" | tr '\n' ' ')
{
  echo "**The mibz2 version ladder**, attempt \`$stamp\`: $total steps, bench-i9 each"
  echo
  echo "| Step | mibz2 | Before | After | Outcome | Submitted |"
  echo "|---|---|---|---|---|---|"
  awk -F'\t' 'NR > 1 {printf "| %s | %s | `%s` | `%s` | %s | %s |\n", $1, $2, substr($3, 1, 8), substr($4, 1, 8), $5, $6}' "$DIR/steps.tsv"
} > "$DIR/summary.md"
echo
cat "$DIR/summary.md"
if [ -n "$bad" ]; then outcome="INCOMPLETE: ${bad% }"; else outcome=COMPLETE; fi
echo "$outcome" > "$DIR/outcome.txt"
{
  echo "harness $(git rev-parse HEAD)"
  echo "note the mibz2 version ladder (bench/i9-ladder.sh): $total steps; each step's own run is posted on its pull request or here"
} > "$DIR/meta.txt"
echo "bench-i9-ladder: $outcome"

# --- this attempt's bundle, submitted on the queue issue (its meta.txt names no pull request)
exec 1>&3 2>&4
wait "$tee_pid" 2>/dev/null
bash "$root/bench/bundle.sh" "$DIR"
bash "$submit" "$DIR.tar.gz" || echo "bench-i9-ladder: the summary wasn't submitted; retry with just submit-bundle $DIR.tar.gz"
[ "$outcome" = COMPLETE ]
