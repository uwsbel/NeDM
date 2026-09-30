#!/usr/bin/env bash
# Serial OFAT architecture-ablation sweep runner.
#
# Trains every trainable entry in the manifest ONE AT A TIME on the single 4090
# (dataset load is ~21 GB into RAM per run and is released between runs, so runs
# must be serial). Idempotent + crash-resilient:
#   - already-complete runs (metrics.jsonl reached NUM_EPOCHS) are skipped;
#   - an incomplete run with a last.pt is resumed, not restarted;
#   - each run gets up to MAX_ATTEMPTS tries (the box has a history of random
#     native SIGABRT/SIGSEGV — see the newton-instability memory — so retry-until-
#     done is the documented pattern, and resume + skip make re-runs cheap).
# A single run failure does NOT abort the sweep (no `set -e`); it is logged and
# the next config proceeds. Free RAM is logged before each run so a memory
# regression is visible in the trace.
#
# Re-run this script any time to pick up whatever is still incomplete.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:-configs/ablation_ofat/manifest.json}"
# Portable across boxes: newton uses ~/anaconda3, luffy uses ~/miniconda3. Override
# PYTHON + CONDA_SH per host. RUN_ONLY restricts this box to a subset of the sweep
# (space/comma-separated spec names) so two machines can split the queue disjointly.
PYTHON="${PYTHON:-/home/harry/anaconda3/envs/nedm/bin/python}"
CONDA_SH="${CONDA_SH:-/home/harry/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-nedm}"
RUN_ONLY="${RUN_ONLY:-}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
DEVICE="${DEVICE:-cuda}"
SWEEP_LOG="${SWEEP_LOG:-artifacts/training_runs/ablation_ofat/sweep.log}"
mkdir -p "$(dirname "$SWEEP_LOG")"

export CONDA_NO_PLUGINS=true
# shellcheck disable=SC1091
if [[ -f "$CONDA_SH" ]]; then source "$CONDA_SH" && conda activate "$CONDA_ENV"; fi

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$SWEEP_LOG"; }

# Return 0 if run_dir's metrics.jsonl has reached NUM_EPOCHS.
is_complete() {
  local run_dir="$1"
  "$PYTHON" - "$run_dir/metrics.jsonl" "$NUM_EPOCHS" <<'PY'
import json, sys
from pathlib import Path
p, target = Path(sys.argv[1]), int(sys.argv[2])
if not p.exists():
    sys.exit(1)
last = None
for line in p.read_text().splitlines():
    line = line.strip()
    if line:
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            pass
sys.exit(0 if last and int(last.get("epoch", 0)) >= target else 1)
PY
}

# Emit "name|config|run_dir" for each trainable manifest entry (train==true),
# optionally filtered to RUN_ONLY (space/comma-separated spec names for this box).
mapfile -t ENTRIES < <(RUN_ONLY="$RUN_ONLY" "$PYTHON" - "$MANIFEST" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
only = os.environ.get("RUN_ONLY", "").replace(",", " ").split()
for e in m["entries"]:
    if not e.get("train"):
        continue
    if only and e["name"] not in only and e["spec"] not in only:
        continue
    print(f"{e['name']}|{e['config']}|{e['run_dir']}")
PY
)

log "=== OFAT ablation sweep start: ${#ENTRIES[@]} trainable configs, ${NUM_EPOCHS} epochs each ==="
log "manifest=$MANIFEST device=$DEVICE max_attempts=$MAX_ATTEMPTS run_only='${RUN_ONLY:-<all>}'"

completed=0; skipped=0; failed=0
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r name config run_dir <<<"$entry"
  mkdir -p "$run_dir/logs"

  if is_complete "$run_dir"; then
    log "SKIP  $name (already reached epoch $NUM_EPOCHS)"
    skipped=$((skipped + 1)); continue
  fi

  ok=0
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    log "RUN   $name attempt $attempt/$MAX_ATTEMPTS  (free RAM: $(free -g | awk '/Mem:/{print $7"G avail"}'))"
    args=(scripts/training/train_hmmwv_dynamics.py --config "$config" --device "$DEVICE" --output-dir "$run_dir")
    if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
      log "      resuming from $run_dir/checkpoints/last.pt"
      args+=(--resume-from-checkpoint "$run_dir/checkpoints/last.pt")
    fi
    start=$(date +%s)
    "$PYTHON" "${args[@]}" >>"$run_dir/logs/run.log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start ))
    if is_complete "$run_dir"; then
      log "DONE  $name in ${dur}s (attempt $attempt, exit $rc)"
      ok=1; break
    fi
    log "FAIL  $name attempt $attempt exit=$rc after ${dur}s (see $run_dir/logs/run.log); will resume"
    sleep 10
  done
  if [[ "$ok" -eq 1 ]]; then completed=$((completed + 1)); else failed=$((failed + 1)); fi
done

log "=== sweep end: completed=$completed skipped=$skipped failed=$failed ==="
[[ "$failed" -eq 0 ]]
