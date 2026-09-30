#!/usr/bin/env bash
# Data-mix ablation at the OFAT Stage-A depth winner (L8_H8_E256_ctx128).
#
# Trains TWO specialists at the SAME L8 architecture as the mix25 generalist,
# changing ONLY the training data mix:
#   1) L8_H8_E256_ctx128_mix00  -> flat (rigid) only  (0% CRM)
#   2) L8_H8_E256_ctx128_mix100 -> CRM (deformable) only (0% flat)
# Everything else (one-hot conditioning over [flat,crm], flat normalization,
# equal-domain-combined-std Huber, rollout_sel selection, 80x2000 schedule,
# seed 2026061801) is identical to L8_H8_E256_ctx128, so the data mix is the
# only variable. These are intentionally NOT in configs/ablation_ofat/manifest.json
# because a specialist's rollout_sel (single-domain) is not comparable to the
# generalist Stage-A ranking (dual-domain).
#
# Runs on NEWTON (idle 4090, anaconda nedm). Serial (mmap-stream, load into
# memory disabled per the sweep memory rule). Idempotent + crash-resilient:
# complete runs are skipped, incomplete runs resume from last.pt, up to
# MAX_ATTEMPTS tries each (newton-instability retry pattern).
#
#   ssh into newton is not needed (this IS newton). Launch detached:
#   tmux new-session -d -s l8_datamix_ablation \
#     'cd ~/NeDM && bash scripts/ablations/run_l8_datamix_ablation.sh; exec bash'
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/harry/anaconda3/envs/nedm/bin/python}"
CONDA_SH="${CONDA_SH:-/home/harry/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-nedm}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
DEVICE="${DEVICE:-cuda}"
SWEEP_LOG="${SWEEP_LOG:-artifacts/training_runs/ablation_ofat/l8_datamix.log}"
mkdir -p "$(dirname "$SWEEP_LOG")"

export CONDA_NO_PLUGINS=true
# shellcheck disable=SC1091
if [[ -f "$CONDA_SH" ]]; then source "$CONDA_SH" && conda activate "$CONDA_ENV"; fi

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$SWEEP_LOG"; }

is_complete() {  # $1=run_dir : 0 if metrics.jsonl reached NUM_EPOCHS
  "$PYTHON" - "$1/metrics.jsonl" "$NUM_EPOCHS" <<'PY'
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

# name|config|run_dir
ENTRIES=(
  "L8_H8_E256_ctx128_mix00|configs/ablation_ofat/L8_H8_E256_ctx128_mix00.json|artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128_mix00"
  "L8_H8_E256_ctx128_mix100|configs/ablation_ofat/L8_H8_E256_ctx128_mix100.json|artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128_mix100"
)

log "=== L8 data-mix ablation start: ${#ENTRIES[@]} runs, ${NUM_EPOCHS} epochs each, device=$DEVICE ==="

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

log "=== L8 data-mix ablation end: completed=$completed skipped=$skipped failed=$failed ==="
[[ "$failed" -eq 0 ]]
