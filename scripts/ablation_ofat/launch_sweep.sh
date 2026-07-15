#!/usr/bin/env bash
# Launch the serial OFAT ablation sweep in a detached tmux session so it survives
# the shell. Idempotent: if the session is already running, it just reports.
#
#   bash scripts/ablation_ofat/launch_sweep.sh          # start the sweep
#   tmux attach -t ofat_ablation_sweep                  # watch it
#   tail -f artifacts/training_runs/ablation_ofat/sweep.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SESSION_NAME="${SESSION_NAME:-ofat_ablation_sweep}"
SWEEP_LOG="artifacts/training_runs/ablation_ofat/sweep.log"
mkdir -p "$(dirname "$SWEEP_LOG")"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for this long-running sweep" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "sweep already running in tmux session '$SESSION_NAME'"
  echo "attach:  tmux attach -t $SESSION_NAME"
  echo "log:     $SWEEP_LOG"
  exit 0
fi

# Explicitly forward configuration env vars into the detached session so RUN_ONLY
# (per-box subset) and the conda/python overrides reliably reach run_sweep.sh
# regardless of tmux environment-inheritance quirks.
ENV_PREFIX=""
for v in RUN_ONLY CONDA_SH CONDA_ENV PYTHON MANIFEST NUM_EPOCHS MAX_ATTEMPTS DEVICE SWEEP_LOG; do
  if [[ -n "${!v:-}" ]]; then ENV_PREFIX+="$v=$(printf '%q' "${!v}") "; fi
done

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$REPO_ROOT' && ${ENV_PREFIX}bash scripts/ablation_ofat/run_sweep.sh; echo '--- sweep runner exited, leaving shell open ---'; exec bash"
echo "started OFAT ablation sweep in tmux session '$SESSION_NAME'"
echo "attach:  tmux attach -t $SESSION_NAME"
echo "log:     $SWEEP_LOG"
