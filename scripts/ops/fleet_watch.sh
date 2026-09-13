#!/bin/bash
# Long-horizon watcher. Wakes on a TRANSITION against a baseline captured at arm time,
# never on a condition that is already true -- that mistake fired a tick in a loop twice.
set -u
CEIL=${1:-2400}; STEP=180; n=$((CEIL/STEP))
# d33 joined the rotation 2026-09-10: Radeon RX 9070 XT on ROCm/HIP gfx1201,
# validated against the CUDA fleet (mean mae_vx 0.161272 vs a CUDA centroid of
# 0.161307 +/- 0.001678) and the FASTEST box we have at ~70 min per policy.
H_ALL="sbel a3 sliger"
cnt () { ssh -o ConnectTimeout=10 kyle@${1}-ubuntu "ls /home/kyle/sbel-artifacts/finetune_crm_*/policy_ts.pt 2>/dev/null | wc -l; ls /home/kyle/sbel-artifacts/crmtrack_*.json 2>/dev/null | wc -l" 2>/dev/null | tr '\n' ' '
}
BASE_FT=0; BASE_SC=0
for H in $H_ALL; do set -- $(cnt $H); BASE_FT=$((BASE_FT+${1:-0})); BASE_SC=$((BASE_SC+${2:-0})); done
echo "ARM baseline: ft=$BASE_FT sc=$BASE_SC"
for i in $(seq 1 $n); do
  UNREACH=""; LINE=""; TOT_FT=0; TOT_SC=0; LIVE=0
  for H in $H_ALL; do
    if ! ssh -o ConnectTimeout=8 -o BatchMode=yes kyle@${H}-ubuntu true 2>/dev/null; then UNREACH="$UNREACH $H"; continue; fi
    set -- $(cnt $H); f=${1:-0}; s=${2:-0}
    TOT_FT=$((TOT_FT+f)); TOT_SC=$((TOT_SC+s)); LINE="$LINE  $H[ft=$f sc=$s]"
    # EVERY session name used by a dispatcher must appear here. This list is the
    # watcher's only notion of 'busy', so a name missing from it reads as idle and
    # fires a false wake. It has now happened twice: first chain*, then ablate/traj/
    # bias, which reported the whole fleet idle while two boxes were training
    # surrogate ablations and a third was scoring a checkpoint trajectory.
    # chain* must count as live. It was missing, so once the crmscore sessions ended and
    # the queued work moved into chain/chain2/chain3, this reported the whole fleet idle
    # while all four boxes were scoring. The comment also has to live OUT HERE: putting it
    # inside the single-quoted remote command swallowed `done; exit 1` onto a comment line,
    # which malformed the remote loop so it ALWAYS failed and LIVE stayed 0 regardless.
    # LIVE MEANS A RUNNING JOB, NOT A KNOWN SESSION NAME.
    #
    # This was an allowlist of tmux session names and it went stale three times: first
    # chain*, then ablate/traj/bias, then lco. Each time the watcher declared the fleet
    # idle while boxes were training, because the list is written once and the session
    # names keep being invented. The processes are the thing that actually means work is
    # happening, and they are a closed set -- the trainer, the fine-tuner, the scorer and
    # the collector -- that has not changed all project.
    ssh -o ConnectTimeout=10 kyle@${H}-ubuntu 'ps -eo args --no-headers | grep -qE "[t]rain_hmmwv_dynamics|[f]inetune_go2_shortbranch|[s]core_crm_tracking|[c]ollect_go2_smoke|[d]rive_go2_collection"' 2>/dev/null && LIVE=$((LIVE+1))
  done
  # A SINGLE failed connect is not a dead box. sbel dropped one probe at 22:33 with a
  # load average of 2.04, six live tmux sessions and both its jobs running, and that
  # woke the watcher for nothing. Require two CONSECUTIVE misses before believing it.
  if [ -n "$UNREACH" ]; then
    if [ "$UNREACH" = "${PREV_UNREACH:-}" ]; then
      echo "[$(date +%H:%M)] UNREACHABLE twice in a row:$UNREACH"
      echo "WAKE: fleet unreachable"; exit 0
    fi
    echo "[$(date +%H:%M)] transient unreachable:$UNREACH (will re-probe)"
  fi
  PREV_UNREACH="$UNREACH"
  echo "[$(date +%H:%M)]$LINE   ft=$TOT_FT(+$((TOT_FT-BASE_FT))) sc=$TOT_SC(+$((TOT_SC-BASE_SC))) live=$LIVE"
  [ "$TOT_SC" -gt "$BASE_SC" ] && { echo "WAKE: new tracking scores (sc $BASE_SC -> $TOT_SC)"; exit 0; }
  # Desktop idleness is not fleet idleness. Work migrated to the cluster during the
  # MuJoCo and holdout phases, and this fired three times while six scoring jobs were
  # running. Count SLURM jobs as live work before declaring the fleet idle.
  CLU=$(ssh -o ConnectTimeout=10 -o BatchMode=yes hpcfund 'squeue -u $USER -h | wc -l' 2>/dev/null || echo 0)
  [ "$LIVE" = "0" ] && [ "${CLU:-0}" -eq 0 ] && { echo "WAKE: nothing running on desktops or cluster"; exit 0; }
  [ "$LIVE" = "0" ] && echo "           (desktops idle, but $CLU cluster jobs running)"
  sleep $STEP
done
echo "WAKE: tick ceiling"
