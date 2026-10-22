#!/bin/bash
# Real2sim student: SMOKE test of the three stages on one MI350X (<= 15 min): (a) round-based start with 64 + 32 tasks
# and 60 / 20 training steps, (b) 24 decisions of the online loop with 64 worlds, (c) a CPU MuJoCo evaluation of 8 HOME
# development tasks with randomised visuals. Episodes cut to 3 s (--horizon 150) and 4 tasks per in-run evaluation so
# that the untrained students' time-outs fit. Checks that the code, data and environment work; no result is meaningful.
# R2S_SCENE: the scene file (default the final one; the smoke of 2026-10-21 used real_scene_placeholder_tonemap.json).
#
#   sbatch scripts/so101_policy/cluster/r2s_smoke.sbatch
#SBATCH -J r2s-smoke
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o /work1/dannegrut/harry/so101_r2s_claude/logs/%x_%j.out
set -uo pipefail
set +u; source /work1/dannegrut/harry/so101_r2s_claude/repo/scripts/so101_policy/cluster/r2s_env.sh; set -u
cd "$REPO"
S=$R2S/runs/smoke_${SLURM_JOB_ID:-$$}
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) gpu $(rocminfo | grep -m1 -o 'gfx9[0-9a-z]*') out $S"
COMMON=(--jt "$R2S_JT" --nrd-ckpt "$R2S_NRD" --mj-python "$R2S_PY" --horizon 150)
CFG=(--config configs/so101_policy/vision/viral_r2s_v1.json --set "scene=\"$R2S_SCENE\"" "calib=\"$R2S_CALIB\"")
t0=$(date +%s)
python3.12 -P scripts/so101_policy/vision_viral.py run --out "$S/start" "${COMMON[@]}" --tasks "$R2S_TRAIN" \
    --eval-tasks $R2S_DEV --eval-n 4 --guard-tasks $R2S_FRESH "${CFG[@]}" --r0-tasks 64 --rounds 1 --round-tasks 32 \
    --train-steps0 60 --train-steps 20 --batch 64 --collect-randomise --eval-randomise --eval-workers 8 \
    --render-worlds 64 --seed 0 || { echo "SMOKE FAIL start"; exit 1; }
t1=$(date +%s); echo "SMOKE stage a ok $((t1 - t0)) s"
python3.12 -P scripts/so101_policy/vision_viral.py online --init "$S/start/r1/last.pt" --out "$S/online" "${COMMON[@]}" \
    --tasks "$R2S_TRAIN" --eval-tasks $R2S_DEV --eval-n 4 --guard-tasks $R2S_FRESH "${CFG[@]}" \
    --online-config configs/so101_policy/vision/viral_r2s_online_v1.json \
    --oset envs=64 buffer=4000 iterations=24 eval_every=24 save_every=24 log_every=8 \
    --eval-randomise --eval-workers 8 --render-worlds 64 --seed 1 || { echo "SMOKE FAIL online"; exit 1; }
t2=$(date +%s); echo "SMOKE stage b ok $((t2 - t1)) s"
python3.12 -P scripts/so101_policy/vision_viral.py eval --student "$S/online/last.pt" "${COMMON[@]}" \
    --tasks "$R2S_DATA/tasks/home_dev100.npz" --n 8 --world mujoco --workers 8 --render-worlds 64 --randomise \
    --out "$S/eval_home_dev8_mujoco_dr.npz" || { echo "SMOKE FAIL eval"; exit 1; }
t3=$(date +%s); echo "SMOKE stage c ok $((t3 - t2)) s"
echo "SMOKE OK total $((t3 - t0)) s $(date -Is)"
