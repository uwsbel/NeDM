#!/bin/bash
# No-PPO-data ablation (plan ~/so101_policy/docs/abl_noppo_plan.md): one CPU MuJoCo evaluation (Newton images for the
# students), with exactly the commands of the build9 final fresh100 runs (~/so101_policy/runs/viral/bin/final_eval.sh and
# runs/viral/teacher_ref), on luffy (RTX 5090 + CPU MuJoCo, 8 workers, env so101_vision). All tasks of the file run as one
# batch (as the fresh100 runs did: batch size changes bf16 outcomes, so keep the same task file for every policy).
#
#   bash abl_noppo_mj_eval.sh student STUDENT.pt JT.pt TASKS.npz OUT.npz [--randomise]   # camera student, lead from its cfg
#   bash abl_noppo_mj_eval.sh jt      JT.pt TASKS.npz OUT.npz                            # Markov joint teacher (teacher B)
#   bash abl_noppo_mj_eval.sh expert  TEACHER_A.pt TASKS.npz OUT.npz                     # state PPO teacher A + decoder v3
#
# In world mujoco no NRD checkpoint is loaded (vec_env.make_world: MujocoWorlds only), so the --nrd-ckpt default does not
# enter these numbers. The script echoes the command and the sha256 of every checkpoint and task file it passes.
set -euo pipefail
W=${W:-/home/harry/.t3/worktrees/NeDM-push-ppo/t3code-2f613f84}
PY=${PY:-$HOME/miniconda3/envs/so101_vision/bin/python}
cd "$W"
export PYTHONPATH=src
kind=$1
shift
case "$kind" in
  student)
    sha256sum "$1" "$2" "$3"
    cmd=("$PY" -P scripts/so101_policy/vision_viral.py eval --student "$1" --jt "$2" --tasks "$3" --world mujoco --out "$4" ${5:+"$5"});;
  jt)
    sha256sum "$1" "$2"
    cmd=("$PY" -P scripts/so101_policy/vision_viral_teacher.py eval --jt "$1" --tasks "$2" --world mujoco --out "$3");;
  expert)
    sha256sum "$1" "$2"
    cmd=("$PY" -P scripts/so101_policy/vision_viral_teacher.py eval-expert --teacher "$1" --tasks "$2" --world mujoco --out "$3");;
  *)
    echo "kind: student | jt | expert"; exit 2;;
esac
echo "host $(hostname) start $(date -Is) HEAD $(git rev-parse --short HEAD)"
echo "cmd: ${cmd[*]}"
"${cmd[@]}"
echo "exit $? $(date -Is)"
