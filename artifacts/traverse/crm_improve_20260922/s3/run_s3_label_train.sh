#!/bin/bash
# S3: when the soil continuation drives finish, sync both worlds, label them into training rows, ship the rows and
# submit the offline arms that test whether moving-state data closes the remaining 3 s gap.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=
K2=artifacts/traverse/crm_improve_20260922; A=$K2/a5data; S3=$K2/s3; L=$A/cont_twin_L3
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
log() { echo "$(date +%H:%M:%S) $*"; }
while true; do n=$(ssh -o ConnectTimeout=20 amd "ls $C/crm_improve/cont_twin_L3/out/runs/*/episode_complete.json 2>/dev/null | wc -l" || echo 0)
  q=$(ssh -o ConnectTimeout=20 amd "squeue -u \$USER -h -n ci_cont3 | wc -l" || echo 1)
  [ "$n" -ge 7182 ] && break
  if [ "$q" -eq 0 ] && [ "$n" -ge 7000 ]; then log "soil continuations ended at $n/7182"; break; fi
  log "soil continuations $n/7182"; sleep 240; done
INC="--include=*/ --include=trajectory.npz --include=outcome.json --include=episode_complete.json --include=command_reference.npz --include=case.json --include=crm_extra.npz --exclude=*"
mkdir -p $L/runs_crm $L/runs_rigid
rsync -az $INC amd:$C/crm_improve/cont_twin_L3/out/runs/ $L/runs_crm/
rsync -az $INC amd:$G2/rigid_cont_twin_L3/runs/ $L/runs_rigid/
log "synced: soil $(ls $L/runs_crm | wc -l), rigid $(ls $L/runs_rigid | wc -l)"
for W in crm rigid; do
  $PY scripts/ga_branch_dataset.py --runs $L/runs_$W --anchors $L/anchors_${W}_L3.json --world $W \
      --map-root artifacts/traverse/crm_f104_v1/map_root --out $K2/datasets/cont3_$W.npz > $S3/label_$W.log 2>&1
  log "labelled $W rc=$? $(tail -2 $S3/label_$W.log | head -1 | cut -c1-160)"
done
$PY scripts/ga_branch_dataset.py --out $K2/datasets/cont3_crm.npz --merge $K2/datasets/cont3_rigid.npz --merged-out $K2/datasets/cont3_both.npz > $S3/merge.log 2>&1
log "merged rc=$? $(grep -o '\"rows\": [0-9]*' $S3/merge.log | head -1)"
rsync -az $K2/datasets/cont3_both.npz amd:$G2/data/
ssh amd "G2=$G2; cd \$G2/train && sed -e 's#--ds \$G2/data/anchor_k60.npz#--ds \$G2/data/anchor_k60.npz --ds \$G2/data/cont3_both.npz#' ci_arms.sbatch > ci_arms_cont3.sbatch && j=\$(sbatch --parsable --array=0,2,4 --export=ALL,ARM_MODE=holdout,ARM_SEEDS=3,ARM_OUT=offline_cont3 ci_arms_cont3.sbatch 2>/dev/null | tail -1); echo submitted offline_cont3 \$j"
log "S3 offline arms submitted"
