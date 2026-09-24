#!/bin/bash
# S2 analysis: sync the short-approach pass-2 drives (existing and new ensembles), index them, and compare every arm
# with the K1 3 s moving-anchor baseline and the K1 standing-start arms on the same 800 groups, per world.
# usage: bash analyze_s2.sh [crm|rigid|both]
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=
K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922; S2=$K2/s2; IDX=$S2/idx
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; G2=/work1/dannegrut/harry/experiments/crm_improve_20260922
WHICH=${1:-both}
INC="--include=*/ --include=trajectory.npz --include=outcome.json --include=episode_complete.json --exclude=*"
mkdir -p $IDX $S2/runs_crm $S2/runs_rigid
rsync -az $INC amd:$C/crm_improve/s2/out/runs/ $S2/runs_crm/ 2>/dev/null
rsync -az $INC amd:$C/crm_improve/s2n/out/runs/ $S2/runs_crm/ 2>/dev/null
rsync -az $INC amd:$G2/rigid_s2/runs/ $S2/runs_rigid/ 2>/dev/null
rsync -az $INC amd:$G2/rigid_s2n/runs/ $S2/runs_rigid/ 2>/dev/null
echo "runs: crm $(ls $S2/runs_crm | wc -l), rigid $(ls $S2/runs_rigid | wc -l)"
for W in crm rigid; do
  [ "$WHICH" != both ] && [ "$WHICH" != $W ] && continue
  files=$(ls $S2/tasks_s2_$W.json $S2/tasks_s2n_$W.json 2>/dev/null)
  $PY - $IDX/tasks_all_$W.json $files <<'EOF'
import json, sys
rows = [r for f in sys.argv[2:] for r in json.load(open(f))]
ids = [r['id'] for r in rows]; assert len(ids) == len(set(ids))
json.dump(rows, open(sys.argv[1], 'w'), indent=1); print(len(rows), 'rows indexed')
EOF
  models=""
  for T in L1 L0p5; do for d in $S2/$T/picks_${W}_*; do [ -d $d ] || continue; case $d in *_named) continue;; esac
    arm=${d##*/picks_${W}_}; ln -sfn ../$T/picks_${W}_$arm $IDX/picks_${W}_${T}_$arm; models="$models,${T}_$arm"; done; done
  models=${models#,}
  $PY scripts/ga_a3_index.py --world $W --models $models --a3-dir $IDX --tasks $IDX/tasks_all_$W.json --out $IDX/run_index_$W.json | tail -1
  if [ $W = crm ]; then
    base="--arm H3s=$K1/a5/picks_crm_H_named:B:$K1/a5/crm_pass2_runs --arm Sp3s=$K1/a5/picks_crm_Spcrm_named:B:$K1/a5/crm_pass2_runs"
    stand="--arm Hstand=$K1/a3/picks_crm_H_named:B:$K1/mixed_out_crm/runs,$K1/suite_out_crm/runs,artifacts/traverse/crm_night2_v1/planner/eval_iter_crm/runs --arm Scrm_stand=$K1/suite/picks_crm_Scrm:B:$K1/mixed_out_crm/runs,$K1/suite_out_crm/runs,artifacts/traverse/crm_night2_v1/planner/eval_iter_crm/runs"
    ridx="$K1/a3/run_index_crm_all.json"; ref=H3s
  else
    base="--arm H3s=$K1/a5/picks_rigid_H_named:B:$K1/a5/rigid_pass2_runs --arm Sp3s=$K1/a5/picks_rigid_Sprigid_named:B:$K1/a5/rigid_pass2_runs"
    stand="--arm Hstand=$K1/a3/picks_rigid_H_named:B:$K1/suite_out_rigid/runs --arm Srigid_stand=$K1/suite/picks_rigid_Srigid:B:$K1/suite_out_rigid/runs"
    ridx="$K1/a3/run_index_rigid_all.json"; ref=H3s
  fi
  arms=""; con=""
  for m in ${models//,/ }; do arms="$arms --arm $m=$IDX/picks_${W}_${m}_named:B:$S2/runs_$W"; con="$con,$m:$ref"; done
  con=${con#,}
  best=$(echo $models | tr ',' '\n' | grep -E '^L1_(X|Hn)$' | head -1); [ -z "$best" ] && best=L1_H
  for cmp in vs3s vsstand; do
    if [ $cmp = vs3s ]; then extra="$base"; bidx=$K1/a5/run_index_${W}_pass2.json; r=H3s; else extra="$stand"; bidx=$ridx; r=Hstand; fi
    $PY - $IDX/run_index_${cmp}_$W.json $IDX/run_index_$W.json $bidx <<'EOF2'
import json, sys
m = {}
for f in sys.argv[2:]:
    m.update(json.load(open(f)))
json.dump(m, open(sys.argv[1], 'w')); print(len(m), 'index entries')
EOF2
    cc=$(echo $con | sed "s/:$ref/:$r/g")
    $PY scripts/ga_analyze.py $extra $arms --primary $best:$r --contrasts $cc --label fail \
        --suite $K1/suite/suite.json --run-index $IDX/run_index_${cmp}_$W.json --margin-pts 3.0 --time-bound 1.10 \
        --cluster-ci --cases $K1/suite/cases --world $W --out $S2/results_s2_${W}_$cmp.json > $S2/results_s2_${W}_$cmp.txt 2>&1
    echo "== $W $cmp (rc $?)"; grep -E "groups with every arm|^  \[all\]|^     " $S2/results_s2_${W}_$cmp.txt | head -24
  done
done
