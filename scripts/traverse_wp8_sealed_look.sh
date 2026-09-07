#!/bin/bash
# The single pre-registered look at the fresh sealed pair (notes §13.12, written before the collection finished).
# Primary: head = GRU on the imagined trajectory (state, pose, progress; no tokens) of wp8e_mom_s1 driven from rest by the WP3
# tracker, trained on f101-f105 outcomes (LOAO early stopping), gate threshold = the f105 threshold rejecting 50 % of infeasible
# routes (fixed here from the validation run), decision = fastest accepted candidate (fallback fastest). Comparators with the same
# rule: the profile-only head (cheap predictor inputs, true terrain), the frozen model's imagined-state head, the fastest heuristic.
# Endpoints: feasible picks vs heuristic with fixes:breaks (paired sign test), AUCs with layout-cluster CIs, contact-only and
# no-solution layouts reported apart. One run; nothing is re-tuned afterwards.
set -e
cd /home/harry/NeDM; export PYTHONPATH=src; PY=/home/harry/miniconda3/envs/nedm/bin/python; O=artifacts/traverse/wp8_head
C2=artifacts/traverse/wp8_cache_sealed2; V1=artifacts/traverse/wp7_cache_v1
FR=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt; MS=artifacts/traverse/wp8e_mom_s1/ckpt_best.pt
PAIR=$($PY -c "import json; print(' '.join(json.load(open('$C2/sealed_pair.json'))['pair']))")
echo "sealed pair: $PAIR"
# classify the sealed-2 runs (classes for the stall label) -- labels only, no model
$PY scripts/traverse_wp7_stall_diagnosis.py classify --caches $V1 artifacts/traverse/wp7_cache_sealed $C2 > /dev/null 2>&1
# imagined sequences on the pair (the only time a model touches these arenas) and the nominal features
$PY -u scripts/traverse_wp8_head.py dump --model $MS --policy tracker --caches $C2 --arenas $PAIR --out $O/img_moms1_trk > $O/dump_sealed2_moms1.log 2>&1
$PY -u scripts/traverse_wp8_head.py dump --model $FR --policy tracker --caches $C2 --arenas $PAIR --out $O/img_frozen_trk > $O/dump_sealed2_frozen.log 2>&1
$PY -u scripts/traverse_wp8_head.py nominal --model $FR --caches $C2 --arenas $PAIR --out $O/nom_frozen > $O/nom_sealed2.log 2>&1
# fixed threshold from the validation run (seed-averaged state head, 50 % infeasible rejection on f105)
THR=$($PY -c "
import json,numpy as np; R=json.load(open('$O/results.json')); print(np.mean([r['threshold'] for r in R['imagined moms1+tracker:state']]))")
echo "gate threshold (from f105): $THR"
$PY -u scripts/traverse_wp8_head.py train --seeds 3 --caches $V1 $C2 --train-arenas arena_f101 arena_f102 arena_f103 arena_f104 arena_f105 --eval-arenas $PAIR --threshold $THR \
   --arms "imagined moms1+tracker:state=$O/img_moms1_trk" "imagined moms1+tracker=$O/img_moms1_trk" "imagined frozen+tracker:state=$O/img_frozen_trk" "nominal:profile=$O/nom_frozen" "nominal:speed=$O/nom_frozen" \
   --out $O/results_sealed2.json > $O/train_sealed2.log 2>&1
grep -v Warning $O/train_sealed2.log
echo "SEALED LOOK DONE"
