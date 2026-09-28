#!/bin/bash
# Why does analytic fine-tuning fail in the rebuilt pipeline when it worked in the old one?
# Hypotheses in order: (1) budget, (2) horizon. Surrogate: nnrom_v2h_s8_ms100_s0 (1 s
# rollout-trained, a CRM comparator arm). an_h15_dw1 (0.30 s, dw 1.0) already exists.
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=4
P=~/miniconda3/envs/nedm/bin/python
M=~/qrc/nnrom_v2h_s8_ms100_s0/best.pt
run(){ L=$1; shift
  $P -u finetune.py --model $M --policy ~/qrun/policy/policy.pt --corpus ~/qrun/quadruped/data/go2_crm_v2 \
    --out ~/qrc/$L --method analytic --device cuda --seed 0 --branches 64 --iters 5000 "$@" > ~/qrc/logs/$L.log 2>&1
  echo "FT $L exit $? $(date +%H:%M)"; }
run an_h15_dw05 --steps 15 --target-dw 0.5
run an_h15_dw2  --steps 15 --target-dw 2.0
run an_h15_dw4  --steps 15 --target-dw 4.0
run an_h50_dw1  --steps 50 --target-dw 1.0 --accum 4
run an_h100_dw1 --steps 100 --target-dw 1.0 --accum 8
run an_h100_dw2 --steps 100 --target-dw 2.0 --accum 8
echo GRID_DONE
