#!/bin/bash
# Local evaluation of the wp8 stall-ablation runs as they finish on the cluster (plan §30, notes §13).
# Polls the cluster; for each run with g3_readout.json (finished) or a ckpt_last.pt older than 5 min (walltime hit),
# pulls the run dir, then runs: the stall model tests on the validation arena (f105) and on a training arena (f104),
# the imagination + pick table on f105 (60 deg attitude limits, tracker's own action centre). Writes
# artifacts/traverse/wp8_eval/<run>/{analyze_f105.txt,analyze_f104.txt,pick_f105.json} and a leaderboard.
#   nohup scripts/traverse_wp8_eval_runs.sh > /tmp/wp8_eval.log 2>&1 &
cd /home/harry/NeDM
export PYTHONPATH=src
PY=/home/harry/miniconda3/envs/nedm/bin/python
REMOTE=amd:/work1/dannegrut/harry/nedm/artifacts/traverse
OUT=artifacts/traverse/wp8_eval; mkdir -p $OUT
FROZEN=artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt
eval_one() {  # $1 = run name (dir under artifacts/traverse), $2 = checkpoint path
  local run=$1 ck=$2 od=$OUT/$1; mkdir -p $od
  [ -f $od/done ] && return 0
  echo "[$(date +%H:%M)] evaluating $run"
  $PY scripts/traverse_wp7_stall_diagnosis.py model --arenas arena_f105 --classes launch stop --dynamics-checkpoints $ck --tag eval_$run --roll-limit-deg 60 --pitch-limit-deg 60 > $od/model_f105.log 2>&1 \
    && $PY scripts/traverse_wp7_stall_diagnosis.py analyze --tag eval_$run --examples 0 > $od/analyze_f105.txt 2>/dev/null
  $PY scripts/traverse_wp7_stall_diagnosis.py model --arenas arena_f104 --classes launch stop --dynamics-checkpoints $ck --tag eval_${run}_f104 --roll-limit-deg 60 --pitch-limit-deg 60 > $od/model_f104.log 2>&1 \
    && $PY scripts/traverse_wp7_stall_diagnosis.py analyze --tag eval_${run}_f104 --examples 0 > $od/analyze_f104.txt 2>/dev/null
  rm -f artifacts/traverse/wp7_stall_diag/model_tests_eval_${run}.json artifacts/traverse/wp7_stall_diag/model_tests_eval_${run}_f104.json
  $PY scripts/traverse_wp7_imagine_cache.py --cache artifacts/traverse/wp7_cache_v1 --arenas arena_f105 --dynamics-checkpoints $ck --roll-limit-deg 60 --pitch-limit-deg 60 --out $od/imagine_f105 > $od/imagine_f105.log 2>&1
  $PY scripts/traverse_wp7_pick_table.py --cache artifacts/traverse/wp7_cache_v1 --arenas arena_f105 --imagine $run=$od/imagine_f105/rows.json --json $od/pick_f105.json > $od/pick_f105.txt 2>&1
  touch $od/done
  $PY scripts/traverse_wp8_leaderboard.py > $OUT/leaderboard.txt 2>/dev/null
  echo "[$(date +%H:%M)] done $run"
}
# references first (frozen and the wp7 selected model), so the leaderboard has baselines
eval_one wp2_mapv2_pt_dag_ro8_amd $FROZEN
eval_one wp7_ft_mix_amd artifacts/traverse/wp7_ft_mix_amd/ckpt_best.pt
while true; do
  runs=$(ssh amd 'cd /work1/dannegrut/harry/nedm/artifacts/traverse && for d in wp8*/; do d=${d%/}; if [ -f $d/g3_readout.json ]; then echo $d; elif [ -f $d/ckpt_last.pt ] && [ $(( $(date +%s) - $(stat -c %Y $d/ckpt_last.pt) )) -gt 900 ] && [ -z "$(squeue -u $USER -h -n $d)" ]; then echo $d; fi; done' 2>/dev/null)
  for r in $runs; do
    [ -f $OUT/$r/done ] && continue
    rsync -az $REMOTE/$r/ artifacts/traverse/$r/ 2>/dev/null || continue
    ck=artifacts/traverse/$r/ckpt_best.pt; [ -f $ck ] || ck=artifacts/traverse/$r/ckpt_last.pt
    [ -f $ck ] && eval_one $r $ck
  done
  n_q=$(ssh amd 'squeue -u $USER -h -n "$(cd /work1/dannegrut/harry/nedm/artifacts/traverse; ls -d wp8* 2>/dev/null | tr "\n" ",")" | wc -l' 2>/dev/null)
  n_done=$(ls $OUT/wp8*/done 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] queue: $n_q  evaluated: $n_done"
  if [ "$n_q" = "0" ] && [ -z "$(ssh amd 'squeue -u $USER -h' 2>/dev/null)" ]; then
    # one last sweep for stragglers, then stop
    sleep 60
    runs=$(ssh amd 'cd /work1/dannegrut/harry/nedm/artifacts/traverse && ls -d wp8*' 2>/dev/null)
    all_done=1; for r in $runs; do [ -f $OUT/$r/done ] || all_done=0; done
    [ $all_done = 1 ] && { echo "ALL WP8 EVALUATED"; break; }
  fi
  sleep 300
done
