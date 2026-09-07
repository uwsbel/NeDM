#!/bin/bash
# Plan 31 step 3: fresh sealed arenas chosen model-free. Generates arena_f108..f111 at a harder family setting, builds the
# same candidate banks as f106/f107, ships them to newton and starts the Chrono collection there (the models never touch
# these arenas). Afterwards traverse_wp8_sealed_count.py builds the cache and counts the layouts on which the fastest
# candidate fails while a feasible alternative exists; the two arenas with the most such layouts become the sealed pair.
#   scripts/traverse_wp8_sealed_prep.sh [difficulty]
set -e
cd /home/harry/NeDM; export PYTHONPATH=src; PY=/home/harry/miniconda3/envs/nedm/bin/python
D=${1:-1.25}
$PY scripts/traverse_wp7_arenas.py --seeds 108 109 110 111 --difficulty $D --family-json assets/traverse/arena_fsealed2_family.json
for s in 108 109 110 111; do
  $PY scripts/traverse_wp7_collection_tasks.py --arena assets/traverse/arena_f$s --out artifacts/traverse/wp8_collect_f$s --seed $s \
     --headings 0 45 90 135 180 225 270 315 --n-sequences 12 --seq-speeds 8 --n-freeform 8 --wide-detours > artifacts/traverse/wp8_collect_f$s.log 2>&1
  echo "tasks f$s: $(grep -c '"key"' artifacts/traverse/wp8_collect_f$s/tasks.json 2>/dev/null) runs"
done
rsync -az assets/traverse/arena_f108 assets/traverse/arena_f109 assets/traverse/arena_f110 assets/traverse/arena_f111 newton:~/NeDM/assets/traverse/
for s in 108 109 110 111; do rsync -az artifacts/traverse/wp8_collect_f$s/ newton:~/NeDM/artifacts/traverse/wp8_collect_f$s/; done
rsync -az scripts/traverse_wp3_chrono_eval.py src/nedm/traverse/ newton:~/NeDM/scripts/ 2>/dev/null || true
rsync -az scripts/traverse_wp3_chrono_eval.py newton:~/NeDM/scripts/; rsync -az src/ newton:~/NeDM/src/
cat > /tmp/run_collect_sealed2.sh <<'EOS'
#!/bin/bash
cd ~/NeDM
for s in 108 109 110 111; do
  echo "=== arena_f$s $(date)"
  ~/run_collect.sh artifacts/traverse/wp8_collect_f$s/tasks.json artifacts/traverse/wp8_collect_f$s --procs 20
done
echo "SEALED2 DONE $(date)"
EOS
scp -q /tmp/run_collect_sealed2.sh newton:~/run_collect_sealed2.sh; ssh newton 'chmod +x ~/run_collect_sealed2.sh; nohup ~/run_collect_sealed2.sh > ~/wp8_collect_sealed2.log 2>&1 &'
echo "SEALED PREP LAUNCHED on newton (log ~/wp8_collect_sealed2.log)"
