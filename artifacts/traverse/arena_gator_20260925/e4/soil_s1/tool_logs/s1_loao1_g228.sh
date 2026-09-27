source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1; unset NEDM_VEHICLE
cd /work1/dannegrut/harry/experiments/arena_gator_20260925/tools/s1; export PYTHONPATH=/work1/dannegrut/harry/experiments/arena_gator_20260925/tools/s1/src:/work1/dannegrut/harry/experiments/arena_gator_20260925/tools/s1/scripts:${PYTHONPATH:-}
nice -n 10 $NRD_PYTHON -u scripts/ag_subset.py --world crm --tiers 0-6 --eval-rows filtered --preset LOAO1_g228 --allow-short --ds /work1/dannegrut/harry/experiments/arena_gator_20260925/e4/soil_s1/g228_hmmwv/ci_g228_hmmwv_crm.npz --out /work1/dannegrut/harry/experiments/arena_gator_20260925/e4/soil_s1/subsets/LOAO1_g228_hmmwv_soil.npz > logs/subset_LOAO1_g228_soil.log 2>&1
echo "exit: $?" >> logs/subset_LOAO1_g228_soil.log
