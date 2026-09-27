source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925; T=$G3/tools/bf; cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
nice -n 10 $NRD_PYTHON -u scripts/ag_s1_gator_soil_qa.py --tasks $G3/e5/ids_bf/soil_v3_gator_tiers0-12.json --runs $G3/soil_v1/runs --ref-runs /work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1/runs --out $G3/e5/ids_bf/gator_soil_qa_all_summary.json
echo "qa exit: $?"
