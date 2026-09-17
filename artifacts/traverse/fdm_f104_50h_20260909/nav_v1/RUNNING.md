# Running nav_v1

The campaign below ran on the AMD cluster (`ssh amd`). Since 2026-09-16 the same runner also runs on the workstation
(luffy) with the user's source-built Chrono (OptiX, depth-FOV fix), about 400x faster per depth frame — see
"Running locally" at the end.
Campaign root: `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/nav_v1` (`$N` below). The cluster job
scripts (`nav_run.sbatch`, `nav_run32.sbatch`, `nav_pilot.sbatch`, `nav_bench.sbatch`, `nav_range*.sbatch`,
`nav_maskds.sbatch`, `nav_train_r20.sbatch`) are copied as run into `scripts/`; they use the cluster paths above.

```
$N/code/      nav_runner.py nav_online.py nav_array.py nav_tasks.py nav_analyze.py nav_moving_diag.py
              nav_video.py nav_figure.py nav_latency_bench.py + sensor_map_v2/sensor_dataset_v2/
              vehicle_corridor/gen_planner/gen_riskmodel/f104_n2_sampler/gen_mission_runner
$N/missions/  30 mission files (scripts/nav_missions.py)
$N/models/    frozen matched_{H,H0,Dabs,Drel}_s{0,1,2}.pt, copied from sensor_v2/matched
$N/source/    copy of the frozen collector tree, with RenderSpec.with_rgb added (gen_v1/source untouched)
$N/main/      the reported campaign (runs/, logs/)
```

## One mission
```
sbatch -p mi2101x -o logs/pilot_%j.out \
  --export=ALL,NAV_MISSION=g216_nav_001,NAV_MODE=periodic,NAV_PERIOD=1.0,NAV_TAG=r1 $N/nav_pilot.sbatch
```
or directly, inside the Chrono environment
(`source /work1/dannegrut/harry/nrd/env.sh; nrd_pychrono; nrd_use_lavapipe`):
```
NAV_MODELS="$N/models/matched_Dabs_s*.pt" $NRD_PYTHON -P -u $N/code/nav_runner.py \
  --mission $N/missions/g216_nav_001.json --mode periodic --period 1.0 \
  --out $N/out/g216_nav_001__R1 --source-root $N/source \
  --chrono-data /work1/dannegrut/harry/nrd/chrono-build/data --torch-threads 8
```
Useful flags: `--latency-s -2` (charge the algorithmic latency), `--sense-radius-m 20` (limited-range sensing),
`--pick random` (control), `--keep-current --switch-margin X` (stickiness), `--save-frames --rgb on` (video),
`--path-heights map` (use the heightmap for the follower's path z instead of the sensed one), `--no-mask`.

## A campaign
```
python3 code/nav_tasks.py --missions missions --out tasks_main.json --arms W R2 R1 R1L --shards 30
# the reported campaign (jobs 421779/421780) split the shards over two partitions:
sbatch -p mi2104x --array=0-12  -o logs/main_%A_%a.out \
  --export=ALL,NAV_TASKS=$N/tasks_main.json,NAV_OUT=$N/main,NAV_WORKERS=4 nav_run32.sbatch
sbatch -p mi2101x --array=13-29 -o logs/main_%A_%a.out \
  --export=ALL,NAV_TASKS=$N/tasks_main.json,NAV_OUT=$N/main,NAV_WORKERS=1 nav_run.sbatch
```
All arms of a mission share a shard on purpose: Chrono on this cluster is deterministic per node, not across nodes.

## Read-out
```
python3 code/nav_analyze.py     --root main --out summary.json
python3 code/nav_moving_diag.py --root main --arm R1 --out diag_R1.json
python3 code/nav_figure.py      --root main --summary summary.json --mission g216_nav_001 --out results.png
python3 code/nav_video.py       --run video/runs/g216_nav_001__R1 --out video_R1.mp4     # needs --save-frames
python3 code/nav_video_compare.py --runs video/runs/M__W video/runs/M__R2 video/runs/M__R1 \
    video/runs/M__R1L --labels W R2 R1 R1L --out compare.mp4                           # four planners, one screen
python3 code/nav_latency_bench.py --models "$N/models/matched_Dabs_s*.pt" \
  --frame $CAMPAIGN/sensor_v2/vehicle_frames/g216_v2_group_0000 --device cuda --out bench_cuda.json
```

## Costs (16-core mi2101x node, depth-only render, 8 torch threads)
| arm | decisions per mission | wall per mission |
|---|---|---|
| W    | 6-8   | ~2.5 min |
| R2   | 25-40 | ~4.5 min |
| R1   | 50-90 | ~8-12 min |

## Running locally (luffy, OptiX)
The Chrono build needs the system Python 3.12 with numpy 1.26; torch comes from the `nedm` conda env.
`scripts/nav_local.py` imports numpy first and then appends the conda site-packages, then runs `nav_runner.py`.
```
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 -u scripts/nav_local.py \
  --mission artifacts/traverse/fdm_f104_50h_20260909/nav_v1/missions/g216_nav_001.json --mode periodic --period 1.0 \
  --out /tmp/g216_nav_001__R1 --source-root . --chrono-data /home/harry/chrono/data \
  --models 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/matched/matched_Dabs_s*.pt' --device cuda --torch-threads 1
```
Whole task list, 6 at a time (finished runs are skipped, so an interrupted batch can simply be restarted):
```
L=artifacts/traverse/fdm_f104_50h_20260909/nav_v1/local_luffy
python3 scripts/nav_local_batch.py --tasks $L/tasks_main.json --out $L --workers 6
```
Add `--video` to record the overhead RGB camera and a chase camera (`--video-dir`, `--chase-cam`); recording leaves
the rollout bit-identical. Median wall time per run with six in parallel on the 8-core CPU: W ~2.5 min, R2 ~3.2 min, R1 ~3.7 min, R1L ~4 min
(physics at ~0.6x real time dominates; per decision: render 0.01 s, back-projection 0.05 s, candidates 0.4-0.7 s,
corridors 0.3 s, model 0.05 s).
