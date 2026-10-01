# Traversing evaluation

`TraversalEval` evaluates one **arm** (an `EvalConfig`: what is driven and how it plans) on a list of **tasks**
(`Task`: where). For each task it plans a route with a learned route-risk model (or takes a given route), drives it in
Project Chrono with the chosen vehicle and controller, records the 50 ms trajectory and labels the drive with the
study's codes: `S` goal reached safely, `s` goal reached with an event, `U` not reached and unsafe, `F` not reached and
safe. Every released input is checked against the pinned release manifest, and the released drives can be replayed
(see [Bitwise or comparable](#bitwise-or-comparable)).

Modules: `config` (arm, machine paths, release checks), `suites` (tasks), `routes`, `planner` and `refine` (route
search), `sim`, `vehicles`, `controllers` and `episode` (one Chrono drive), `nav` (M1 live navigation), `labels`,
`runner` (run folders, one process per drive, resume).

## The switches

| Switch | `EvalConfig` field(s) | Values |
|---|---|---|
| vehicle | `vehicle` | `hmmwv`, `gator`, `polaris`, `polaris_pc` (power-corrected); smoke-only `polaris_4wd`, `polaris_w08` |
| ground | `ground`, `soil_config` | `rigid`, or `soil` (CRM deformable soil) with `soil_config = "crm_main"` (1 ms step) or a JSON file |
| arena and tasks | suite and subset (passed to `load_suite`, not an arm field) | `f104_800` (M2, M4b: 800 start-goal pairs on the training arena), `unseen_soil1000` / `unseen_rigid2000` (M4a: 8 unseen arenas), `tracker423` (M3 reference routes), `missions30` (M1 waypoint missions), `smoke144` (vehicle smoke routes), or `custom:<dir>` of cases written by `make_case` (any start and goal on an 80 m arena, optionally a route; planning needs a released planner map). Subsets: `key=v1\|v2`, `lowest_md5=N`, `lowest_md5_per_arena=N`, comma-joined |
| planner | `planner`, `models` | `given` (the task's reference route), `straight` (the straight route at `speed`), `cem` (sampling route search scored by the ensemble `models`), `cem_grad` (the search, then gradient refinement), `live` (M1: re-plans inside the drive from an overhead depth camera) |
| route search | `rounds`, `samples`, `update`, `speed` | 4 x 64 as recorded; `update = "cem"` (cumulative-elite CEM) or `"mppi"` (ESS-tempered exponential weights); `speed = 2.0` searches the fixed-speed family |
| recorded picks | `picks` | drive the study's locked picks instead of planning; omit it to re-plan |
| decision timing | `approach_s`, `decisions` | `0`: plan at a standing start. `> 0`: drive the released approach route, plan at frame F = `approach_s` / 0.05 from that moving state, then switch to the planned route at F (planning time is not charged). `decisions`: plan from the released states at F instead of driving the approach first (the drive still runs the approach to F) |
| M1 replanning | `replan`, `latency_replay` | `"waypoint"` or a period in s; `latency_replay` replays the recorded planning delays |
| controller | `controller`, `actor` | `pid` (the path follower), `pid_held` (held every 50 ms), `tracker` (the learned tracker; `actor` = the released `actor.npz`), `nav_pid` (M1) |
| label | `label` | `rollback` (S: goal reached without rolling back), `rollback_belly` (and no belly flag: hull > 5 cm under the surface for > 1 s; Polaris), `goal_belly` (goal and belly only; vehicle smoke), `tracker` (M3: unsafe by status, F = timeout, plus cross-track and speed errors), `mission` (M1: every waypoint, no backward slide) |
| build | `build_lock`, `allow_unvalidated` | the parity-checked Chrono build the arm must run on; a combination outside the validated matrix runs only with `allow_unvalidated = true` and is stamped `validated: false` in every record |

`cfg.validate()` lists every problem at once; with an `Env` and the tasks it also checks every input file, which
`prepare` does before any drive starts. The headline arms are in `configs/traversing/evaluation/*.toml`: `[defaults]`
merged under each `[[arm]]`, one arm per results column. They name the cluster builds' locks, so on any other Chrono
build their drives are refused; copy an arm without `build_lock` and with `allow_unvalidated = true` to run it there.
`run.py` and `slurm.py` print a refusal as `refused:` and the list of problems; an unknown planner name points to
`cem` with `update = "mppi"` (MPPI) and `cem_grad` (gradient).

Environment: `NEDM_DATA` (the release restore base; `data:<path>` references resolve under it), `NEDM_RELEASE_CACHE`
(the Hub mirror holding the tar index files; default `$NEDM_DATA/artifacts/hf_release/download`), `NEDM_CHRONO_DATA`
(the data folder of the Chrono build; drives only) and `NEDM_DEVICE` (default `cuda`).

## Examples

```bash
export PYTHONPATH=src NEDM_DATA=... NEDM_RELEASE_CACHE=... NEDM_CHRONO_DATA=...
R=scripts/traversing/evaluation/run.py; A=configs/traversing/evaluation

# 1-2 use the headline arms, whose build locks pin the cluster builds: run them on an mi3501x node (slurm.py, below)
# 1. Tracker vs PID on rigid ground, the 10 lowest-md5 reference routes (8 drive processes)
python $R --out out/m3 --arms $A/m3_tracker.toml --ground rigid --suite tracker423 --subset lowest_md5=10 --workers 8

# 2. One soil arm on 10 fresh pairs: its locked picks driven as recorded, one drive process per GPU
python $R --out out/m2 --arms $A/m2_shared_risk.toml --ground soil --arm shared_hist_3s \
    --suite f104_800 --subset 'stratum=fresh,lowest_md5=10' --gpus 0

# 3. Your own arms (below) on 20 pairs of an unseen arena: plan only (no Chrono), then drive the same run folder
python $R --out out/mine --arms my_arms.toml --suite unseen_rigid2000 --subset arena=g260,lowest_md5=20 --stage plan
python $R --out out/mine --block 0 --workers 8
```

`my_arms.toml`: re-planning instead of the locked picks, the MPPI update and gradient refinement on the Gator. These
combinations were not recorded, so they run with `allow_unvalidated`. `--stage plan` refuses an arm that decides after
its own approach drive (`approach_s > 0` without `decisions` or `picks`): its pass 1 is a Chrono drive.

```toml
[defaults]
ground = "rigid"
models = "data:artifacts/traverse/crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt"
allow_unvalidated = true

[[arm]]
name = "search_cem"
planner = "cem"

[[arm]]
name = "search_mppi"
planner = "cem"
update = "mppi"

[[arm]]
name = "gator_grad"
vehicle = "gator"
planner = "cem_grad"
```

Python API (4: plan and drive one task in this process; 5: a custom task in a resumable run folder):

```python
import os
from nedm.traversing.evaluation import Decision, EvalConfig, TraversalEval, load_suite, make_case
from nedm.traversing.evaluation.runner import process_env

cfg = EvalConfig(name='grad', planner='cem_grad', allow_unvalidated=True,      # rigid, HMMWV, stock PID
                 models='data:artifacts/traverse/crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt')
ev = TraversalEval(cfg, 'out/api')
task = load_suite('f104_800', 'stratum=fresh,lowest_md5=1')[0]
pick = ev.plan(task, Decision.standing(task))   # no Chrono: pick.route, pick.z_mean, the search in pick.record
os.environ.update(process_env('rigid'))         # a drive runs only in the recorded thread setup
rec = ev.drive(task, pick.route)                # one Chrono episode -> Record (status, frames, arrays)

make_case('my_cases', 'cross_1', 'f104', start_xy=(-25., -10.), start_yaw=0., goal_xy=(25., 10.))
records = ev.run(load_suite('custom:my_cases'), workers=1)    # one process per drive; labels: runner.cells('out/api')
```

M1 (`m1_navigation.toml`, suite `missions30`) runs on the workstation with the Chrono fork that has the OptiX
depth-camera fix; the recipe is in the arm file's header.

## On the cluster

`slurm.py` prepares the run folder, writes `<out>/job/run.sbatch` (one array task per block of whole pairs, each
running `run.py --block`) and `<out>/job/code_sha.json`, and submits with `--submit`. Run it on the AMD login node from
a clean checkout (or a copy carrying `GIT_COMMIT`); the default partition is `mi3501x`; soil uses one drive process per
GPU, rigid CPUs - 2 workers. Size `--block-size` so a block finishes within `--time`; a `--time` that cannot fit one
pair of every arm (400 s rigid, 500 s soil per arm, `runner.GUARD_S`) is refused.

```bash
python scripts/traversing/evaluation/slurm.py --out $OUT --arms configs/traversing/evaluation/m2_shared_risk.toml \
    --ground soil --suite f104_800 --subset 'stratum=fresh,lowest_md5=20' --block-size 5 --time 00:55:00 --submit
# after a timeout or a node failure: the same command with --resume (finished drives are kept)
```

A block is pinned to the node and build it first ran on. Resumed elsewhere, each unfinished pair is moved to
`superseded/` and driven again whole (rigid physics repeats on one node only). Only runs with a `DONE` file count.

Run folder: `config/<arm>.json`, `tasks.jsonl` + `TASKS.sha256`, `blocks.json`, `blocks/<i>.pin.json`,
`runs/<task>/<arm>/` (`input.json`, `drive.log`, `attempts.jsonl`, `pick.json`, `pass1/`, `record.json`,
`trajectory.npz`, `crm_extra.npz`, `vehicle_extra.npz`, M1 `decisions.json` and `routes.json`, `DONE` last),
`superseded/`, `job/`.

## Checking against the released drives

`compare.py` compares one arm of a run folder with the release (drives of M2 to M4; per-pair codes of every table):

```bash
T=data:artifacts/traverse
python scripts/traversing/evaluation/compare.py --out out/m2 --arm shared_hist_3s \
    --released "$T/generalist_20260921/A_adapt/a5/crm_pass2_runs/{task}__H_B" \
    --csv traversing/results/m2_shared_risk_soil.csv --picks $T/generalist_20260921/A_adapt/a5/picks_crm_H
```

It writes `<out>/compare_<arm>.json`: per drive, every npz key against the released one (dtype and values; `bitwise`
needs state, action and pose), status, frames and positive work; the per-pair code against the results CSV: the arm's
column of a wide table (`--column` when the column has another name, e.g. `polaris_own_model_sampling_grad` for the
`_unseen` arm) or, in the long M1 and M3 tables, the `outcome_code` of the arm's rows on this ground (`--candidate DIR`
writes a results copy of a wide table for `recount_milestones.py`); the pick (and a gradient pick's search result)
against `picks/<task>.json`; with `--decisions`, the run's own decision state against the released one. A drive the
study shared between arms has no `outcome.json` in this arm's folder and counts as `reference_missing`: compare it
with `--task` and the other arm's folder. M1 drives have no `outcome.json`: only their codes are compared.

## Bitwise or comparable

As measured by the parity and headline checks (2026-09-30 to 10-01; 440 headline drives on mi3501x):

Bitwise (array-equal to the release):
- labels: all 105,100 released drive records relabelled with 0 mismatches; the 9 results tables rebuilt byte for
  byte; `recount_milestones` 251/251;
- planning: route search, gradient refinement, decision states and corridors, re-planned from the released inputs on
  the record environment (RTX 5090, torch 2.12.0+cu130; `pick.record['numerics']['env'] == 'bitwise_env'`);
- rigid drives with the stock and held PID on the locked rigid build: 180/180 headline drives;
- M1 drives on the workstation's OptiX fork build; the decisions of all 120 released M1 runs replayed offline.

Not bitwise, and why:
- soil drives: the CRM solver on the GPU does not repeat run to run, even on one GPU and in the frozen collector. 230/240
  headline soil drives (PID, all vehicles) were bitwise; the other 10 matched for their first 103-286 frames and then
  diverged, keeping their status and code. Short drives (183-320 frames) diverged too: no soil drive is sure to be
  bitwise;
- the learned tracker: its float64 network arithmetic depends on the CPU type, so it is bitwise only against the
  frozen collector on the same build and CPU (rigid 10/10 on one node; soil 9/10, status 10/10). Its recordings ran on
  other CPUs, and against them it can change the outcome: one of 20 headline tracker drives did (soil
  `f104_v2_group_0502_route_02` rolls over at frame 180, as the frozen collector does on mi3501x; the recording
  reached the goal);
- planning on another GPU or torch version (`numerics.env == 'comparable'`): a pick can differ;
- rigid drives on another node or build (rigid physics repeats on one node only); M1 on any other build than the
  OptiX fork; the latency-charged M1 arm is a replay only.

So the acceptance test of a soil or tracker drive is its status and per-pair code (`status_equal`, `code_equal` of
`compare.py`), not its bits. On the headline checks 439 of 440 drives kept the released status and per-pair code, and
every small-sample rate lies inside the 95 % binomial band of the documented full-set rate.

Tests: `python -m unittest discover -s tests -t .` (the data, GPU and Chrono tests run when `NEDM_DATA`,
`NEDM_RELEASE_CACHE`, the record GPU and `NEDM_CHRONO_DATA` are available).
