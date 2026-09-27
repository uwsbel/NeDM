# E5: training on the AMD cluster (arena_gator_20260925)

`G3=/work1/dannegrut/harry/experiments/arena_gator_20260925`, `K3=artifacts/traverse/arena_gator_20260925`.
Every submission is also a line in `$G3/e3/submissions.tsv` and in `$K3/LOG.md`.

## 1. How a training job runs

- Job script: `$G3/source/scripts/ag_train.sbatch` (copy of `scripts/ag_train.sbatch`, sha256 `c5db4eb7069fa698...`).
  - One MI350X node (`mi3501x`, 24 cores, 3 h 50 min limit), environment as in the cluster training skill
    (`module load pytorch/2.10.0`, venv `nedm`, `python3.12`, `PYTHONPATH=$G3/source/src:$G3/source/scripts:$PYTHONPATH`).
  - `NEDM_VEHICLE` is unset in the job.
  - It reads a job list (`AG_JOBS`, tab-separated: tag, output folder relative to G3, trainer arguments) and starts every
    line at once on the one GPU, with the 24 cores shared evenly (`OMP_NUM_THREADS = 24 / lines`).
  - Each run's log: `$G3/e5/logs/<tag>_<job id>.log`, ending with `exit: <code>`. The job log
    `$G3/e5/logs/<job name>_<job id>.out` lists every run's exit code; the job fails if any run fails.
- Trainer: `$G3/source/scripts/ci_train.py` (sha256 `7a4f2d67316ea47a...`, the worktree file with E1's blacklist line;
  `ga_train.py` `dc73d0bb2ac83521...`). Checkpoints `<out>/<tag>_s<seed>.pt`, summary `<out>/<tag>.json`, logits
  `<out>/<tag>_logits.npz` (ids, ensemble and member logits, fit / dev / held-out masks).
- Data: `$G3/e4/subsets/*.npz` (made locally by `scripts/ag_subset.py` from `$K3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz`,
  synced with `rsync -a --exclude work $K3/e4/ amd:$G3/e4/`; sha256 equal on both sides).

Common trainer arguments (the deploy_a1 recipe of `crm_improve_20260922/deploy_v1/deploy_a1_haux_gru.json` where it
applies; PLAN 1.1 for the rest):

```
--arch gru --cond none --domain-filter rigid --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5
(lr and weight decay: the trainer defaults for the GRU, 2e-3 and 1e-4, as deploy_a1)
```

Not carried over from deploy_a1 (they only act with a history input or with both worlds): `--cond hist_aux`,
`--hist-drop`, `--aux-weight`, `--crm-batch-frac`.

## 2. Submissions

| job | what | job list | status |
|---|---|---|---|
| 436133 | smoke: M1 rigid deploy 1 epoch 1 seed (with checkpoint round trip), LC272 rigid holdout 1 epoch | `$G3/e5/jobs/smoke_rigid_f104.tsv` | done 02:50, 49 s, both exit 0, round trip exact |
| 436135 | rigid f104: M1a, M1b (deploy), M1 / LC545 / LC272 (holdout) | `$G3/e5/jobs/rigid_f104_v1.tsv` | running since 02:52 on k007-005-v8; first seed (LC272 s0) 03:00, 0.23 s/step with 5 runs sharing the GPU; expected end about 05:15 (limit 06:42) |

Exact commands (run on the login node):

```bash
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
squeue -u $USER -h -r | wc -l        # must stay <= 47 after the submission (50 per user, 3 kept free)
sbatch -J ag_train_smoke      -t 00:20:00 --export=ALL,AG_JOBS=$G3/e5/jobs/smoke_rigid_f104.tsv $G3/source/scripts/ag_train.sbatch   # 436133
sbatch -J ag_train_rigid_f104 -t 03:50:00 --export=ALL,AG_JOBS=$G3/e5/jobs/rigid_f104_v1.tsv    $G3/source/scripts/ag_train.sbatch   # 436135
```

The five runs of 436135 (`$G3/e5/jobs/rigid_f104_v1.tsv`; each line gets `--out $G3/<out> --tag <tag>` appended):

| tag | out | data | mode, seeds | fitted rows / groups |
|---|---|---|---|---|
| `M1a_rigid_deploy` | `e5/train/M1_rigid` | `e4/subsets/M1_f104_hmmwv_rigid.npz` | deploy, seeds 0-4, `--roundtrip-check` | 84,787 / 1,089 |
| `M1b_rigid_deploy` | `e5/train/M1_rigid` | same | deploy, seeds 5-9, `--roundtrip-check` | 84,787 / 1,089 |
| `M1_rigid_holdout` | `e5/train/offline_rigid` | same (= learning-curve point 1,089) | holdout, seeds 0-4 | 67,368 / 865 |
| `LC545_rigid_holdout` | `e5/train/offline_rigid` | `e4/subsets/LC545_f104_hmmwv_rigid.npz` | holdout, seeds 0-4 | 33,577 / 431 |
| `LC272_rigid_holdout` | `e5/train/offline_rigid` | `e4/subsets/LC272_f104_hmmwv_rigid.npz` | holdout, seeds 0-4 | 16,612 / 213 |

Written as e.g.
`python3.12 -u scripts/ci_train.py --ds $G3/e4/subsets/M1_f104_hmmwv_rigid.npz --mode deploy --arch gru --cond none
--domain-filter rigid --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 0 --roundtrip-check
--out $G3/e5/train/M1_rigid --tag M1a_rigid_deploy`.

Reading the holdout runs:
- Holdout mode leaves ga_train's dev fold (training groups with md5 % 5 == 0) out of the fit and scores it. The two
  learning-curve files keep all 224 dev-fold groups of f104, so every point of the curve is scored on the same rows:
  dev fold 224 groups (17,419 rows) and the val split 56 groups (4,358 rows). Fitted groups are therefore about 80 % of
  the nominal count (213 / 431 / 865 for 272 / 545 / 1,089).
- The LC files are for holdout mode only: in deploy mode they would also fit the dev fold.
- Per-arena or per-row read-outs later: the `_logits.npz` of each run carries the ids, so rows of other arenas can be
  scored post hoc once they are added as evaluation-only rows.

## 3. Watching and fetching

```bash
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
squeue -j 436135; tail -n 3 $G3/e5/logs/ag_train_rigid_f104_436135.out
grep -h "ENSEMBLE .* rigid startup\|exit:" $G3/e5/logs/*_436135.log
sacct -j 436135 --format=JobID,State,Elapsed,ExitCode
# local, after the job
rsync -a amd:$G3/e5/train/ $K3/e5/train/ && rsync -a amd:$G3/e5/logs/ $K3/e5/logs/
```

## 4. Not trained yet

- Soil models: their tier range depends on how far the new arenas' soil collection gets (PLAN 7.8: every model is
  trained on the same tier range as its comparison partner).
- Rigid M2, M3, A3, the leave-one-arena-out sets and task B's H and G: they need the g203 / g228 rigid runs (job
  436075, still waiting for mi2104x nodes at 02:50) or the Gator collection.

## 5. E5a rigid runs after the 05:50-10:50 shutdown (added 11:30; details in `$K3/NOTES_E5a.md`)

| job | what | job list | status |
|---|---|---|---|
| 436234 / 436235 | E5a TA1 / TA2 on mi3501x (2 lanes each, `tools/e5a/scripts/ag_train_lanes.sbatch`) | `$G3/e5/jobs/rigid_A_v1{a,b}.tsv` | TIMEOUT 09:37: M2 deploy, M2 / LOAO1_g203 / LOAO2_f104_g203 holdout done by 06:11; the M3 / A3 lines waited 2 h for subset files that were never written (session cut), then failed; the g228 leave-one-out lines never started |
| 436351 | smoke of `tools/e5a/scripts/ag_train_lanes_multi.sbatch` on an 8-GPU MI250X node (mi2508x), 2 lanes on 2 GPUs, 1 epoch | `$G3/e5/jobs/smoke_multi_mi250.tsv` | done 11:04, both exit 0, round trip exact |
| 436352 | A3 deploy, A3 holdout, M3a / M3b deploy, M3 holdout + LOAO2_g203_g228, LOAO1_g228 + LOAO2_f104_g228, G deploy, G holdout; one lane per GPU | `$G3/e5/jobs/rigid_AB_v2_mi250.tsv` | submitted 11:05 (every MI350X / MI300X node was taken by another user) |

```bash
sbatch -p mi2508x -c 128 -t 03:50:00 -J ag_train_rigid_AB_v2 --export=ALL,AG_JOBS=$G3/e5/jobs/rigid_AB_v2_mi250.tsv \
  $G3/tools/e5a/scripts/ag_train_lanes_multi.sbatch      # 436352 (submitted with env -u NEDM_VEHICLE)
```

Deploy ensembles are copied with their trainer summary and training logits to `$K3/e5/deploy/<model>/` with a
`SHA256SUMS` file (`scripts/ag_deploy_sync.sh <model> <train subdir> <tag>`, which also requires the cluster hashes to be
equal). H is the M1a ensemble (`$K3/e5/deploy/H/README.md`).
