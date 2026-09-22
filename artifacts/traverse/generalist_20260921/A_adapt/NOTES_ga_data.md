# Module note: mixed rigid/CRM re-anchored dataset with history windows, and the domain identifiability probe (PLAN A1, A1b)

Written 2026-09-21. Everything here was run on the local machine (no cluster jobs, no Chrono runs). Nothing under
`crm_night2_v1`, `crm_f104_v1` or `fdm_f104_50h_20260909` was modified.

## What was built

1. `scripts/ga_build_mixed.py` - builds `A_adapt/datasets/mixed_reanchor.npz` from the two night-2 re-anchored files
   (`crm_night2_v1/datasets/reanchor_rigid.npz`, `reanchor_crm.npz`) and the raw local episodes.
2. `scripts/ga_domain_probe.py` - the identifiability probes; writes `A_adapt/probe/probe.json`.

## The dataset (`A_adapt/datasets/mixed_reanchor.npz`, 2.10 GB compressed, 115,868 rows)

Rows are the rigid rows first (58,424, domain 0) then the CRM rows (57,444, domain 1), each block in the source file's
original order. Every original array is kept byte-identical (`X, ctx, E, T, id, episode, group, split, source, status,
profile, fail, unsafe, event_idx, route_len, anchor_frame, vx_anchor, rem_m, time_to_event_s`; `id` is the only one changed,
suffixed `@rigid` / `@crm`; `group` and `episode` unchanged). New arrays:

| key | shape / dtype | content |
|---|---|---|
| `domain` | (n,) int8 | 0 rigid, 1 crm |
| `hist` | (n,40,15) float16 | causal window ending at the anchor k: `hist[t] = [state[k-39+t][cols 0-6, 11-15], action[k-40+t]]`, t = 0..39; masked steps are 0 |
| `hmask` | (n,40) bool | True where the action frame `k-40+t >= 0`; k = 0 rows are all-masked; the state frame `k-39+t` is then >= 1 |
| `privileged` | (n,8) float32 | means over the state frames `max(0,k-39)..k` of tyre Fz fl/fr/rl/rr (state cols 7-10), motorshaft torque (col 16); for CRM the wheel-mean slip ratio and the wheel-mean spindle height above the BMP ground (`crm_extra.npz: spindle_z_m - bmp_ground_z_m`), zeros for rigid; last column `is_crm` |
| `hist_cols` | (12,) int16 | `[0,1,2,3,4,5,6,11,12,13,14,15]` |
| `priv_names` | (8,) str | column names of `privileged` |

A build summary with all counts is written next to it (`mixed_reanchor_build.json`).

Counts (domain | split | startup = k 0 / established = k >= 40):

| domain | train startup | train established | val startup | val established | test startup | test established |
|---|---|---|---|---|---|---|
| rigid | 13,629 | 39,391 | 702 | 2,027 | 693 | 1,982 |
| crm | 13,629 | 38,544 | 702 | 1,963 | 693 | 1,913 |

1,200 groups (1,089 train / 56 val / 55 test), each with one split, identical in both domains. 15,024 episodes per domain,
all found in the raw run dirs (CRM `crm_f104_v1/collect_v1/runs`; rigid 9,135 in `production_v3/runs` + 5,889 in
`production_v4/runs`): raw episode found for 115,868 / 115,868 rows = 100 %. (episode, anchor_frame) pairs present in
both worlds: 44,990, of which 29,966 established (the probe set). No id / group / episode matches
`f104_crm_eval_group_*`, `f104_g1_test_group_*` or `f104_pair_group_*` (asserted).

Choices that the contract left open, stated plainly:

- The window mask follows the action index: at k = 0 the anchor state exists but no previous action does, so the row is
  all-masked as the plan requires; the same rule gives partial windows for 0 < k < 40 (none occur here: all anchors are
  multiples of 40, so every established row has a full window).
- `privileged` for all-masked rows is the mean over the single frame k (the frame-0 state, whose columns 7-10 and 16
  are already in `ctx`); nothing new leaks at startup. For CRM the frame-0 slip ratio and spindle height are included.
- The "sinkage" column is the spindle height above the BMP ground surface as the plan spelled it (`spindle_z - bmp_ground_z`),
  so a lower value means a deeper wheel (CRM p1/50/99 = 0.34 / 0.49 / 0.77 m).
- The CRM slip ratio has heavy tails (window means p1/50/99 = -0.72 / 0.51 / 6.9; p99.9 = 18.5, max 75.9; 244 rows with
  |mean slip| > 10). It is stored raw as the plan says; a trainer that standardises it should clip or use a robust scale.
- Storage: `hist` float16 (max relative rounding error against the float32 raw data 4.8e-4 on 354 rows checked by hand).

Memory and time: metadata for both files is concatenated first; the history is cut by a process pool of 8 workers (one
task per episode per domain); the corridor tensor is preallocated once and filled from one source file at a time.
Full build 73 s; peak RSS 5.7 GB in the main process, 0.4 GB per worker.

## Probe results (`A_adapt/probe/probe.json`, `probe_run.log`)

Established probe: the 59,932 rows (29,966 anchors present in both worlds, k >= 40), split by group
(train / val / test = 54,408 / 2,794 / 2,730, 50 % CRM by construction). GRU(15 -> 32) + linear head on the standardised
window, Adam 2e-3, 20 epochs, batch 512, 3 seeds; ridge-logistic (IRLS, l2 = 1) baselines; tie-aware rank AUC.

| probe | val AUC | test AUC |
|---|---|---|
| established GRU(15->32) on the 2 s history, 3-seed mean | 1.000 | 1.000 |
| GRU, anchor vx < 1 m/s (n val/test 414/374, CRM share 0.55/0.55) | 0.999 | 1.000 |
| GRU, anchor 1-3 m/s (1,324/1,321, CRM share 0.56/0.55) | 1.000 | 1.000 |
| GRU, anchor vx > 3 m/s (1,056/1,035, CRM share 0.41/0.43) | 1.000 | 1.000 |
| established hand-feature logistic (mean vx, mean throttle, throttle/vx, mean spindle slip speed, yaw-rate variance) | 0.741 | 0.768 |
| hand-feature logistic by bin: vx < 1 / 1-3 / > 3 | 0.790 / 0.796 / 0.712 | 0.763 / 0.821 / 0.763 |
| startup joint logistic, frame-0 12-column state | 0.642 | 0.669 |
| startup single columns (best): pitch rate / engine speed / vy | 0.581 / 0.620 / 0.581 | 0.646 / 0.496 / 0.495 |

Attribution (`--ablate`, 1 seed, 10 epochs, GRU on channel subsets with the others zeroed; logistic on window statistics):

| channels available to the GRU | val AUC | test AUC |
|---|---|---|
| actions only (steer, throttle, brake) | 0.919 | 0.912 |
| body velocity only (vx, vy) | 0.920 | 0.940 |
| attitude only (roll, pitch) | 0.604 | 0.610 |
| angular rates only (roll, pitch, yaw rate) | 0.977 | 0.981 |
| spindle omegas only | 0.931 | 0.930 |
| engine speed only | 0.845 | 0.865 |
| vx only | 0.804 | 0.858 |
| 12 state columns, no actions | 1.000 | 1.000 |
| all but engine speed | 1.000 | 1.000 |
| all but spindle omegas | 1.000 | 1.000 |
| logistic on the last window step, 15 channels | 0.866 | 0.868 |
| logistic on per-channel window mean + std (30 features) | 0.982 | 0.978 |
| logistic on per-channel window std only (15) | 0.814 | 0.807 |

Reading: at a moving anchor the 2 s history identifies the world essentially perfectly, in every speed bin, and the
signal is redundant across channel groups (no single group is necessary; angular rates alone reach 0.98). Most of it
is amplitude-level (window mean + std logistic 0.98; largest standardised coefficients: mean vx -5.6, mean engine speed
+4.0, std vx -3.0, mean throttle +2.9, std of the front-left omega -2.4): the CRM vehicle runs at higher throttle and
engine speed for a lower and less variable vx, with less wheel-speed chatter. The temporal structure adds the rest
(1.000 vs 0.98). The hand-feature baseline at 0.74-0.77 shows the five summary features from the plan are far from
sufficient. The plan's stop rule (established AUC < 0.7) is not triggered. At startup (rest state after the settle) the
frame-0 state is nearly uninformative (joint 0.64-0.67; single columns at chance except a weak pitch-rate signal), as
expected; the all-masked window is 0.5 by definition.

Caveat a reviewer should weigh: the probe cannot tell physical soil response from collector/integrator differences
(CRM 1 ms physics with FSI contact vs rigid 2 ms with TMEASY tyres; a different steering-rate clamp in the first frames).
Both are real properties of the recordings the shared model will train on, so the identifiability holds for the risk
model either way, but the adaptation story should not be read as "soil response" alone until the A4 branch rows are
probed with the same script.

## How to run

```
cd /home/harry/NeDM-traverse_mppi
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/ga_build_mixed.py \
    --out artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz --workers 8
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/ga_domain_probe.py \
    --dataset artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz \
    --out artifacts/traverse/generalist_20260921/A_adapt/probe/probe.json --seeds 3 --epochs 20 --ablate
```
`ga_build_mixed.py` defaults to the night-2 files and the local run dirs; `--limit-episodes N` builds a small file for
smoke tests; `--no-compress` writes ~2x larger but faster. `ga_domain_probe.py` works on any npz with the same keys
(the A4 branch rows later); `--all-established` uses every established row instead of the coinciding anchors (adds the
selection confound between the two worlds' anchor choices); exit code 2 from the builder means missing raw episodes.

## Tests run (all under `A_adapt/selftest/` unless noted)

| command | checked | result |
|---|---|---|
| `ga_build_mixed.py --limit-episodes 50 --workers 4 --out selftest/mixed_small.npz` | end-to-end on 50 episodes per domain; internal assertions (uniqueness, blacklist, k = 0 all-masked, last window step == ctx anchor state) | 389 rows, 100 % raw found, 9 s |
| by-hand recompute on 60 random rows of the small file from `trajectory.npz` / `crm_extra.npz` | hist, hmask, privileged (Fz, torque, slip, spindle height, is_crm), id suffix | all equal; max relative f16 error 4.7e-4 |
| full build (`build_full.log`) | 115,868 rows; assertions; raw found 100 %; counts table above | 73 s, peak RSS 5.7 GB main / 0.4 GB worker, 2.10 GB file |
| `verify_full.log`: 200 random rows of the full file recomputed by hand; one split per group; blacklist; uniqueness; finiteness; hist means per domain | as listed | all pass; groups 1,089/56/55 |
| `verify_preserved.log`: every original array compared to the two source files on all 115,868 rows (NaN-aware), X on 2,000 sampled rows | "all original arrays kept", row order = source order | identical |
| `ga_domain_probe.py ... --seeds 3 --epochs 20 --ablate` (`probe/probe_run.log`) | probe (a) and (b) + ablations | 80 s on the 5090; numbers above |

## Known limits

- Only anchors at multiples of 40 frames exist, so partial windows (0 < k < 40) are untested on real rows (the code
  path is exercised by the k = 0 rows and the mask rule; the invariant `hmask.sum() == min(k, 40)` is asserted).
- `production_v2` (night-1 rigid episodes, no CRM twin) is not included, as in the night-2 re-anchored file.
- The probe's speed bins are by each row's own anchor speed, so the bins are not domain-balanced (CRM share 0.41-0.56);
  AUC within a bin is still a valid ranking statistic.
- The GRU probe reports the final-epoch model; the best-val epoch and its test AUC are in the json (best epochs 16-18,
  identical numbers).
