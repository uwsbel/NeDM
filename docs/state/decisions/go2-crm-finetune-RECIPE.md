# CRM fine-tuning: the recipe that works

**Status: reproducible, verified in Chrono, 2026-09-10.** This is the first configuration
in this project to beat the base policy on CRM. Everything needed to reproduce it is
pinned below. If only one document survives from this line of work, it should be this one.

## Result

`w_h15r0` against `go2_cts_150k`, paired on 74 CRM episodes, same machine, same episode
list, scored in **real Chrono SPH** (`score_crm_tracking.py` shells out to
`collect_go2_smoke.py --terrain crm`, which imports `pychrono.fsi`; the surrogate appears
nowhere in the scoring path).

| axis | BASE | w_h15r0 | change | episodes won |
|---|---|---|---|---|
| `mae_vx` | 0.1574 | 0.0937 | **-40.5%** | 67/74 |
| `mae_wz` | 0.1030 | 0.0820 | -20.4% | 43/74 |
| `mae_vy` | 0.0575 | 0.0504 | -12.3% | 37/74 |

Paired t = -11.12, exact sign test p = 2.1e-13, sign-flip permutation p < 1e-5.
`completed` 1.000 for both: no falls. Mean achieved |vx| 0.5268 -> 0.6156 against a
commanded 0.6549, i.e. 80% -> 94% of commanded speed.

Wins in **every** command bin and in 7 of 8 command families. The two families where
forward velocity is not the point are the strongest wins on their own axis: **pivot**
halves yaw error (0.219 -> 0.112, 10/10) and **lateral** cuts sideways error 39%
(0.155 -> 0.095, 9/9). Known trade: on lateral episodes `mae_vx` worsens slightly
(0.033 -> 0.039) and on pure-forward families `mae_vy` drifts up a little. The policy
pushes harder along the commanded axis and is marginally less tidy on the near-zero axes.

## Exact provenance

| | |
|---|---|
| repo commit | `0df6995` (branch `kyle/locomotion`) |
| base policy | `sbel-artifacts/checkpoints/go2_cts_150k.pt` sha256 `3e102d0d5e69aebd…` |
| surrogate | `training_runs/go2_crm_baseline_s1/checkpoints/best_val.pt` sha256 `d5c57edc8b7ea870…` |
| corpus | `datasets/go2_crm_merged`, 795 episodes |
| scoring set | `datasets/go2_crm_merged/score_subset_index.json`, 80 val episodes |
| output policy | sha256 `20c084424570341b…` |
| machine | any CUDA box; trained on sbel, scored on north |

## The command

```bash
cd /home/kyle/Documents/sbel/NeDM
export PYTHONPATH=$PWD/src NEDM_REPO=$PWD

python scripts/training/finetune_go2_shortbranch_upstream_reward.py \
    --surrogate /home/kyle/sbel-artifacts/training_runs/go2_crm_baseline_s1/checkpoints/best_val.pt \
    --policy    /home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt \
    --root      /home/kyle/sbel-artifacts/datasets/go2_crm_merged \
    --out       /home/kyle/sbel-artifacts/finetune_crm_w_h15r0 \
    --seed 0 --branch-steps 15 --reg-scale 0 \
    --target-dw 1.0 --height-target 0.58 --min-upright 0.7

python scripts/evaluation/export_finetuned_policy.py \
    --base /home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt \
    --ckpt /home/kyle/sbel-artifacts/finetune_crm_w_h15r0/best.pt \
    --out  /home/kyle/sbel-artifacts/finetune_crm_w_h15r0/policy_ts.pt

# verdict, ~90 min for 80 episodes at concurrency 4
python scripts/evaluation/score_crm_tracking.py \
    --policy /home/kyle/sbel-artifacts/finetune_crm_w_h15r0/policy_ts.pt \
    --index  /home/kyle/sbel-artifacts/datasets/go2_crm_merged/score_subset_index.json \
    --split val --out crmtrack_w_h15r0_$(hostname -s).json --concurrency 4
```

Run terminates at **update 94**, `||dW|| 1.002`, surrogate val `-1.438482`. Takes about a
minute on a 3090. The Chrono verdict is the expensive part, not the fine-tune.

## Why each flag is there

- `--branch-steps 15` (0.30 s). Covers a trot cycle (0.30-0.50 s); the CRM surrogate beats
  predict-zero-delta to ~0.50 s on velocity. **This exact horizon was the WORST arm in the
  project before the fixes** (`h15_dw4`, +0.735 m/s, 1/37 episodes won). Same horizon,
  correct forward pass, no `dof_acc`: best arm. Horizon was never the problem.
- `--reg-scale 0`. **The necessary ingredient.** Zeroes the negative-weight reward terms in
  the gradient. Measured at the base policy: `dof_acc` takes 47% of gradient norm at a
  5-step branch and 63% at 15, against 6.7% of the reward's value, because it is a squared
  finite difference over dt so its path through the surrogate is stiffer by (1/0.02)^2.
  Backprop weights a term by path stiffness, not by contribution to return. The ablation
  is causal: with penalties on, the policy gets **worse** (`w_r1`, +0.0546, reproduced on
  two machines to 4 decimal places).
- `--target-dw 1.0`. Stop at a fixed weight displacement. Not a good trust region (see
  Known weaknesses) but it is what these numbers were produced under. Do not change it
  when reproducing.
- `--height-target 0.58`. Upstream's `correct_base_height` target is a FIXED 0.38 above
  ground (`go2_config.py`); the soil top here is 0.20. Passing the base policy's own mean
  (0.5491, as every arm before 2026-09-10 did) turns a restoring force into a pure
  variance penalty.
- `--min-upright 0.7`. Rejects branch starts with the robot tilted past ~45 deg. A no-op on
  CRM (100% of rows accepted) but not on flat, where 9.9% of starts begin fully inverted.
- **No `--freeze-encoder`.** Tested and it HURTS: `w_r0fz` -0.0258 vs `w_r0` -0.0412, and
  yaw degrades to 142% of base instead of 95%. The argument for freezing (the encoder is
  55% of parameters, absorbed 57% of displacement, and is distillation-trained on a teacher
  latent nothing here preserves) sounded good and was wrong. The encoder needs to adapt.

## The ablation this rests on

| arm | branch | penalties | encoder | delta mae_vx | yaw vs base |
|---|---|---|---|---|---|
| `w_r1` | 0.10 s | **on** | trainable | **+0.0546 worse** | 286-318% |
| `w_r0fz` | 0.10 s | off | frozen | -0.0258 | 142% |
| `w_r0` | 0.10 s | off | trainable | -0.0412 | 95% |
| **`w_h15r0`** | **0.30 s** | **off** | **trainable** | **-0.0637** | **80%** |

Monotone on both columns. `w_r1` was scored independently on two machines: +0.0546 and
+0.0548, identical 26/75 win counts. That cross-build agreement is the strongest evidence
that the harness resolves effects far smaller than the ones being claimed.

## Prerequisites: four bugs that had to be fixed first

None of this works on a pipeline without these. All in commit `491e7fe` and `0df6995`.

1. **The branch fed the surrogate a duplicated state token that occurs nowhere in training.**
   `newa` is built to REPLACE the last action and was APPENDED, so the action ran a step
   ahead of the state and a duplicate state was inserted to compensate. One-step, driving
   recorded actions: `vel_body_x_mps` nRMSE 0.119 -> 0.684, **5.75x**, while joint channels
   moved 1.15x. Velocity is what the tracking terms read.
2. **The previous-action observation channel was invented** -- seeded to zeros and self-fed
   through the warm-up while the true value sat in the `policy_raw_*` columns. First branch
   action RMS error 1.6455 against a signal std of 1.539.
3. **Control-row parity was hardcoded to ODD.** It is a per-episode property; 16 of 30 CRM
   episodes are EVEN. Now read off the data.
4. **`--target-dw` performed no checkpoint selection at all**, writing `NaN` into `best.pt`
   and overwriting whatever validation had chosen.

## Known weaknesses of this recipe

Recorded so nobody has to rediscover them, and so the result is not oversold.

- **It optimises a modified objective.** `--reg-scale 0` deletes reward terms because our
  estimator mishandles them. It works, but it is a workaround for a defect in the method,
  not a fix to it. A correct estimator should not need it.
- **`||dW||` is not a trust region.** Going from 0.52 to 4.0 is an 8x displacement and moves
  behaviour only 15% -> 39%, and under Adam it mostly counts update steps
  (lr*sqrt(P) = 1e-4*679 = 0.068/update, matching the observed stops). A KL bound on the
  action distribution over visited states would be the right control.
- **Model exploitation is unquantified here.** The policy is optimised against a learned
  model and nothing bounds how far it drifts from the data that model was fitted on. This
  arm stops at `||dW|| 1.0`, which is small, and it transfers -- but that is evidence, not
  a guarantee, and larger displacements are untested at this configuration.
- **The 0.50 s horizon is unreachable on this path.** Analytic backprop retains the graph
  across `branch_steps * DECIM` surrogate forwards, so memory grows with horizon and 25
  steps OOMs a 24 GB card at batch 64. Horizon helped monotonically up to 0.30 s, so this
  is a real ceiling. PPO retains no graph and runs 25 steps at batch 128 comfortably.
- **Run-to-run noise: MEASURED, and it does not threaten the result.** Re-running the
  same policy on the same box, BASE reproduces bit-exactly 7/7 episodes while `w_h15r0`
  reproduces only 4/7 -- the granular solver's atomic accumulation is non-associative.
  Propagated to the 74-episode mean: same-box SE **+/-0.00037 m/s**, so the -0.0637 effect
  is **172x the noise**, 95% interval **-40.0% to -40.9%**. Cross-box (a different machine
  entirely): SE +/-0.00161, still 40x, interval **-38.5% to -42.5%**. The divergence is
  symmetric, not directional (every replicate |z| < 2), so the run mean is unbiased, and
  only 1 of 74 episodes has an effect smaller than the run-to-run noise. Quote the interval
  rather than the bare point estimate, and prefer the cross-box interval when comparing
  numbers produced on different machines.

- **Reproducibility is ARM-DEPENDENT and must be measured per arm, not assumed.** `w_r1`
  reproduces only 28/75 episodes across the same box pair where BASE reproduces 62/77,
  with sd 0.120 and a **maximum single-episode disagreement of 0.948 m/s**. Its *mean* is
  stable (+0.0546 on a3 vs +0.0548 on sbel) because the noise averages out over 75
  episodes, but a single-run per-episode number from an arm like that means little. Before
  trusting any new arm's score -- especially at larger displacements -- measure its own
  reproducibility rather than inheriting `w_h15r0`'s.

- **Model exploitation: TESTED, and the evidence contradicts it.** `w_h15r0` does walk off
  the training distribution (8.72% of visited states beyond the corpus p99, against BASE's
  0.97%), and the surrogate is correspondingly 1.8-2.1x less accurate there, surviving a
  speed-confound control. But that degradation is a property of WHERE THE POLICY GOES, not
  of the model it was optimised against:
    - `w_h15r0` was fitted against `go2_crm_baseline_s1` alone, so s2/s3 are held out. At
      h=15 the error ratio is **1.82 for s1, 2.02 for s2, 2.11 for s3** -- the two models
      the optimiser never touched are slightly WORSE. Difference-in-differences is negative;
      95% upper bound puts at most ~10% of the degradation on s1 specifically.
    - The error has no reward-direction. `|true-cmd| - |pred-cmd|` at h=50 is **-0.087 m/s**
      for `w_h15r0` against -0.018 for BASE: the surrogate OVER-states how badly the
      fine-tuned policy tracks. The model is pessimistic about this policy, not optimistic.
    - Ensemble disagreement rises in lockstep (2.23x, 16/16 episodes), i.e. the ensemble
      correctly signals its own ignorance in that region.
  This is ordinary epistemic uncertainty, honestly signalled -- not exploitable error the
  optimiser sought out. Note the test was STACKED toward finding exploitation: the CRM
  corpus was collected with the base policy itself, so the BASE arm is on-manifold by
  construction. Full tables at `north-ubuntu:/home/kyle/exploitation_audit/`.

- **Single seed.** Everything above is `--seed 0`. Seed variance on this pipeline has been
  large enough to matter before and has not been measured for this configuration.
